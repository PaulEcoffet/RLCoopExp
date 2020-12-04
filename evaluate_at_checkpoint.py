import glob
import logging
import re
from copy import copy
from pathlib import Path
from typing import List
from typing.re import Pattern
import numpy as np
import pickle
import tqdm

from ray.tune import register_env
from ray.tune.logger import pretty_print
from timer import timer
from PartnerChoiceEnv import PartnerChoiceFakeSites
import ray
from cma_test import CMAESTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
import pandas as pd
from main_test import init_setup, select_policy, MyCallbacks


logging.basicConfig(level=logging.DEBUG)

policies = init_setup()

config = {
    "num_envs_per_worker": 16,
    "num_workers": 0,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": select_policy,
    },
    "clip_actions": True,
    "framework": "torch",
    "no_done_at_end": True,
    "gamma": 1,
    "lr": 5e-3,
    "num_sgd_iter": 10,
    "callbacks": MyCallbacks,
    "env": "partner_choice",
    "env_config":
        {
            "good_site_prob": 1,
            "max_it": 100
        }
}


@timer
def bench(path):
    if "cma" in path:
        agent = loadcma(path)
    else:
        agent = loadppo(path)
    config["env_config"]["eval_mode"] = True
    config["env_config"]["good_site_prob"] = 1
    config["env_config"]["max_it"] = 100
    env = PartnerChoiceFakeSites(config["env_config"])
    logs = []
    for i_opp in range(env.nb_sites):
        obs = env.reset(force_cur_opp=i_opp)
        done = {"__all__": False}
        while not done["__all__"]:
            act = {}
            for key in obs:
                act[key] = agent.get_policy(key).compute_actions(obs[key].reshape(1, -1))[0].flatten()[0]
            obs, reward, done, info = env.step(act)
            if "inv00" in info:
                assert (isinstance(info["inv00"]["other"], float))
                logs.append(copy(info["inv00"]))
    df = pd.DataFrame(logs)
    return df


@timer
def loadppo(path):
    agent = PPOTrainer(config)
    agent.load_checkpoint(path)
    return agent

@timer
def loadcma(path):
    # hackish lookalike
    class FakeAgentDict(dict):
        def get_policy(self, policy):
            return self[policy]
    agent = FakeAgentDict()
    with open(path, "rb") as f:
        bests = pickle.load(f)
    for key, params in config["multiagent"]["policies"].items():
        agent[key] = CMAESTorchPolicy(*params[1:])
        agent[key].set_flat_weights(bests[key])
    return agent


def get_highest(vals: List[str], *, pattern: Pattern = ""):
    if vals is None or len(vals) == 0:
        return None
    m = np.argmax([float(re.search(pattern, val).group("target")) for val in vals])
    return vals[m]


if __name__ == "__main__":
    ray.init(local_mode=True)
    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))

    main_path = Path("/Users/paulecoffet/Documents/isir/These/data/RLCoopExp/logs/paperrun2/e1000000/ppobiglr/")
    glob_path = main_path
    alldfs = []
    with timer("glob"):
        allpaths = list(main_path.rglob("**/*"))
    for path in tqdm.tqdm(allpaths):
        res = re.search(r"partner_choice_(?P<trialid>.*)_(?P<runid>\d+)_good_site_prob=(?P<prob>[0-9.]*),", str(path))
        if not res:
            continue
        run_id = res.group("runid")
        trial_id = res.group("trialid")
        good_site_prob = res.group("prob")
        if "cma" in str(path):
            checkpoint_path = path / "checkpoint200000/best.pkl"
        else:
            checkpoint_path = get_highest([str(c) for c in path.glob("checkpoint*/*")
                                       if "tune_metadata" not in str(c) and ".is_check" not in str(c)],
                                      pattern=r"checkpoint[-_]?(?P<target>[-0-9]+)/")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print("no checkpoint for", path)
            continue
        try:
            df = bench(str(checkpoint_path))
        except Exception as e:
            print(checkpoint_path)
            print(type(e), e)
        else:
            df["run_id"] = run_id
            df["trial_id"] = trial_id
            df["good_site_prob"] = df["p"] = good_site_prob
            df["checkpoint_path"] = checkpoint_path
            alldfs.append(df)
    with timer("saving"):
        fuldf = pd.concat(alldfs)
        fuldf.to_csv(main_path / "postmortem.csv.gz")
