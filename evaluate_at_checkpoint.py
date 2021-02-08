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

policies = init_setup(256, 2)

config = {
    "num_envs_per_worker": 1,
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
def bench_score(path):
    if "cma" in path:
        agent = loadcma(path)
    else:
        agent = loadppo(path)
    config["env_config"]["good_site_prob"] = 1
    config["env_config"]["max_it"] = 100
    config["env_config"]["eval_mode"] = False
    env = PartnerChoiceFakeSites(config["env_config"])
    logs = []
    rewardslog = []
    for ep in range(1000):
        obs = env.reset()
        done = {"__all__": False}
        totreward = 0
        while not done["__all__"]:
            act = {}
            for key in obs:
                act[key] = agent.get_policy(key).compute_actions(obs[key].reshape(1, -1))[0].flatten()[0]
            obs, reward, done, info = env.step(act)
            if "inv00" in reward:
                totreward = reward["inv00"]
            if "inv00" in info:
                assert (isinstance(info["inv00"]["other"], float))
                logs.append(copy(info["inv00"]))
        rewardslog.append({"ep": ep, "reward": totreward})
    df = pd.DataFrame(rewardslog)
    df_logs = pd.DataFrame(logs)
    return df, df_logs


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
    i = 0
    rangeparams = {"choice00": range(4, 34), "inv00": range(0, 4)}
    for key, params in config["multiagent"]["policies"].items():
        agent[key] = CMAESTorchPolicy(*params[1:])
        agent[key].set_flat_weights(bests[rangeparams[key]])
        i += agent[key].num_params
    return agent


def get_highest(vals: List[str], *, pattern: Pattern = ""):
    if vals is None or len(vals) == 0:
        return None
    m = np.argmax([float(re.search(pattern, val).group("target")) for val in vals])
    return vals[m]


def tie(gen):
    for i, elem in enumerate(gen):
        print(i, elem)
        yield elem


if __name__ == "__main__":
    analysis_mode = False
    ray.init(local_mode=True)
    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))
    conds = [("ppo_mlp", False), ("ppo_mlp", True),
             ("ppo_deep", False), ("ppo_deep", True),
             ("cmaes", False), ("cmaes", True)]
    for cond, analysis_mode in conds:
        main_path = Path(f"logs/paperrun/{cond}/")
        glob_path = main_path
        alldfs = []
        alldfs_logs = []
        with timer("glob"):
            print("getting all path")
            if "cma" in str(main_path):
                print("cma style")
                allpaths = list(main_path.glob("*/*/"))
                print("cma done")
            else:
                allpaths = list(main_path.rglob("**/*"))
            print("done")

        for path in tqdm.tqdm(allpaths):
            res = re.search(r"partner_choice_(?P<trialid>.*)_(?P<runid>\d+).*_good_site_prob=(?P<prob>[0-9.]*),", str(path))
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
               #print("no checkpoint for", path)
                continue
            try:
                if analysis_mode:
                    df = bench(str(checkpoint_path))
                else:
                    df, df_logs = bench_score(str(checkpoint_path))
            except Exception as e:
                print(checkpoint_path)
                print(type(e), e)
            else:
                df["run_id"] = run_id
                df["trial_id"] = trial_id
                df["good_site_prob"] = df["p"] = good_site_prob
                df["checkpoint_path"] = checkpoint_path
                alldfs.append(df)
                if not analysis_mode:
                    df_logs["run_id"] = run_id
                    df_logs["trial_id"] = trial_id
                    df_logs["good_site_prob"] = df["p"] = good_site_prob
                    df_logs["checkpoint_path"] = checkpoint_path
                    alldfs_logs.append(df_logs)
        with timer("saving"):
            fuldf = pd.concat(alldfs)
            if analysis_mode:
                fuldf.to_csv(main_path / "postmortem.csv.gz")
            else:
                fuldf_logs = pd.concat(alldfs_logs)
                fuldf.to_csv(main_path / "evalatend.csv.gz")
                fuldf_logs.to_csv(main_path / "evalatend_log.csv.gz")
