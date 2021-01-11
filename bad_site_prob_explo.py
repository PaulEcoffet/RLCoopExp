from datetime import datetime
from typing import Dict

import torch
import multiprocessing
import numpy as np
import ray
from gym.spaces import Box, Discrete
from hyperopt import hp
from hyperopt.pyll import scope
from ray import tune
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.tune.logger import TBXLogger
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from pathlib import Path

from PartnerChoiceEnv import PartnerChoiceFakeSites
from main_test import MyCallbacks, get_it_from_prob, select_policy, init_setup

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episode", type=int, default=200000)
    parser.add_argument("goodprob", type=float, nargs="*", default=[1])
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--num-per-layers", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="paperrun3/")
    parser.add_argument("--subdir", type=str, default="")

    outparse = parser.parse_args()
    if outparse.local:
        ray.init(local_mode=True, num_cpus=24)
    else:
        ray.init(num_cpus=24)

    policies = init_setup(outparse.num_per_layers, outparse.num_layers)

    config = {
        "num_envs_per_worker": 16,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        "clip_actions": True,
        "framework": "torch",
        "no_done_at_end": True,
        "gamma": tune.grid_search([outparse.gamma]),
        "lr": 5e-3,
        "num_sgd_iter": 10,
        "callbacks": MyCallbacks,
        "env": "partner_choice",
        #"num_cpus_per_worker": 0,
        "num_workers": 6,
        "env_config":
            {
                "good_site_prob": tune.grid_search(outparse.goodprob),
                "max_it": tune.sample_from(get_it_from_prob)
            },
        #"sgd_minibatch_size": tune.sample_from(get_it_from_prob),
    }

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path("logs") / outparse.outdir / outparse.subdir
    totdata = []
    for k, elem in sorted(vars(outparse).items()):
        if k not in ["outdir", "subdir", "goodprob", "local"]:
            totdata.append(str(k) + "_" + str(elem))
    logdir /= "+".join(totdata)

    analysis = tune.run(
        "PPO",
        name="goodsiteprob_" + date_str,
        stop={
            "episodes_total": outparse.episode
        },
        config=config,
        loggers=[TBXLogger], checkpoint_at_end=True, local_dir=logdir,
        num_samples=24,
        verbose=1
    )
    print("ending")
    analysis.trial_dataframes.to_pickle(f"./good_site_res.df.{date_str}.pkl")
