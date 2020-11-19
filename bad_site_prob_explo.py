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

from PartnerChoiceEnv import PartnerChoiceFakeSites
from main_test import MyCallbacks, get_it_from_prob, select_policy, init_setup

if __name__ == "__main__":
    ray.init(num_cpus=24)
    policies = init_setup()

    config = {
        "num_envs_per_worker": 16,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        "clip_actions": True,
        "framework": "torch",
        "no_done_at_end": True,
        "gamma": 1,
        "callbacks": MyCallbacks,
        "env": "partner_choice",
        "env_config":
            {
                #"good_site_prob": tune.grid_search([1, 0.5, 0.3, 0.2, 0.1]),
                "good_site_prob": tune.grid_search([0.3, 0.2, 0.1]),
                "max_it": tune.sample_from(get_it_from_prob)
            }
    }

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    analysis = tune.run(
        "PPO",
        name="goodsiteprob_" + date_str,
        stop={
            "episodes_total": 200000
        },
        config=config,
        loggers=[TBXLogger], checkpoint_at_end=True, local_dir="./logs/paperrun/ppo/",
        num_samples=24,
        verbose=1
    )
    print("ending")
    analysis.trial_dataframes.to_pickle(f"./good_site_res.df.{date_str}.pkl")
