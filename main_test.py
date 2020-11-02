from typing import Dict

from hyperopt.pyll import scope
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from PartnerChoiceEnv import PartnerChoiceFakeSites
import numpy as np
import ray
from ray import tune
from ray.tune.logger import TBXLogger

from gym.spaces import Box, Discrete
from ray.tune.registry import register_env


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data["inv"] = []
        episode.user_data["accept"] = []
        episode.hist_data["inv"] = []
        episode.hist_data["accept"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        info = episode.last_info_for("inv00")
        if info and "inv" in info:
            inv = info["inv"]
            accept = info["accept"]
            other = info["other"]
            if accept:
                episode.user_data["accept"].append(other)
            episode.user_data["inv"].append(inv)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        episode.hist_data["inv"] = episode.user_data["inv"]
        episode.hist_data["accept"] = episode.user_data["accept"]


if __name__ == "__main__":
    ray.init(local_mode=True)
    nb_agents = 1
    inv_id = ['inv' + '{:02d}'.format(i) for i in range(nb_agents)]
    choice_id = [f'choice{i:02d}' for i in range(nb_agents)]

    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))

    choice_act_space = Discrete(2)
    choice_obs_space = Box(np.array([0, 0], dtype=np.float32), np.array([15, 15], dtype=np.float32))
    inv_act_space = Box(np.array([0], dtype=np.float32), np.array([15], dtype=np.float32))
    inv_obs_space = Box(np.array([0], dtype=np.float32), np.array([1], np.float32))

    choicemodel_dict = {
        "fcnet_hiddens": [3],
    }

    investormodel_dict = {
        "fcnet_hiddens": []
    }

    policies = {inv_id[i]: (None, inv_obs_space, inv_act_space, investormodel_dict) for i in range(nb_agents)}
    policies.update(
        {choice_id[i]: (None, choice_obs_space, choice_act_space, choicemodel_dict) for i in range(nb_agents)})


    def select_policy(agent_id):
        return agent_id


    def get_it_from_prob(spec):
        bad_prob = spec['config']['env_config']['bad_site_prob']
        base_it = 100
        if bad_prob == 0:
            return base_it
        else:
            return np.round(1 / (1 - bad_prob)) * base_it


    config = {
        "num_gpus": 0,
        'num_workers': 0,
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
                "bad_site_prob": 0,
                "max_it": 100
            }
    }
    space = {
        "lr": hp.loguniform("lr", np.log(1e-7), np.log(0.1)),
        "horizon": scope.int(hp.quniform("horizon", 100, 5000, q=1)),
        "num_sgd_iter": scope.int(hp.quniform("num_sgd_iter", 3, 30, q=1)),
        "train_batch_size": scope.int(hp.quniform("train_batch_size", 1024, 4000, q=1)),
        "sgd_minibatch_size": scope.int(hp.quniform("sgd_minibatch_size", 16, 1024, q=1)),
    }
    hyperopt_search = HyperOptSearch(space, metric="episode_reward_mean", mode="max")
    hyperband = ASHAScheduler(metric="episode_reward_mean", mode="max", grace_period=10)

    analysis = tune.run(
        "PPO",
        name="gridsearch",
        stop={
            "training_iteration": 1000,
        },
        config=config,
        loggers=[TBXLogger], checkpoint_at_end=True, local_dir="./logs/",
        search_alg=hyperopt_search,
        scheduler=hyperband,
        num_samples=1000,
        verbose=1
    )

    analysis.trial_dataframes.to_pickle("./res.df.pkl")
