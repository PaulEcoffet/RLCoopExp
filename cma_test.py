from collections import defaultdict
from datetime import datetime

import cma
import ray
from gym.spaces import Discrete, Box
from ray.rllib.evaluation import collect_metrics
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import build_torch_policy
from ray.rllib.utils.filter import get_filter
from ray.tune import register_env, tune
import ray.rllib.agents.es.es
import numpy as np
from ray.tune.logger import TBXLogger
import torch

from PartnerChoiceEnv import PartnerChoiceFakeSites
from main_test import select_policy


def before_init(policy, observation_space, action_space, config):
    policy.preprocessor = ModelCatalog.get_preprocessor_for_space(
        observation_space)
    policy.observation_filter = get_filter(config["observation_filter"],
                                           policy.preprocessor.shape)
    policy.single_threaded = config.get("single_threaded", False)

    def _set_flat_weights(policy, theta):
        pos = 0
        theta_dict = policy.model.state_dict()
        new_theta_dict = {}

        for k in sorted(theta_dict.keys()):
            shape = policy.param_shapes[k]
            num_params = int(np.prod(shape))
            new_theta_dict[k] = torch.from_numpy(
                np.reshape(theta[pos:pos + num_params], shape))
            pos += num_params
        policy.model.load_state_dict(new_theta_dict)

    def _get_flat_weights(policy):
        # Get the parameter tensors.
        theta_dict = policy.model.state_dict()
        # Flatten it into a single np.ndarray.
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1, )))
        cat = torch.cat(theta_list, dim=0)
        return cat.numpy()

    type(policy).set_flat_weights = _set_flat_weights
    type(policy).get_flat_weights = _get_flat_weights


def after_init(policy, observation_space, action_space, config):
    state_dict = policy.model.state_dict()
    policy.param_shapes = {
        k: tuple(state_dict[k].size())
        for k in sorted(state_dict.keys())
    }
    policy.num_params = sum(np.prod(s) for s in policy.param_shapes.values())


def make_model_and_action_dist(policy, observation_space, action_space,
                               config):
    # Policy network.
    dist_class, dist_dim = ModelCatalog.get_action_dist(
        action_space,
        config["model"],  # model_options
        dist_type="deterministic",
        framework="torch")
    model = ModelCatalog.get_model_v2(
        policy.preprocessor.observation_space,
        action_space,
        num_outputs=dist_dim,
        model_config=config["model"],
        framework="torch")
    # Make all model params not require any gradients.
    for p in model.parameters():
        p.requires_grad = False
    return model, dist_class


CMAESTorchPolicy = build_torch_policy(
    name="CMAESTorchPolicy",
    loss_fn=None,
    get_default_config=lambda: ray.rllib.agents.es.es.DEFAULT_CONFIG,
    before_init=before_init,
    after_init=after_init,
    make_model_and_action_dist=make_model_and_action_dist)


register_env("partner_choice",
             lambda config: PartnerChoiceFakeSites(config))


def train(config, reporter):
    env = PartnerChoiceFakeSites(config['env_config'])
    policies = {}
    es = {}
    solutions = {}
    tell = {}
    counter = {}
    best = {}
    for key, params in config["multiagent"]["policies"].items():
        policies[key] = CMAESTorchPolicy(*params[1:])
        es[key] = cma.CMAEvolutionStrategy(np.zeros(policies[key].num_params), 1)
        solutions[key] = es[key].ask()
        tell[key] = np.zeros(len(solutions[key]))
        counter[key] = 0
        best[key] = solutions[key][0]

    timestep_total = 0
    for i_episode in range(100000):
        # set the solutions
        for key in policies:
            policies[key].set_flat_weights(solutions[key][counter[key]])
        # test env
        obs = env.reset()
        done = {"__all__": False}
        totrewards = defaultdict(lambda: 0)
        while not done["__all__"]:
            timestep_total += 1
            act = {}
            for key in obs:
                act[key] = policies[key].compute_actions([obs[key]])[0][0]
            obs, reward, done, info = env.step(act)
            for key in reward:
                totrewards[key] += reward[key]
        #print(totrewards.items())

        should_evaluate = False
        for key in policies:
            tell[key][counter[key]] = totrewards[key]
            counter[key] += 1
            if counter[key] == len(solutions[key]):
                # print("*" * 30)
                # print("episode", _)
                # print("new batch for", key)
                # print("best score was", max(tell[key]))
                # print("mean score was", np.mean(tell[key]))
                # print("pop size for", key, "is", len(solutions[key]))
                # print("genome size for", key, "is", len(solutions[key][0]))
                best.update({key: solutions[key][np.argmax(tell[key])]})
                es[key].tell(solutions[key], [-x for x in tell[key]])
                solutions[key] = es[key].ask()
                tell[key] = np.zeros(len(solutions[key]))
                counter[key] = 0

        if i_episode % 100 == 0 and i_episode != 0:
            reward_through_eval = []
            inv_through_eval = []
            accept_through_eval = []
            stepcount_through_eval = []
            for _ in range(10):
                # set the solutions
                for key in policies:
                    policies[key].set_flat_weights(best[key])
                # test env
                obs = env.reset()
                stepcount = 0
                done = {"__all__": False}
                totrewards = defaultdict(lambda: 0)
                while not done["__all__"]:
                    stepcount +=1
                    act = {}
                    for key in obs:
                        act[key] = policies[key].compute_actions([obs[key]])[0][0]
                    if "inv00" in act:
                        inv_through_eval.append(act["inv00"])
                    obs, reward, done, info = env.step(act)
                    true_info = info.get("inv00", None)
                    if true_info and "accept" in true_info:
                        inv = true_info["inv"]
                        accept = true_info["accept"]
                        other = true_info["other"]
                        if accept and inv >= other:
                            accept_through_eval.append(other)
                    for key in reward:
                        totrewards[key] += reward[key]
                reward_through_eval.append(totrewards["inv00"])
                stepcount_through_eval.append(stepcount)

            reporter(
                episodes_total=i_episode,
                timesteps_total=timestep_total,
                episode_reward_max=np.max(reward_through_eval),
                episode_reward_min=np.min(reward_through_eval),
                episode_reward_mean=np.mean(reward_through_eval),
                episode_len_mean=np.mean(stepcount_through_eval),
                custom_metrics={"inv": np.mean(inv_through_eval), "accept": np.mean(accept_through_eval),
                                "good_site_prob": env.good_site_prob},
                hist_stats=dict(inv=inv_through_eval, accept=accept_through_eval)
                )


if __name__ == "__main__":
    ray.init(cpu=24)
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
        "model": {
            "fcnet_hiddens": [3],
            "max_seq_len": 999999
        },
        "use_critic": False
    }

    investormodel_dict = {
        "model": {
            "fcnet_hiddens": [],
            "max_seq_len": 999999

        },
        "use_critic": False
    }

    policies = {inv_id[i]: (None, inv_obs_space, inv_act_space, investormodel_dict) for i in range(nb_agents)}
    policies.update(
        {choice_id[i]: (None, choice_obs_space, choice_act_space, choicemodel_dict) for i in range(nb_agents)})


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
        #"callbacks": MyCallbacks,
        "env": "partner_choice",
        "env_config":
            {
                "good_site_prob": tune.grid_search([1, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001]),
                "max_it": 10000
            }
    }

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    analysis = tune.run(
        train,
        name="goodsiteprob_" + date_str,
        stop={
            "episodes_total": 100_000
        },
        config=config,
        loggers=[TBXLogger], checkpoint_at_end=True, local_dir="./logs/paperrun/cma/",
        num_samples=1,
        verbose=3
    )
    print("ending")
    #analysis.trial_dataframes.to_pickle(f"./good_site_res.df.{date_str}.pkl")
