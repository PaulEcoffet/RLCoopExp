from PartnerChoiceEnv import PartnerChoiceFakeSites
import numpy as np
import ray
from ray import tune
from ray.tune.logger import pretty_print

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy

from gym.spaces import Box, Discrete
import numpy as np
from negotiate_model import InvestorModel
from ray.tune.registry import register_env


if __name__ == "__main__":
    ray.init(local_mode=True)
    nb_agents = 1
    inv_id = ['inv' + '{:02d}'.format(i) for i in range(nb_agents)]
    choice_id = [f'choice{i:02d}' for i in range(nb_agents)]


    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))
    ModelCatalog.register_custom_model("investor_model", InvestorModel)

    choice_act_space = Discrete(2)
    choice_obs_space = Box(np.array([0, 0], dtype=np.float32), np.array([15, 15], dtype=np.float32))
    inv_act_space = Box(np.array([-1], dtype=np.float32), np.array([1], dtype=np.float32))
    inv_obs_space = Box(np.array([0], dtype=np.float32), np.array([1], np.float32))

    choicemodel_dict = {
                            "fcnet_hiddens": [3],
                            "model":
                                {
                                    "custom_model": None,
                                },
                          }

    investormodel_dict = {
        "model": {
        "custom_model": "investor_model",
        }
    }

    policies = {inv_id[i]: (None, inv_obs_space, inv_act_space, investormodel_dict) for i in range(nb_agents)}
    policies.update({choice_id[i]: (None, choice_obs_space, choice_act_space, choicemodel_dict) for i in range(nb_agents)})

    def select_policy(agent_id):
        return agent_id

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
        "env_config":
            {
                "bad_site_prob": 0.999,
                "max_it": 10000
            }
    }
    
    trainer = PPOTrainer(env="partner_choice", config=config)

    while True:
        print(trainer.train())
