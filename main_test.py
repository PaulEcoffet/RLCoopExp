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
from pprint import pprint
from datetime import datetime

if __name__ == "__main__":
    ray.init()
    nb_agents = 1
    inv_id = ['inv' + '{:02d}'.format(i) for i in range(nb_agents)]
    choice_id = [f'choice{i:02d}' for i in range(nb_agents)]


    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))
    ModelCatalog.register_custom_model("investor_model", InvestorModel)

    choice_act_space = Discrete(2)
    choice_obs_space = Box(np.array([0, 0], dtype=np.float32), np.array([15, 15], dtype=np.float32))
    inv_act_space = Box(np.array([0], dtype=np.float32), np.array([15], dtype=np.float32))
    inv_obs_space = Box(np.array([0], dtype=np.float32), np.array([1], np.float32))

    choicemodel_dict = {
        "model": {
            "fcnet_hiddens": [3],
        },
    }

    investormodel_dict = {
        "model": {
            "fcnet_hiddens": [],  # linear mapping
        },
    }

    policies = {inv_id[i]: (None, inv_obs_space, inv_act_space, investormodel_dict)
                for i in range(nb_agents)}
    policies.update({choice_id[i]: (None, choice_obs_space, choice_act_space, choicemodel_dict)
                     for i in range(nb_agents)})

    def select_policy(agent_id):
        return agent_id

    def choose_max_it(spec):
        bad_site_prob = spec['config']['env_config']['bad_site_prob']
        if bad_site_prob == 0:
            return 100
        else:
            return int(np.ceil(1 / (1 - bad_site_prob))) * 100

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
            },
        "clip_actions": True,
        "framework": "torch",
        "no_done_at_end": True,
        "gamma": 1,
        "env": "partner_choice",
        "env_config":
            {
                "bad_site_prob": tune.grid_search([0, 0.99, 0.999, 0.9999]),
                "max_it": tune.sample_from(lambda spec: choose_max_it(spec))
            }
    }
    
    trainer = PPOTrainer(env="partner_choice", config=config)

    now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    tune.run(PPOTrainer, name="pc", config=config, stop={"training_iteration": 1_000_000}, local_dir=f"logs/{now_str}/",
             checkpoint_at_end=True)
