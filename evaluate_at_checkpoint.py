from ray.tune import register_env

from PartnerChoiceEnv import PartnerChoiceFakeSites
import ray
from cma_test import CMAESTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer

from main_test import init_setup, select_policy, MyCallbacks

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

def main():
    ray.init(local_mode=True)
    register_env("partner_choice",
                 lambda config: PartnerChoiceFakeSites(config))
    agent = PPOTrainer(config)
    agent.restore("/Users/paulecoffet/Documents/isir/These/data/RLCoopExp/logs/paperrun/e200000/ppobiglr/goodsiteprob_20201120-221749/PPO_partner_choice_d7c84_00023_23_good_site_prob=1.0,max_it=100.0_2020-11-21_00-43-53/checkpoint_594/checkpoint-594")

    print(agent)
    agent.compute_action()


if __name__ == "__main__":
    main()
