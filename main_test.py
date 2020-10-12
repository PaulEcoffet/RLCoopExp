from PartnerChoiceEnv import PartnerChoice
import ray
from ray import tune
from ray.tune.logger import pretty_print

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy



from ray.tune.registry import register_env


if __name__ == "__main__":
    ray.init()  
    n_players = 5
    agents_id = ['player' + '{:02d}'.format(i) for i in range(n_players)]
    actions = {agents_id[i]:1 for i in range(n_players)}

    register_env("partner_choice",
            lambda _: PartnerChoice(n_players))
    env = PartnerChoice(n_players)
    act_space = env.action_space
    obs_space = env.observation_space
    print(env.reset())
    action_test = {'agent00': [4.666666666666667, 1], 'agent01': [4.666666666666667, 1], 'agent02': [9.333333333333334, 1], 'agent03': [4.666666666666667, 1], 'agent04': [9.333333333333334, 1]}
    print(env.step(action_test))

    #print(obs_space.sample())


"""
    policies = {agents_id[i]: (PPOTorchPolicy, obs_space, act_space, {}) for i in range(n_players)}

    def select_policy(agent_id):
        return agent_id

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy
            },
        "simple_optimizer":True
    }
    
    trainer = PPOTrainer(env="partner_choice", config=config)

    stop_iter = 20
    for i in range(stop_iter):
        print("== Iteration", i, "==")

        #print(trainer.workers.local_worker().env)
        result_ppo = trainer.train()
        print(pretty_print(result_ppo))
"""