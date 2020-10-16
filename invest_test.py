import gym, ray
from negotiate_model import InvestorModel
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
import numpy as np


class InvestTestEnv(gym.Env):
    observation_space = gym.spaces.Box(np.asarray([0]), np.asarray([1]))
    action_space = gym.spaces.Box(np.asarray([0]), np.asarray([15]))

    def __init__(self, config):
        super().__init__()

    def reset(self):
        return np.asarray([0])

    def render(self, mode='human'):
        pass

    def step(self, action):
        return [0], action, True, {}


if __name__ == "__main__":
    ray.init()

    ModelCatalog.register_custom_model("invest_model", InvestorModel)
    trainer = ppo.PPOTrainer(env=InvestTestEnv, config={
        "env_config": {},  # config to pass to env class
        "framework": "torch",
        "model": {
            "custom_model": "invest_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {},
        },
    })

    while True:
        print(trainer.train())
