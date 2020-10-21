import gym, ray
from negotiate_model import InvestorModel
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
import numpy as np


class InvestTestEnv(gym.Env):
    observation_space = gym.spaces.Box(np.array([0]), np.array([1]))
    action_space = gym.spaces.Box(np.array([0]), np.array([1]))

    def __init__(self, config):
        super().__init__()

    def reset(self):
        return np.array([0.0])

    def render(self, mode='human'):
        pass

    def step(self, action: np.ndarray):
        if np.random.rand() < 0.01:
            print(action[0] * 15)
        return [0.0], action[0], True, {}


if __name__ == "__main__":
    ray.init(local_mode=True)

    ModelCatalog.register_custom_model("invest_model", InvestorModel)
    trainer = ppo.PPOTrainer(env=InvestTestEnv, config={
        "env_config": {},  # config to pass to env class
        "framework": "torch",
        "num_workers": 0,
        "model": {
            "custom_model": "invest_model",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {},
        },
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        "gamma": 1,
        "monitor": True,
    })

    while True:
        print(trainer.train())
