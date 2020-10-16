from gym.spaces import Box, Discrete
from random import randrange
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def payoff(xi, xj):
    a = 5
    b = 10
    pg = a*0.5*(xi+xj)
    pd = b*0.5*xj
    c = 0.5*xi**2
    return pg + pd - c

class PartnerChoice(MultiAgentEnv):

    def __init__(self, nb_agents=1, nb_sites=10):

        self.nb_agents = nb_agents
        self.nb_sites = nb_sites

        self.iteration_count = 0

        self.agents_names = ['agent' + '{:02d}'.format(i) for i in range(self.nb_agents)]

        self.max_action = 15

        self.action_space = Box(low=np.asarray([0, 0]), high=np.asarray([self.max_action, 1]),
                                shape=(2,), dtype=np.float64)
        self.observation_space = Box(low=0, high=self.max_action, shape=(1,), dtype=np.float64)

        self.previous_obs = {}

        self.site_action = np.linspace(0, self.max_action, self.nb_sites)
        #print(self.site_action)

        self.site_acceptance_threshold = np.copy(self.site_action)

        self.previous_interaction_id = {self.agents_names[i]: 0 for i in range(self.nb_agents)}

        self.rewards = {}
        self.obs = {}

    def reset(self):
        #print('Hey RESET')
        self.iteration_count = 0
        return {
            agent_name: [self.site_action[self.previous_interaction_id[agent_name]]] for agent_name in self.agents_names
        }

    def step(self, action_dict):
        self.iteration_count += 1

        # All agents receive their reward for previous site and observe a new site
        for agent_id, action_value in action_dict.items():
            #print(action_value)
            # Reward of the previous action
            print(action_value)
            if action_value[0] > self.site_acceptance_threshold[self.previous_interaction_id[agent_id]] and action_value[1]>= self.max_action/2:
                self.rewards[agent_id] = payoff(action_value[1], self.site_action[self.previous_interaction_id[agent_id]])
            else:
                self.rewards[agent_id] = 0

            # Agent new observation
            self.previous_interaction_id[agent_id] = randrange(0, self.nb_sites)
            self.obs[agent_id] = [self.site_action[self.previous_interaction_id[agent_id]]]
 

        # done after 10 moves
        done = {"__all__": self.iteration_count >= 100}
        #print(self.obs, self.rewards)
        return self.obs, self.rewards, done, {}
