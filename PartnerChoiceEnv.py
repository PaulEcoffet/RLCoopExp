from gym.spaces import Box, Discrete
from random import randrange
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def get_id(agent_name):
    inv = 'inv'
    choice = 'choice'
    if agent_name.startswith(inv):
        return int(agent_name[len(inv):])
    elif agent_name.startswith(choice):
        return int(agent_name[len(choice):])
    else:
        raise ValueError


def payoff(xi, xj):
    a = 5
    b = 5
    pg = a*xi
    pd = b*xj
    c = 0.5*xi**2
    return pg + pd - c


class PartnerChoice(MultiAgentEnv):

    def __init__(self, nb_agents=1, nb_sites=100):

        self.nb_agents = nb_agents
        self.nb_sites = nb_sites

        self.iteration_count = 0

        self.agents_names = ['inv' + '{:02d}'.format(i) for i in range(self.nb_agents)]
        self.agents_names += ['choice' + '{:02d}'.format(i) for i in range(self.nb_agents)]
        self.inv = [0 for i in range(self.nb_agents)]
        self.cur_opp = [-1 for i in range(self.nb_agents)]

        self.max_action = 15
        self.all_dones = [False for i in range(self.nb_agents)]

        self.action_space = Discrete(1)
        self.observation_space = Box(low=0, high=self.max_action, shape=(2,), dtype=np.float64)

        self.site_action = np.linspace(0.1, self.max_action, self.nb_sites)

        self.site_acceptance_threshold = np.copy(self.site_action)

    def reset(self):
        self.iteration_count = 0
        self.all_dones = [False for i in range(self.nb_agents)]
        self.cur_opp = [None for i in range(self.nb_agents)]
        # Make all individuals make their investment choice
        return {
            f'inv{i:02}': np.array([0]) for i in range(self.nb_agents)
        }

    def step(self, action_dict):
        self.iteration_count += 1
        obs = {}
        reward = {}
        done = {}

        for agent_name in action_dict:
            ind = get_id(agent_name)
            # If we get a investment action
            if agent_name.startswith('inv'):
                self.inv[ind] = action_dict[agent_name][0] * 15
                if np.random.rand() < 0.05:
                    print(self.inv[ind])
                self.cur_opp[ind] = np.random.randint(self.nb_sites)
                obs[f'choice{ind:02d}'] = np.array([self.site_action[self.cur_opp[ind]], self.inv[ind]])
                reward[f'choice{ind:02d}'] = 0  # dummy reward at init
            else:  # if it's a choice action
                # if they both agree
                curopp = self.cur_opp[ind]
                if action_dict[agent_name] == 1 and self.inv[ind] >= self.site_acceptance_threshold[curopp]:
                    curpayoff = payoff(self.inv[ind], self.site_action[curopp])
                    if np.random.rand() < 0.05:
                        print(f"payoff({self.inv[ind]}, {self.site_action[curopp]}) = {curpayoff}")
                    # give payoff to both module and end interaction
                    reward[agent_name] = curpayoff
                    reward[f'inv{ind:02d}'] = curpayoff
                    obs[agent_name] = np.array([0, 0])
                    done[agent_name] = True
                    done[f'inv{ind:02d}'] = True
                    obs[f'inv{ind:02d}'] = np.array([0])
                    self.all_dones[ind] = True
                else:  # if at least one disagree
                    reward[agent_name] = 0
                    done[agent_name] = False
                    self.cur_opp[ind] = np.random.randint(self.nb_sites)
                    obs[f'choice{ind:02d}'] = np.array([self.site_action[self.cur_opp[ind]], self.inv[ind]])

        done["__all__"] = all(self.all_dones)  # Everyone has finished
        if self.iteration_count >= 20:  # or max iter count
            for i in range(self.nb_agents):
                if not self.all_dones[i]:
                    obs[f'inv{i:02d}'] = np.array([0])
                    reward[f'inv{i:02d}'] = 0
                    done[f'inv{i:02d}'] = True
                    obs[f'choice{i:02d}'] = np.array([0, 0])
                    reward[f'choice{i:02d}'] = 0
                    done[f'choice{i:02d}'] = True
            done['__all__'] = True
        return obs, reward, done, {}
