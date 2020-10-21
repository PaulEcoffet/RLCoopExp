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


class PartnerChoiceFakeSites(MultiAgentEnv):

    def __init__(self, env_config):
        self.nb_agents = env_config.get("nb_agents", 1)
        self.nb_sites = env_config.get("nb_sites", 31)
        self.max_it = env_config.get("max_it", 100)
        self.new_x_each_interaction = env_config.get("new_x_each_interaction", True)
        self.max_action = env_config.get("max_action", 15.0)
        self.bad_site_prob = env_config.get("bad_site_prob", 0)
        self.iteration_count = 0
        self.agents_names = ['inv' + '{:02d}'.format(i) for i in range(self.nb_agents)]
        self.agents_names += ['choice' + '{:02d}'.format(i) for i in range(self.nb_agents)]
        self.inv = [0.0 for i in range(self.nb_agents)]
        self.cur_opp = [-1 for i in range(self.nb_agents)]

        self.cur_its = np.array([0 for i in range(self.nb_agents)], dtype=int)

        self.action_space = Discrete(1)
        self.observation_space = Box(low=0, high=self.max_action, shape=(2,), dtype=np.float64)

        self.site_action = np.linspace(0, self.max_action, self.nb_sites)
        self.site_acceptance_threshold = np.copy(self.site_action)

    def reset(self):
        self.cur_its = np.array([0 for i in range(self.nb_agents)], dtype=int)
        self.cur_opp = [None for i in range(self.nb_agents)]
        # Make all individuals make their investment choice
        return {
            f'inv{i:02}': np.array([0], dtype=np.float32) for i in range(self.nb_agents)
        }

    def _find_opp(self):
        if self.bad_site_prob == 0:
            return np.random.randint(self.nb_sites)
        elif np.random.rand() < self.bad_site_prob:
            return 0
        else:
            return np.random.randint(1, self.nb_sites)

    def step(self, action_dict):
        obs = {}
        reward = {}
        done = {}

        for agent_name in action_dict:
            ind = get_id(agent_name)
            inv = f"inv{ind:02d}"
            choice = f"choice{ind:02d}"
            # If we get a investment action
            if agent_name.startswith('inv'):
                self.inv[ind] = (action_dict[agent_name][0] + 1) / 2 * self.max_action
                if np.random.rand() < 0.001:
                    print(self.inv[ind])
                self.cur_opp[ind] = self._find_opp()
                obs[choice] = np.array([self.site_action[self.cur_opp[ind]], self.inv[ind]])
                reward[choice] = 0  # dummy reward at init
            else:  # if it's a choice action
                self.cur_its[ind] += 1

                # if they both agree
                curopp = self.cur_opp[ind]
                if action_dict[choice] == 1 and self.inv[ind] >= self.site_acceptance_threshold[curopp]\
                        and (not self.bad_site_prob != 0 or curopp != 0):  # if bad_site_prob then no nowhere
                    curpayoff = payoff(self.inv[ind], self.site_action[curopp])
                    if np.random.rand() < 0.005:
                        print(f"payoff({self.inv[ind]}, {self.site_action[curopp]}) = {curpayoff}")
                    # give payoff to both module and end interaction
                    reward[agent_name] = curpayoff
                    reward[f'inv{ind:02d}'] = curpayoff
                    obs[agent_name] = np.array([0, 0], dtype=np.float32)
                    done[agent_name] = True
                    done[f'inv{ind:02d}'] = True
                    obs[f'inv{ind:02d}'] = np.array([0], dtype=np.float32)
                    self.cur_its[ind] = self.max_it
                else:  # if at least one disagree
                    done[agent_name] = False
                    self.cur_opp[ind] = self._find_opp()
                    if self.new_x_each_interaction:
                        obs[f'inv{ind:02d}'] = np.array([0], dtype=np.float32)
                        reward[f'inv{ind:02d}'] = 0
                    else:
                        obs[f'choice{ind:02d}'] = np.array([self.site_action[self.cur_opp[ind]], self.inv[ind]],
                                                           dtype=np.float32)
                        reward[agent_name] = 0

        done["__all__"] = all(self.cur_its == self.max_it)  # Everyone has finished

        # Tell everyone for each its over that it is over
        for i in range(self.nb_agents):
            if self.cur_its[i] >= self.max_it and (f"choice{i:02d}" not in done or not done[f"choice{i:02d}"]):
                obs[f'inv{i:02d}'] = np.array([0], dtype=np.float32)
                reward[f'inv{i:02d}'] = 0.0
                done[f'inv{i:02d}'] = True
                obs[f'choice{i:02d}'] = np.array([0, 0], dtype=np.float32)
                reward[f'choice{i:02d}'] = 0.0
                done[f'choice{i:02d}'] = True
        return obs, reward, done, {}
