import numpy as np
import os
import sys
# To access the file in the parent file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents_policies import Agent


class GameTheoretic:
    def __init__(self, size=4,
                 nb_agents=1,
                 agent_configs=None,
                 initial_resources=100,
                 regen_rate=1.0):
        self.size = size
        self.nb_agents = nb_agents
        self.number_actions = 2  # 0: Not Exploit, 1: Exploit
        self.actions = np.arange(self.number_actions)
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        self.initial_resources = initial_resources
        self.regen_rate = regen_rate

        self.init_agents()
        self.new_episode()

    def init_agents(self):
        self.agents = []
        for i, config in enumerate(self.agent_configs):
            agent = Agent(agent_id=i, **config)
            self.agents.append(agent)

    def new_episode(self):
        self.time_step = 0
        self.resource = self.initial_resources
        for agent in self.agents:
            agent.reset()

    def update_environment(self, consumed):
        self.time_step += 1
        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)

    def make_step(self, actions):
        rewards = []
        consumed = 0

        for i, act in enumerate(actions):
            if act == 1 and self.resource > 0:
                reward = 1.0
                consumed += 1
                self.agents[i].record_meal(True, reward)
            else:
                reward = 0.0
                self.agents[i].record_meal(False, reward)
            rewards.append(reward)

        self.update_environment(consumed)
        return rewards

    def get_agent_meal_stats(self, agent_idx):
        agent = self.agents[agent_idx]
        return {
            'recent_meals': agent.get_recent_meals(),
            'total_meals': agent.total_meals,
            'meal_history': list(agent.meal_history)
        }

    def get_all_agents_meal_stats(self):
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
