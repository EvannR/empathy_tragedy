import numpy as np
from agent_policies_game_theoretic import QAgent, DQNAgent, SocialRewardCalculator


class GameTheoreticEnv:
    """
    Multi-agent environment where each agent can exploit a shared resource or not.
    Agents can optionally observe emotional states of others (average or full vector), or see nothing.
    """
    def __init__(self, nb_agents,
                 initial_resources=100,
                 regen_rate=1.0,
                 env_type="deterministic",
                 emotion_type="average",
                 see_emotions=True,
                 alpha=0.5, 
                 beta=0.5, 
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.7,
                 smoothing='linear',
                 sigmoid_gain=10.0):

        # Environment settings
        self.nb_agents = nb_agents
        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector" (ignored if see_emotions=False)
        self.see_emotions = see_emotions      # If False, agents receive zero observations
        self.alpha = alpha                    # empathic weight on others
        self.beta = beta                      # weight of last vs history
        self.number_actions = 2               # 0: Not Exploit, 1: Exploit
        self.actions = np.arange(self.number_actions)

        # Agent setup
        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        # Social reward calculator
        self.reward_calculator = SocialRewardCalculator(nb_agents,
                                                        alpha=alpha,
                                                        beta=beta,
                                                        threshold=threshold,
                                                        smoothing=smoothing,
                                                        sigmoid_gain=sigmoid_gain
                                                        )

        # Initialize agents and reset environment state
        self._init_agents()
        self.reset()

    def _init_agents(self):
        # Determine input dimension for agents
        if not self.see_emotions:
            self.state_size = 1
        else:
            if self.emotion_type == "average":
                self.state_size = 1
            elif self.emotion_type == "vector":
                self.state_size = self.nb_agents
            else:
                raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

        # Instantiate agent objects
        self.agents = []
        for idx, config in enumerate(self.agent_configs):
            agent = self.agent_class(
                state_size=self.state_size,
                action_size=self.number_actions,
                agent_id=idx,
                **config
            )
            self.agents.append(agent)

    def reset(self):
        """Reset environment to start a new episode."""
        self.resource = self.initial_resources
        self.time_step = 0
        # Recreate agents to clear internal states
        self._init_agents()
        # Initial observation and agent episode start
        obs = self.get_observation()
        for agent in self.agents:
            agent.start_episode(obs)
            if hasattr(agent, 'memory'):  # clear DQN replay buffer
                agent.memory.buffer.clear()
        return obs

    def get_observation(self):
        """Return list of observations for each agent."""
        if not self.see_emotions:
            # Agents have no emotional info
            return [np.zeros(self.state_size, dtype=float) for _ in range(self.nb_agents)]

        # Compute each agent's personal satisfaction
        emotions = np.array([
            self.reward_calculator.calculate_personal_satisfaction(agent)
            for agent in self.agents
        ], dtype=float)

        if self.emotion_type == "average":
            avg = np.mean(emotions)
            return [np.array([avg], dtype=float) for _ in range(self.nb_agents)]
        elif self.emotion_type == "vector":
            return [emotions.copy() for _ in range(self.nb_agents)]
        else:
            raise ValueError(f"Unsupported emotion_type: {self.emotion_type}")

    def make_step(self, actions):
        """
        Execute one timestep: agents choose actions, environment updates.
        actions: list of 0/1 for each agent.
        Returns: next_observations, rewards, done, info
        """
        consumed = 0
        immediate_rewards = []
    
        for idx, act in enumerate(actions):
            reward = 0.0
            success = False
            if act == 1 and self.resource > 0:
                if self.env_type == "stochastic":
                    prob = self.resource / self.initial_resources
                    success = np.random.rand() < prob
                else:
                    success = True
                if success:
                    reward = 1.0
                    consumed += 1
    
            if hasattr(self.agents[idx], 'record_meal'):
                self.agents[idx].record_meal(success, reward)
            immediate_rewards.append(reward)
    
        # Calculate components from the reward calculator
        emotions, personal, empathic, total = self.reward_calculator.calculate_rewards(self.agents)
    
        # Combine exploitation reward with internal reward system (personal + empathic)
        combined_rewards = [im + tot for im, tot in zip(immediate_rewards, total)]
    
        # Update environment state
        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)
        self.time_step += 1
        next_obs = self.get_observation()
        done = self.resource <= 0
    
        info = {
            'emotions': emotions,
            'personal_satisfaction': personal,
            'empathic_reward': empathic,
            'internal_total_reward': total,
            'exploitation_reward': immediate_rewards,
            'combined_reward': combined_rewards
        }
    
        return next_obs, combined_rewards, done, info

    def get_agent_meal_stats(self, agent_idx):
        """Return recent and total meals for one agent."""
        a = self.agents[agent_idx]
        return {
            'recent_meals': a.get_recent_meals(),
            'total_meals': a.total_meals,
            'meal_history': list(a.meal_history)
        }

    def get_all_agents_meal_stats(self):
        """Return meal stats for all agents."""
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
