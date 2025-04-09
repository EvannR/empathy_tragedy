import numpy as np
from agent_policies_game_theoretic import Agent, QAgent, DQNAgent, SocialRewardCalculator


class GameTheoreticEnv:
    def __init__(self, size=4,
                 nb_agents=1,
                 agent_configs=None,
                 initial_resources=100,
                 regen_rate=1.0,
                 env_type="deterministic",
                 emotion_type="average"):  # emotion_type peut être "average" ou "vector"
        self.size = size
        self.nb_agents = nb_agents
        self.number_actions = 2  # 0: Not Exploit, 1: Exploit
        self.actions = np.arange(self.number_actions)
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type  # "deterministic" ou "stochastic"
        self.emotion_type = emotion_type  # "average" ou "vector"

        self.init_agents()
        self.reward_calculator = SocialRewardCalculator(nb_agents)
        self.new_episode()

    def init_agents(self):
        self.agents = []
        for i, config in enumerate(self.agent_configs):
            # L'agent peut être soit un QAgent, soit un DQNAgent
            agent = DQNAgent(state_size=4, action_size=self.number_actions, agent_id=i, **config)
            self.agents.append(agent)

    def reset(self):
        # Réinitialiser l'état de l'environnement
        self.resource = 100  # Par exemple, réinitialiser la ressource
        self.agents = []  # Réinitialiser la liste des agents
        self.init_agents()  # Réinitialiser les agents
        return self.get_observation()
    
    def new_episode(self):
        self.time_step = 0
        self.resource = self.initial_resources
        for agent in self.agents:
            agent.start_episode(self.get_observation())

    def update_environment(self, consumed):
        self.time_step += 1
        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)

    def get_observation(self):
        # Calculer l'état émotionnel de chaque agent (satisfaction personnelle)
        emotions = np.array([self.reward_calculator.calculate_personal_satisfaction(agent) for agent in self.agents])

        # Option 1 : Emotion en fonction de la moyenne des émotions des autres agents
        if self.emotion_type == "average":
            avg_emotion = np.mean(emotions)
            emotions = np.array([avg_emotion] * self.nb_agents)

        # Option 2 : Emotion comme un vecteur
        elif self.emotion_type == "vector":
            emotions = np.delete(emotions, np.s_[:])  # Si on choisit "vector", inclure tous les scores individuels


        return emotions

    def make_step(self, actions):
        rewards = []
        consumed = 0

        for i, act in enumerate(actions):
            reward = 0.0
            success = False

            if act == 1 and self.resource > 0:
                if self.env_type == "stochastic":
                    p = self.resource / self.initial_resources
                    success = np.random.rand() < p
                else:
                    success = True

                if success:
                    reward = 1.0
                    consumed += 1

            self.agents[i].record_meal(success, reward)
            rewards.append(reward)

        social_rewards = self.reward_calculator.calculate_rewards(self.agents)
        for i in range(self.nb_agents):
            rewards[i] += social_rewards[i]

        self.update_environment(consumed)

        next_obs = self.get_observation()
        done = self.resource <= 0
        info = {} 

        return next_obs, rewards, done, info

    def get_agent_meal_stats(self, agent_idx):
        agent = self.agents[agent_idx]
        return {
            'recent_meals': agent.get_recent_meals(),
            'total_meals': agent.total_meals,
            'meal_history': list(agent.meal_history)
        }

    def get_all_agents_meal_stats(self):
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
