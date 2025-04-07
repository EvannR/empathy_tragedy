import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class Agent:
    def __init__(self,
                 agent_id,
                 memory_size=10):
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0

    def record_meal(self, has_eaten, reward_value=0):
        """Enregistre si l'agent a mangé (1) ou non (0) à ce tour"""
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1

    def get_recent_meals(self):
        """Retourne le nombre de reward dans l'historique"""
        return sum(self.meal_history)

    def reset(self):
            """Réinitialise l'état de l'agent pour un nouvel épisode"""
            self.state = self.env.reset()
            self.meal_history = deque([0] * self.memory_size, maxlen=self.memory_size)
            self.total_meals = 0


Experience = namedtuple('Experience', ['state',
                                       'action',
                                       'reward',
                                       'next_state',
                                       'done'])


class ReplayBuffer:
    """mémoire d'expériences pour le DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """ajoute une expérience à la mémoire"""
        # convertir l'action en int et s'assurer qu'elle est dans un format compatible
        if isinstance(action, (list, tuple)):
            action = np.array([action[0]], dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = np.array([action.item()], dtype=np.int64)
        else:
            action = np.array([int(action)], dtype=np.int64)

        # conversion des autres données en format approprié
        state = np.array(state, dtype=np.float32)
        if not isinstance(reward, (int, float)):
            reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32)
        if not isinstance(done, bool):
            done = bool(done)

        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """échantillonne aléatoirement un batch d'expériences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # extraction des données avec les conversions appropriées
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.tensor([[e.reward] for e in experiences], dtype=torch.float)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.tensor([[int(e.done)] for e in experiences], dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QAgent:
    """agent utilisant l'algorithme Q-Learning"""
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # facteur d'actualisation
        self.epsilon = epsilon  # paramètre d'exploration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # table Q-values : mapping état -> valeurs d'action
        self.q_table = {}

        # état courant et action précédente
        self.current_state = None
        self.previous_action = None

    def get_state_key(self, state):
        """convertit un état en clé hashable pour la table Q"""
        # exemple simple : concatenation des valeurs
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        return tuple(state)

    def get_q_values(self, state):
        """récupère les valeurs Q pour un état donné"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            # initialisation des valeurs Q à zéro pour un nouvel état
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, state):
        """sélectionne une action selon la politique epsilon-greedy"""
        if np.random.random() < self.epsilon:
            # exploration : action aléatoire
            return int(np.random.choice(self.action_size))

        # exploitation : meilleure action selon les valeurs Q
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done):
        """met à jour la table Q en fonction de l'expérience"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        target = reward + self.gamma * np.max(next_q_values)
        
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])
        self.q_table[state_key] = q_values

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def start_episode(self, state):
        """initialise l'état au début d'un épisode"""
        self.current_state = state
        self.previous_action = None

    def step(self, next_state, reward, done):
        """effectue une étape d'apprentissage"""
        if self.current_state is not None and self.previous_action is not None:
            self.learn(self.current_state, self.previous_action, reward, next_state, done)

        self.current_state = next_state

        action = self.select_action(next_state)
        self.previous_action = action

        return action


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, agent_id=0, hidden_size=64,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64,
                 update_target_every=10):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0

        self.policy_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()

        self.current_state = None
        self.previous_action = None

        self.meal_history = []

    def select_action(self, state):
        """sélectionne une action selon la politique epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        self.policy_network.train()

        return int(np.argmax(action_values.cpu().data.numpy()))

    def learn(self, experiences):
        """apprentissage à partir d'un batch d'expériences"""
        states, actions, rewards, next_states, dones = experiences

        try:
            q_expected = self.policy_network(states).gather(1, actions)
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

            loss = F.mse_loss(q_expected, q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.steps += 1
            if self.steps % self.update_target_every == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
        except Exception as e:
            print(f"erreur lors de l'apprentissage: {e}")
            raise

    def remember(self, state, action, reward, next_state, done):
        """stocke une expérience dans la mémoire"""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(reward[0])
        else:
            reward = float(reward)

        if isinstance(done, (list, tuple, np.ndarray)):
            done = bool(done[0])
        else:
            done = bool(done)

        self.memory.add(state, action, reward, next_state, done)

    def step(self, next_state, reward, done):
        """effectue une étape d'apprentissage"""
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        if self.current_state is not None and self.previous_action is not None:
            self.remember(self.current_state, self.previous_action, reward, next_state, done)

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

        self.current_state = next_state

        action = self.select_action(next_state)
        self.previous_action = action

        self.meal_history.append(action)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def start_episode(self, state):
        """initialise l'état au début d'un épisode"""
        self.current_state = state
        self.previous_action = None


class SocialRewardCalculator:
    """
    classe pour calculer les récompenses sociales basées sur la consommation
    et l'empathie entre agents
    """
    def __init__(self, nb_agents, alpha=0.5, beta=0.5):
        """
        initialise le calculateur de récompense sociale

        parameters:
        -----------
        nb_agents : int
            nombre d'agents dans l'environnement
        alpha : float
            pondération entre la satisfaction personnelle et l'empathie (0-1)
        beta : float
            pondération du dernier repas par rapport à l'historique (0-1)
        """
        self.nb_agents = nb_agents
        self.alpha = alpha  # balance entre soi et les autres
        self.beta = beta    # balance entre dernier repas et historique

    def calculate_personal_satisfaction(self, agent):
        """
        calcule la satisfaction personnelle d'un agent basée sur son historique de repas

        parameters:
        -----------
        agent : Agent
            l'agent dont on calcule la satisfaction

        returns:
        --------
        float
            score de satisfaction personnelle
        """

        if len(agent.meal_history) > 0:
            last_meal = 1 if agent.meal_history[-1] > 0 else 0
            history_weight = sum(agent.meal_history) / len(agent.meal_history)
        else:
            # Si meal_history est vide, définir des valeurs par défaut
            last_meal = 0  # Valeur par défaut pour le dernier repas
            history_weight = 0  # Valeur par défaut pour le poids de l'historique

        # Calcul de la satisfaction personnelle
        satisfaction = self.beta * last_meal + (1 - self.beta) * history_weight

        return satisfaction

    def calculate_rewards(self, agents):
        """
        calcule les récompenses émotionnelles pour tous les agents

        parameters:
        -----------
        agents : list
            liste des agents

        returns:
        --------
        list
            liste des récompenses pour chaque agent
        """
        personal_satisfactions = [self.calculate_personal_satisfaction(agent) for agent in agents]

        rewards = []
        for idx, satisfaction in enumerate(personal_satisfactions):
            own_satisfaction = satisfaction

            others_satisfaction = np.mean([s for i, s in enumerate(personal_satisfactions) if i != idx])

            emotional_reward = self.alpha * own_satisfaction + (1 - self.alpha) * others_satisfaction
            rewards.append(emotional_reward)

        return rewards
