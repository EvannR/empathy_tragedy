import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class Agent:
    def __init__(self, agent_id, position, memory_size=10):
        self.agent_id = agent_id
        self.position = position
        self.memory_size = memory_size
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0
    
    def record_meal(self, has_eaten, reward_value=0):
        """Enregistre si l'agent a mangé (1) ou non (0) à ce tour"""
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1
    
    def get_recent_meals(self):
        """Retourne le nombre de repas dans l'historique"""
        return sum(self.meal_history)
    
    def update_position(self, new_position):
        """Met à jour la position de l'agent"""
        self.position = new_position



Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

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
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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
        
        # valeurs Q actuelles
        q_values = self.get_q_values(state)
        
        # valeurs Q de l'état suivant
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        
        # calcul de la cible (target Q-value)
        target = reward + self.gamma * np.max(next_q_values)
        
        # mise à jour de la valeur Q
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])
        
        # mise à jour de la table Q
        self.q_table[state_key] = q_values
        
        # décroissance de epsilon (exploration)
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
        
        # mise à jour de l'état courant
        self.current_state = next_state
        
        # sélection d'une nouvelle action
        action = self.select_action(next_state)
        self.previous_action = action
        
        return action


class DQNNetwork(nn.Module):
    """réseau de neurones pour le DQN"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """agent utilisant l'algorithme Deep Q-Network"""
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
        
        # réseaux de neurones (policy et target)
        self.policy_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # optimiseur
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # replay buffer
        self.memory = ReplayBuffer()
        
        # état et action courants
        self.current_state = None
        self.previous_action = None
    
    def select_action(self, state):
        """sélectionne une action selon la politique epsilon-greedy"""
        if np.random.random() < self.epsilon:
            # exploration : action aléatoire
            return int(np.random.choice(self.action_size))
        
        # exploitation : meilleure action selon le réseau de neurones
        # assurons-nous que l'état est correctement formaté
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        self.policy_network.train()
        
        # retourner un entier simple, pas un array
        return int(np.argmax(action_values.cpu().data.numpy()))
    
    def learn(self, experiences):
        """apprentissage à partir d'un batch d'expériences"""
        states, actions, rewards, next_states, dones = experiences
        
        try:
            # calcul des prédictions Q actuelles
            q_expected = self.policy_network(states).gather(1, actions)
            
            # calcul des cibles Q
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
            
            # calcul de la perte
            loss = F.mse_loss(q_expected, q_targets)
            
            # mise à jour des poids
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # mise à jour du réseau cible périodiquement
            self.steps += 1
            if self.steps % self.update_target_every == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
        except Exception as e:
            # afficher des informations de débogage en cas d'erreur
            print(f"erreur lors de l'apprentissage: {e}")
            print(f"états shape: {states.shape}")
            print(f"actions shape: {actions.shape}, type: {actions.dtype}")
            print(f"récompenses shape: {rewards.shape}")
            print(f"états suivants shape: {next_states.shape}")
            print(f"terminés shape: {dones.shape}")
            raise
    
    def remember(self, state, action, reward, next_state, done):
        """stocke une expérience dans la mémoire"""
        # conversion en types corrects avant de stocker
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        # assurer que reward et done sont des scalaires
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
        # s'assurer que next_state est un numpy array
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        if self.current_state is not None and self.previous_action is not None:
            # stocke l'expérience
            self.remember(self.current_state, self.previous_action, reward, next_state, done)
            
            # apprentissage si suffisamment d'expériences
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
        
        # mise à jour de l'état courant
        self.current_state = next_state
        
        # sélection d'une nouvelle action
        action = self.select_action(next_state)
        self.previous_action = action
        
        # décroissance de epsilon (exploration)
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
        # poids du dernier repas
        last_meal = 1 if agent.meal_history[-1] > 0 else 0
        
        # poids de l'historique récent
        history_weight = sum(agent.meal_history) / len(agent.meal_history)
        
        # combinaison pondérée
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
        # calcul des satisfactions personnelles
        personal_satisfactions = [self.calculate_personal_satisfaction(agent) for agent in agents]
        
        # calcul des récompenses émotionnelles
        rewards = []
        for idx, satisfaction in enumerate(personal_satisfactions):
            # satisfaction personnelle
            own_satisfaction = satisfaction
            
            # satisfaction moyenne des autres agents (empathie)
            others_satisfaction = np.mean([s for i, s in enumerate(personal_satisfactions) if i != idx])
            
            # combinaison pondérée des deux composantes
            emotional_reward = self.alpha * own_satisfaction + (1 - self.alpha) * others_satisfaction
            rewards.append(emotional_reward)
        
        return rewards