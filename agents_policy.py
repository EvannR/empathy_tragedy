import numpy as np

class QLearning:
    def __init__(
            self,
            environment,
            gamma=0.9,
            alpha=0.3,
            beta=1.0):
        
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros(self.shape_SA)
        self.Q_probas = np.ones(self.shape_SA) / self.size_actions
    
    def choose_action(self, current_state):
        """Sélectionne une action en utilisant une politique softmax"""
        max_Q = np.max(self.Q[current_state])
        tmp_Q = self.Q[current_state] - max_Q  # Éviter les valeurs trop grandes
        exp_Q = np.exp(tmp_Q * self.beta)
        self.Q_probas[current_state, :] = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions), p=self.Q_probas[current_state])
        return action
    
    def learn(self, old_state, reward, new_state, action):
        """Met à jour la table Q selon l'équation de Bellman"""
        update = reward + self.gamma * np.max(self.Q[new_state])
        self.Q[old_state][action] *= (1 - self.alpha)
        self.Q[old_state][action] += self.alpha * update


class DiffQLearning:
    def __init__(
            self,
            environment,
            gamma=0.9,
            alpha_plus=0.3,
            alpha_minus=0.3,
            beta=1.0):
        
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.beta = beta
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.gamma = gamma
        self.Q = np.zeros(self.shape_SA)
        self.Q_probas = np.ones(self.shape_SA) / self.size_actions
    
    def choose_action(self, current_state):
        """Sélectionne une action avec une politique softmax"""
        max_Q = np.max(self.Q[current_state])
        tmp_Q = self.Q[current_state] - max_Q  # eviter les valeurs trop grandes
        exp_Q = np.exp(tmp_Q * self.beta)
        self.Q_probas[current_state, :] = exp_Q / np.sum(exp_Q)
        action = np.random.choice(np.arange(self.size_actions), p=self.Q_probas[current_state])
        return action
    
    def learn(self, old_state, reward, new_state, action):
        """Met à jour la table Q avec des coefficients d'apprentissage asymétriques"""
        update = reward + self.gamma * np.max(self.Q[new_state])
        rpe = update - self.Q[old_state][action]
        alpha = self.alpha_plus if rpe > 0 else self.alpha_minus
        self.Q[old_state][action] *= (1 - alpha)
        self.Q[old_state][action] += alpha * update


class DQN:
    def __init__(self, environment, gamma=0.99, alpha=0.001, beta=1.0):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.gamma = gamma
        self.alpha = alpha
        
        # reseau de neurones aec une couche cachée lol 
        self.model = nn.Sequential(
            nn.Linear(self.size_environment, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.size_actions)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, state):
        import torch
        """Choisit une action en utilisant une politique epsilon-greedy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        action = torch.argmax(q_values).item()
        return action
    
    def learn(self, old_state, reward, new_state, action):
        import torch
        """Met à jour le réseau de neurones avec une mise à jour Q-learning"""
        old_state_tensor = torch.FloatTensor(old_state).unsqueeze(0)
        new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
        q_values = self.model(old_state_tensor)
        target_q_values = q_values.clone().detach()
        
        max_new_q = torch.max(self.model(new_state_tensor)).item()
        target_q_values[0, action] = reward + self.gamma * max_new_q
        
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
