# simple_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class ReplayBuffer:
    """Simple memory buffer for experience replay."""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays for easier processing
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple Q-Network for deep reinforcement learning."""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """Forward pass through network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleAgent:
    """
    A basic reinforcement learning agent that uses
    deep Q-learning to make decisions.
    """
    def __init__(self, state_size, action_size, agent_id=0, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.1, batch_size=64):
        """
        Initialize agent parameters.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            agent_id: Identifier for the agent
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Number of samples to learn from at once
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural network for Q-function approximation
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer()
        
        # Step counter for target network update
        self.update_counter = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return torch.argmax(action_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Update Q-network based on stored experiences."""
        # Need enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q = self.q_network(states_tensor).gather(1, actions_tensor)
        
        # Next Q-values from target network
        with torch.no_grad():
            max_next_q = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
        
        # Target Q-values
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 10 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """Save model weights."""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model weights."""
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(self.q_network.state_dict())