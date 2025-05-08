# agents_policies.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class Agent:
    """
    Base Agent class representing an agent in the grid world.
    Keeps track of position, meals consumed, and history.
    """
    def __init__(self, agent_id, position:tuple, memory_size=10):
        self.agent_id = agent_id
        self.position = position
        self.memory_size = memory_size
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0

    def record_meal(self, has_eaten:bool, reward_value=0):
        """Record whether agent has eaten and update meal history"""
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1
        return reward_value  # Return reward for chaining

    def get_recent_meals(self):
        """Return number of meals in recent history"""
        return sum(self.meal_history)

    def update_position(self, new_position):
        """Update agent's position on the grid"""
        self.position = new_position
        return self  # Return self for method chaining

    def get_state(self, env): 
        """
        Get the state representation for the agent
        Includes position and surrounding resources
        """
        pos_i, pos_j = self.position
        state = np.zeros(10, dtype=np.float32)
        
        # Normalize position coordinates
        state[0] = pos_i / env.size
        state[1] = pos_j / env.size
        
        # Check resources in surrounding cells (8 directions)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for idx, (di, dj) in enumerate(directions):
            ni, nj = pos_i + di, pos_j + dj
            if 0 <= ni < env.size and 0 <= nj < env.size:
                state[2 + idx] = env.rewards[ni, nj]
                
        return state


# Named tuple for storing experiences in the replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Memory buffer for storing agent experiences used in DQN learning
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer with proper type conversion"""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array([int(action)], dtype=np.int64)
        reward = float(reward)
        done = bool(done)
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Explicit conversion to avoid errors with numpy 2.x
        states = torch.tensor(np.stack([e.state for e in experiences], axis=0).astype(np.float32))
        actions = torch.tensor(np.stack([e.action for e in experiences], axis=0).astype(np.int64))
        rewards = torch.tensor(np.array([[e.reward] for e in experiences], dtype=np.float32))
        next_states = torch.tensor(np.stack([e.next_state for e in experiences], axis=0).astype(np.float32))
        dones = torch.tensor(np.array([[int(e.done)] for e in experiences], dtype=np.float32))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of the buffer"""
        return len(self.buffer)



class QAgent(Agent):

    """
    Q-Learning agent that uses a table to store action values
    """
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # State-action value table
        self.current_state = None
        self.previous_action = None

        super().__init__(agent_id)

    def get_state_key(self, state):
        """Convert state to hashable key for the Q-table"""
        # Round every entry to 2 decimals before making the key for stability
        rounded = np.round(state, 2)
        return tuple(rounded.flatten())

    def get_q_values(self, state):
        """Get Q-values for a state, initializing if not yet present"""
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key]

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_size)
        # Exploitation: best known action
        return int(np.argmax(self.get_q_values(state)))

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values based on transition"""
        state_key = self.get_state_key(state)
        q_values = self.get_q_values(state)
        
        # If terminal state, no future reward
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        
        # Q-learning update rule
        target = reward + self.gamma * np.max(next_q_values)
        old_q_value = q_values[action]
        q_values[action] += self.learning_rate * (target - q_values[action])
        
        # Log update for debugging
        print(f"Q-update: State={state_key}, Action={action}, Old={old_q_value:.4f}, New={q_values[action]:.4f}, Reward={reward:.4f}")
        
        # Update Q-table
        self.q_table[state_key] = q_values
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def start_episode(self, state):
        """Initialize agent state at the start of an episode"""
        self.current_state = state
        action = self.select_action(state)
        self.previous_action = action
        return action

    def step(self, reward, next_state, done):
        """Process one step of the environment and learn"""
        # Update Q-table on last transition if there was one
        if self.current_state is not None and self.previous_action is not None:
            self.learn(self.current_state, self.previous_action, reward, next_state, done)

        # Move to next state and choose next action
        self.current_state = next_state
        action = self.select_action(self.current_state)
        self.previous_action = action
        return action


class DQNNetwork(nn.Module):
    """
    Neural network for Deep Q-Learning
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(Agent):
    """
    Deep Q-Network agent using neural networks to approximate Q-values
    """
    def __init__(self, state_size, action_size, agent_id=0, hidden_size=64,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.6, batch_size=64,
                 update_target_every=10):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0

        super().__init__(agent_id)
        
        # Neural networks
        self.policy_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()
        self.current_state = None
        self.previous_action = None

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_size)
            
        # Exploitation: use policy network to select best action
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        return int(np.argmax(action_values.cpu().data.numpy()))

    def learn(self, experiences):
        """Update policy network using batch of experiences"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get Q values for current states using policy network
        q_expected = self.policy_network(states).gather(1, actions)
        
        # Get max Q values for next states using target network
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute target Q values using Bellman equation
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Compute loss and update policy network
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)

    def start_episode(self, state):
        """Initialize agent state at the start of an episode"""
        self.current_state = state
        action = self.select_action(state)
        self.previous_action = action
        return action

    def step(self, reward, next_state, done):
        """Process one step of the environment and learn"""
        # Store experience in memory
        if self.current_state is not None and self.previous_action is not None:
            self.remember(self.current_state, self.previous_action, reward, next_state, done)
            
            # Learn if we have enough samples
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
                
        # Move to next state and choose next action
        self.current_state = next_state
        action = self.select_action(next_state)
        self.previous_action = action
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return action


class SocialRewardCalculator:
    """
    Calculates social rewards based on individual and group satisfaction
    alpha: weight between own satisfaction and others' satisfaction
    beta: weight between immediate satisfaction and historical satisfaction
    """
    def __init__(self, 
                 nb_agents, 
                 alpha=0.5, 
                 beta=0.5, 
                 threshold = 0.7,
                 smoothing = 'linear',
                 sigmoid_gain = 10):
        

        self.nb_agents = nb_agents
        self.alpha = alpha  
        self.beta = beta 
        self.threshold = threshold
        self.smoothing = smoothing
        self.sigmoid_gain = sigmoid_gain  


    def consumption_rate(self,agent): 
        """return average consumption [0,1] based on meal_history"""
        if not agent.meal_history: 
            return 0.0 
        return sum(agent.meal_history)/len(agent.meal_history)
    
    def linear_emotions(self,rate):
         """Linear mapping from rate to emotion signal [-1,1] based on threshold."""
         threshold = self.threshold
         if rate >= threshold :
             if threshold < 1 :  
                return (rate - threshold)/( 1-threshold)
             else : 
                 return 1    
         return - (threshold - rate )/threshold if threshold > 0 else -1 
             
def sigmoid_emotion(self, rate):
        """Sigmoid mapping from rate to emotion signal [-1,1] centered at threshold."""
        g = self.sigmoid_gain
        t = self.threshold
        norm = (rate - t) / (1 - t) if rate >= t else (rate - t) / t
        exp_val = np.exp(-g * norm)
        return 2.0 / (1.0 + exp_val) - 1.0

def emotion_from_rate(self, rate):
        """Compute emotion signal from consumption rate using smoothing."""
        if self.smoothing == 'sigmoid':
            return self.sigmoid_emotion(rate)
        return self.linear_emotion(rate)

def calculate_personal_satisfaction(self, agent):
        """Compute personal satisfaction from last meal and history."""
        if agent.meal_history:
            last = 1.0 if agent.meal_history[-1] > 0 else 0.0
            hist = sum(agent.meal_history) / len(agent.meal_history)
        else:
            last, hist = 0.0, 0.0
        return self.beta * last + (1 - self.beta) * hist

def calculate_emotions(self, agents):
        """Return list of emotion signals for each agent."""
        return [self.emotion_from_rate(self.consumption_rate(a)) for a in agents]

def calculate_rewards(self, agents):
        """
        Compute and return:
        - emotions   : list of emotion signals ([-1,1])
        - personal   : list of personal satisfaction values ([0,1])
        - empathic   : list of empathic signals ([-1,1])
        - total      : list of total rewards (personal + empathic)
        """
        personal = [self.calculate_personal_satisfaction(a) for a in agents]

        emotions = self.calculate_emotions(agents)

        # 3) empathic reward
        empathic_reward = []
        for idx, emo in enumerate(emotions):
            #  other agents mean emotion
            others_emo = np.mean([e for j, e in enumerate(emotions) if j != idx])
            # formation of the list of empathic reward
            empathic_reward.append(others_emo)

        # 4) total reward: we keep personal satisfaction + empathic reward
        total = [(1 - self.alpha) * pers + self.alpha * emp for pers, emp in zip(personal, empathic_reward)]

        return emotions, personal, empathic_reward, total

