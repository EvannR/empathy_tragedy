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
                 memory_size=10):  # may be 7
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0

    def record_meal(self, success: bool, reward: float):
        """Record whether the agent successfully ate in this timestep."""
        self.meal_history.append(int(success))
        if success:
            self.total_meals += 1

    def get_recent_meals(self):
        """Returns the sum of recent meals in the history"""
        return sum(self.meal_history)

    def reset(self, observation=None):
        self.current_state = observation
        self.previous_action = None
        self.meal_history = deque([0]*10, maxlen=10)


Experience = namedtuple('Experience', ['state',
                                       'action',
                                       'reward',
                                       'next_state',
                                       'done'])


class ReplayBuffer:
    """Memory for the DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        if isinstance(action, (list, tuple)):
            action = np.array([action[0]], dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = np.array([action.item()], dtype=np.int64)
        else:
            action = np.array([int(action)], dtype=np.int64)

        state = np.array(state, dtype=np.float32)
        if not isinstance(reward, (int, float)):
            reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32)
        if not isinstance(done, bool):
            done = bool(done)

        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample in the batch"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.tensor([[e.reward] for e in experiences], dtype=torch.float)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.tensor([[int(e.done)] for e in experiences], dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QAgentMaze(Agent):
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, memory_size=10):
        super().__init__(agent_id, memory_size=memory_size)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Using a dictionary as Q-table for continuous state space
        self.q_table = {}
        
        # For discretizing continuous state space
        self.position_bins = 10  # Number of bins for x and y coordinates
        self.resource_bins = 6   # Number of bins for resource values (0-5)
        self.emotion_bins = 10   # Number of bins for emotion values (-1 to 1)

        self.current_state = None
        self.previous_action = None

    def discretize_state(self, state):
        """
        Convert continuous state values to discrete bins for Q-table lookup.
        Handles different state structures (with/without emotions).
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
            
        # Basic state includes (x, y, resource)
        x_bin = min(self.position_bins - 1, max(0, int(state[0] * self.position_bins)))
        y_bin = min(self.position_bins - 1, max(0, int(state[1] * self.position_bins)))
        resource_bin = min(self.resource_bins - 1, max(0, int(state[2] * self.resource_bins)))
        
        # Initialize discretized state
        discretized = [x_bin, y_bin, resource_bin]
        
        # Add emotion bins if present
        for i in range(3, len(state)):
            # Emotion values are typically between -1 and 1
            emotion_value = state[i]
            # Map from [-1, 1] to [0, emotion_bins-1]
            emotion_bin = min(self.emotion_bins - 1, 
                             max(0, int((emotion_value + 1) * self.emotion_bins / 2)))
            discretized.append(emotion_bin)
            
        return tuple(discretized)

    def get_q_values(self, state):
        """obtains the q values for a given agent and state"""
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, state):
        """selects an action according to epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done):
        """Update of the Q-table"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        target = reward + self.gamma * np.max(next_q_values) 
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])
        self.q_table[state_key] = q_values

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def start_episode(self, state):
        """Initialize a new simulation"""
        self.current_state = state
        self.previous_action = None

    def step(self, next_state, reward, done):
        """Initiate a new step in a simulation"""
        if self.current_state is not None and self.previous_action is not None:
            self.learn(self.current_state, self.previous_action, reward, next_state, done)

        self.current_state = next_state
        action = self.select_action(next_state)
        self.previous_action = action

        return action


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()

        # Enhanced network architecture for maze environment
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, agent_id=0, hidden_size=64,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64,
                 update_target_every=10):
        super().__init__(agent_id)
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

        # Use the enhanced network for maze environment
        self.policy_network = DQN(state_size, action_size, hidden_size)
        self.target_network = DQN(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()

        self.current_state = None
        self.previous_action = None

    def select_action(self, state):
        """Select an action according to epsilon-greedy"""
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
        """Learn from experience"""
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
        """Stock the experience in memory"""
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
        """Make a new step in the simulation"""
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def start_episode(self, state):
        """Make a new episode by initiating variables"""
        self.current_state = state
        self.previous_action = None


# Keep the SocialRewardCalculator from the original file
class SocialRewardCalculator:
    """
    Calculate rewards based on agents' consumption and configurable emotional smoothing.
    """
    def __init__(self, nb_agents, alpha=0.5, beta=0.5, threshold=0.7,
                 smoothing='linear', sigmoid_gain=10.0):
        """
        parameters:
        -----------
        nb_agents : int
            number of agents in the environment
        alpha : float
            weight on others' satisfaction vs. personal satisfaction (0-1)
        beta : float
            balance last vs. history consumption for personal satisfaction (0-1)
        threshold : float
            consumption rate threshold for neutral emotion (0-1)
        smoothing : str
            'linear' or 'sigmoid', type of mapping from rate to emotion signal
        sigmoid_gain : float
            steepness for sigmoid mapping
        """
        self.nb_agents = nb_agents
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.smoothing = smoothing
        self.sigmoid_gain = sigmoid_gain

    def _consumption_rate(self, agent):
        """Return average consumption rate [0,1] based on binary meal_history."""
        if not agent.meal_history:
            return 0.0
        return sum(agent.meal_history) / len(agent.meal_history)

    def _linear_emotion(self, rate):
        """Linear mapping from rate to emotion signal [-1,1] based on threshold."""
        t = self.threshold
        if rate >= t:
            return (rate - t) / (1 - t) if t < 1 else 1.0
        return - (t - rate) / t if t > 0 else -1.0

    def _sigmoid_emotion(self, rate):
        """Sigmoid mapping from rate to emotion signal [-1,1] centered at threshold."""
        g = self.sigmoid_gain
        t = self.threshold
        norm = (rate - t) / (1 - t) if rate >= t else (rate - t) / t
        exp_val = np.exp(-g * norm)
        return 2.0 / (1.0 + exp_val) - 1.0

    def emotion_from_rate(self, rate):
        """Compute emotion signal from consumption rate using smoothing."""
        if self.smoothing == 'sigmoid':
            return self._sigmoid_emotion(rate)
        return self._linear_emotion(rate)

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
        return [self.emotion_from_rate(self._consumption_rate(a)) for a in agents]

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
            # moyenne des émotions des autres agents
            other_emotions = [e for j, e in enumerate(emotions) if j != idx]
            if len(other_emotions) == 0:
                others_emo = 0.0  # <- safe default for solo agent
            else:
                others_emo = np.mean(other_emotions)
            empathic_reward.append(others_emo)

        # 4) total reward: on garde la satisfaction perso + empathic reward
        total = [(1 - self.alpha) * pers + self.alpha * emp for pers, emp in zip(personal, empathic_reward)]

        return emotions, personal, empathic_reward, total