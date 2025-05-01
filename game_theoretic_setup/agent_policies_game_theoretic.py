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

    def record_meal(self, has_eaten, reward_value=0):
        """Register whether the agent has eater this turn (1) or not (0) at this step"""
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1

    def get_recent_meals(self):
        """Sends the reward in the historic"""
        return sum(self.meal_history)

    def reset(self):
        """Reseting the agent for a new episode"""
        self.state = self.env.reset()
        self.meal_history = deque([0] * self.memory_size, maxlen=self.memory_size)
        self.total_meals = 0


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


class QAgent(Agent):
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, memory_size=10):
        super().__init__(agent_id, memory_size=memory_size)  # Hérite de Agent

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}

        self.current_state = None
        self.previous_action = None

    def get_state_key(self, state):
        """Convert the state for the Q-table."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        else:

            return (state,)

    def get_q_values(self, state):
        """obtains the q values for a given agent and state"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, state):
        """sélectionne une action selon la politique epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done):
        """Update of the Q-table"""
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

        self.policy_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()

        self.current_state = None
        self.previous_action = None

    def select_action(self, state):
        """Select and action according to epsilon-greedy"""
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

        self.meal_history.append(action)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def start_episode(self, state):
        """Make a new episode by initiating variables"""
        self.current_state = state
        self.previous_action = None


class SocialRewardCalculator:
    """
    Calculate rewards based on agents' consumption and configurable emotional smoothing.
    """
    def __init__(self, nb_agents, alpha=0.5, beta=0.5, threshold=0.5,
                 smoothing='linear', sigmoid_gain=10.0):
        """
        parameters:
        -----------
        nb_agents : int
            number of agents in the environment
        alpha : float
            weight on others' emotions vs. personal satisfaction (0-1)
        beta : float
            balance last vs. history consumption for personal satisfaction (0-1)
        threshold : float
            consumption rate threshold for neutral emotion (0-1)
        smoothing : str
            'linear' or 'sigmoid', type of mapping from rate to emotion
        sigmoid_gain : float
            gain ( steepness ) for sigmoid mapping
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
        """Linear mapping from rate to emotion [-1,1] based on threshold."""
        t = self.threshold
        if rate >= t:
            return (rate - t) / (1 - t) if t < 1 else 1.0
        else:
            return - (t - rate) / t if t > 0 else -1.0

    def _sigmoid_emotion(self, rate):
        """Sigmoid mapping from rate to emotion [-1,1] centered at threshold."""
        # logistic: 2/(1+exp(-g*(r-t)/(1-t))) - 1 maps rate=t -> 0, rate=1 -> +1, rate=0 -> -1
        g = self.sigmoid_gain
        t = self.threshold
        # normalize rate offset by threshold, scaled by available range
        norm = (rate - t) / (1 - t) if rate >= t else (rate - t) / t
        exp_val = np.exp(-g * norm)
        return 2.0 / (1.0 + exp_val) - 1.0

    def emotion_from_rate(self, rate):
        """Map a consumption rate [0,1] to emotion [-1,1] using chosen smoothing."""
        if self.smoothing == 'sigmoid':
            return self._sigmoid_emotion(rate)
        else:
            return self._linear_emotion(rate)

    def calculate_personal_satisfaction(self, agent):
        """
        Calculate personal satisfaction based on last meal and history.
        """
        if agent.meal_history:
            last_meal = 1 if agent.meal_history[-1] > 0 else 0
            history_w = sum(agent.meal_history) / len(agent.meal_history)
        else:
            last_meal = history_w = 0.0
        return self.beta * last_meal + (1 - self.beta) * history_w

    def calculate_emotions(self, agents):
        """
        Compute the emotion of each agent from their consumption rate.

        returns:
        --------
        list of float
            emotions in [-1,1]
        """
        return [self.emotion_from_rate(self._consumption_rate(a)) for a in agents]

    def calculate_rewards(self, agents):
        """
        Combine personal satisfaction and empathy into final rewards.

        returns:
        --------
        list of float
            reward for each agent
        """
        personal = [self.calculate_personal_satisfaction(a) for a in agents]
        emotions = self.calculate_emotions(agents)
        rewards = []

        for idx, (persat, emo) in enumerate(zip(personal, emotions)):
            others = np.mean([personal[i] for i in range(len(agents)) if i != idx])
            base = self.alpha * others + (1 - self.alpha) * persat
            # integrate emotion as additive factor
            rewards.append(base + emo)

        return rewards
