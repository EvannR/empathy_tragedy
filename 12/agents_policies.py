# agents_policies.py
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
        self.meal_history.append(1 if has_eaten else 0)
        if has_eaten:
            self.total_meals += 1

    def get_recent_meals(self):
        return sum(self.meal_history)

    def update_position(self, new_position):
        self.position = new_position

    def get_state(self, env):
        pos_i, pos_j = self.position
        state = np.zeros(10, dtype=np.float32)
        state[0] = pos_i / env.size
        state[1] = pos_j / env.size
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


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array([int(action)], dtype=np.int64)
        reward = float(reward)
        done = bool(done)
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

    # Conversion explicite pour éviter les erreurs avec numpy 2.x
        states = torch.tensor(np.stack([e.state for e in experiences], axis=0).astype(np.float32))
        actions = torch.tensor(np.stack([e.action for e in experiences], axis=0).astype(np.int64))
        rewards = torch.tensor(np.array([[e.reward] for e in experiences], dtype=np.float32))
        next_states = torch.tensor(np.stack([e.next_state for e in experiences], axis=0).astype(np.float32))
        dones = torch.tensor(np.array([[int(e.done)] for e in experiences], dtype=np.float32))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QAgent:
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.agent_id = agent_id
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
        return tuple(state.flatten()) if isinstance(state, np.ndarray) else tuple(state)

    def get_q_values(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key]

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))
        return int(np.argmax(self.get_q_values(state)))

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        target = reward + self.gamma * np.max(next_q_values)
        q_values[action] += self.learning_rate * (target - q_values[action])
        self.q_table[state_key] = q_values
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def start_episode(self, state):
        self.current_state = state
        self.previous_action = None

    def step(self, next_state, reward, done):
        if self.current_state is not None and self.previous_action is not None:
            self.learn(self.current_state, self.previous_action, reward, next_state, done)
        self.current_state = next_state
        self.previous_action = self.select_action(next_state)
        return self.previous_action


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        return int(np.argmax(action_values.cpu().data.numpy()))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def step(self, next_state, reward, done):
        if self.current_state is not None and self.previous_action is not None:
            self.remember(self.current_state, self.previous_action, reward, next_state, done)
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
        self.current_state = next_state
        self.previous_action = self.select_action(next_state)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.previous_action

    def start_episode(self, state):
        self.current_state = state
        self.previous_action = None


class SocialRewardCalculator:
    def __init__(self, nb_agents, alpha=0.5, beta=0.5):
        self.nb_agents = nb_agents
        self.alpha = alpha
        self.beta = beta

    def calculate_personal_satisfaction(self, agent):
        last_meal = 1 if agent.meal_history[-1] > 0 else 0
        history_weight = sum(agent.meal_history) / len(agent.meal_history)
        return self.beta * last_meal + (1 - self.beta) * history_weight

    def calculate_rewards(self, agents):
        personal_satisfactions = [self.calculate_personal_satisfaction(agent) for agent in agents]
        rewards = []
        for idx, satisfaction in enumerate(personal_satisfactions):
            own_satisfaction = satisfaction
            others_satisfaction = np.mean([s for i, s in enumerate(personal_satisfactions) if i != idx])
            emotional_reward = self.alpha * own_satisfaction + (1 - self.alpha) * others_satisfaction
            rewards.append(emotional_reward)
        return rewards
