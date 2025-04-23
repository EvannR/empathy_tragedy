import numpy as np
import matplotlib.pyplot as plt
from agents_policies import QAgent

class MultiAgentGrid:
    def __init__(self, size=5, nb_agents=5, reward_density=0.1):
        self.size = size
        self.nb_agents = nb_agents
        self.nb_actions = 5  # UP, DOWN, LEFT, RIGHT, EXPLOIT
        self.reward_density = reward_density
        self.reset()

    def reset(self):
        self.rewards = np.zeros((self.size, self.size))
        num_rewards = int(self.size * self.size * self.reward_density)
        positions = np.random.choice(self.size * self.size, num_rewards, replace=False)
        for pos in positions:
            i, j = divmod(pos, self.size)
            self.rewards[i, j] = 1.0  # simple binaire
        self.agent_positions = [tuple(np.random.randint(0, self.size, 2)) for _ in range(self.nb_agents)]
        self.meals = [0 for _ in range(self.nb_agents)]
        return self.get_all_states()

    def move(self, pos, action):
        i, j = pos
        if action == 0 and i > 0: i -= 1
        elif action == 1 and i < self.size - 1: i += 1
        elif action == 2 and j > 0: j -= 1
        elif action == 3 and j < self.size - 1: j += 1
        return (i, j)

    def step(self, agent_idx, action):
        pos = self.agent_positions[agent_idx]
        if action == 4:  # EXPLOIT
            reward = self.rewards[pos]
            if reward > 0:
                self.rewards[pos] = 0
                self.meals[agent_idx] += 1
            return pos, reward, False
        else:
            new_pos = self.move(pos, action)
            self.agent_positions[agent_idx] = new_pos
            return new_pos, -0.01, False

    def get_state(self, pos):
        on_reward = int(self.rewards[pos] > 0)
        return np.array([pos[0]/self.size, pos[1]/self.size, on_reward], dtype=np.float32)

    def get_all_states(self):
        return [self.get_state(pos) for pos in self.agent_positions]

# === Simulation ===
EPISODES = 100
STEPS = 50
grid = MultiAgentGrid()
agents = [QAgent(state_size=3, action_size=5, agent_id=i) for i in range(grid.nb_agents)]

total_rewards = []
inequality = []

for ep in range(EPISODES):
    grid.reset()
    states = grid.get_all_states()
    for agent in agents:
        agent.start_episode(np.array([0.0, 0.0, 0.0]))  # dummy init

    episode_rewards = [0] * grid.nb_agents

    for _ in range(STEPS):
        for i, agent in enumerate(agents):
            state = grid.get_state(grid.agent_positions[i])
            action = agent.select_action(state)
            new_pos, reward, _ = grid.step(i, action)
            next_state = grid.get_state(new_pos)
            agent.learn(state, action, reward, next_state, False)
            episode_rewards[i] += reward

    total_rewards.append(np.mean(episode_rewards))
    inequality.append(np.std(grid.meals))  # inégalité entre agents

    if (ep+1) % 10 == 0:
        print(f"Épisode {ep+1} | Récompense moyenne: {np.mean(episode_rewards):.2f} | Injustice: {inequality[-1]:.2f}")

# === Affichage ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_rewards)
plt.title("Récompense moyenne par épisode")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(inequality)
plt.title("Inégalité sociale (écart-type des repas)")
plt.grid()
plt.tight_layout()
plt.show()
