# vizualize_exploration.py
import pygame
import time
from agents_policies import QAgent, DQNAgent, SocialRewardCalculator
from env import RandomizedGridMaze
import numpy as np

# === PARAMÈTRES MODIFIABLES ===
agent_type = "DQN"  # "QLearning" ou "DQN"
empathy = "high_empathy"  # "high_empathy", "balanced", "low_empathy"

env_config = {
    'size': 6,
    'nb_agents': 3,
    'agent_configs': [{'memory_size': 10} for _ in range(3)],
    'reward_density': 0.2,
    'respawn_prob': 0.2,
    'simple_mode': True,
    'auto_consume': True,
    'exploit_only': False
}

agent_params = {
    "QLearning": {
        "learning_rate": 0.1,
        "gamma": 0.99,
        "epsilon": 0.8,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01
    },
    "DQN": {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "batch_size": 16,
        "hidden_size": 64,
        "update_target_every": 5
    }
}

emotions_params = {
    "high_empathy": {"alpha": 0.3, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "low_empathy": {"alpha": 0.8, "beta": 0.7}
}

# === INITIALISATION ===
agents_classes = {"QLearning": QAgent, "DQN": DQNAgent}
agents_class = agents_classes[agent_type]
agents_cfg = agent_params[agent_type]
emotion_cfg = emotions_params[empathy]

env = RandomizedGridMaze(**env_config)
agents = [agents_class(state_size=10, action_size=env.number_actions, agent_id=i, **agents_cfg) for i in range(env_config['nb_agents'])]
reward_calc = SocialRewardCalculator(env_config['nb_agents'], **emotion_cfg)

# === INIT PYGAME ===
pygame.init()
tile_size = 80
grid_size = env_config['size']
screen = pygame.display.set_mode((tile_size * grid_size, tile_size * grid_size + 100))
pygame.display.set_caption("Exploration des agents")
font = pygame.font.SysFont(None, 24)
colors = {
    'background': (30, 30, 30),
    'agent': (255, 255, 0),
    'resource': (0, 200, 0),
    'text': (255, 255, 255)
}

def draw_grid():
    screen.fill(colors['background'])
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * tile_size, i * tile_size, tile_size, tile_size)
            pygame.draw.rect(screen, (80, 80, 80), rect, 1)
            if env.rewards[i, j] > 0:
                pygame.draw.circle(screen, colors['resource'], rect.center, tile_size // 4)

    for agent in env.agents:
        x, y = agent.position[1], agent.position[0]
        center = (x * tile_size + tile_size // 2, y * tile_size + tile_size // 2)
        pygame.draw.circle(screen, colors['agent'], center, tile_size // 3)

    stats = " | ".join([f"A{i}: {agent.get_recent_meals()} repas" for i, agent in enumerate(env.agents)])
    text_surface = font.render(stats, True, colors['text'])
    screen.blit(text_surface, (10, tile_size * grid_size + 10))
    pygame.display.flip()

def main_loop(steps=100):
    running = True
    env.new_episode()
    for i, agent in enumerate(agents):
        agent.start_episode(env.agents[i].get_state(env))

    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        for idx, agent in enumerate(agents):
            state = env.agents[idx].get_state(env)
            action = agent.select_action(state)
            current_pos = env.agents[idx].position
            if env.rewards[current_pos] > 0 and action != 4:
                print(f"[OBSERVE] Agent {idx} a ignoré une ressource à {current_pos} (reward={env.rewards[current_pos]:.2f})")
            reward, _ = env.make_step(idx, action)
            next_state = env.agents[idx].get_state(env)
            agent.step(next_state, reward, False)

        env.update_environment()
        draw_grid()
        time.sleep(0.3)

    pygame.quit()

if __name__ == "__main__":
    main_loop(steps=100)
