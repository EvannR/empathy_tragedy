import pygame
import numpy as np
from env import RandomizedGridMaze  # À adapter selon ton nom de fichier

# === PARAMÈTRES ===
CELL_SIZE = 60
GRID_SIZE = 6
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5
NB_AGENTS = 3

# === COULEURS ===
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (66, 135, 245)
GREEN = (50, 200, 50)
GREY = (180, 180, 180)

# === INITIALISATION PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("RL Grid Environment - Pygame")
clock = pygame.time.Clock()

# === INITIALISATION ENVIRONNEMENT ===
agent_configs = [
    {'memory_size': 5},
    {'memory_size': 10},
    {'memory_size': 15},
]

env = RandomizedGridMaze(
    size=GRID_SIZE,
    nb_agents=NB_AGENTS,
    agent_configs=agent_configs,
    reward_density=0.2,
    respawn_prob=0.2,
    simple_mode=True,
    auto_consume=True,
    exploit_only=False
)

# === FONCTION POUR AFFICHER LA GRILLE ===
def draw_grid(env):
    screen.fill(WHITE)

    for i in range(env.size):
        for j in range(env.size):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREY, rect, 1)

            # Ressources
            if env.rewards[i, j] > 0:
                pygame.draw.circle(screen, GREEN, rect.center, CELL_SIZE // 6)

    # Agents
    for idx, agent in enumerate(env.agents):
        i, j = agent.position
        center = (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, BLUE, center, CELL_SIZE // 3)

        # ID d'agent
        font = pygame.font.SysFont(None, 20)
        label = font.render(f"{idx+1}", True, WHITE)
        label_rect = label.get_rect(center=center)
        screen.blit(label, label_rect)

# === BOUCLE PRINCIPALE ===
running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Actions aléatoires pour chaque agent
    for idx in range(env.nb_agents):
        action = np.random.choice(env.actions)
        env.make_step(idx, action)

    env.update_environment()
    draw_grid(env)
    pygame.display.flip()

pygame.quit()
