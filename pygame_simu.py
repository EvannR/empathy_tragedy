import pygame
import numpy as np
from env import RandomizedGridMaze  # Adapté au bon module

# === PARAMÈTRES ===
CELL_SIZE = 60
GRID_SIZE = 6
STATS_WIDTH = 300
VISIBLE_STATS_HEIGHT = GRID_SIZE * CELL_SIZE
STATS_SCROLL_AREA = 1000  # Espace total pour scroller dans la stats area
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + STATS_WIDTH
WINDOW_HEIGHT = VISIBLE_STATS_HEIGHT
FPS = 5
NB_AGENTS = 8

# === COULEURS ===
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (66, 135, 245)
GREEN = (50, 200, 50)
GREY = (180, 180, 180)
DARKGREY = (100, 100, 100)

# === INITIALISATION PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("RL Grid Environment - Pygame")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# === INITIALISATION ENVIRONNEMENT ===
agent_configs = [{'memory_size': 10} for _ in range(NB_AGENTS)]

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

# === SCROLLING ===
scroll_offset = 0
scroll_speed = 20

# === FONCTION POUR AFFICHER LA GRILLE + STATS ===
def draw_grid(env, scroll_offset):
    screen.fill(WHITE)

    # --- GRILLE ---
    for i in range(env.size):
        for j in range(env.size):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREY, rect, 1)

            # Ressources
            if env.rewards[i, j] > 0:
                pygame.draw.circle(screen, GREEN, rect.center, CELL_SIZE // 6)

    # --- AGENTS ---
    for idx, agent in enumerate(env.agents):
        i, j = agent.position
        center = (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, BLUE, center, CELL_SIZE // 3)

        label = font.render(f"{idx+1}", True, WHITE)
        label_rect = label.get_rect(center=center)
        screen.blit(label, label_rect)

    # --- STATS PANEL ---
    stats_surface = pygame.Surface((STATS_WIDTH, STATS_SCROLL_AREA))
    stats_surface.fill(WHITE)

    stats_x = 10
    y_offset = 10
    stats_surface.blit(font.render(f"Time Step: {env.time_step}", True, BLACK), (stats_x, y_offset))
    y_offset += 30
    total_resources = np.sum(env.rewards > 0)
    stats_surface.blit(font.render(f"Ressources restantes: {int(total_resources)}", True, BLACK), (stats_x, y_offset))
    y_offset += 30

    for idx, agent in enumerate(env.agents):
        stats = env.get_agent_meal_stats(idx)
        stats_surface.blit(font.render(f"Agent {idx+1}:", True, DARKGREY), (stats_x, y_offset))
        y_offset += 20
        stats_surface.blit(font.render(f"  Total meals: {stats['total_meals']}", True, BLACK), (stats_x, y_offset))
        y_offset += 20
        stats_surface.blit(font.render(f"  Recent: {stats['recent_meals']}", True, BLACK), (stats_x, y_offset))
        y_offset += 30

    screen.blit(stats_surface, (GRID_SIZE * CELL_SIZE, -scroll_offset))

# === BOUCLE PRINCIPALE ===
running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                scroll_offset = min(scroll_offset + scroll_speed, STATS_SCROLL_AREA - VISIBLE_STATS_HEIGHT)
            elif event.key == pygame.K_UP:
                scroll_offset = max(scroll_offset - scroll_speed, 0)

    for idx in range(env.nb_agents):
        action = np.random.choice(env.actions)
        env.make_step(idx, action)

    env.update_environment()
    draw_grid(env, scroll_offset)
    pygame.display.flip()

pygame.quit()