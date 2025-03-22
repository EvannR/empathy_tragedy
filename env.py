import numpy as np
from agents import Agent
import time 
import os


class GridMaze:
    def __init__(self, size=4, nb_agents=1, agent_configs=None):
        self.size = size
        self.nb_agents = nb_agents
        self.number_actions = 5  # UP, DOWN, LEFT, RIGHT, EXPLOIT
        self.actions = np.arange(self.number_actions)
        
        # Initialisation des positions des agents
        self.agents_positions = self.initialize_positions()
        
        # Création des agents avec leurs positions
        if agent_configs is None:
            agent_configs = [{'memory_size': 10} for _ in range(nb_agents)]
        
        self.agents = []
        for i in range(nb_agents):
            config = agent_configs[i] if i < len(agent_configs) else {'memory_size': 10}
            memory_size = config.get('memory_size', 10)
            self.agents.append(Agent(i, self.agents_positions[i], memory_size))
        
        # Initialisation des récompenses
        self.rewards = np.zeros((size, size))

        self.init_transitions()
        self.time_step = 0

    def initialize_positions(self):
        """Place les agents aléatoirement sur la grille."""
        positions = set()
        while len(positions) < self.nb_agents:
            i, j = np.random.randint(0, self.size, size=2)
            positions.add((i, j))
        return list(positions)

    def init_transitions(self):
        UP, DOWN, LEFT, RIGHT, EXPLOIT = 0, 1, 2, 3, 4
        self.P = {}
        
        for i in range(self.size):
            for j in range(self.size):
                self.P[(i, j)] = {}
                
                # UP action
                if i == 0:
                    self.P[(i, j)][UP] = (i, j)
                else:
                    self.P[(i, j)][UP] = (i - 1, j)
                
                # DOWN action
                if i == self.size - 1:
                    self.P[(i, j)][DOWN] = (i, j)
                else:
                    self.P[(i, j)][DOWN] = (i + 1, j)
                
                # LEFT action
                if j == 0:
                    self.P[(i, j)][LEFT] = (i, j)
                else:
                    self.P[(i, j)][LEFT] = (i, j - 1)
                
                # RIGHT action
                if j == self.size - 1:
                    self.P[(i, j)][RIGHT] = (i, j)
                else:
                    self.P[(i, j)][RIGHT] = (i, j + 1)
                
                # EXPLOIT action
                self.P[(i, j)][EXPLOIT] = (i, j)

    def new_episode(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        self.time_step = 0
        self.agents_positions = self.initialize_positions()
        
        # Mise à jour des positions des agents
        for i, agent in enumerate(self.agents):
            agent.update_position(self.agents_positions[i])

    def update_environment(self):
        """Met à jour l'environnement à chaque pas de temps"""
        self.time_step += 1
        # Dans la classe de base, rien n'est mis à jour

    def make_step(self, agent_idx, action):
        """Met à jour l'état d'un agent spécifique et enregistre les repas"""
        agent = self.agents[agent_idx]
        current_pos = agent.position
        new_pos = self.P[current_pos][action]
        
        # Met à jour la position de l'agent
        agent.update_position(new_pos)
        self.agents_positions[agent_idx] = new_pos
        
        # Vérifie s'il y a une récompense à cette position
        reward = self.rewards[new_pos]
        has_eaten = reward > 0
        
        # Enregistre si l'agent a mangé
        agent.record_meal(has_eaten, reward)
        
        return reward, new_pos
    
    def get_agent_meal_stats(self, agent_idx):
        """Retourne les statistiques de repas d'un agent"""
        agent = self.agents[agent_idx]
        return {
            'recent_meals': agent.get_recent_meals(),
            'total_meals': agent.total_meals,
            'meal_history': list(agent.meal_history)
        }
    
    def get_all_agents_meal_stats(self):
        """Retourne les statistiques de repas de tous les agents"""
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]


class RandomizedGridMaze(GridMaze):
    def __init__(self, size=4, nb_agents=1, agent_configs=None, reward_density=0.2, 
                 respawn_prob=0.1, simple_mode=True, auto_consume=True, 
                 exploit_only=False):
        super().__init__(size, nb_agents, agent_configs)
        self.reward_density = reward_density
        self.respawn_prob = respawn_prob
        self.simple_mode = simple_mode  # Mode simple ou complexe
        self.auto_consume = auto_consume  # Consommation automatique ou non
        self.exploit_only = exploit_only  # Consommation uniquement avec EXPLOIT
        self.initialize_rewards()
    
    def initialize_rewards(self):
        """Générer des récompenses aléatoires dans la grille."""
        self.rewards = np.zeros((self.size, self.size))
        num_rewards = int(self.size * self.size * self.reward_density)
        reward_positions = np.random.choice(self.size * self.size, num_rewards, replace=False)
        for pos in reward_positions:
            i, j = divmod(pos, self.size)
            self.rewards[i, j] = np.random.uniform(0.1, 1.0)
    
    def update_environment(self):
        """Met à jour l'environnement à chaque pas de temps"""
        super().update_environment()
        
        # En mode simple, on fait juste apparaître des ressources aléatoirement
        if self.simple_mode:
            # Probabilité d'apparition de nouvelles ressources
            if np.random.rand() < self.respawn_prob:
                empty_cells = np.argwhere(self.rewards == 0)
                if empty_cells.size > 0:
                    # Choisir une cellule vide au hasard
                    i, j = empty_cells[np.random.choice(len(empty_cells))]
                    self.rewards[i, j] = np.random.uniform(0.1, 1.0)
    
    def make_step(self, agent_idx, action):
        """Faire un pas et gérer la consommation des ressources."""
        agent = self.agents[agent_idx]
        current_pos = agent.position
        new_pos = self.P[current_pos][action]
        
        # Met à jour la position de l'agent
        agent.update_position(new_pos)
        self.agents_positions[agent_idx] = new_pos
        
        # Vérifier si l'agent peut consommer une ressource
        can_consume = True
        if self.exploit_only and action != 4:  # Si seul EXPLOIT permet de consommer
            can_consume = False
        
        reward = 0
        has_eaten = False
        
        if can_consume:
            reward = self.rewards[new_pos]
            has_eaten = reward > 0
            
            # Si l'agent a mangé et que la consommation est activée, retirer la ressource
            if has_eaten and self.auto_consume:
                self.rewards[new_pos] = 0
        
        # Enregistrer si l'agent a mangé
        agent.record_meal(has_eaten, reward)
        
        return reward, new_pos








# # test de l'environnement
# # fonction d'affichage avec les statistiques


# def display_grid_with_stats(env):
#     """Affiche la grille avec les agents, les ressources et les statistiques"""
#     grid = np.full((env.size, env.size), '.', dtype=str)  # Grille vide
    
#     # Ajouter les ressources avec leur valeur
#     for i, j in np.argwhere(env.rewards > 0):
#         # Afficher la valeur de la ressource
#         grid[i, j] = f'R{env.rewards[i, j]:.1f}'
    
#     # Ajouter les agents
#     for idx, agent in enumerate(env.agents):
#         i, j = agent.position
#         grid[i, j] = f'A{idx+1}' if grid[i, j] == '.' else f'A{idx+1}{grid[i, j][1:]}'
    
#     # Nettoyage écran
#     os.system('clear' if os.name == 'posix' else 'cls')
    
#     # Affichage de la grille
#     print(f"Grille à t={env.time_step} (R = ressource, A = agent):")
#     for row in grid:
#         print(' '.join([item.ljust(5) for item in row]))
#     print("\n")
    
#     # Statistiques des ressources
#     total_resources = np.sum(env.rewards > 0)
#     print(f"Ressources: {total_resources} (densité actuelle: {total_resources/(env.size*env.size):.2f})")
    
#     # Mode de fonctionnement
#     mode_str = "simple" if hasattr(env, 'simple_mode') and env.simple_mode else "complexe"
#     consume_str = "automatique" if hasattr(env, 'auto_consume') and env.auto_consume else "manuel"
#     exploit_str = "EXPLOIT uniquement" if hasattr(env, 'exploit_only') and env.exploit_only else "sur déplacement"
#     print(f"Mode: {mode_str}, Consommation: {consume_str}, Récolte: {exploit_str}")
    
#     # Affichage des statistiques de repas
#     print("\nStatistiques des repas :")
#     for idx, agent in enumerate(env.agents):
#         stats = env.get_agent_meal_stats(idx)
#         print(f"Agent {idx+1}: {stats['recent_meals']} repas récents, {stats['total_meals']} repas au total")
#         print(f"  Historique: {stats['meal_history']}")
#     print("\n")

# # Fonction de test de l'environnement
# def test_environment():
#     # Paramètres de la simulation
#     size = 6  # Taille de la grille
#     nb_agents = 3  # Nombre d'agents
#     steps = 30  # Nombre de pas à simuler
    
#     # Configurations individuelles des agents
#     agent_configs = [ {'memory_size': 10} for _ in range(nb_agents) ]

    
#     # Création de l'environnement
#     env = RandomizedGridMaze(
#         size=size, 
#         nb_agents=nb_agents,
#         agent_configs=agent_configs,
#         reward_density=0.2,    # 20% de densité initiale
#         respawn_prob=0.2,      # 20% de chance d'apparition par tour
#         simple_mode=True,      # Mode simple (pas de croissance complexe)
#         auto_consume=True,     # Consommation automatique des ressources
#         exploit_only=False     # Pas besoin d'utiliser EXPLOIT pour consommer
#     )
    
#     # Simulation
#     for step in range(steps):
#         display_grid_with_stats(env)  # Afficher l'état de la grille et les stats
        
#         print(f"Étape {step+1}/{steps}")
        
#         # Actions des agents
#         for agent_idx in range(nb_agents):
#             action = np.random.choice(env.actions)  # Action aléatoire pour chaque agent
#             reward, _ = env.make_step(agent_idx, action)
#             if reward > 0:
#                 print(f"Agent {agent_idx+1} a obtenu une récompense de {reward:.2f}")
        
#         # Mise à jour de l'environnement (apparition aléatoire de ressources)
#         env.update_environment()
        


# # Exécuter le test si ce fichier est exécuté directement
# if __name__ == "__main__":
#     test_environment()
