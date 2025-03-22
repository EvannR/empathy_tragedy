import numpy as np
import time
import os
from env import RandomizedGridMaze
from policies import QAgent, DQNAgent, SocialRewardCalculator

class MARLSimulation:
    def __init__(self, env_size=6, nb_agents=3, algorithm="q_learning", 
                 episodes=100, max_steps=50, alpha=0.5, beta=0.7):
        """
        Initialise la simulation MARL
        
        Parameters:
        -----------
        env_size : int
            Taille de l'environnement (grille carrée)
        nb_agents : int
            Nombre d'agents
        algorithm : str
            Algorithme d'apprentissage ("q_learning" ou "dqn")
        episodes : int
            Nombre d'épisodes pour l'apprentissage
        max_steps : int
            Nombre maximum d'étapes par épisode
        alpha : float
            Pondération entre satisfaction personnelle et empathie (0-1)
        beta : float
            Pondération du dernier repas par rapport à l'historique (0-1)
        """
        # Configuration de l'environnement
        self.env_size = env_size
        self.nb_agents = nb_agents
        self.max_steps = max_steps
        self.episodes = episodes
        
        # Configurations des agents
        agent_configs = [{'memory_size': 10} for _ in range(nb_agents)]
        
        # Création de l'environnement
        self.env = RandomizedGridMaze(
            size=env_size,
            nb_agents=nb_agents,
            agent_configs=agent_configs,
            reward_density=0.3,
            respawn_prob=0.1,
            simple_mode=True,
            auto_consume=True,
            exploit_only=False
        )
        
        # Taille de l'état pour un agent
        # Format: [position_i, position_j, ressources environnantes (8 directions)]
        self.state_size = 10
        
        # Création des agents d'apprentissage
        self.algorithm = algorithm
        self.rl_agents = []
        
        if algorithm == "q_learning":
            for i in range(nb_agents):
                self.rl_agents.append(QAgent(
                    state_size=self.state_size,
                    action_size=self.env.number_actions,
                    agent_id=i,
                    learning_rate=0.1,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01
                ))
        elif algorithm == "dqn":
            for i in range(nb_agents):
                self.rl_agents.append(DQNAgent(
                    state_size=self.state_size,
                    action_size=self.env.number_actions,
                    agent_id=i,
                    hidden_size=64,
                    learning_rate=0.001,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    epsilon_min=0.01,
                    batch_size=32
                ))
        else:
            raise ValueError(f"Algorithme inconnu: {algorithm}")
        
        # Calculateur de récompense sociale
        self.reward_calculator = SocialRewardCalculator(
            nb_agents=nb_agents,
            alpha=alpha,
            beta=beta
        )
        
        # Statistiques
        self.episode_rewards = []
        self.social_welfare = []
    
    def get_state_representation(self, agent_idx):
        """
        Construit une représentation de l'état pour un agent
        
        Returns:
        --------
        numpy.ndarray
            Représentation de l'état de l'agent
        """
        agent = self.env.agents[agent_idx]
        pos_i, pos_j = agent.position
        
        # Initialisation de l'état
        state = np.zeros(self.state_size, dtype=np.float32)
        
        # Position normalisée de l'agent
        state[0] = pos_i / self.env_size
        state[1] = pos_j / self.env_size
        
        # Ressources environnantes (8 directions)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for idx, (di, dj) in enumerate(directions):
            ni, nj = pos_i + di, pos_j + dj
            if 0 <= ni < self.env_size and 0 <= nj < self.env_size:
                state[2 + idx] = self.env.rewards[ni, nj]
        
        # Assurons-nous que le state est du bon type
        return state.astype(np.float32)
        
    def debug_info(self, episode, step):
        """
        Affiche des informations de débogage
        
        Parameters:
        -----------
        episode : int
            Numéro de l'épisode
        step : int
            Numéro de l'étape
        """
        print(f"\n==== Débogage - Épisode {episode}, Étape {step} ====")
        
        for idx, agent in enumerate(self.rl_agents):
            print(f"Agent {idx}:")
            print(f"  Position: {self.env.agents[idx].position}")
            
            if self.algorithm == "dqn":
                if hasattr(agent, 'memory') and len(agent.memory) > 0:
                    print(f"  Taille du buffer: {len(agent.memory)}")
                    
                    # Vérifiez le dernier élément du buffer
                    state, action, reward, next_state, done = agent.memory.buffer[-1]
                    print(f"  Dernier état: shape={state.shape}, type={state.dtype}")
                    print(f"  Dernière action: value={action}, type={type(action)}")
                    print(f"  Dernière récompense: {reward}")
                
                if hasattr(agent, 'current_state') and agent.current_state is not None:
                    print(f"  État courant: shape={agent.current_state.shape}, type={agent.current_state.dtype}")
                
                if hasattr(agent, 'previous_action') and agent.previous_action is not None:
                    print(f"  Action précédente: {agent.previous_action}, type={type(agent.previous_action)}")
        
        # Vérifiez l'état de l'environnement
        print(f"Environnement:")
        print(f"  Nombre de ressources: {np.sum(self.env.rewards > 0)}")
        print(f"  Temps écoulé: {self.env.time_step}")
        
        print("====================================\n")
    
    def run_episode(self, visualize=True, debug=False, episode_num=0):
        """
        Exécute un épisode complet
        
        Parameters:
        -----------
        visualize : bool
            Si True, affiche l'état de l'environnement
        debug : bool
            Si True, affiche des informations de débogage
        episode_num : int
            Numéro de l'épisode (pour le débogage)
            
        Returns:
        --------
        float
            Récompense totale pour cet épisode
        """
        # Réinitialisation de l'environnement
        self.env.new_episode()
        
        # États initiaux pour les agents RL
        for idx, rl_agent in enumerate(self.rl_agents):
            initial_state = self.get_state_representation(idx)
            rl_agent.start_episode(initial_state)
        
        episode_reward = 0
        
        # Boucle principale de l'épisode
        for step in range(self.max_steps):
            if debug and (step == 0 or step == self.max_steps - 1):
                self.debug_info(episode_num, step)
                
            if visualize:
                self.visualize()
                time.sleep(0.5)
            
            # Chaque agent effectue une action
            for idx, rl_agent in enumerate(self.rl_agents):
                try:
                    # Obtenir l'état actuel
                    current_state = self.get_state_representation(idx)
                    
                    # Sélectionner une action
                    action = rl_agent.select_action(current_state)
                    
                    # Exécuter l'action dans l'environnement
                    immediate_reward, _ = self.env.make_step(idx, action)
                    
                    # Stockage de l'expérience pour les agents DQN à chaque étape
                    # avec une récompense immédiate (option possible)
                    # if self.algorithm == "dqn":
                    #     next_state = self.get_state_representation(idx)
                    #     rl_agent.remember(current_state, action, immediate_reward, next_state, False)
                except Exception as e:
                    print(f"Erreur durant l'exécution de l'action de l'agent {idx} à l'étape {step}: {e}")
                    raise
            
            # Mise à jour de l'environnement
            self.env.update_environment()
        
        # Calcul des récompenses sociales à la fin de l'épisode
        social_rewards = self.reward_calculator.calculate_rewards(self.env.agents)
        
        # Mise à jour des agents RL avec les récompenses sociales
        for idx, rl_agent in enumerate(self.rl_agents):
            try:
                final_state = self.get_state_representation(idx)
                social_reward = social_rewards[idx]
                episode_reward += social_reward
                
                # Apprentissage avec la récompense sociale
                if self.algorithm == "q_learning":
                    if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                        rl_agent.learn(rl_agent.current_state, rl_agent.previous_action, 
                                      social_reward, final_state, True)
                else:  # DQN
                    if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                        rl_agent.remember(rl_agent.current_state, rl_agent.previous_action,
                                         social_reward, final_state, True)
                    
                    # Apprentissage à partir du replay buffer
                    if len(rl_agent.memory) > rl_agent.batch_size:
                        try:
                            experiences = rl_agent.memory.sample(rl_agent.batch_size)
                            rl_agent.learn(experiences)
                        except Exception as e:
                            print(f"Erreur pendant l'apprentissage de l'agent {idx}: {e}")
                            # Continuer malgré l'erreur
            except Exception as e:
                print(f"Erreur pendant la mise à jour de l'agent {idx} en fin d'épisode: {e}")
                raise
        
        # Bien-être social = somme des récompenses
        social_welfare = sum(social_rewards)
        
        return episode_reward, social_welfare
    
    def train(self, visualize_every=10, debug_first_episodes=2):
        """
        Entraîne les agents sur plusieurs épisodes
        
        Parameters:
        -----------
        visualize_every : int
            Fréquence de visualisation des épisodes
        debug_first_episodes : int
            Nombre d'épisodes initiaux pour lesquels activer le débogage
        """
        for episode in range(1, self.episodes + 1):
            visualize = (episode % visualize_every == 0)
            debug = (episode <= debug_first_episodes)  # Débogage pour les premiers épisodes
            
            try:
                # Exécuter l'épisode
                episode_reward, social_welfare = self.run_episode(
                    visualize=visualize, 
                    debug=debug,
                    episode_num=episode
                )
                
                # Enregistrer les statistiques
                self.episode_rewards.append(episode_reward)
                self.social_welfare.append(social_welfare)
                
                # Affichage des progrès
                print(f"Épisode {episode}/{self.episodes}, "
                      f"Récompense: {episode_reward:.2f}, "
                      f"Bien-être social: {social_welfare:.2f}")
                
                # Si on visualise, attendre avant le prochain épisode
                if visualize:
                    print("\nAppuyez sur Entrée pour continuer...")
                    input()
            
            except Exception as e:
                print(f"Erreur dans l'épisode {episode}: {e}")
                import traceback
                traceback.print_exc()
                
                # Continuer avec le prochain épisode
                print("Passage au prochain épisode...")
                continue
    
    def visualize(self):
        """Affiche l'état de l'environnement et les statistiques"""
        grid = np.full((self.env_size, self.env_size), '.', dtype=str)
        
        # Ajouter les ressources avec leur valeur
        for i, j in np.argwhere(self.env.rewards > 0):
            grid[i, j] = f'R{self.env.rewards[i, j]:.1f}'
        
        # Ajouter les agents
        for idx, agent in enumerate(self.env.agents):
            i, j = agent.position
            grid[i, j] = f'A{idx+1}' if grid[i, j] == '.' else f'A{idx+1}{grid[i, j][1:]}'
        
        # Nettoyage écran
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Affichage de la grille
        print(f"Grille à t={self.env.time_step} (R = ressource, A = agent):")
        for row in grid:
            print(' '.join([item.ljust(5) for item in row]))
        print("\n")
        
        # Statistiques des agents
        print("Statistiques des agents:")
        for idx, agent in enumerate(self.env.agents):
            stats = self.env.get_agent_meal_stats(idx)
            satisfaction = self.reward_calculator.calculate_personal_satisfaction(agent)
            epsilon = self.rl_agents[idx].epsilon if hasattr(self.rl_agents[idx], 'epsilon') else 0
            
            print(f"Agent {idx+1}: {stats['recent_meals']} repas récents, "
                  f"{stats['total_meals']} repas au total, "
                  f"Satisfaction: {satisfaction:.2f}, "
                  f"Epsilon: {epsilon:.2f}")
            print(f"  Historique des repas: {stats['meal_history']}")
        
        # Statistiques de l'algorithme
        print(f"\nAlgorithme: {self.algorithm}")
        
        # Statistiques globales
        if len(self.social_welfare) > 0:
            print(f"Bien-être social moyen: {np.mean(self.social_welfare):.2f}")


if __name__ == "__main__":
    # Paramètres de la simulation
    simulation = MARLSimulation(
        env_size=6,
        nb_agents=3,
        algorithm="q_learning",  # "q_learning" ou "dqn"
        episodes=50,
        max_steps=30,
        alpha=0.6,  # 60% satisfaction personnelle, 40% empathie
        beta=0.7    # 70% dernier repas, 30% historique
    )
    
    # Lancement de l'entraînement
    simulation.train(visualize_every=5)