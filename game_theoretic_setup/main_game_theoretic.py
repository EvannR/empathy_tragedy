from agent_policies_game_theoretic import QAgent, DQNAgent, SocialRewardCalculator
from env_game_theoretic import GameTheoreticEnv
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# definitions
agent_policy_name_to_class = {
    "QLearning": QAgent,
    "DQN": DQNAgent
}

env_name_to_class = {
    "Game_Theoretic": GameTheoreticEnv
}

emotions_params = {
    "high_empathy": {"alpha": 0.3, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "low_empathy": {"alpha": 0.8, "beta": 0.7}
}

###########
# change the parameters of the agents here
params_QLearning = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01
}

params_DQN = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 16,  # réduit pour des mises à jour plus fréquentes
    "hidden_size": 64,
    "update_target_every": 5  # réduit pour des mises à jour plus fréquentes
}

agent_params = {
    "QLearning": params_QLearning,
    "DQN": params_DQN
}
###########

###########
# choose what agent to test here. "QLearning" or "DQN"
agent_to_test = "DQN"
###########

###########
# choose what environment to test here.
env_to_test = "GameTheoreticEnv"
###########

###########
# choose which empathy level to test
empathy_to_test = "balanced"  # "high_empathy", "balanced", "low_empathy"
###########

###########
# choose the number of steps, the number of trials and the number of tests here
episodes = 50
steps = 30
nb_tests = 3
nb_agents = 3
env_size = nb_agents
np.random.seed(42)
###########

state_size = 1 # 1 emotionnal value


#########################################################################################################################

def run_single_test(agent_class, 
                    env_class, 
                    agent_config, 
                    env_config, 
                    emotion_config):
    """exécute un seul test avec les configurations données"""
    
    # création de l'environnement
    env = env_class(**env_config)
    
    # initialisation des agents d'apprentissage
    rl_agents = []
    
    for i in range(nb_agents):
        if agent_to_test == "QLearning":
            rl_agent = agent_class(
                state_size=state_size,
                action_size=env.number_actions,
                agent_id=i,
                **agent_config
            )
        else:  # DQN
            rl_agent = agent_class(
                state_size=state_size,
                action_size=env.number_actions,
                agent_id=i,
                **agent_config
            )
        rl_agents.append(rl_agent)
    
    # calculateur de récompense sociale
    reward_calculator = SocialRewardCalculator(
        nb_agents=nb_agents,
        alpha=emotion_config["alpha"],
        beta=emotion_config["beta"]
    )
    
    # variables pour stocker les résultats
    episode_rewards = []
    social_welfare = []
    
    # simulation des épisodes
    for episode in range(episodes):
        # réinitialisation de l'environnement
        env.new_episode()
        
        # initialisation des états des agents
        for idx, rl_agent in enumerate(rl_agents):
            initial_state = get_state_representation(env, idx)
            if hasattr(rl_agent, 'start_episode'):
                rl_agent.start_episode(initial_state)
            else:
                rl_agent.current_state = initial_state
                rl_agent.previous_action = None
        
        episode_reward = 0
        
        # boucle principale de l'épisode
        for step in range(steps):
            for idx, rl_agent in enumerate(rl_agents):
                current_state = get_state_representation(env, idx)
                
                action = rl_agent.select_action(current_state)
                immediate_reward, _ = env.make_step(idx, action)
                next_state = get_state_representation(env, idx)
                
                # pour DQN: apprentissage à chaque pas
                if agent_to_test == "DQN":
                    if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                        rl_agent.remember(rl_agent.current_state, rl_agent.previous_action,
                                         immediate_reward, next_state, False)
                    
                        if len(rl_agent.memory) > rl_agent.batch_size:
                            experiences = rl_agent.memory.sample(rl_agent.batch_size)
                            rl_agent.learn(experiences)
                
                    rl_agent.current_state = current_state
                    rl_agent.previous_action = action
                
            env.update_environment()
        
        social_rewards = reward_calculator.calculate_rewards(env.agents)
        
        for idx, rl_agent in enumerate(rl_agents):
            final_state = get_state_representation(env, idx)
            social_reward = social_rewards[idx]
            episode_reward += social_reward
            
            if agent_to_test == "QLearning":
                if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                    rl_agent.learn(rl_agent.current_state, rl_agent.previous_action, 
                                  social_reward, final_state, True)
            else:  # DQN - apprentissage final avec récompense sociale
                if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                    # stocker l'expérience terminale avec la récompense sociale
                    rl_agent.remember(rl_agent.current_state, rl_agent.previous_action,
                                     social_reward, final_state, True)
                
                    # un dernier apprentissage à la fin de l'épisode
                    if len(rl_agent.memory) > rl_agent.batch_size:
                        experiences = rl_agent.memory.sample(rl_agent.batch_size)
                        rl_agent.learn(experiences)
        
        episode_rewards.append(episode_reward)
        social_welfare.append(sum(social_rewards))
        
        if episode % 10 == 0:
            print(f"épisode {episode}/{episodes}, récompense: {episode_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'social_welfare': social_welfare
    }


# à modifier et réfléchir
def get_state_representation(env, agent_idx):
    """construit une représentation de l'état pour un agent"""
    agent = env.agents[agent_idx]
    
    state = np.zeros(10, dtype=np.float32)
    
    state[0] = pos_i / env.size
    state[1] = pos_j / env.size

    
    for idx, (di, dj) in enumerate(directions):
        ni, nj = pos_i + di, pos_j + dj
        if 0 <= ni < env.size and 0 <= nj < env.size:
            state[2 + idx] = env.rewards[ni, nj]
    
    return state.astype(np.float32)


# à modifier et réfléchir
def visualize_episode(agent_class, env_class, agent_config, env_config, emotion_config, num_episodes_training=30):
    """visualise un épisode avec les politiques apprises après un certain nombre d'épisodes d'entraînement"""
    # créer l'environnement
    env = env_class(**env_config)
    
    # créer et entraîner les agents
    rl_agents = []
    state_size = 10
    
    for i in range(nb_agents):
        rl_agent = agent_class(
            state_size=state_size,
            action_size=env.number_actions,
            agent_id=i,
            **agent_config
        )
        rl_agents.append(rl_agent)
    
    # calculateur de récompense sociale
    reward_calculator = SocialRewardCalculator(
        nb_agents=nb_agents,
        alpha=emotion_config["alpha"],
        beta=emotion_config["beta"]
    )
    
    # entraîner les agents pendant num_episodes_training épisodes
    print(f"entraînement des agents pendant {num_episodes_training} épisodes...")
    for episode in range(num_episodes_training):
        # réinitialisation de l'environnement
        env.new_episode()
        
        # initialisation des états des agents
        for idx, rl_agent in enumerate(rl_agents):
            initial_state = get_state_representation(env, idx)
            if hasattr(rl_agent, 'start_episode'):
                rl_agent.start_episode(initial_state)
            else:
                rl_agent.current_state = initial_state
                rl_agent.previous_action = None
        
        # boucle principale de l'épisode
        for step in range(steps):
            # chaque agent effectue une action
            for idx, rl_agent in enumerate(rl_agents):
                current_state = get_state_representation(env, idx)
                
                # sélection et exécution de l'action
                action = rl_agent.select_action(current_state)
                immediate_reward, _ = env.make_step(idx, action)
                next_state = get_state_representation(env, idx)
                
                # pour DQN: apprentissage à chaque pas
                if agent_to_test == "DQN":
                    if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                        rl_agent.remember(rl_agent.current_state, rl_agent.previous_action,
                                         immediate_reward, next_state, False)
                    
                        if len(rl_agent.memory) > rl_agent.batch_size:
                            experiences = rl_agent.memory.sample(rl_agent.batch_size)
                            rl_agent.learn(experiences)
                
                    rl_agent.current_state = current_state
                    rl_agent.previous_action = action
                
            env.update_environment()
        
        # calcul des récompenses sociales à la fin de l'épisode
        social_rewards = reward_calculator.calculate_rewards(env.agents)
        
        # mise à jour des agents RL avec les récompenses sociales
        for idx, rl_agent in enumerate(rl_agents):
            final_state = get_state_representation(env, idx)
            social_reward = social_rewards[idx]
            
            if agent_to_test == "QLearning":
                if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                    rl_agent.learn(rl_agent.current_state, rl_agent.previous_action, 
                                  social_reward, final_state, True)
            else:  # DQN
                if rl_agent.current_state is not None and rl_agent.previous_action is not None:
                    rl_agent.remember(rl_agent.current_state, rl_agent.previous_action,
                                     social_reward, final_state, True)
                
                    if len(rl_agent.memory) > rl_agent.batch_size:
                        experiences = rl_agent.memory.sample(rl_agent.batch_size)
                        rl_agent.learn(experiences)
                        
        if episode % 5 == 0:
            print(f"épisode d'entraînement {episode}/{num_episodes_training} terminé")
    
    print("\nentraînement terminé, visualisation du comportement des agents...")
    
    # réduire l'exploration pour mieux observer le comportement appris
    for agent in rl_agents:
        agent.epsilon = 0.05  # exploration minimale
    
    # réinitialiser l'environnement pour la démonstration
    env.new_episode()
    
    # initialiser les états des agents
    for idx, rl_agent in enumerate(rl_agents):
        initial_state = get_state_representation(env, idx)
        if hasattr(rl_agent, 'start_episode'):
            rl_agent.start_episode(initial_state)
        else:
            rl_agent.current_state = initial_state
            rl_agent.previous_action = None
    
    # exécuter un épisode en visualisant
    for step in range(steps):
        # afficher l'état actuel
        display_grid(env)
        
        # imprimer l'historique des repas actuel pour voir si les agents ont déjà mangé
        print("\nhistorique des repas avant action:")
        for idx, agent in enumerate(env.agents):
            has_eaten_recently = any(agent.meal_history[-3:])
            print(f"agent {idx+1}: {'a mangé récemment' if has_eaten_recently else 'n\'a pas mangé récemment'}")
        
        # chaque agent exécute une action selon sa politique apprise
        print("\nactions des agents:")
        for idx, rl_agent in enumerate(rl_agents):
            current_state = get_state_representation(env, idx)
            
            # utiliser epsilon=0 pour forcer l'exploitation (pas d'aléatoire)
            old_epsilon = rl_agent.epsilon
            rl_agent.epsilon = 0
            
            # sélectionner la meilleure action selon la politique apprise
            action = rl_agent.select_action(current_state)
            
            # restaurer epsilon
            rl_agent.epsilon = old_epsilon
            
            # montrer l'action choisie
            action_names = ["HAUT", "BAS", "GAUCHE", "DROITE", "EXPLOITER"]
            print(f"agent {idx+1} choisit l'action: {action_names[action]}")
            
            # exécuter l'action
            reward, _ = env.make_step(idx, action)
            
            # indiquer si l'agent a obtenu une récompense
            if reward > 0:
                print(f"agent {idx+1} a obtenu une récompense de {reward:.2f}")
        
        # mettre à jour l'environnement
        env.update_environment()
        
        time.sleep(1)  # pause pour observer
        print("\nappuyez sur entrée pour continuer au prochain pas...")
        input()


# à modifier et réfléchir
def display_grid(env):
    """affiche la grille avec les agents, les ressources et les statistiques"""
    grid = np.full((env.size, env.size), '.', dtype=str)
    
    # ajouter les ressources avec leur valeur
    for i, j in np.argwhere(env.rewards > 0):
        grid[i, j] = f'R{env.rewards[i, j]:.1f}'
    
    # ajouter les agents
    for idx, agent in enumerate(env.agents):
        i, j = agent.position
        grid[i, j] = f'A{idx+1}' if grid[i, j] == '.' else f'A{idx+1}{grid[i, j][1:]}'
    
    # nettoyage écran
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # affichage de la grille
    print(f"grille à t={env.time_step} (R = ressource, A = agent):")
    for row in grid:
        print(' '.join([item.ljust(5) for item in row]))
    print("\n")
    
    # statistiques des agents
    print("statistiques des agents:")
    for idx, agent in enumerate(env.agents):
        print(f"agent {idx+1}: {agent.get_recent_meals()} repas récents, {agent.total_meals} repas au total")
        print(f"  historique des repas: {list(agent.meal_history)}")

def plot_all():
    """exécute tous les tests et affiche les résultats"""
    # obtenir les classes et configurations
    agent_class = agent_policy_name_to_class[agent_to_test]
    env_class = env_name_to_class[env_to_test]
    agent_config = agent_params[agent_to_test]
    emotion_config = emotions_params[empathy_to_test]
    
    # configuration de l'environnement
    env_config = {
        'size': env_size,
        'nb_agents': nb_agents,
        'agent_configs': [{'memory_size': 10} for _ in range(nb_agents)],
        'reward_density': 0.2,
        'respawn_prob': 0.1,
        'simple_mode': True,
        'auto_consume': True,
        'exploit_only': False
    }
    
    # exécuter les tests
    all_rewards = []
    all_welfare = []
    
    print(f"exécution de {nb_tests} tests avec {agent_to_test} sur {env_to_test}")
    print(f"empathie: {empathy_to_test} (alpha={emotion_config['alpha']}, beta={emotion_config['beta']})")
    
    for test in range(nb_tests):
        print(f"\ntest {test+1}/{nb_tests}")
        results = run_single_test(agent_class, env_class, agent_config, env_config, emotion_config)
        all_rewards.append(results['episode_rewards'])
        all_welfare.append(results['social_welfare'])
    
    # calculer les moyennes et écarts-types
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    mean_welfare = np.mean(all_welfare, axis=0)
    std_welfare = np.std(all_welfare, axis=0)
    
    # afficher les résultats
    plt.figure(figsize=(15, 10))
    
    # graphique des récompenses
    plt.subplot(2, 1, 1)
    plt.plot(mean_rewards, label=f'{agent_to_test} - récompense moyenne')
    plt.fill_between(range(len(mean_rewards)), 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     alpha=0.2)
    plt.title(f'récompenses moyennes par épisode - {agent_to_test}')
    plt.xlabel('épisode')
    plt.ylabel('récompense')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # graphique du bien-être social
    plt.subplot(2, 1, 2)
    plt.plot(mean_welfare, label=f'{agent_to_test} - bien-être social moyen')
    plt.fill_between(range(len(mean_welfare)), 
                     mean_welfare - std_welfare, 
                     mean_welfare + std_welfare, 
                     alpha=0.2)
    plt.title(f'bien-être social moyen par épisode - {agent_to_test}')
    plt.xlabel('épisode')
    plt.ylabel('bien-être social')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results_{agent_to_test}_{empathy_to_test}.png')
    plt.show()
    
    # afficher les statistiques finales
    print("\n===== résultats finaux =====")
    print(f"agent: {agent_to_test}, empathie: {empathy_to_test}")
    print(f"récompense finale moyenne: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"bien-être social final moyen: {mean_welfare[-1]:.2f} ± {std_welfare[-1]:.2f}")
    
    # visualiser un épisode avec la politique apprise
    print("\nvisualisation d'un épisode avec la politique apprise:")
    visualize_episode(agent_class, env_class, agent_config, env_config, emotion_config)

# créer l'agent et exécuter les tests
if __name__ == "__main__":
    plot_all()