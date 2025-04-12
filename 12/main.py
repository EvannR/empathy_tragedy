# main.py (modifié pour utiliser agent.get_state(env))
from agents_policies import QAgent, DQNAgent, SocialRewardCalculator
from env import RandomizedGridMaze
import numpy as np
import matplotlib.pyplot as plt
import time
import os

agent_policy_name_to_class = {
    "QLearning": QAgent,
    "DQN": DQNAgent
}

env_name_to_class = {
    "random_Maze": RandomizedGridMaze
}

emotions_params = {
    "high_empathy": {"alpha": 0.3, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "low_empathy": {"alpha": 0.8, "beta": 0.7}
}

params_QLearning = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0.8,
    "epsilon_decay": 0.9,
    "epsilon_min": 0.01
}

params_DQN = {
    "learning_rate": 0.01,
    "gamma": 0.99,
    "epsilon": 0.8,
    "epsilon_decay": 0.9,
    "epsilon_min": 0.01,
    "batch_size": 16,
    "hidden_size": 64,
    "update_target_every": 5
}

agent_params = {
    "QLearning": params_QLearning,
    "DQN": params_DQN
}

agent_to_test = "DQN"
env_to_test = "random_Maze"
empathy_to_test = "high_empathy"

episodes = 50
steps = 30
nb_tests = 3
env_size = 6
nb_agents = 3
np.random.seed(42)

def run_single_test(agent_class, env_class, agent_config, env_config, emotion_config):
    env = env_class(**env_config)
    rl_agents = []
    state_size = 10

    for i in range(nb_agents):
        rl_agent = agent_class(state_size=state_size, action_size=env.number_actions, agent_id=i, **agent_config)
        rl_agents.append(rl_agent)

    reward_calculator = SocialRewardCalculator(nb_agents=nb_agents, alpha=emotion_config["alpha"], beta=emotion_config["beta"])

    episode_rewards = []
    social_welfare = []

    for episode in range(episodes):
        env.new_episode()
        for idx, rl_agent in enumerate(rl_agents):
            initial_state = env.agents[idx].get_state(env)
            rl_agent.start_episode(initial_state)

        episode_reward = 0  # récompense immédiate cumulée pendant les steps

        for step in range(steps):
            for idx, rl_agent in enumerate(rl_agents):
                current_state = env.agents[idx].get_state(env)
                action = rl_agent.select_action(current_state)
                immediate_reward, _ = env.make_step(idx, action)
                episode_reward += immediate_reward
                next_state = env.agents[idx].get_state(env)

                if agent_to_test == "DQN":
                    rl_agent.step(next_state, immediate_reward, False)
                else:
                    rl_agent.step(next_state, immediate_reward, False)

            env.update_environment()

        social_rewards = reward_calculator.calculate_rewards(env.agents)
        for idx, rl_agent in enumerate(rl_agents):
            final_state = env.agents[idx].get_state(env)
            social_reward = social_rewards[idx]
            # on ne cumule plus les récompenses sociales dans episode_reward

            if agent_to_test == "DQN":
                rl_agent.step(final_state, social_reward, True)
            else:
                rl_agent.step(final_state, social_reward, True)

        episode_rewards.append(episode_reward)
        social_welfare.append(sum(social_rewards))

        if episode % 10 == 0:
            print(f"épisode {episode}/{episodes}, récompense: {episode_reward:.2f}")

    return {
        'episode_rewards': episode_rewards,
        'social_welfare': social_welfare
    }

def plot_all():
    agent_class = agent_policy_name_to_class[agent_to_test]
    env_class = env_name_to_class[env_to_test]
    agent_config = agent_params[agent_to_test]
    emotion_config = emotions_params[empathy_to_test]

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

    all_rewards = []
    all_welfare = []

    print(f"exécution de {nb_tests} tests avec {agent_to_test} sur {env_to_test}")
    print(f"empathie: {empathy_to_test} (alpha={emotion_config['alpha']}, beta={emotion_config['beta']})")

    for test in range(nb_tests):
        print(f"\ntest {test+1}/{nb_tests}")
        results = run_single_test(agent_class, env_class, agent_config, env_config, emotion_config)
        all_rewards.append(results['episode_rewards'])
        all_welfare.append(results['social_welfare'])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_welfare = np.mean(all_welfare, axis=0)
    std_welfare = np.std(all_welfare, axis=0)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(mean_rewards, label=f'{agent_to_test} - récompense moyenne')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.title(f'récompenses moyennes par épisode - {agent_to_test}')
    plt.xlabel('épisode')
    plt.ylabel('récompense')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(mean_welfare, label=f'{agent_to_test} - bien-être social moyen')
    plt.fill_between(range(len(mean_welfare)), mean_welfare - std_welfare, mean_welfare + std_welfare, alpha=0.2)
    plt.title(f'bien-être social moyen par épisode - {agent_to_test}')
    plt.xlabel('épisode')
    plt.ylabel('bien-être social')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results_{agent_to_test}_{empathy_to_test}.png')
    plt.show()

    print("\n===== résultats finaux =====")
    print(f"agent: {agent_to_test}, empathie: {empathy_to_test}")
    print(f"récompense finale moyenne: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"bien-être social final moyen: {mean_welfare[-1]:.2f} ± {std_welfare[-1]:.2f}")

if __name__ == "__main__":
    plot_all()