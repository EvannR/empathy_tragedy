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


episodes = 5000
steps = 1000
nb_tests = 1
env_size = 6
nb_agents = 5

np.random.seed(42)

def run_single_test(agent_class, env_class, agent_config, env_config, emotion_config):
    # Initialize environment and agents
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
        
        # --- Start of episode: get first actions ---
        actions = []  
        for idx, rl_agent in enumerate(rl_agents):
            state = env.agents[idx].get_state(env)
            action = rl_agent.start_episode(state)
            actions.append(action)

        episode_reward = 0.0  # récompense immédiate cumulée pendant les steps

        # --- Steps within episode ---
        for step in range(steps):
            next_actions = [None] * nb_agents
            for idx, rl_agent in enumerate(rl_agents):
                # Execute the chosen action
                #current_state = env.agents[idx].get_state(env)
                #action = rl_agent.select_action(current_state)
                immediate_reward, _ = env.make_step(idx, actions[idx])
                episode_reward += immediate_reward
                next_state = env.agents[idx].get_state(env)

                # Learn and choose next action
                next_actions[idx] = rl_agent.step(immediate_reward, next_state, False)

            # Update environment dynamics
            env.update_environment()
            actions = next_actions

        # --- End of episode: social reward update ---
        social_rewards = reward_calculator.calculate_rewards(env.agents)

        for idx, rl_agent in enumerate(rl_agents):
            final_state = env.agents[idx].get_state(env)
            #social_reward = social_rewards[idx]
            rl_agent.step(social_rewards[idx], final_state, True)
        
        # Record metrics    
        episode_rewards.append(episode_reward)
        social_welfare.append(sum(social_rewards))

        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}")

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

    print(f"Running {nb_tests} tests with {agent_to_test} on {env_to_test}")
    print(f"Empathy: {empathy_to_test} (alpha={emotion_config['alpha']}, beta={emotion_config['beta']})")

    for test in range(nb_tests):
        print(f"\nTest {test+1}/{nb_tests}")
        results = run_single_test(agent_class, env_class, agent_config, env_config, emotion_config)
        all_rewards.append(results['episode_rewards'])
        all_welfare.append(results['social_welfare'])

    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_welfare = np.mean(all_welfare, axis=0)
    std_welfare = np.std(all_welfare, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(mean_rewards, label=f'{agent_to_test} - Avg Reward')
    ax1.fill_between(range(len(mean_rewards)),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2)
    ax1.set_title(f'Mean Episode Reward per Episode - {agent_to_test}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(mean_welfare, label=f'{agent_to_test} - Avg Social Welfare')
    ax2.fill_between(range(len(mean_welfare)),
                     mean_welfare - std_welfare,
                     mean_welfare + std_welfare,
                     alpha=0.2)
    ax2.set_title(f'Mean Social Welfare per Episode - {agent_to_test}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Social Welfare')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'results_{agent_to_test}_{empathy_to_test}_{episodes}_episodes.png')
    #plt.show()
    plt.close(fig)

    print("\n===== final results =====")
    print(f"agent: {agent_to_test}, empathy: {empathy_to_test}")
    print(f"mean final reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"mean final social welfare: {mean_welfare[-1]:.2f} ± {std_welfare[-1]:.2f}")

if __name__ == "__main__":
    plot_all()