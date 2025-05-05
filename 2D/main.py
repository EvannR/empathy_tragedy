# 
from agents_policies import QAgent, DQNAgent, SocialRewardCalculator
from env import RandomizedGridMaze
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging

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



# Default configuration for simulation
default_config = {
    "agent_type": "DQN",              # Agent learning algorithm
    "env_type": "RandomMaze",         # Environment type
    "empathy_level": "high_empathy",  # Social reward profile
    "episodes": 1000,                  # Number of episodes to run
    "steps_per_episode": 500,         # Steps per episode
    "num_tests": 3,                   # Number of test runs
    "env_size": 6,                    # Environment grid size
    "num_agents": 3,                  # Number of agents
    "seed": 42                        # Random seed for reproducibility
}


def run_single_test(agent_class, env_class, agent_config, env_config, emotion_config):
    """
    Run a single reinforcement learning test with the specified configuration.
    
    Args:
        agent_class: Class of the RL agent to use
        env_class: Class of the environment to use
        agent_config: Configuration parameters for the agents
        env_config: Configuration parameters for the environment
        emotion_config: Configuration parameters for social rewards
        
    Returns:
        Dictionary containing episode rewards and social welfare metrics
    """
    # Initialize environment and agents
    env = env_class(**env_config)
    rl_agents = []
    state_size = 10
    nb_agents = env_config['nb_agents']

    for i in range(nb_agents):
        rl_agent = agent_class(
            state_size=state_size, 
            action_size=env.number_actions,
            agent_id=i, 
            **agent_config)
        rl_agents.append(rl_agent)

    reward_calculator = SocialRewardCalculator(nb_agents=nb_agents,
                                             alpha=emotion_config["alpha"],
                                               beta=emotion_config["beta"])

    episode_rewards = []
    social_welfare = []
    episode_actions = []  # Track action distributions

    logger.info(f"Starting test with {agent_class.__name__} agents")

    for episode in range(env_config['episodes']):
        env.new_episode()
                # Track actions for this episode
        actions_this_episode = {i: [0, 0, 0, 0, 0] for i in range(nb_agents)}  # Initialize counters
        
        # --- Start of episode: get first actions ---
        actions = []  
        for idx, rl_agent in enumerate(rl_agents):
            state = env.agents[idx].get_state(env)
            action = rl_agent.start_episode(state)
            actions.append(action)
            actions_this_episode[idx][action] += 1

        episode_reward = 0.0  # récompense immédiate cumulée pendant les steps

        # --- Steps within episode ---
        for step in range(env_config['steps_per_episode']):
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
                actions_this_episode[idx][next_actions[idx]] += 1

            # Update environment dynamics
            env.update_environment()
            actions = next_actions

        # End of episode: social reward update
        social_rewards = reward_calculator.calculate_rewards(env.agents)

        for idx, rl_agent in enumerate(rl_agents):
            final_state = env.agents[idx].get_state(env)
            #social_reward = social_rewards[idx]
            rl_agent.step(social_rewards[idx], final_state, True)
        
        # Record metrics    
        episode_rewards.append(episode_reward)
        social_welfare.append(sum(social_rewards))
        episode_actions.append(actions_this_episode)

        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{env_config.get('episodes', 100)}, "
                       f"Reward: {episode_reward:.2f}, "
                       f"Social Welfare: {sum(social_rewards):.2f}")

    return {
        'episode_rewards': episode_rewards,
        'social_welfare': social_welfare,
        'actions': episode_actions
    }


def analyze_actions(all_actions, nb_agents, action_names=None):
    """
    Analyze action distributions across episodes.
    
    Args:
        all_actions: List of action distributions per episode
        nb_agents: Number of agents
        action_names: Dictionary mapping action indices to names
        
    Returns:
        Dictionary with action analysis
    """
    if action_names is None:
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "EXPLOIT"}
    
    # Initialize analysis dictionary
    analysis = {
        'by_agent': {},
        'overall': {name: [] for name in action_names.values()}
    }
    
    # Process each episode
    for episode_idx, episode_actions in enumerate(all_actions):
        # Process each agent
        for agent_idx in range(nb_agents):
            if agent_idx not in analysis['by_agent']:
                analysis['by_agent'][agent_idx] = {name: [] for name in action_names.values()}
            
            if agent_idx in episode_actions:
                agent_data = episode_actions[agent_idx]
                # Record percentage of each action type
                total_actions = sum(agent_data)
                for action_idx, count in enumerate(agent_data):
                    action_name = action_names.get(action_idx, f"ACTION_{action_idx}")
                    percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                    analysis['by_agent'][agent_idx][action_name].append(percentage)
                    
                    # Also update overall statistics
                    if episode_idx == 0:
                        analysis['overall'][action_name].append(percentage)
                    else:
                        analysis['overall'][action_name][episode_idx] += percentage
    
    # Average overall statistics across agents
    for action_name in action_names.values():
        analysis['overall'][action_name] = [x / nb_agents for x in analysis['overall'][action_name]]
    
    return analysis


def run_simulation(config=None):
    """
    Run a full simulation with multiple tests based on configuration.
    
    Args:
        config: Configuration dictionary (uses default_config if None)
        
    Returns:
        Dictionary with combined results from all tests
    """
    if config is None:
        config = default_config
    
    # Set random seed for reproducibility
    np.random.seed(config.get("seed", 42))
    
    # Get agent and environment classes
    agent_type = config.get("agent_type", "DQN")
    env_type = config.get("env_type", "RandomMaze")
    empathy_level = config.get("empathy_level", "high_empathy")
    
    agent_class = agent_policy_name_to_class[agent_type]
    env_class = env_name_to_class[env_type]
    
    # Get configuration parameters
    agent_config = agent_params[agent_type]
    emotion_config = emotions_params[empathy_level]
    
    env_config = {
        'size': config.get("env_size", 6),
        'nb_agents': config.get("num_agents", 3),
        'agent_configs': [{'memory_size': 10} for _ in range(config.get("num_agents", 3))],
        'reward_density': 0.2,
        'respawn_prob': 0.1,
        'simple_mode': True,
        'auto_consume': True,
        'exploit_only': False,
        'episodes': config.get("episodes", 100),
        'steps_per_episode': config.get("steps_per_episode", 500)
    }
    
    num_tests = config.get("num_tests", 3)
    
    # Storage for results from all tests
    all_rewards = []
    all_welfare = []
    all_actions = []
    
    
    # Calculate statistics across all tests
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_welfare = np.mean(all_welfare, axis=0)
    std_welfare = np.std(all_welfare, axis=0)
    
    # Analyze action distributions
    action_analysis = analyze_actions(all_actions, config.get("num_agents", 3))
    
    # Create results dictionary
    results = {
        'config': config,
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'mean_welfare': mean_welfare,
        'std_welfare': std_welfare,
        'action_analysis': action_analysis,
        'final_reward': mean_rewards[-1],
        'final_reward_std': std_rewards[-1],
        'final_welfare': mean_welfare[-1],
        'final_welfare_std': std_welfare[-1]
    }
    
    return results


def plot_results(results):
    """
    Create plots for simulation results.
    
    Args:
        results: Results dictionary from run_simulation
    """
    config = results['config']
    agent_type = config.get("agent_type", "DQN")
    empathy_level = config.get("empathy_level", "high_empathy")
    episodes = config.get("episodes", 100)
    
    mean_rewards = results['mean_rewards']
    std_rewards = results['std_rewards']
    mean_welfare = results['mean_welfare']
    std_welfare = results['std_welfare']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean episode rewards
    ax1.plot(mean_rewards, label=f'{agent_type} - Avg Reward', color='blue')
    ax1.fill_between(
        range(len(mean_rewards)),
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        color='blue'
    )
    ax1.set_title(f'Mean Episode Reward per Episode - {agent_type}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot mean social welfare
    ax2.plot(mean_welfare, label=f'{agent_type} - Avg Social Welfare', color='green')
    ax2.fill_between(
        range(len(mean_welfare)),
        mean_welfare - std_welfare,
        mean_welfare + std_welfare,
        alpha=0.2,
        color='green'
    )
    ax2.set_title(f'Mean Social Welfare per Episode - {agent_type}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Social Welfare')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add overall title and adjust layout
    plt.suptitle(f'Results for {agent_type} with {empathy_level} ({episodes} episodes)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for overall title
    
    # Save figure
    filename = f'results_{agent_type}_{empathy_level}_{episodes}_episodes.png'
    plt.savefig(filename, dpi=300)
   
    
    # Optional: show figure
    plt.show()
    plt.close(fig)
    
    # Plot action distribution over time
    action_analysis = results['action_analysis']
    
    # Plot overall action distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for action_name, percentages in action_analysis['overall'].items():
        ax.plot(percentages, label=action_name)
    
    ax.set_title(f'Action Distribution Evolution ({agent_type} with {empathy_level})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action Usage (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save action distribution figure
    action_filename = f'action_distribution_{agent_type}_{empathy_level}.png'
    plt.savefig(action_filename, dpi=300)
    logger.info(f"Action distribution plot saved as {action_filename}")
    
    # Optional: show figure
    plt.show()
    plt.close(fig)


def print_results_summary(results):
    """
    Print a summary of simulation results.
    
    Args:
        results: Results dictionary from run_simulation
    """
    config = results['config']
    agent_type = config.get("agent_type", "DQN")
    empathy_level = config.get("empathy_level", "high_empathy")
    
    final_reward = results['final_reward']
    final_reward_std = results['final_reward_std']
    final_welfare = results['final_welfare']
    final_welfare_std = results['final_welfare_std']
    
    print("\n===== SIMULATION RESULTS =====")
    print(f"Agent: {agent_type}, Empathy: {empathy_level}")
    print(f"Final Mean Reward: {final_reward:.2f} ± {final_reward_std:.2f}")
    print(f"Final Social Welfare: {final_welfare:.2f} ± {final_welfare_std:.2f}")
    
    # Action distribution at the end
    action_analysis = results['action_analysis']
    last_episode = len(results['mean_rewards']) - 1
    
    print("\nFinal Action Distribution:")
    for action_name, percentages in action_analysis['overall'].items():
        print(f"  {action_name}: {percentages[last_episode]:.1f}%")
    print("\nAgent-specific behavior:")
    
    for agent_idx, actions in action_analysis['by_agent'].items():
        print(f"Agent {agent_idx+1}:")
        for action_name, percentages in actions.items():
            print(f"  {action_name}: {percentages[last_episode]:.1f}%")


