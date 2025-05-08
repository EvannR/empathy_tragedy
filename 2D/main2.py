from agents_policies import QAgent, DQNAgent, SocialRewardCalculator
from env import RandomizedGridMaze
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(episodes=1000, agent_type="DQN", empathy_level="balanced", 
                   env_size=6, num_agents=3, steps_per_episode=500, seed=42):
    """
    Run a simplified simulation with the specified parameters.
    
    Args:
        episodes: Number of episodes to run
        agent_type: Type of agent ("DQN" or "QLearning")
        empathy_level: Social reward profile ("high_empathy", "balanced", or "low_empathy")
        env_size: Environment grid size
        num_agents: Number of agents
        steps_per_episode: Steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Define parameters for each agent type
    agent_params = {
        "QLearning": {
            "learning_rate": 0.1,
            "gamma": 0.99,
            "epsilon": 0.8,
            "epsilon_decay": 0.9,
            "epsilon_min": 0.01
        },
        "DQN": {
            "learning_rate": 0.01,
            "gamma": 0.99,
            "epsilon": 0.8,
            "epsilon_decay": 0.9,
            "epsilon_min": 0.01,
            "batch_size": 16,
            "hidden_size": 64,
            "update_target_every": 5
        }
    }
    
    # Define emotional parameters
    emotions_params = {
        "high_empathy": {"alpha": 0.3, "beta": 0.7},
        "balanced": {"alpha": 0.5, "beta": 0.5},
        "low_empathy": {"alpha": 0.8, "beta": 0.7}
    }
    
    # Get agent class and parameters
    agent_class = QAgent if agent_type == "QLearning" else DQNAgent
    agent_config = agent_params[agent_type]
    emotion_config = emotions_params[empathy_level]
    
    # Initialize environment
    env_config = {
        'size': env_size,
        'nb_agents': num_agents,
        'agent_configs': [{'memory_size': 10} for _ in range(num_agents)],
        'reward_density': 0.2,
        'respawn_prob': 0.1,
        'simple_mode': True,
        'auto_consume': True,
        'exploit_only': False
    }
    env = RandomizedGridMaze(**env_config)
    
    # Initialize agents
    rl_agents = []
    state_size = 10  # Fixed state size for our environment
    
    for i in range(num_agents):
        rl_agent = agent_class(
            state_size=state_size, 
            action_size=env.number_actions,
            agent_id=i, 
            **agent_config)
        rl_agents.append(rl_agent)
    
    # Initialize reward calculator
    reward_calculator = SocialRewardCalculator(
        nb_agents=num_agents,
        alpha=emotion_config["alpha"],
        beta=emotion_config["beta"]
    )
    
    # Storage for results
    episode_rewards = []
    social_welfare = []
    agent_meals = [[] for _ in range(num_agents)]
    action_counts = []  # Track action distributions
    
    # Run simulation for specified number of episodes
    for episode in range(episodes):
        env.new_episode()
        
        # Track actions for this episode
        episode_actions = {i: [0, 0, 0, 0, 0] for i in range(num_agents)}
        
        # Start of episode: get first actions
        actions = []
        for idx, rl_agent in enumerate(rl_agents):
            state = env.agents[idx].get_state(env)
            action = rl_agent.start_episode(state)
            actions.append(action)
            episode_actions[idx][action] += 1
        
        episode_reward = 0.0
        
        # Steps within episode
        for step in range(steps_per_episode):
            next_actions = [None] * num_agents
            
            for idx, rl_agent in enumerate(rl_agents):
                # Execute chosen action
                immediate_reward, _ = env.make_step(idx, actions[idx])
                episode_reward += immediate_reward
                next_state = env.agents[idx].get_state(env)
                
                # Learn and choose next action
                next_actions[idx] = rl_agent.step(immediate_reward, next_state, False)
                episode_actions[idx][next_actions[idx]] += 1
            
            # Update environment
            env.update_environment()
            actions = next_actions
        
        # End of episode: social reward update
        emotions, personal, empathic, total_rewards = reward_calculator.calculate_rewards(env.agents)
        
        # Final learning step with social rewards
        for idx, rl_agent in enumerate(rl_agents):
            final_state = env.agents[idx].get_state(env)
            rl_agent.step(total_rewards[idx], final_state, True)
            agent_meals[idx].append(env.agents[idx].get_recent_meals())
        
        # Record metrics
        episode_rewards.append(episode_reward)
        social_welfare.append(sum(total_rewards))
        action_counts.append(episode_actions)
        
        # Optional: Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes} completed. Current reward: {episode_reward:.2f}")
    
    # Process results
    results = {
        'episode_rewards': episode_rewards,
        'social_welfare': social_welfare,
        'agent_meals': agent_meals,
        'action_counts': action_counts,
        'config': {
            'agent_type': agent_type,
            'empathy_level': empathy_level,
            'episodes': episodes,
            'env_size': env_size,
            'num_agents': num_agents,
            'steps_per_episode': steps_per_episode
        }
    }
    
    return results


def plot_simulation_results(results):
    """
    Plot the results of a simulation.
    
    Args:
        results: Dictionary with simulation results
    """
    config = results['config']
    agent_type = config['agent_type']
    empathy_level = config['empathy_level']
    episodes = config['episodes']
    
    # Extract data
    episode_rewards = results['episode_rewards']
    social_welfare = results['social_welfare']
    agent_meals = results['agent_meals']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards
    axes[0, 0].plot(episode_rewards, label='Total Reward', color='blue')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Social welfare
    axes[0, 1].plot(social_welfare, label='Social Welfare', color='green')
    axes[0, 1].set_title('Social Welfare')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Welfare Value')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Agent meals
    for i, meals in enumerate(agent_meals):
        axes[1, 0].plot(meals, label=f'Agent {i+1}')
    axes[1, 0].set_title('Agent Meals')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Recent Meals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Action distribution (last episode)
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "EXPLOIT"}
    last_actions = results['action_counts'][-1]
    agent_ids = list(last_actions.keys())
    
    width = 0.15
    x = np.arange(len(action_names))
    
    for i, agent_id in enumerate(agent_ids):
        actions = last_actions[agent_id]
        total = sum(actions)
        percentages = [count/total*100 for count in actions]
        axes[1, 1].bar(x + i*width, percentages, width, label=f'Agent {agent_id+1}')
    
    axes[1, 1].set_title('Final Action Distribution')
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_xticks(x + width * (len(agent_ids) - 1) / 2)
    axes[1, 1].set_xticklabels(action_names.values())
    axes[1, 1].set_ylabel('Usage (%)')
    axes[1, 1].legend()
    
    # Add overall title
    plt.suptitle(f'Simulation Results: {agent_type} with {empathy_level} ({episodes} episodes)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save and show
    plt.savefig(f'simulation_{agent_type}_{empathy_level}_{episodes}.png', dpi=300)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Run a simple simulation with default parameters
    results = run_simulation(episodes=500)
    
    # Plot results
    plot_simulation_results(results)