# simple_main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simple_env import SimpleGridWorld
from simple_agent import SimpleAgent
import os
from datetime import datetime
import time

def run_simulation(num_episodes=100, max_steps=1000, render_interval=None, save_results=True):
    """
    Run the MARL simulation.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_interval: How often to render the environment (None=never)
        save_results: Whether to save results to CSV
    
    Returns:
        DataFrame with results
    """
    # Create environment
    env = SimpleGridWorld(size=7, num_agents=3, resource_density=0.6, respawn_prob=0.1)
    
    # Create agents
    agents = []
    state_size = 8  # Position (2) + Nearby resources (4) + Own emotion (1) + Avg emotion (1)
    action_size = 5  # UP, DOWN, LEFT, RIGHT, CONSUME
    
    for i in range(env.num_agents):
        agent = SimpleAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id=i,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.1,
            batch_size=32
        )
        agents.append(agent)
    
    # Initialize results storage
    results = []
    
    # Main training loop
    for episode in range(num_episodes):
        # Reset environment
        observations = env.reset()
        episode_rewards = np.zeros(env.num_agents)
        consumption_count = np.zeros(env.num_agents)
        
        for step in range(max_steps):
            # Select actions
            actions = [agent.select_action(obs) for agent, obs in zip(agents, observations)]
            
            # Take step in environment
            next_observations, rewards, done, info = env.step(actions)
            
            # Store experience in each agent's memory
            for i, agent in enumerate(agents):
                agent.remember(
                    observations[i], 
                    actions[i], 
                    rewards[i], 
                    next_observations[i], 
                    done
                )
            
            # Update each agent's Q-network
            losses = [agent.learn() for agent in agents]
            
            # Update observations
            observations = next_observations
            
            # Accumulate rewards
            episode_rewards += rewards
            
            # Track consumption
            for i, consumed in enumerate(info['consumed']):
                if consumed:
                    consumption_count[i] += 1
            
            # Render environment
            if render_interval and episode % render_interval == 0 and step % 10 == 0:
                print(f"Episode {episode}, Step {step}")
                env.render()
                time.sleep(0.2)  # Pause for visual clarity
        
        # Store episode results
        for i in range(env.num_agents):
            results.append({
                'episode': episode,
                'agent_id': i,
                'total_reward': episode_rewards[i],
                'consumption_count': consumption_count[i],
                'final_emotional_state': env.emotional_states[i],
                'exploration_rate': agents[i].epsilon
            })
        
        # Print progress
        if episode % 10 == 0 or episode == num_episodes - 1:
            avg_reward = np.mean(episode_rewards)
            avg_emotion = np.mean(env.emotional_states)
            print(f"Episode {episode}/{num_episodes} - Avg Reward: {avg_reward:.2f}, Avg Emotion: {avg_emotion:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if requested
    if save_results:
        save_dir = 'results'
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"simulation_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
        # Save agent models
        model_dir = os.path.join(save_dir, f"models_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        for i, agent in enumerate(agents):
            agent.save(os.path.join(model_dir, f"agent_{i}.pt"))
        print(f"Agent models saved to {model_dir}")
        
        # Generate and save plots
        plot_results(results_df, save_dir, timestamp)
    
    return results_df

def plot_results(results_df, save_dir, timestamp):
    """Create and save plots from simulation results."""
    # Create per-episode summary
    episode_summary = results_df.groupby('episode').agg({
        'total_reward': 'mean',
        'consumption_count': 'mean',
        'final_emotional_state': 'mean',
        'exploration_rate': 'mean'
    }).reset_index()
    
    # Plot rewards over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_summary['episode'], episode_summary['total_reward'])
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"rewards_{timestamp}.png"))
    
    # Plot consumption count over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_summary['episode'], episode_summary['consumption_count'])
    plt.title('Average Consumption Count per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Consumption Count')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"consumption_{timestamp}.png"))
    
    # Plot emotional states over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_summary['episode'], episode_summary['final_emotional_state'])
    plt.title('Average Emotional State per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Emotional State')
    plt.axhline(y=0, color='r', linestyle='--')  # Reference line at neutral emotion
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"emotions_{timestamp}.png"))
    
    # Plot exploration rate over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episode_summary['episode'], episode_summary['exploration_rate'])
    plt.title('Exploration Rate (Epsilon) over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"exploration_{timestamp}.png"))
    
    # Per-agent comparison for final 10 episodes
    last_episodes = results_df[results_df['episode'] >= results_df['episode'].max() - 9]
    per_agent = last_episodes.groupby('agent_id').agg({
        'total_reward': 'mean',
        'consumption_count': 'mean',
        'final_emotional_state': 'mean'
    }).reset_index()
    
    # Plot comparison between agents
    plt.figure(figsize=(12, 6))
    
    x = per_agent['agent_id']
    width = 0.25
    
    plt.bar(x - width, per_agent['total_reward'], width, label='Avg Reward')
    plt.bar(x, per_agent['consumption_count'], width, label='Avg Consumption')
    plt.bar(x + width, per_agent['final_emotional_state'], width, label='Avg Emotion')
    
    plt.xlabel('Agent ID')
    plt.title('Agent Performance (Last 10 Episodes)')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, f"agent_comparison_{timestamp}.png"))
    
    print("Plots saved to results directory")

def analyze_agent_behavior(results_df):
    """Analyze and print insights about agent behavior."""
    # Overall statistics
    print("\n===== OVERALL STATISTICS =====")
    print(f"Total Episodes: {results_df['episode'].max() + 1}")
    print(f"Number of Agents: {results_df['agent_id'].nunique()}")
    
    # Calculate overall averages
    avg_reward = results_df['total_reward'].mean()
    avg_consumption = results_df['consumption_count'].mean()
    avg_emotion = results_df['final_emotional_state'].mean()
    
    print(f"Average Reward per Episode: {avg_reward:.2f}")
    print(f"Average Consumption per Episode: {avg_consumption:.2f}")
    print(f"Average Emotional State: {avg_emotion:.2f}")
    
    # First vs last episodes comparison
    first_10 = results_df[results_df['episode'] < 10]
    last_10 = results_df[results_df['episode'] >= results_df['episode'].max() - 9]
    
    print("\n===== LEARNING PROGRESS =====")
    print(f"First 10 Episodes Avg Reward: {first_10['total_reward'].mean():.2f}")
    print(f"Last 10 Episodes Avg Reward: {last_10['total_reward'].mean():.2f}")
    print(f"First 10 Episodes Avg Consumption: {first_10['consumption_count'].mean():.2f}")
    print(f"Last 10 Episodes Avg Consumption: {last_10['consumption_count'].mean():.2f}")
    print(f"First 10 Episodes Avg Emotion: {first_10['final_emotional_state'].mean():.2f}")
    print(f"Last 10 Episodes Avg Emotion: {last_10['final_emotional_state'].mean():.2f}")
    
    # Per-agent analysis
    print("\n===== PER-AGENT ANALYSIS (Last 10 Episodes) =====")
    agent_stats = last_10.groupby('agent_id').agg({
        'total_reward': 'mean',
        'consumption_count': 'mean',
        'final_emotional_state': 'mean'
    }).reset_index()
    
    for _, row in agent_stats.iterrows():
        agent_id = int(row['agent_id'])
        print(f"Agent {agent_id}:")
        print(f"  - Avg Reward: {row['total_reward']:.2f}")
        print(f"  - Avg Consumption: {row['consumption_count']:.2f}")
        print(f"  - Avg Emotional State: {row['final_emotional_state']:.2f}")
    
    # Check correlation between emotional state and consumption
    correlation = np.corrcoef(
        last_10['final_emotional_state'],
        last_10['consumption_count']
    )[0, 1]
    
    print(f"\nCorrelation between Emotional State and Consumption: {correlation:.2f}")
    
    # Effectiveness of social reward mechanism
    if correlation < 0:
        print("Agents have learned to balance consumption (negative correlation between emotion and consumption)")
    else:
        print("Agents may not have fully learned optimal social behavior yet")

if __name__ == "__main__":
    print("Starting Multi-Agent Reinforcement Learning Simulation...")
    print("Agents will learn when to consume resources based on emotional states")
    
    # Run simulation
    results = run_simulation(
        num_episodes=100,
        max_steps=50,
        render_interval=20,  # Render every 20 episodes
        save_results=True
    )
    
    # Analyze results
    analyze_agent_behavior(results)
    
    print("\nSimulation complete!")