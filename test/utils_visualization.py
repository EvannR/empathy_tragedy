import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import ffmpeg

def plot_resource_consumption(csv_path, output_dir=None):
    """
    Plot resource consumption over time from a step data CSV file.
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    data = pd.read_csv(csv_path)
    
    # Group by episode and step
    grouped = data.groupby(['episode', 'step']).first().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot for each episode
    episodes = grouped['episode'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))
    
    for i, episode in enumerate(episodes):
        episode_data = grouped[grouped['episode'] == episode]
        ax.plot(episode_data['step'], episode_data['resource_remaining'], 
                label=f'Episode {episode}', color=colors[i], linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Remaining Resources')
    ax.set_title('Resource Consumption Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'resource_consumption.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_agent_rewards(csv_path, output_dir=None):
    """
    Plot reward components (personal, empathic, combined) for each agent
    across episodes.
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    data = pd.read_csv(csv_path)
    
    # Get number of agents from column names
    reward_columns = [col for col in data.columns if col.startswith('total_personal_reward_')]
    num_agents = len(reward_columns)
    
    # Create figure
    fig, axes = plt.subplots(num_agents, 3, figsize=(15, 3 * num_agents), sharex=True)
    
    # Ensure axes is 2D even with one agent
    if num_agents == 1:
        axes = axes.reshape(1, -1)
    
    # Plot types and their column prefixes
    reward_types = {
        'Personal Rewards': 'total_personal_reward_',
        'Empathic Rewards': 'total_empathic_reward_',
        'Combined Rewards': 'total_combined_reward_'
    }
    
    for i in range(num_agents):
        for j, (title, prefix) in enumerate(reward_types.items()):
            column = f"{prefix}{i}"
            ax = axes[i, j]
            
            # Group by episode and calculate mean and std
            grouped = data.groupby('episode')[column].agg(['mean', 'std']).reset_index()
            
            # Plot
            ax.errorbar(grouped['episode'], grouped['mean'], yerr=grouped['std'], 
                       marker='o', linestyle='-', capsize=5)
            
            ax.set_title(f'{title} - Agent {i}')
            ax.set_xlabel('Episode' if i == num_agents - 1 else '')
            ax.set_ylabel('Reward')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'agent_rewards.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_maze_animation(step_csv, env, output_path=None, fps=10, agent_data_columns=None):
    """
    Create an animation of maze environment from step data.
    
    Parameters:
    -----------
    step_csv : str
        Path to the step data CSV file
    env : Maze2DEnv
        The environment object (used to get maze dimensions)
    output_path : str
        Path to save the animation (if None, just displays)
    fps : int
        Frames per second for the animation
    agent_data_columns : dict
        Dictionary mapping column names to plots (e.g., {'action_0': 'Agent 0 Action'})
    """
    # Read data
    data = pd.read_csv(step_csv)
    
    # Filter for a single episode (first one)
    episode_data = data[data['episode'] == data['episode'].min()]
    
    # Get maze dimensions
    maze_width, maze_height = env.maze_size
    
    # Create figure with two subplots - maze and data
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[4, 1])
    
    # Maze subplot
    ax_maze = fig.add_subplot(gs[0, 0])
    ax_maze.set_xlim(-0.5, maze_width - 0.5)
    ax_maze.set_ylim(-0.5, maze_height - 0.5)
    ax_maze.set_xticks(np.arange(0, maze_width, 1))
    ax_maze.set_yticks(np.arange(0, maze_height, 1))
    ax_maze.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax_maze.set_title('Maze Environment')
    
    # Resource counter
    ax_resource = fig.add_subplot(gs[0, 1])
    ax_resource.set_xlim(0, episode_data['step'].max() + 1)
    ax_resource.set_ylim(0, episode_data['resource_remaining'].max() * 1.1)
    ax_resource.set_title('Resources Remaining')
    ax_resource.set_xlabel('Step')
    ax_resource.set_ylabel('Resource Units')
    ax_resource.grid(True)
    
    # Agent data subplot (if provided)
    ax_data = fig.add_subplot(gs[1, :])
    if agent_data_columns:
        ax_data.set_xlim(0, episode_data['step'].max() + 1)
        ax_data.set_ylim(-0.1, 1.1)  # Assuming actions are 0 or 1
        ax_data.set_title('Agent Actions')
        ax_data.set_xlabel('Step')
        ax_data.set_ylabel('Action (0=Don\'t Consume, 1=Consume)')
        ax_data.grid(True)
    
    # Plot elements that will be updated in animation
    resource_line, = ax_resource.plot([], [], 'r-', linewidth=2)
    
    # Initialize agent patches and data lines
    agent_patches = []
    agent_colors = ['red', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
    num_agents = len([col for col in episode_data.columns if col.startswith('agent_0')])
    
    for i in range(num_agents):
        color = agent_colors[i % len(agent_colors)]
        circle = plt.Circle((0, 0), 0.3, color=color, alpha=0.7)
        ax_maze.add_patch(circle)
        agent_patches.append(circle)
    
    # Data lines for agent actions
    data_lines = []
    if agent_data_columns:
        for col, label in agent_data_columns.items():
            line, = ax_data.plot([], [], 'o-', label=label)
            data_lines.append((col, line))
        ax_data.legend()
    
    # Title for animation
    title = ax_maze.text(maze_width/2, -1, "", ha="center", fontsize=12)
    
    # Initialization function
    def init():
        for patch in agent_patches:
            patch.center = (0, 0)
        resource_line.set_data([], [])
        if agent_data_columns:
            for _, line in data_lines:
                line.set_data([], [])
        title.set_text("")
        return [resource_line, *agent_patches, title, *[line for _, line in data_lines]]
    
    # Animation function
    def animate(i):
        step_record = episode_data.iloc[i]
        step = step_record['step']
        resource = step_record['resource_remaining']
        
        # Update title
        title.set_text(f"Step: {step}, Resources: {resource:.1f}")
        
        # Update agent positions
        for j in range(num_agents):
            x = step_record[f'agent_{j}_x']
            y = step_record[f'agent_{j}_y']
            agent_patches[j].center = (x, y)
        
        # Update resource graph
        steps = episode_data['step'].values[:i+1]
        resources = episode_data['resource_remaining'].values[:i+1]
        resource_line.set_data(steps, resources)
        
        # Update agent data lines
        if agent_data_columns:
            for col, line in data_lines:
                line.set_data(
                    episode_data['step'].values[:i+1],
                    episode_data[col].values[:i+1]
                )
        
        return [resource_line, *agent_patches, title, *[line for _, line in data_lines]]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(episode_data), interval=1000/fps, blit=True
    )
    
    plt.tight_layout()
    
    if output_path:
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close()
    else:
        plt.show()
    
    return anim

def plot_agent_learning_curves(step_csv, output_dir=None):
    """
    Plot learning curves showing how agent policies evolve over time.
    Specifically shows:
    1. Percentage of exploit decisions when resources are available
    2. Resource utilization efficiency
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    data = pd.read_csv(step_csv)
    
    # Get number of agents
    action_columns = [col for col in data.columns if col.startswith('action_')]
    num_agents = len(action_columns)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Group data by episode and calculate statistics
    episodes = data['episode'].unique()
    window_size = 50  # For moving average
    
    # Colors for agents
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Plot 1: Exploitation rate when resources are available
    ax = axes[0]
    
    for i in range(num_agents):
        exploit_rates = []
        
        for episode in episodes:
            episode_data = data[data['episode'] == episode]
            
            # Get agent's position columns
            x_col = f'agent_{i}_x'
            y_col = f'agent_{i}_y'
            action_col = f'action_{i}'
            
            # Calculate exploitation rate when resources available
            exploit_decisions = []
            
            for _, row in episode_data.iterrows():
                # Skip if we don't have position data
                if x_col not in row or y_col not in row:
                    continue
                
                # We assume that if personal reward is > 0, there was a resource available
                # This is a simplification - in a real implementation you'd track the maze state
                if row[f'personal_{i}'] > 0 or (row[action_col] == 1 and row[f'personal_{i}'] == 0):
                    exploit_decisions.append(row[action_col])
            
            if exploit_decisions:
                rate = sum(exploit_decisions) / len(exploit_decisions)
                exploit_rates.append(rate)
            else:
                exploit_rates.append(0)
        
        # Plot the exploitation rate
        ax.plot(episodes, exploit_rates, 'o-', color=colors[i], label=f'Agent {i}')
    
    ax.set_ylabel('Exploitation Rate\nWhen Resources Available')
    ax.set_title('Agent Learning: Resource Exploitation Strategy')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot 2: Personal reward efficiency
    ax = axes[1]
    
    for i in range(num_agents):
        reward_efficiency = []
        
        for episode in episodes:
            episode_data = data[data['episode'] == episode]
            action_col = f'action_{i}'
            personal_col = f'personal_{i}'
            
            # Calculate reward efficiency: rewards / exploitation attempts
            exploit_attempts = sum(episode_data[action_col] == 1)
            rewards = sum(episode_data[personal_col])
            
            if exploit_attempts > 0:
                efficiency = rewards / exploit_attempts
            else:
                efficiency = 0
                
            reward_efficiency.append(efficiency)
        
        # Plot the reward efficiency
        ax.plot(episodes, reward_efficiency, 's-', color=colors[i], label=f'Agent {i}')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Efficiency\n(Rewards / Exploit Attempts)')
    ax.set_title('Agent Learning: Resource Acquisition Efficiency')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # These functions should be imported and used in other scripts
    print("This is a utility module for creating visualizations.")
    print("Import and use these functions in your main script.")