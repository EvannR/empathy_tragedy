from env_maze import Maze2DEnv
from agent_policies_maze import QAgentMaze, DQNAgent
from utils_visualization import (
    plot_resource_consumption, 
    plot_agent_rewards, 
    create_maze_animation, 
    plot_agent_learning_curves
)

from tqdm import tqdm  # for nice progress bars
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import ffmpeg


"""
Simulation parameter

First experiment : 2 conditions
    empathic : SEE_EMOTIONS = TRUE AND ALPHA = 0.5
    standard : SEE_EMOTIONS = FALSE AND ALPHA = 0

    Episodes = 5000
    NB_AGENTS = 6
    STEP = 1000

    MAZE_SIZE = (5, 5) # has to be sufficiently big for the amount of agents
    RESOURCE_DENSITY = 0.3 # to reduce ?

"""

# ----------------------------------------
# Constants for the simulation
# ----------------------------------------

SIMULATION_NUMBER = 1       # number of simulation runs (also used as seed per run)
EPISODE_NUMBER = 5000          # number of episodes per simulation
NB_AGENTS = 2
MAX_STEPS = 1000             # number of steps per episode
MAZE_SIZE = (10, 10)        # size of the 2D maze
INITIAL_RESOURCES = 50     # total resource units in the environment at start
RESOURCE_DENSITY = 0.3      # percentage of cells with resources
RESOURCE_REGEN_RATE = 0.0  # resource regeneration rate per step
RESOURCE_DISTRIBUTION = "random" # "random" or "clustered"
ENVIRONMENT_TYPE = "stochastic"   # 'deterministic' or 'stochastic'

# Agent & emotion settings
AGENT_TO_TEST = "DQN"       # 'DQN' or 'QLearning'
EMOTION_TYPE = "average"    # 'average' or 'vector'
SEE_EMOTIONS = False        # whether agents observe others' emotions
ALPHA = 0.5                 # empathy degree (0.0 - 1.0)
BETA = 0.5                  # valuation of last meal
SMOOTHING = 'linear'        # function transforming the meal history into an emotion : "sigmoid" OR "linear"
SIGMOID_GAIN = 5.0
THRESHOLD = 0.5             # Ratio of reward in meal history for the emotion to be neutral
EMOTION_ROUNDER = 2         # emotion's number of decimals

# RL agents Hyperparameters
PARAMS_QLEARNING = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.075
}
PARAMS_DQN = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 16,
    "hidden_size": 64,
    "update_target_every": 5
}

POLICY_CLASSES = {
    "QLearning": QAgentMaze,
    "DQN": DQNAgent
}

# ----------------------------------------
# Initialization of agents and environment for a new simulation
# ----------------------------------------

def initialize_agents_and_env():
    """
    Create environment and agents for a new simulation.
    """
    try:
        # Determine state size based on environment settings
        base_state_size = 3  # x, y, resource at location
        if not SEE_EMOTIONS:
            state_size = base_state_size
        else:
            if EMOTION_TYPE == "average":
                state_size = base_state_size + 1
            else:  # "vector"
                state_size = base_state_size + (NB_AGENTS - 1)
                
        # Create agent configurations
        agent_configs = []
        for i in range(NB_AGENTS):
            if AGENT_TO_TEST == "DQN":
                config = PARAMS_DQN.copy()
            else:  # "QLearning"
                config = PARAMS_QLEARNING.copy()
            agent_configs.append(config)
        
        # Create environment
        env = Maze2DEnv(
            nb_agents=NB_AGENTS,
            maze_size=MAZE_SIZE,
            initial_resources=INITIAL_RESOURCES,
            resource_regen_rate=RESOURCE_REGEN_RATE,
            resource_distribution=RESOURCE_DISTRIBUTION,
            resource_density=RESOURCE_DENSITY,
            env_type=ENVIRONMENT_TYPE,
            emotion_type=EMOTION_TYPE,
            see_emotions=SEE_EMOTIONS,
            agent_class=POLICY_CLASSES[AGENT_TO_TEST],
            agent_configs=agent_configs,
            alpha=ALPHA,
            beta=BETA,
            smoothing=SMOOTHING,
            sigmoid_gain=SIGMOID_GAIN,
            threshold=THRESHOLD,
            round_emotions=EMOTION_ROUNDER
        )
        
        return env

    except KeyError:
        raise ValueError(f"Invalid AGENT_TO_TEST value: {AGENT_TO_TEST}. Choose 'DQN' or 'QLearning'.")

# ----------------------------------------
# Simulation logic for one simulation with multiple episodes
# ----------------------------------------

def run_simulation_with_progressive_saving(simulation_index, step_file, summary_file, seed,
                                           episode_number=EPISODE_NUMBER, step_count=MAX_STEPS,
                                           show_progress=True, save_step_data=False):
    np.random.seed(seed)
    env = initialize_agents_and_env()
    episode_iter = tqdm(range(episode_number), desc=f"Sim {simulation_index+1}/{SIMULATION_NUMBER}") if show_progress else range(episode_number)

    for episode in episode_iter:
        obs = env.reset()
        for i, agent in enumerate(env.agents):
            if hasattr(agent, "start_episode"):
                agent.start_episode(obs[i])

        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)
        episode_step_records = []

        last_rewards = [0.0] * NB_AGENTS
        last_dones = [False] * NB_AGENTS

        for step in range(step_count):
            actions = []
            for i, agent in enumerate(env.agents):
                if step == 0:
                    action = agent.select_action(obs[i])
                    agent.current_state = obs[i]
                    agent.previous_action = action
                else:
                    action = agent.step(obs[i], last_rewards[i], last_dones[i])
                actions.append(action)
            next_obs, rewards, done, info = env.make_step(actions)

            prs = np.array(info['personal_satisfaction'])
            ers = np.array(info['empathic_reward'])
            crs = np.array(info['combined_reward'])

            total_personal += prs
            total_empathic += ers
            total_combined += crs

            if save_step_data:
                episode_step_records.append({
                    'simulation_number': simulation_index,
                    'seed': seed,
                    #'episode': episode,
                    'step': step,
                    'resource_remaining': info['remaining_resources'],
                    'positions': info['agent_positions'],
                    'actions': actions,
                    'personal': prs.tolist(),
                    'empathic': ers.tolist(),
                    'combined': crs.tolist()
                })

            last_rewards = rewards
            last_dones = [done for _ in range(NB_AGENTS)]
            obs = next_obs
            if done:
                break

        # Save step data for this episode
        if save_step_data and episode_step_records:
            write_step_csv(episode_step_records, simulation_index, filename=step_file)

        # Write summary for this episode immediately
        summary = {
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'resource_remaining': info['remaining_resources'],
            'personal_totals': total_personal.tolist(),
            'empathic_totals': total_empathic.tolist(),
            'combined_totals': total_combined.tolist()
        }
        write_summary_csv([summary], simulation_index, filename=summary_file)
        
        # Optional: print progress information (or use tqdm bar postfix)
        if show_progress and ((episode + 1) % 100 == 0 or episode == 0 or episode == episode_number - 1):
            divisor = (step + 1)
            #print("DEBUG:", NB_AGENTS, total_personal, total_empathic, total_combined)
            # Fallback to nan if no steps in episode
            if divisor > 0 and len(total_personal) > 0:
                avg_personal = np.sum(total_personal) / NB_AGENTS
                avg_empathic = np.sum(total_empathic) / NB_AGENTS if np.any(total_empathic) else 0.0
                avg_combined = np.sum(total_combined) / NB_AGENTS if np.any(total_combined) else 0.0
            else:
                avg_personal = float('nan')
                avg_empathic = float('nan')
                avg_combined = float('nan')
            print(f"Sim {simulation_index+1}, Ep {episode+1}/{episode_number} | "
                  f"Steps: {step+1}, "
                  f"Avg Pers: {avg_personal:.3f}, "
                  f"Avg Emp: {avg_empathic:.3f}, "
                  f"Avg Comb: {avg_combined:.3f}, "
                  f"Resources left: {info['remaining_resources']}")


# ----------------------------------------
# Visualization functions
# ----------------------------------------

def visualize_maze(env, step_data=None, episode=0, step=0, save_path=None):
    """
    Create a visualization of the maze at a given step.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define custom colormap for resources
    cmap = LinearSegmentedColormap.from_list('resource_cmap', ['white', 'forestgreen'], N=6)
    
    # Plot resources
    maze_data = env.maze.copy()
    im = ax.imshow(maze_data.T, cmap=cmap, origin='upper', vmin=0, vmax=5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Resource Amount')
    
    # Plot agents
    agent_colors = ['red', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
    for i, (x, y) in enumerate(env.agent_positions):
        color = agent_colors[i % len(agent_colors)]
        circle = plt.Circle((x, y), 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, f'A{i}', ha='center', va='center', color='white', fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, env.maze_size[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.maze_size[1], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Set labels and title
    ax.set_xticks(np.arange(0, env.maze_size[0], 1))
    ax.set_yticks(np.arange(0, env.maze_size[1], 1))
    ax.set_xticklabels(np.arange(0, env.maze_size[0], 1))
    ax.set_yticklabels(np.arange(0, env.maze_size[1], 1))
    ax.set_title(f'Maze State - Episode {episode}, Step {step}')
    
    # Add legend for agents
    handles = [patches.Patch(color=agent_colors[i % len(agent_colors)], label=f'Agent {i}') 
              for i in range(env.nb_agents)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_rewards(summaries, save_path=None):
    """
    Visualize reward trends across episodes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Extract data
    episodes = [s['episode'] for s in summaries]
    
    # Plot personal rewards
    ax = axes[0]
    for i in range(NB_AGENTS):
        personal_rewards = [s['personal_totals'][i] for s in summaries]
        ax.plot(episodes, personal_rewards, marker='o', label=f'Agent {i}')
    ax.set_ylabel('Personal Rewards')
    ax.set_title('Personal Rewards per Episode')
    ax.legend()
    ax.grid(True)
    
    # Plot empathic rewards
    ax = axes[1]
    for i in range(NB_AGENTS):
        empathic_rewards = [s['empathic_totals'][i] for s in summaries]
        ax.plot(episodes, empathic_rewards, marker='s', label=f'Agent {i}')
    ax.set_ylabel('Empathic Rewards')
    ax.set_title('Empathic Rewards per Episode')
    ax.legend()
    ax.grid(True)
    
    # Plot combined rewards
    ax = axes[2]
    for i in range(NB_AGENTS):
        combined_rewards = [s['combined_totals'][i] for s in summaries]
        ax.plot(episodes, combined_rewards, marker='^', label=f'Agent {i}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Combined Rewards')
    ax.set_title('Combined Rewards per Episode')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ----------------------------------------
# Functions used to write CSV
# ----------------------------------------

def write_step_csv(step_records, simulation_index, filename):
    """
    Write detailed per-step data to CSV, including seed and episode.
    """
    file_exists = os.path.exists(filename)
    header = (
        ["simulation_number", "seed", "episode", "step", "resource_remaining"] +
        [f"agent_{i}_x" for i in range(NB_AGENTS)] +
        [f"agent_{i}_y" for i in range(NB_AGENTS)] +
        [f"action_{i}" for i in range(NB_AGENTS)] +
        [f"personal_{i}" for i in range(NB_AGENTS)] +
        [f"empathic_{i}" for i in range(NB_AGENTS)] +
        [f"combined_{i}" for i in range(NB_AGENTS)]
    )
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for record in step_records:
            pos_values = []
            for x, y in record['positions']:
                pos_values.extend([x, y])
            row = (
                [record['seed'], record['episode'], record['step'], record['resource_remaining']]
                + pos_values
                + record['actions']
                + record['personal']
                + record['empathic']
                + record['combined']
            )
            writer.writerow(row)


def write_summary_csv(summaries, simulation_index, filename=None):
    """
    Write per-episode summary data to CSV, including seed and episode.
    """
    if filename is None:
        filename = filename_definer(simulation_index, suffix="episode_summary")
    file_exists = os.path.exists(filename)
    header = (
        ['simulation_number', 'seed', 'episode', 'total_steps', 'resource_remaining'] +
        [f"total_personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_combined_reward_{i}" for i in range(NB_AGENTS)]
    )
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for rec in summaries:
            row = (
                [rec['simulation_number'], rec['seed'], rec['episode'], rec['total_steps'], rec['resource_remaining']]
                + rec['personal_totals']
                + rec['empathic_totals']
                + rec['combined_totals']
            )
            writer.writerow(row)


# ----------------------------------------
# Filename builder
# ----------------------------------------

def filename_definer(simulation_index: int, suffix: str) -> str:
    """
    Builds a filename matching the original format:
    results_<sim>_<episodes>_<agent>_<emotion>_<see_emotions>_<alpha>_<beta>_
             <smoothing>_<threshold>_<rounder>_<params>_<random>_<suffix>.csv

             To ensure not overwriting, a version number is added if a file already exists.
    """
    if AGENT_TO_TEST == "DQN":
        param_order = ["learning_rate", "gamma", "epsilon", "epsilon_decay", "epsilon_min",
                       "batch_size", "hidden_size", "update_target_every"]
        params = PARAMS_DQN
    elif AGENT_TO_TEST == "QLearning":
        param_order = ["learning_rate", "gamma", "epsilon", "epsilon_decay", "epsilon_min"]
        params = PARAMS_QLEARNING
    else:
        raise ValueError("Wrong agent type")

    param_values = "_".join(str(params[key]) for key in param_order)
    random_suffix = ''.join(str(random.randint(0, 9)) for _ in range(6))
    see_emotions_str = str(SEE_EMOTIONS)

    filename = (
        f"results_{simulation_index:03d}_"
        f"{EPISODE_NUMBER}_"
        f"{AGENT_TO_TEST}_"
        f"{EMOTION_TYPE}_"
        f"{see_emotions_str}_"
        f"{ALPHA}_"
        f"{BETA}_"
        f"{SMOOTHING}_"
        f"{THRESHOLD}_"
        f"{EMOTION_ROUNDER}_"
        f"{param_values}_"
        f"{random_suffix}_"
        f"{suffix}.csv"
    )

    # To ensure no file deletion
    base_filename = filename.replace('.csv', '')
    version = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{version}.csv"
        version += 1

    return filename

# ----------------------------------------
# Main entry
# ----------------------------------------
if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    SHOW_SIMULATION_PROGRESS = True
    SAVE_STEP_DATA = False  # Set to True if you want step-by-step CSVs

    simulation_iter = tqdm(range(SIMULATION_NUMBER), desc="All simulations") if SHOW_SIMULATION_PROGRESS else range(SIMULATION_NUMBER)
    for simulation_number in simulation_iter:
        seed = simulation_number + 1
        np.random.seed(seed)

        step_csv_name = filename_definer(simulation_index=simulation_number, suffix="step_data")
        summary_csv_name = filename_definer(simulation_index=simulation_number, suffix="episode_summary")
        step_csv_path = os.path.join("results", step_csv_name)
        summary_csv_path = os.path.join("results", summary_csv_name)

        run_simulation_with_progressive_saving(
            simulation_index=simulation_number,
            step_file=step_csv_path,
            summary_file=summary_csv_path,
            seed=seed,
            episode_number=EPISODE_NUMBER,
            step_count=MAX_STEPS,
            show_progress=SHOW_SIMULATION_PROGRESS,
            save_step_data=SAVE_STEP_DATA
        )
    
        '''
        # Generate visualizations for first simulation only
        if simulation_number == 0:
            print("Generating visualizations...")
            viz_dir = f"visualizations/sim_{simulation_number}"
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create a fresh environment for visualizations
            env = initialize_agents_and_env()
            
            # Plot resource consumption
            plot_resource_consumption(step_csv_path, output_dir=viz_dir)
            print(f"Generated resource consumption plot in {viz_dir}")
            
            # Plot agent rewards
            plot_agent_rewards(summary_csv_path, output_dir=viz_dir)
            print(f"Generated agent rewards plot in {viz_dir}")
            
            # Plot learning curves
            plot_agent_learning_curves(step_csv_path, output_dir=viz_dir)
            print(f"Generated learning curves in {viz_dir}")
            
            # Create animation of first episode
            animation_path = os.path.join(viz_dir, "maze_animation.mp4")
            agent_data_columns = {f'action_{i}': f'Agent {i} Action' for i in range(NB_AGENTS)}
            
            try:
                create_maze_animation(
                    step_csv_path, 
                    env, 
                    output_path=animation_path,
                    fps=5,
                    agent_data_columns=agent_data_columns
                )
                print(f"Generated maze animation at {animation_path}")
            except Exception as e:
                print(f"Could not create animation: {e}")
                print("Note: Animation creation requires ffmpeg to be installed.")
            
            # Visualize initial and final maze states for the first episode
            for episode_idx, episode_data in enumerate(detailed[:1]):  # Just first episode
                # Initial state
                env.reset()  # Reset to get a new environment state
                initial_viz_path = os.path.join(viz_dir, f"maze_episode_{episode_idx}_initial.png")
                visualize_maze(env, episode=episode_idx, step=0, save_path=initial_viz_path)
                
                # Run through steps to get to final state
                for step_data in episode_data:
                    actions = step_data['actions']
                    env.make_step(actions)
                
                # Final state
                final_step = len(episode_data) - 1
                final_viz_path = os.path.join(viz_dir, f"maze_episode_{episode_idx}_final.png")
                visualize_maze(env, episode=episode_idx, step=final_step, save_path=final_viz_path)
        
        '''
        print("=== ALL SIMULATIONS COMPLETE ===")
        # print(f"Completed simulation {simulation_number + 1}/{SIMULATION_NUMBER}")
    
    # print("All simulations completed successfully!")
    # print("\nSuggested next steps:")
    # print("1. Run experiments with different parameters:")
    # print("   - Try different maze sizes (MAZE_SIZE)")
    # print("   - Adjust resource regeneration rates (RESOURCE_REGEN_RATE)")
    # print("   - Test different emotion settings (ALPHA, BETA, THRESHOLD)")
    # print("2. Compare Q-Learning vs DQN performance")
    # print("3. Analyze the effect of seeing emotions (SEE_EMOTIONS) on agent behavior")
    # print("4. Experiment with different resource distributions (RESOURCE_DISTRIBUTION)")