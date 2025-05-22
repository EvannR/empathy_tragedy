from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import random
import csv
import os
import pandas as pd
from tqdm import tqdm


# ----------------------------------------
# Constants for the simulation
# ----------------------------------------
'''
First experiment : 2 conditions
    empathic : SEE_EMOTIONS = TRUE AND ALPHA = 0.5
    standard : SEE_EMOTIONS = FALSE AND ALPHA = 0

    Episodes = 5000
    NB_AGENTS = 6
    STEP = 1000
    INITIAL_RESOURCES = 500
'''

'''
Second experiment : ANOVA
Third experiment : multiple ALPHA
'''

SIMULATION_NUMBER = 30      # number of simulation runs (also used as seed per run)
EPISODE_NUMBER = 5000        # number of episodes per simulation
NB_AGENTS = 6
MAX_STEPS = 1000         # number of steps per episode
INITIAL_RESOURCES = 500   # number of ressource at the beginning of each episode
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

# Agent & emotion settings
AGENT_TO_TEST = "DQN"      # 'DQN' or 'QLearning'
EMOTION_TYPE = "average"   # 'average' or 'vector'
SEE_EMOTIONS = True        # whether agents observe others' emotions : First experiment - True or False
ALPHA = 0.5                # empathy degree (0.0 - 1.0) First experiment - 0.5 or 0 (non empathic)
BETA = 0.5                 # valuation of last meal
SMOOTHING = 'linear'       # function transforming the meal history into an emotion : "sigmoid" OR "linear"
SIGMOID_GAIN = 5.0
THRESHOLD = 0.5            # Ratio of reward in meal history for the emotion to be neutral
EMOTION_ROUNDER = 2        # emotion's number of decimals

# RL agents Hyperparameters
PARAMS_QLEARNING = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01
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
    "QLearning": QAgent,
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
        env = GameTheoreticEnv(
            nb_agents=NB_AGENTS,
            env_type=ENVIRONMENT_TYPE,
            initial_resources=INITIAL_RESOURCES,
            emotion_type=EMOTION_TYPE,
            see_emotions=SEE_EMOTIONS,
            agent_class=POLICY_CLASSES[AGENT_TO_TEST],
            alpha=ALPHA,
            beta=BETA,
            smoothing=SMOOTHING,
            sigmoid_gain=SIGMOID_GAIN,
            threshold=THRESHOLD,
            round_emotions=EMOTION_ROUNDER
        )

    except KeyError:
        raise ValueError(f"Invalid AGENT_TO_TEST value: {AGENT_TO_TEST}. Choose 'DQN' or 'QLearning'.")

    state_size = 1 if (not SEE_EMOTIONS or EMOTION_TYPE == "average") else (NB_AGENTS - 1)
    action_size = env.number_actions
    params = PARAMS_DQN if AGENT_TO_TEST == "DQN" else PARAMS_QLEARNING
    AgentClass = POLICY_CLASSES[AGENT_TO_TEST]

    agents = [AgentClass(state_size, action_size, agent_id=i, **params) for i in range(NB_AGENTS)]
    return env, agents

# ----------------------------------------
# Step processing
# ----------------------------------------


def run_step(env, agents, simulation_index, episode, step, obs):
    """
    Execute one step:
        select actions,
        apply to env,
        collect info,
        update agents
    Return :
        the step record,
        rewards arrays,
        next observation
    """

    actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]

    next_obs, rewards, done, info = env.make_step(actions)

    # Extract reward components
    personal_arr = np.array(info['exploitation_reward'])
    empathic_arr = np.array(info['empathic_reward'])
    combined_arr = np.array(info['combined_reward'])

    record = {
        'simulation_number': simulation_index,
        'seed': simulation_index,
        'episode': episode,
        'step': step,
        'resource': env.resource,
        'observations': obs.copy(),
        'actions': actions,
        'personal': personal_arr.tolist(),
        'empathic': empathic_arr.tolist(),
        'combined_reward': combined_arr.tolist()
    }
    # Learning updates
    for i, agent in enumerate(agents):
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done)

    return record, personal_arr, empathic_arr, combined_arr, next_obs, done

# ----------------------------------------
# Simulation logic for one simulation with multiple episodes
# ----------------------------------------


def run_simulation_with_progressive_saving(episode_count, simulation_index, step_file, summary_file, seed, verbose=True):
    """
    Run a full simulation consisting of multiple episodes and steps, progressively saving step-wise 
    and episode summary data to CSV files.

    Each simulation is initialized with a new environment and agents using the specified seed.
    Agents interact with the environment by selecting actions, receiving rewards, and learning from feedback.
    Step-level data (observations, actions, rewards) and episode-level summaries (cumulative rewards, resources) 
    are written to separate CSV files to prevent memory overload and support analysis.

    Parameters:
        episode_count (int): Number of episodes to run in this simulation.
        simulation_index (int): Index identifying this simulation run (also used in filenames and random seed).
        step_file (str): Path to the CSV file where step-by-step records will be saved.
        summary_file (str): Path to the CSV file where per-episode summaries will be saved.
        seed (int): Random seed for reproducibility of the simulation.
        verbose (bool): If True, displays progress bars using `tqdm`; otherwise runs silently.

    Outputs:
        Writes two CSV files:
            - Step-level data including actions, rewards, and environment state at each step.
            - Episode-level summary data aggregating total rewards and remaining resources per episode.
    """
    np.random.seed(seed)
    env, agents = initialize_agents_and_env()
    summaries = []

    episode_iter = tqdm(range(episode_count), desc="Episodes", disable=not verbose)
    for episode in episode_iter:
        obs = env.reset()
        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)

        episode_step_records = []  # <-- store records here

        step_iter = tqdm(range(MAX_STEPS), desc=f"Episode {episode}", leave=False, disable=not verbose)
        for step in step_iter:
            record, prs, ers, crs, obs, done = run_step(env, agents, simulation_index, episode, step, obs)

            episode_step_records.append(record)  # <-- collect record in memory

            total_personal += prs
            total_empathic += ers
            total_combined += crs

            if done:
                break

        # After episode ends, write all step records at once
        write_step_csv(episode_step_records, simulation_index, seed=seed, filename=step_file)

        summary = {
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'resource_remaining': env.resource,
            'personal_totals': total_personal.tolist(),
            'empathic_totals': total_empathic.tolist(),
            'combined_totals': total_combined.tolist()
        }
        summaries.append(summary)

        write_summary_csv([summary], simulation_index, filename=summary_file, seed=seed)


# ----------------------------------------
# Functions used to write CSV
# ----------------------------------------


def append_step_record(record, simulation_index, filename):
    """
    Append a single step record to the simulation's step CSV file.
    Writes header only once (if file doesn't exist).
    """
    file_exists = os.path.exists(filename)

    header = (
        ["simulation_number", "seed", "episode", "step", "resource_remaining", "initial_resources", "max_step"] +
        [f"observation_{i}" for i in range(NB_AGENTS)] +
        [f"action_{i}" for i in range(NB_AGENTS)] +
        [f"personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_reward_{i}" for i in range(NB_AGENTS)]
    )

    row = (
        [simulation_index,
         record.get('seed', simulation_index),
         record['episode'],
         record['step'],
         record['resource'],
         INITIAL_RESOURCES,
         MAX_STEPS]
        + record['observations']
        + record['actions']
        + record['personal']
        + record['empathic']
        + record['combined_reward']
    )

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def write_step_csv(step_records, simulation_index, seed, filename):
    """
    Write a list of step-level records to a CSV file for a given simulation run.

    Each record contains detailed information for a single environment step, including:
    - Episode and step number
    - Observations and actions of each agent
    - Rewards (personal, empathic, combined) for each agent
    - Remaining shared resources

    The function appends to the CSV file if it already exists, writing the header only once. 
    This allows progressive saving across episodes and simulations.

    Parameters:
        step_records (list of dict): List of dictionaries, each representing a single step's data.
        simulation_index (int): The index identifying the current simulation (used in file naming).
        seed (int): Random seed used for the simulation (stored in the file for reproducibility).
        filename (str): Full path to the CSV file where data should be written.

    Outputs:
        Appends step-level data to the specified CSV file.
        If the file does not exist, it creates a new file with the appropriate header.
    """
    file_exists = os.path.exists(filename)
    header = (
        ["simulation_number", "seed", "episode", "step", "resource_remaining", "initial_resources", "max_step"] +
        [f"observation_{i}" for i in range(NB_AGENTS)] +
        [f"action_{i}" for i in range(NB_AGENTS)] +
        [f"personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_reward_{i}" for i in range(NB_AGENTS)]
    )

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for record in step_records:
            row = (
                [record['simulation_number'],
                 record.get('seed', seed),
                 record['episode'],
                 record['step'],
                 record['resource'],
                 INITIAL_RESOURCES,
                 MAX_STEPS]
                + record['observations']
                + record['actions']
                + record['personal']
                + record['empathic']
                + record['combined_reward']
            )
            writer.writerow(row)



def write_summary_csv(summaries, simulation_index, seed, filename=None):
    """
    Append per-episode summary data to CSV. Write header only if file does not exist.
    """
    if filename is None:
        filename = filename_definer(simulation_index, suffix="episode_summary")

    file_exists = os.path.exists(filename)

    header = (
        ["simulation_number", "seed", "episode", "total_steps", "resource_remaining", "initial_resources", "max_steps"] +
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
                [
                    rec['simulation_number'],
                    seed,
                    rec['episode'],
                    rec['total_steps'],
                    rec['resource_remaining'],
                    INITIAL_RESOURCES,
                    MAX_STEPS,
                ]
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
# debug data
# ----------------------------------------


def test_combined_rewards(csv_path, alpha=0.5, tolerance=1e-6, nb_agents=None):
    """
    Validate that the combined rewards in a summary CSV file match the expected formula:
        combined_reward = (1 - alpha) * personal_reward + alpha * empathic_reward

    This function reads a summary CSV file and checks each agent's combined reward against the
    weighted sum of their personal and empathic rewards using the provided alpha (empathy degree).
    It raises an AssertionError if any discrepancies exceed the specified tolerance.

    Parameters:
        csv_path (str): Path to the summary CSV file generated by the simulation.
        alpha (float): Weight for empathic rewards in the combination formula (range 0.0 - 1.0).
        tolerance (float): Allowed numerical error when comparing expected and recorded combined rewards.
        nb_agents (int or None): Number of agents in the simulation. If None, inferred from CSV column headers.

    Raises:
        AssertionError: If any combined reward does not match the expected value within the specified tolerance.
    
    Outputs:
        Prints a success message if all combined rewards match expectations. Otherwise, prints detailed
        mismatch information before raising an error.
    """
    
    df = pd.read_csv(csv_path)

    if nb_agents is None:
        nb_agents = sum(col.startswith('total_personal_reward_') for col in df.columns)

    personal_cols = [f'total_personal_reward_{i}' for i in range(nb_agents)]
    empathic_cols = [f'total_empathic_reward_{i}' for i in range(nb_agents)]
    combined_cols = [f'total_combined_reward_{i}' for i in range(nb_agents)]

    for i in range(nb_agents):
        personal = np.array(df[personal_cols[i]].values, dtype=float)
        empathic = np.array(df[empathic_cols[i]].values, dtype=float)
        combined = np.array(df[combined_cols[i]].values, dtype=float)

        combined_expected = (1 - alpha) * personal + alpha * empathic

        mismatches = np.abs(combined - combined_expected) > tolerance
        if mismatches.any():
            indices = np.where(mismatches)[0]
            for idx in indices:
                print(f"Mismatch at row {idx} for agent {i}:")
                print(f"  CSV combined     = {combined[idx]}")
                print(f"  Expected combined = {combined_expected[idx]}")
                print(f"  Personal          = {personal[idx]}")
                print(f"  Empathic          = {empathic[idx]}")
            raise AssertionError(f"Found mismatches for agent {i}")

    print("✅ All combined rewards match expected values within tolerance.")


# ----------------------------------------
# Main entry
# ----------------------------------------


if __name__ == '__main__':
    folder_name = "GT_simulation_jerome"
    os.makedirs(folder_name, exist_ok=True)

    for simulation_number in range(SIMULATION_NUMBER):
        seed = simulation_number + 1
        np.random.seed(seed)

        step_csv_name = filename_definer(simulation_index=simulation_number,
                                         suffix="step_data")
        summary_csv_name = filename_definer(simulation_index=simulation_number,
                                            suffix="episode_summary")
        step_csv_path = os.path.join(folder_name, step_csv_name)
        summary_csv_path = os.path.join(folder_name, summary_csv_name)

        # Run simulation — records steps & summaries progressively
        run_simulation_with_progressive_saving(
            episode_count=EPISODE_NUMBER,
            simulation_index=simulation_number,
            step_file=step_csv_path,
            summary_file=summary_csv_path,
            seed=seed,
            verbose=True
        )

        '''
        test_combined_rewards(summary_csv_path,
                              alpha=ALPHA,
                              nb_agents=NB_AGENTS)
        '''
