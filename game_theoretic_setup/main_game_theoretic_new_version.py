"""
main entry point for running multi-condition game-theoretic simulations.

this script orchestrates complete experiments for the empathic tragedy-of-the-
commons setup.  it runs multiple simulation runs per empathy condition
(e.g. empathic vs non-empathic), where each run consists of many episodes of
multi-agent resource exploitation.

key features:
  - configurable RL agent type (DQN or tabular Q-learning).
  - configurable empathy parameters (alpha, beta, smoothing, threshold).
  - progressive saving: step-level and episode-summary CSV files are written
    incrementally so data is preserved even if a run is interrupted.
  - deterministic seeding for reproducibility (numpy, random, torch).

usage:
    python main_game_theoretic_new_version.py

output CSV files are saved under  empathy_tragedy/GT_simulation_jerome_thesis_emp/.
"""

from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import random
import csv
import os
import pandas as pd
import torch
from tqdm import tqdm



# ----------------------------------------
# constants for the simulation
# ----------------------------------------
'''
first experiment: 2 conditions
    empathic: SEE_EMOTIONS = TRUE and ALPHA = 0.5
    standard: SEE_EMOTIONS = FALSE and ALPHA = 0

    Episodes = 5000
    NB_AGENTS = 6
    STEP = 1000
    INITIAL_RESOURCES = 500
'''

'''
second experiment: ANOVA -> need and explanation. 
third experiment: multiple ALPHA
forth experiment: 2 by 2 matrix (observation, consideration of others)
'''

<<<<<<< HEAD
SIMULATION_NUMBER = 3      # number of simulation runs (also used as seed per run)
EPISODE_NUMBER = 50        # number of episodes per simulation
NB_AGENTS = 6
MAX_STEPS = 100         # number of steps per episode
INITIAL_RESOURCES = 8   # number of ressource at the beginning of each episode
=======
NUM_RUNS_PER_CONDITION = 2  # number of simulation runs per empathy condition (non-empathic / empathic)
EPISODE_NUMBER = 500        # number of episodes per simulation
NB_AGENTS = 6
MAX_STEPS = 1000        # number of steps per episode
INITIAL_RESOURCES = 500   # number of ressource at the beginning of each episode
>>>>>>> d7e25db5b90c85d454b6c8ec591b2e376142a8da
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

# agent and emotion settings
AGENT_TO_TEST = "DQN"      # 'DQN' or 'QLearning'
EMOTION_TYPE = "average"   # 'average' or 'vector'
SEE_EMOTIONS = True        # whether agents observe others' emotions : first experiment - True or False
ALPHA = 0.5                # empathy degree (0.0 - 1.0) first experiment - 0.5 or 0 (non empathic)
BETA = 0.5                 # valuation of last meal
SMOOTHING = 'linear'       # function transforming the meal history into an emotion : "sigmoid" OR "linear"
SIGMOID_GAIN = 5.0
THRESHOLD = 0.5            # ratio of reward in meal history for the emotion to be neutral
EMOTION_DECIMALS = 2       # emotion's number of decimals

# RL agent hyperparameters
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
# initialization of agents and environment for a new simulation
# ----------------------------------------


def init_env(n_agents=NB_AGENTS,
             env_type=ENVIRONMENT_TYPE,
             initial_resources=INITIAL_RESOURCES,
             emotion_type=EMOTION_TYPE,
             see_emotions=SEE_EMOTIONS,
             agent_to_test=AGENT_TO_TEST,
             alpha=ALPHA,
             beta=BETA,
             smoothing=SMOOTHING,
             sigmoid_gain=SIGMOID_GAIN,
             threshold=THRESHOLD,
             round_emotions=EMOTION_DECIMALS
             ):
    """
    create environment and agents for a new simulation.
    """

    # validate agent type
    if agent_to_test not in POLICY_CLASSES:
        raise ValueError(f"Invalid AGENT_TO_TEST value: {agent_to_test}. Choose from {list(POLICY_CLASSES.keys())}")

    AgentClass = POLICY_CLASSES[agent_to_test]
    params = PARAMS_DQN if agent_to_test == "DQN" else PARAMS_QLEARNING

    env = GameTheoreticEnv(
        nb_agents=n_agents,
        env_type=env_type,
        initial_resources=initial_resources,
        emotion_type=emotion_type,
        see_emotions=see_emotions,
        agent_class=AgentClass,
        agent_configs=[params for _ in range(n_agents)],
        alpha=alpha,
        beta=beta,
        smoothing=smoothing,
        sigmoid_gain=sigmoid_gain,
        threshold=threshold,
        round_emotions=round_emotions
    )

    state_size = 1 if (not see_emotions or emotion_type == "average") else (n_agents - 1)
    action_size = env.n_actions

    agents = [AgentClass(state_size, action_size, agent_id=i, **params) for i in range(n_agents)]
    return env, agents

# ----------------------------------------
# step processing
# ----------------------------------------


def run_step(env, agents, simulation_index, episode, step, obs):
    """
    execute one step:
        select actions,
        apply to env,
        collect info,
        update agents.
    return:
        the step record,
        rewards arrays,
        next observation.
    """

    actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]

    next_obs, rewards, done, info = env.step(actions)

    # extract reward components
    personal_arr = np.array(info['personal_reward'])
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
    # learning updates
    for i, agent in enumerate(agents):
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done)

    return record, personal_arr, empathic_arr, combined_arr, next_obs, done

# ----------------------------------------
# simulation logic for one simulation with multiple episodes
# ----------------------------------------


def run_simulation(simulation_index, step_file, summary_file, seed, episode_number=EPISODE_NUMBER, step_count=MAX_STEPS, verbose=True, save_steps=True, alpha=None):
    if alpha is None:
        alpha = ALPHA
    np.random.seed(seed)
    env, _ = init_env(alpha=alpha)
    agents = env.agents
    summaries = []

    episode_iter = range(episode_number)
    if verbose:
        episode_iter = tqdm(episode_iter, desc=f"  Sim {simulation_index + 1}", unit="ep")

    for episode in episode_iter:

        obs = env.reset()
        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)
        episode_step_records = []

        step_iter = range(step_count)

        for step in step_iter:
            record, pers_r, emp_r, comb_r, obs, done = run_step(env, agents, simulation_index, episode, step, obs)

            if save_steps:
                episode_step_records.append(record)

            total_personal += pers_r
            total_empathic += emp_r
            total_combined += comb_r

            if done:
                break

        if save_steps:
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
# functions used to write CSV
# ----------------------------------------


def append_step_record(record, simulation_index, filename):
    """
    append a single step record to the simulation's step CSV file.
    writes header only once (if file doesn't exist).
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
    write a list of step-level records to a CSV file for a given simulation run.

    each record contains detailed information for a single environment step, including:
    - episode and step number
    - observations and actions of each agent
    - rewards (personal, empathic, combined) for each agent
    - remaining shared resources

    the function appends to the CSV file if it already exists, writing the header only once.
    this allows progressive saving across episodes and simulations.

    parameters:
        step_records (list of dict): list of dictionaries, each representing a single step's data.
        simulation_index (int): the index identifying the current simulation (used in file naming).
        seed (int): random seed used for the simulation (stored in the file for reproducibility).
        filename (str): full path to the CSV file where data should be written.

    outputs:
        appends step-level data to the specified CSV file.
        if the file does not exist, it creates a new file with the appropriate header.
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
    append per-episode summary data to CSV. write header only if file does not exist.
    """
    if filename is None:
        filename = build_filename(simulation_index, suffix="episode_summary")

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
# filename builder
# ----------------------------------------


def build_filename(simulation_index: int, suffix: str, empathy_alpha: float = None) -> str:
    """
    build a filename matching the original format:
    results_<sim>_<episodes>_<agent>_<emotion>_<see_emotions>_<alpha>_<beta>_
             <smoothing>_<threshold>_<rounder>_<params>_<random>_<suffix>.csv

    empathy_alpha: if provided, used in filename (for multi-condition runs); else uses global ALPHA.
    to ensure not overwriting, a version number is added if a file already exists.
    """
    if empathy_alpha is None:
        empathy_alpha = ALPHA
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
        f"{empathy_alpha}_"
        f"{BETA}_"
        f"{SMOOTHING}_"
        f"{THRESHOLD}_"
        f"{EMOTION_DECIMALS}_"
        f"{param_values}_"
        f"{random_suffix}_"
        f"{suffix}.csv"
    )

    # ensure no file overwrite by appending version number
    base_filename = filename.replace('.csv', '')
    version = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{version}.csv"
        version += 1

    return filename

# ----------------------------------------
# debug / data validation
# ----------------------------------------


def validate_rewards(csv_path, alpha=0.5, tolerance=1e-6, nb_agents=None):
    """
    validate that the combined rewards in a summary CSV file match the expected formula:
        combined_reward = (1 - alpha) * personal_reward + alpha * empathic_reward

    this function reads a summary CSV file and checks each agent's combined reward against the
    weighted sum of their personal and empathic rewards using the provided alpha (empathy degree).
    it raises an AssertionError if any discrepancies exceed the specified tolerance.

    parameters:
        csv_path (str): path to the summary CSV file generated by the simulation.
        alpha (float): weight for empathic rewards in the combination formula (range 0.0 - 1.0).
        tolerance (float): allowed numerical error when comparing expected and recorded combined rewards.
        nb_agents (int or None): number of agents in the simulation. if None, inferred from CSV column headers.

    raises:
        AssertionError: if any combined reward does not match the expected value within the specified tolerance.

    outputs:
        prints a success message if all combined rewards match expectations. otherwise, prints detailed
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
# seed management
# ----------------------------------------


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------
# main entry point
# ----------------------------------------

BASE_SEED = 2

# empathy conditions: (alpha, label)
EMPATHY_CONDITIONS = [
    #(0.0, "non_empathic"),
    #(0.15, "slightly_empathic"),
    #(0.2, "somewhat_empathic"),
    (0.25, "moderately_empathic"),
    #(0.3, "fairly_empathic"),
    #(0.4, "quite_empathic"),
    (0.5, "empathic"),
    #(0.6, "very_empathic"),
    #(0.7, "extremely_empathic"),
    (0.75, "highly_empathic"),
    (0.85, "very_highly_empathic"),
    #(0.9, "very_highly_empathic"),
    (0.99, "near_fully_empathic")
]

if __name__ == '__main__':
    # write results to empathy_tragedy/GT_simulation_jerome_thesis_emp so comparison notebook finds CSVs
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(_script_dir, "..", "GT_simulation_2_MultipleAlpha")
    folder_name = os.path.normpath(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    SHOW_SIMULATION_PROGRESS = True

    for condition_idx, (empathy_alpha, condition_label) in enumerate(EMPATHY_CONDITIONS):
        print(f"\n{'='*70}")
        print(f"CONDITION {condition_idx + 1}/{len(EMPATHY_CONDITIONS)}: {condition_label.upper()} (alpha={empathy_alpha})")
        print(f"{'='*70}\n")

        for simulation_number in range(NUM_RUNS_PER_CONDITION):
            seed = BASE_SEED + condition_idx * 1000 + simulation_number
            set_global_seed(seed)

            step_csv_name = build_filename(
                simulation_index=simulation_number,
                suffix="step_data",
                empathy_alpha=empathy_alpha,
            )
            summary_csv_name = build_filename(
                simulation_index=simulation_number,
                suffix="episode_summary",
                empathy_alpha=empathy_alpha,
            )
            step_csv_path = os.path.join(folder_name, step_csv_name)
            summary_csv_path = os.path.join(folder_name, summary_csv_name)

            run_simulation(
                episode_number=EPISODE_NUMBER,
                step_count=MAX_STEPS,
                simulation_index=simulation_number,
                step_file=step_csv_path,
                summary_file=summary_csv_path,
                seed=seed,
                verbose=True,
                save_steps=False,
                alpha=empathy_alpha,
            )
            print(f"  Saved: {summary_csv_name}")

    print(f"\n{'='*70}")
    print(f"Done. {len(EMPATHY_CONDITIONS)} conditions x {NUM_RUNS_PER_CONDITION} runs = {len(EMPATHY_CONDITIONS) * NUM_RUNS_PER_CONDITION} CSVs in {folder_name}/")
    print(f"{'='*70}\n")
