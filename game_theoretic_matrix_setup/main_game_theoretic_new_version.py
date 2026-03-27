"""
main entry point for running the 2x2 matrix empathy experiment.

this script orchestrates a factorial experiment with four conditions defined by two
independent dimensions:
  - observation dimension (see_emotions): do agents observe others' emotions?
  - reward dimension (alpha):             do agents weight others' emotions in their reward?

the 2x2 design yields four conditions:
  A. blind_non_empathic    : see_emotions=False, alpha=0.0  (baseline, replicates original non-empathic)
  B. blind_reward_empathic : see_emotions=False, alpha=0.5  (rewarded for empathy but cannot observe)
  C. sees_ignores          : see_emotions=True,  alpha=0.0  (observes but ignores in reward)
  D. full_empathy          : see_emotions=True,  alpha=0.5  (full empathy, replicates original empathic)

key features:
  - configurable RL agent type (DQN or tabular Q-learning).
  - per-agent see_emotions and alpha passed through agent_configs.
  - progressive saving: episode-summary CSV files written incrementally.
  - deterministic seeding for reproducibility (numpy, random, torch).

usage:
    python main_game_theoretic_new_version.py

output CSV files are saved under  empathy_tragedy/GT_simulation_matrix/.
"""

from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import random
import csv
import os
import pandas as pd
from tqdm import tqdm
import torch



# ----------------------------------------
# constants for the simulation
# ----------------------------------------

NUM_RUNS_PER_CONDITION = 3  # number of simulation runs per condition
EPISODE_NUMBER = 500        # number of episodes per simulation
NB_AGENTS = 6
MAX_STEPS = 1000            # number of steps per episode
INITIAL_RESOURCES = 500     # resources at the beginning of each episode
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

# agent and emotion settings
AGENT_TO_TEST = "DQN"      # 'DQN' or 'QLearning'
EMOTION_TYPE = "average"   # 'average' or 'vector'
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
# 2x2 empathy matrix conditions
# (see_emotions, alpha, label)
# ----------------------------------------

EMPATHY_CONDITIONS = [
    (False, 0.0, "blind_non_empathic"),     # A: baseline — blind and self-interested
    (False, 0.5, "blind_reward_empathic"),  # B: blind altruist — empathic reward without observation
    (True,  0.0, "sees_ignores"),           # C: indifferent observer — sees but alpha=0
    (True,  0.5, "full_empathy"),           # D: full empathy — sees and cares
]

# ----------------------------------------
# initialization of agents and environment for a new simulation
# ----------------------------------------


def init_env(n_agents=NB_AGENTS,
             env_type=ENVIRONMENT_TYPE,
             initial_resources=INITIAL_RESOURCES,
             emotion_type=EMOTION_TYPE,
             see_emotions=True,
             agent_to_test=AGENT_TO_TEST,
             alpha=0.5,
             beta=BETA,
             smoothing=SMOOTHING,
             sigmoid_gain=SIGMOID_GAIN,
             threshold=THRESHOLD,
             round_emotions=EMOTION_DECIMALS
             ):
    """
    create environment and agents for a new simulation.
    see_emotions and alpha are embedded into each agent's config so they become
    per-agent attributes, enabling the 2x2 factorial design.
    """

    if agent_to_test not in POLICY_CLASSES:
        raise ValueError(f"Invalid AGENT_TO_TEST value: {agent_to_test}. Choose from {list(POLICY_CLASSES.keys())}")

    AgentClass = POLICY_CLASSES[agent_to_test]
    base_params = PARAMS_DQN if agent_to_test == "DQN" else PARAMS_QLEARNING

    # embed see_emotions and alpha into each agent's config
    per_agent_config = {**base_params, 'see_emotions': see_emotions, 'alpha': alpha}
    agent_configs = [per_agent_config.copy() for _ in range(n_agents)]

    env = GameTheoreticEnv(
        nb_agents=n_agents,
        env_type=env_type,
        initial_resources=initial_resources,
        emotion_type=emotion_type,
        see_emotions=see_emotions,   # env-level default (also set per-agent via configs)
        agent_class=AgentClass,
        agent_configs=agent_configs,
        alpha=alpha,
        beta=beta,
        smoothing=smoothing,
        sigmoid_gain=sigmoid_gain,
        threshold=threshold,
        round_emotions=round_emotions
    )

    # agents are instantiated inside GameTheoreticEnv._init_agents; use env.agents directly
    return env, env.agents

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


def run_simulation(simulation_index, step_file, summary_file, seed,
                   episode_number=EPISODE_NUMBER, step_count=MAX_STEPS,
                   verbose=True, save_steps=True,
                   alpha=None, see_emotions=None):
    if alpha is None:
        alpha = 0.5
    if see_emotions is None:
        see_emotions = True
    np.random.seed(seed)
    env, _ = init_env(alpha=alpha, see_emotions=see_emotions)
    agents = env.agents
    summaries = []

    episode_iter = tqdm(range(episode_number), desc=f"Simulation {simulation_index + 1}") if verbose else range(episode_number)

    for episode in episode_iter:
        obs = env.reset()
        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)
        episode_step_records = []

        step_iter = tqdm(range(step_count), desc=f"Episode {episode}", leave=False) if verbose else range(step_count)

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
    appends to the CSV file if it already exists, writing the header only once.
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


def build_filename(simulation_index: int, suffix: str,
                   empathy_alpha: float = None,
                   empathy_see: bool = None) -> str:
    """
    build a filename encoding both empathy dimensions:
    results_<sim>_<episodes>_<agent>_<emotion>_<see_emotions>_<alpha>_<beta>_
             <smoothing>_<threshold>_<rounder>_<params>_<random>_<suffix>.csv

    empathy_alpha: alpha value for this condition.
    empathy_see: see_emotions flag for this condition.
    to ensure no overwriting, a version number is added if a file already exists.
    """
    if empathy_alpha is None:
        empathy_alpha = 0.5
    if empathy_see is None:
        empathy_see = True
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

    filename = (
        f"results_{simulation_index:03d}_"
        f"{EPISODE_NUMBER}_"
        f"{AGENT_TO_TEST}_"
        f"{EMOTION_TYPE}_"
        f"{empathy_see}_"
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

    all agents in a homogeneous condition share the same alpha, so a single scalar
    is sufficient for validation.
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

    print("All combined rewards match expected values within tolerance.")


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

BASE_SEED = 1

if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.normpath(os.path.join(_script_dir, "..", "GT_simulation_matrix"))
    os.makedirs(folder_name, exist_ok=True)
    SHOW_SIMULATION_PROGRESS = True

    for condition_idx, (empathy_see, empathy_alpha, condition_label) in enumerate(EMPATHY_CONDITIONS):
        print(f"\n{'='*70}")
        print(f"CONDITION {condition_idx + 1}/{len(EMPATHY_CONDITIONS)}: {condition_label.upper()}")
        print(f"  see_emotions={empathy_see}, alpha={empathy_alpha}")
        print(f"{'='*70}\n")

        for simulation_number in range(NUM_RUNS_PER_CONDITION):
            seed = BASE_SEED + condition_idx * 1000 + simulation_number
            set_global_seed(seed)

            step_csv_name = build_filename(
                simulation_index=simulation_number,
                suffix="step_data",
                empathy_alpha=empathy_alpha,
                empathy_see=empathy_see,
            )
            summary_csv_name = build_filename(
                simulation_index=simulation_number,
                suffix="episode_summary",
                empathy_alpha=empathy_alpha,
                empathy_see=empathy_see,
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
                see_emotions=empathy_see,
            )
            print(f"  Saved: {summary_csv_name}")

    print(f"\n{'='*70}")
    print(f"Done. {len(EMPATHY_CONDITIONS)} conditions x {NUM_RUNS_PER_CONDITION} runs = "
          f"{len(EMPATHY_CONDITIONS) * NUM_RUNS_PER_CONDITION} CSVs in {folder_name}/")
    print(f"{'='*70}\n")
