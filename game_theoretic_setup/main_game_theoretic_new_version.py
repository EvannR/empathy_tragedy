from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import random
import csv
import os

# ----------------------------------------
# Constants for the simulation
# ----------------------------------------


SIMULATION_NUMBER = 2      # number of simulation runs (also used as seed per run)
EPISODE_NUMBER = 15         # number of episodes per simulation
NB_AGENTS = 2
MAX_STEPS = 300            # number of steps per episode
INITIAL_RESOURCES = 200    # number of ressource at the beginning of each episode
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

# Agent & emotion settings
AGENT_TO_TEST = "DQN"      # 'DQN' or 'QLearning'
EMOTION_TYPE = "average"   # 'average' or 'vector'
SEE_EMOTIONS = True        # whether agents observe others' emotions
ALPHA = 0.5                # empathy degree (0.0 - 1.0)
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


def run_step(env, agents, seed, episode, step, obs):
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
    # Select actions
    actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]

    # Environment transition
    next_obs, rewards, done, info = env.make_step(actions)

    # Extract reward components
    personal_arr = np.array(info['personal_satisfaction'])
    empathic_arr = np.array(info['empathic_reward'])
    combined_arr = np.array(info['combined_reward'])

    # Build record
    record = {
        'seed': seed,
        'episode': episode,
        'step': step,
        'resource': env.resource,
        'observations': obs.copy(),
        'actions': actions,
        'personal': personal_arr.tolist(),
        'empathic': empathic_arr.tolist(),
        'combined': combined_arr.tolist()
    }

    # Learning updates
    for i, agent in enumerate(agents):
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done)

    return record, personal_arr, empathic_arr, combined_arr, next_obs, done

# ----------------------------------------
# Simulation logic for one simulation with multiple episodes
# ----------------------------------------


def run_simulation(episode_count, simulation_index):
    """
    Runs `episode_count` episodes in one simulation,
    returns detailed per-step data and per-episode summaries.
    """
    env, agents = initialize_agents_and_env()
    detailed_data = []
    summaries = []

    for episode in range(episode_count):
        obs = env.reset()
        episode_steps = []
        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)

        for step in range(MAX_STEPS):
            actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]
            next_obs, rewards, done, info = env.make_step(actions)

            prs = np.array(info['personal_satisfaction'])
            ers = np.array(info['empathic_reward'])
            crs = np.array(info['combined_reward'])

            total_personal += prs
            total_empathic += ers
            total_combined += crs

            episode_steps.append({
                'seed': simulation_index,
                'episode': episode,
                'step': step,
                'resource': env.resource,
                'observations': obs.copy(),
                'actions': actions,
                'personal': prs.tolist(),
                'empathic': ers.tolist(),
                'combined': crs.tolist()
            })

            for i, agent in enumerate(agents):
                agent.step(next_state=next_obs[i], reward=rewards[i], done=done)
            obs = next_obs
            if done:
                break

        detailed_data.append(episode_steps)
        summaries.append({
            'seed': simulation_index,
            'episode': episode,
            'total_steps': step + 1,
            'resource_remaining': env.resource,
            'personal_totals': total_personal.tolist(),
            'empathic_totals': total_empathic.tolist(),
            'combined_totals': total_combined.tolist()
        })

    return detailed_data, summaries

# ----------------------------------------
# Functions used to write CSV
# ----------------------------------------


def write_step_csv(detailed_data, simulation_index, filename=None):
    """
    Write detailed per-step data to CSV, including seed and episode.
    """
    if filename is None:
        filename = filename_definer(simulation_index, suffix="step_data")

    header = (
        ["seed", "episode", "step", "resource_remaining", "initial_resources"] +
        [f"observation_{i}" for i in range(NB_AGENTS)] +
        [f"action_{i}" for i in range(NB_AGENTS)] +
        [f"reward_{i}" for i in range(NB_AGENTS)] +
        [f"emotion_{i}" for i in range(NB_AGENTS)]
        )

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for episode_steps in detailed_data:
            for record in episode_steps:
                row = [
                    record['seed'], record['episode'],
                    record['step'], record['resource'], INITIAL_RESOURCES
                ] + sum([
                    [record['observations'][i], record['actions'][i],
                     record['personal'][i], record['empathic'][i], record['combined'][i]]
                    for i in range(NB_AGENTS)
                ], [])
                writer.writerow(row)


def write_summary_csv(summaries, simulation_index, filename=None):
    """
    Write per-episode summary data to CSV, including seed and episode.
    """
    if filename is None:
        filename = filename_definer(simulation_index, suffix="episode_summary")

    header = (
        ['seed', 'episode', 'total_steps', 'resource_remaining', 'initial_resources'] +
        [f"total_personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_combined_reward_{i}" for i in range(NB_AGENTS)]
    )

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in summaries:
            row = [
                rec['seed'], rec['episode'],
                rec['total_steps'], rec['resource_remaining'], INITIAL_RESOURCES
            ] + sum([
                [rec['personal_totals'][i], rec['empathic_totals'][i], rec['combined_totals'][i]]
                for i in range(NB_AGENTS)
            ], [])
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
    version = 1
    while os.path.exists(filename):
        filename = filename.replace('.csv', f'_{version}.csv')
        version += 1

    return filename

# ----------------------------------------
# Main entry
# ----------------------------------------


if __name__ == '__main__':
    for simulation_number in range(SIMULATION_NUMBER):
        np.random.seed(simulation_number + 1)

        detailed, summaries = run_simulation(EPISODE_NUMBER, simulation_number)
        write_step_csv(detailed, simulation_number)
        write_summary_csv(summaries, simulation_number)
