from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import random


agent_policy_name_to_class = {
    "QLearning": QAgent,
    "DQN": DQNAgent
}

env_name_to_class = {
    "game_theoretic": GameTheoreticEnv  # Utilisation de GameTheoreticEnv
}

# changed no longer used but can be a reference
emotions_params = {
    "high_empathy": {"alpha": 1, "beta": 0.7},
    "medium_high_empathy": {"alpha": 0.7, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.7},
    "low_empathy": {"alpha": 0.3, "beta": 0.7},
    "no_empathy": {"alpha": 0, "beta": 0.7}
}

emotional_observation_type = {
    "average": "average",
    "vector": 'vector'

}

##############################################################################
# Parameter for the agents
params_QLearning = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01
}

params_DQN = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 16,
    "hidden_size": 64,
    "update_target_every": 5
}

agent_params = {
    "QLearning": params_QLearning,
    "DQN": params_DQN
}

# Choice of the agent and level of empathy
agent_to_test = "DQN"  # "DQN" or "QLearning"
emotion_type = "average"  # can be average or vector
see_emotions = True
alpha = 1  # parameter for the degree of empathy (the higher the value the higher the empathy in range 0 - 1)
beta = 0.3  # parameter for the valuation of the last meal (higher beta = higher valuation)

# Choice of emotion parameters
smoothing_type = 'linear'  # or 'linear'
sigmoid_gain_value = 5.0
threshold_value = 0.5  # ratio of ressource acquisition needed for the emotion to be neutral
emotion_rounder = 1 # number of decimals of the emotion => complexity 


##############################################################################
# Parameter of the episodes
episodes = 1
MAX_STEPS = 1000
nb_tests = 3
nb_agents = 5
initial_amount_ressources = 3000
environnement_type = "stochastic"  # can be deterministic or stochastic #
np.random.seed(42) # interesting to save in csv_file + iterating runs over different seeds
##############################################################################


def initialize_agents_and_env():
    """
    Initialize a new environnement
    """
    env = GameTheoreticEnv(
        nb_agents=nb_agents,
        env_type=environnement_type,
        initial_resources=initial_amount_ressources,
        emotion_type=emotion_type,
        see_emotions=see_emotions,
        agent_class=agent_policy_name_to_class[agent_to_test],
        alpha=alpha,
        beta=beta,
        smoothing=smoothing_type,
        sigmoid_gain=sigmoid_gain_value,
        threshold=threshold_value,
        round_emotions=emotion_rounder
    )

    if not see_emotions:
        state_size = 1
    elif emotion_type == "average":
        state_size = 1
    elif emotion_type == "vector":
        state_size = nb_agents - 1
    else:
        raise ValueError("Unknown emotion_type")
 
    action_size = env.number_actions

    agents = []
    for agent_idx in range(nb_agents):
        if agent_to_test == "QLearning":
            agent = QAgent(state_size, action_size,
                           agent_id=agent_idx,
                           **params_QLearning)
        else:
            agent = DQNAgent(state_size,
                             action_size,
                             agent_id=agent_idx,
                             **params_DQN)
        agents.append(agent)

    return env, agents


def run_simulation():
    env, agents = initialize_agents_and_env()
    states_per_step = []

    # Initial state
    obs = env.reset()

    for step in range(MAX_STEPS):
        actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]

        next_obs, rewards, done, info = env.make_step(actions)

        state_snapshot = {
            'step': step,
            'resource': env.resource,
            'actions': actions,
            'observations': obs,
            'rewards_total': rewards,
            'exploitation_reward': info['exploitation_reward'],
            'personal_reward': info['personal_satisfaction'],
            'empathic_reward': info['empathic_reward'],
            'emotions': info['emotions'],
            'combined_reward': info['combined_reward'],
            'done': done
        }

        states_per_step.append(state_snapshot)
        obs = next_obs

        # learning step for each agent
        for i, agent in enumerate(agents):
            agent.step(next_state=next_obs[i], reward=rewards[i], done=done)

        if done:
            break

    return states_per_step, env, agents


def export_to_csv_episode_data(states_per_step, filename='simulation_data.csv'): # ADD episode number 
    """
    Export episode data to CSV with one line per step, including for each agent:
    - current resource level
    - observation (emotion)
    - action selected
    - personal_reward
    - empathic_reward
    - combined_reward (total internal reward)
    """
    import csv

    # Number of agents
    n_agents = len(states_per_step[0]['observations'])

    # Dynamically construct CSV headers
    fieldnames = ['step', 'resource']
    for i in range(n_agents):
        fieldnames += [
            f'observation_{i}',
            f'action_{i}',
            f'personal_reward_{i}',
            f'empathic_reward_{i}',
            f'combined_reward_{i}'
        ]

    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step_data in states_per_step:
            row = {'step': step_data['step'], 'resource': step_data.get('resource', None)}
            obs = step_data['observations']
            acts = step_data['actions']
            personal = step_data['personal_reward']
            empathic = step_data['empathic_reward']
            # combined reward stored as 'internal_total_reward'
            combined = step_data.get('combined_reward', [])

            for i in range(n_agents):
                row[f'observation_{i}'] = obs[i]
                row[f'action_{i}'] = acts[i]
                row[f'personal_reward_{i}'] = personal[i]
                row[f'empathic_reward_{i}'] = empathic[i]
                # safe fallback to 0 if missing
                row[f'combined_reward_{i}'] = combined[i] if i < len(combined) else None

            writer.writerow(row)

    return filename


def plot_resource_evolution(states_per_step, env, save_path="resource_evolution.png"):
    """
    Generate a static image to visualize the evolution of the ressources
    """
    steps = [step['step'] for step in states_per_step]
    resources = [step['resource'] for step in states_per_step]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, resources, label='Level of ressources', color='green', linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Ressource")
    plt.title(f"Fluctuation of ressources in the environment for agent: {agent_to_test} with empathy level: {alpha} and valuation of last meal: {beta}")
    plt.ylim(0, env.initial_resources * 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_q_table_detailed_to_csv(agents, filename="q_table_detailed.csv"):
    """
    Save each Q-value individually with action separation.
    CSV format: agent_id, state, action, expected_reward
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["agent_id", "state", "action", "expected_reward"])

        for agent_idx, agent in enumerate(agents):
            if hasattr(agent, 'q_table'):  # QAgent only
                for state, actions in agent.q_table.items():
                    for action, value in enumerate(actions):
                        writer.writerow([agent_idx, state, action, value])


def visualize_q_table(filename):
    df = pd.read_csv(filename)
    agent_id = 0
    df_agent = df[df['agent_id'] == agent_id]

    pivot_table = df_agent.pivot(index='state', columns='action', values='expected_reward')

    plt.figure(figsize=(12, 6))
    plt.title(f"Q-table (agent {agent_id})")
    heatmap = plt.imshow(pivot_table.fillna(0), cmap='viridis', aspect='auto')
    plt.colorbar(heatmap, label='Expected Reward')
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.xticks(ticks=range(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index)
    plt.tight_layout()
    plt.show()


def filename_definer(agent_type,
                     episode_number,
                     emotion_type,
                     see_emotions,
                     alpha,
                     beta,
                     smoothing_type,
                     threshold_value,
                     emotion_rounder,
                     params_DQN,
                     params_QLearning):
    """
    name of the file order : 
    episode number
    agent_to_test = "DQN" or "QLearning"
    emotion_type = can be "average" or "vector"
    see_emotions = "False" or "True"
    alpha = 1  # parameter for the degree of empathy (the higher the value the higher the empathy in range 0 - 1)
    beta = 0.3 # valuation of the last meal
    smoothing_type = linear or sigmoid
    threshold_value  proportion of reward in the history necessary to have a positive emotion
    emotion_rounder = decimale of emotions

    the parameters of the agents are in the order :
Params_QL
    "learning_rate"
    "gamma"
    "epsilon"
    "epsilon_decay"
    "epsilon_min"

params_DQN =
    "learning_rate"
    "gamma"
    "epsilon"
    "epsilon_decay"
    "epsilon_min"
    "batch_size"
    "hidden_size"
    "update_target_every"

    return the filename of one episode with a random 6 int suffix
    """
    if agent_type == "DQN":
        params = params_DQN
        param_order = ["learning_rate", "gamma", "epsilon", "epsilon_decay", "epsilon_min", "batch_size", "hidden_size", "update_target_every"]
    elif agent_type == "QLearning":
        params = params_QLearning
        param_order = ["learning_rate", "gamma", "epsilon", "epsilon_decay", "epsilon_min"]
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")

    # Ensure values appear in fixed order (no key names)
    param_values = "_".join(str(params[key]) for key in param_order)

    random_suffix = ''.join(str(random.randint(0, 9)) for _ in range(6))
    see_emotions_str = str(see_emotions)

    filename = (
        f"results_"
        f"{episode_number}_"
        f"{agent_type}_"
        f"{emotion_type}_"
        f"{see_emotions_str}_"
        f"{alpha}_"
        f"{beta}_"
        f"{smoothing_type}_"
        f"{threshold_value}_"
        f"{emotion_rounder}_"
        f"{param_values}_"
        f"{random_suffix}.csv"
    )

    return filename


if __name__ == '__main__':
    for episode_number in range(1, episodes+1): # add a seed number to the loop + name of file (for a full simulation including multiple episodes)
        states, env, agents = run_simulation()
        filename_data = export_to_csv_episode_data(states,
                                                   filename=filename_definer(agent_type=agent_to_test,
                                                                             episode_number=episode_number,
                                                                             emotion_type=emotion_type,
                                                                             see_emotions=see_emotions,
                                                                             alpha=alpha,
                                                                             beta=beta,
                                                                             smoothing_type=smoothing_type,
                                                                             threshold_value=threshold_value,
                                                                             emotion_rounder=emotion_rounder,
                                                                             params_DQN=params_DQN,
                                                                             params_QLearning=params_QLearning
                                                                             )
                                                        )
        # perhaps make a simulation csv_file : one csv for multiple episodes (one simulation) => change csv_builder

        plot_resource_evolution(states,
                                env)

#######added to visualize the fluctuation of the Q_table
        #if agent_to_test == "QLearning":
        #    save_q_table_detailed_to_csv(agents,
        #                                 filename=f"q_table_episode_{episode}.csv")
        #    visualize_q_table(f"q_table_episode_{episode}.csv")
