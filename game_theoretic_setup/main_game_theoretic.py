from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import csv
import matplotlib.pyplot as plt

###########################################################################################################
# General parameter
agent_policy_name_to_class = {
    "QLearning": QAgent,
    "DQN": DQNAgent
}

env_name_to_class = {
    "game_theoretic": GameTheoreticEnv  # Utilisation de GameTheoreticEnv
}

emotions_params = {
    "high_empathy": {"alpha": 0.3, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "low_empathy": {"alpha": 0.8, "beta": 0.7}
}

###########################################################################################################
# Parameter for the agents
emotion_type = "average" # can be : "high_empathy", "balanced" or "low_empathy"

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
agent_to_test = "QLearning"
empathy_to_test = "balanced"


###########################################################################################################
# Parameter of the episodes
episodes = 1
MAX_STEPS = 1000
nb_tests = 3
nb_agents = 5
initial_amount_ressources = 2000
environnement_type = "stochastic"  # can be deterministic or stochastic
np.random.seed(42)
###########################################################################################################


def initialize_agents_and_env():
    """
    Initialize a new environnement
    """
    env = GameTheoreticEnv(nb_agents=nb_agents, 
                           env_type=environnement_type,
                           initial_resources=initial_amount_ressources,
                           emotion_type=emotion_type)

    sample_obs = env.get_observation()
    state_size = len(sample_obs[0]) if isinstance(sample_obs[0], 
                                                  (list, np.ndarray)) else 1
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

    # Initialize the environment
    obs = env.reset()

    for step in range(MAX_STEPS):
        actions = []

        for i, agent in enumerate(agents):
            if isinstance(agent, QAgent):
                actions.append(agent.select_action(obs[i]))
            elif isinstance(agent, DQNAgent):
                actions.append(agent.select_action(obs[i]))

        next_obs, rewards, done, info = env.make_step(actions)

        state_snapshot = {
            'step': step,
            'resource': env.resource,
            'actions': actions,
            'emotions': obs,
            'rewards': rewards,
            'done': done
        }

        states_per_step.append(state_snapshot)
        obs = next_obs

        for i, agent in enumerate(agents):
            if isinstance(agent, QAgent):
                agent.step(next_state=next_obs[i],
                           reward=rewards[i],
                           done=done)

        if done:
            break

    return states_per_step, env


def export_to_csv_episode_data(states_per_step, filename=f'simulation_data.csv'):
    """
    Function used to create the data for each simulation
    """
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'resource'] + \
                     [f'emotion_{i}' for i in range(len(states_per_step[0]['emotions']))] + \
                     [f'reward_{i}' for i in range(len(states_per_step[0]['rewards']))] + \
                     [f'action_{i}' for i in range(len(states_per_step[0]['actions']))]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step_data in states_per_step:
            row = {
                'step': step_data['step'],
                'resource': step_data['resource'],
            }
            for i, val in enumerate(step_data['emotions']):
                row[f'emotion_{i}'] = val
            for i, val in enumerate(step_data['rewards']):
                row[f'reward_{i}'] = val
            for i, val in enumerate(step_data['actions']):
                row[f'action_{i}'] = val
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
    plt.title(f"Fluctuation of ressources in the environment for agent: {agent_to_test} with empathy level: {empathy_to_test}")
    plt.ylim(0, env.initial_resources * 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def export_general_metric_episode(input_file, output_file, episode_number):
    equality_gini_ceoficient = gini_calculator(input_file) # measure of fairness
    social_welfare = social_welfare_calculator(input_file) # sum of marginal gain
    sustainability = sustainability_calculator(input_file) # average rate of the decision to take a ressource

    fieldnames = ['episode_number', 'equality_gini_ceoficient', 'social_welfare', 'sustainability']

    new_line = [episode_number, equality_gini_ceoficient, social_welfare, sustainability]

    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, filednames=fieldnames)
        writer.writerows(new_line)


def gini_calculator(data_file):
    ...


def social_welfare_calculator(data_file):
    ...


def sustainability_calculator(data_file):
    ...


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


if __name__ == '__main__':
    for episode in range(1, episodes+1):
        states, env = run_simulation()
        filename = export_to_csv_episode_data(states,
                                   filename=f'{episode}_simulation_data.csv')
        plot_resource_evolution(states,
                                env)
        # export_general_metric_episode(filename, output_file=f'{}', episode_number=episode) # à définir

        if agent_to_test == "QLearning":
            save_q_table_detailed_to_csv(env.agents,
                                         filename=f"q_table_episode_{episode}.csv") # add length of the episode / potentially do it later for data processing
