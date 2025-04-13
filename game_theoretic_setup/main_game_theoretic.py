from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

###########################################################################################################
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
agent_to_test = "DQN"
empathy_to_test = "balanced"


###########################################################################################################
# Parameter of the episodes
episodes = 50
MAX_STEPS = 30
nb_tests = 3
nb_agents = 5
np.random.seed(42)


def initialize_agents_and_env():
    """
    Initialize a new environnement
    
    """
    env = GameTheoreticEnv(nb_agents=nb_agents, 
                           env_type="deterministic",
                           initial_resources=100,
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

    # Initialize the environnement
    obs = env.reset()
    for step in range(MAX_STEPS):

        actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]

        next_obs, rewards, done, info = env.make_step(actions)

        state_snapshot = {
            'step': step,
            'resource': env.resource,
            'actions': actions,
            'emotions': obs,
            'rewards': rewards
        }
        states_per_step.append(state_snapshot)

        obs = next_obs

        if done:
            break

    return states_per_step, env


def export_to_csv(states_per_step, filename='simulation_data.csv'):
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


def animate_simulation(states_per_step, env, save_path="resource_animation.mp4"):
    """
    Function used to visualize an episode, not yet functionnal
    """
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)

    resource_line, = ax.plot([], [], 
                             label='Resource Level', 
                             color='green', 
                             linewidth=2)

    ax.set_xlim(0, len(states_per_step))
    ax.set_ylim(0, env.initial_resources * 1.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Resource")
    ax.set_title("Fluctuation of Resource Over Time")
    ax.legend(loc='upper right')

    x_data = []
    resource_data = []

    def update(frame):
        step_data = states_per_step[frame]
        x_data.append(step_data['step'])
        resource_data.append(step_data['resource'])
        resource_line.set_data(x_data, resource_data)
        return [resource_line]

    ani = FuncAnimation(fig, update, frames=len(states_per_step), interval=300, blit=True)

    writer = FFMpegWriter(fps=3)
    ani.save(save_path, writer=writer)


if __name__ == '__main__':
    states, env = run_simulation()
    export_to_csv(states)
    animate_simulation(states, env)
