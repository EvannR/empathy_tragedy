from env_game_theoretic import GameTheoreticEnv
from agent_policies_game_theoretic import QAgent, DQNAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
from matplotlib.widgets import Button

# Dictionnaires pour lier le nom des agents à leurs classes
agent_policy_name_to_class = {
    "QLearning": QAgent,
    "DQN": DQNAgent
}

# Dictionnaire pour lier le nom de l'environnement à sa classe
env_name_to_class = {
    "game_theoretic": GameTheoreticEnv  # Utilisation de GameTheoreticEnv
}

# Paramètres des émotions
emotions_params = {
    "high_empathy": {"alpha": 0.3, "beta": 0.7},
    "balanced": {"alpha": 0.5, "beta": 0.5},
    "low_empathy": {"alpha": 0.8, "beta": 0.7}
}

# Paramètres des agents
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

# Choix de l'agent et de l'empathie
agent_to_test = "DQN"
empathy_to_test = "balanced"

# Nombre d'épisodes, d'étapes et de tests
episodes = 50
MAX_STEPS = 30
nb_tests = 3
nb_agents = 5
np.random.seed(42)

def initialize_agents_and_env():
    # Initialisation de l'environnement
    env = GameTheoreticEnv(nb_agents=nb_agents, env_type="deterministic", initial_resources=100)

    # Initialisation des agents
    agents = []
    for agent_idx in range(nb_agents):
        if agent_to_test == "QLearning":
            agent = QAgent(agent_idx, env, **params_QLearning)
        else:
            agent = DQNAgent(agent_idx, env, **params_DQN)
        agents.append(agent)

    return env, agents

def run_simulation():
    env, agents = initialize_agents_and_env()
    states_per_step = []

    # Réinitialiser l'environnement
    obs = env.reset()
    for step in range(MAX_STEPS):
        # Choisir l'action pour chaque agent
        actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]
        # Passer l'action à l'environnement et obtenir les résultats
        next_obs, rewards, done, info = env.make_step(actions)

        # Capturer l'état de la simulation à chaque étape
        state_snapshot = {
            'step': step,
            'resource': env.resource,
            'actions': actions,
            'emotions': [agent.emotion_level for agent in agents],
            'rewards': rewards
        }
        states_per_step.append(state_snapshot)
        obs = next_obs

        if done:
            break

    return states_per_step, env

def export_to_csv(states_per_step, filename='simulation_data.csv'):
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

def animate_simulation(states_per_step, env):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    resource_line, = ax.plot([], [], label='Resource', color='green')
    emotion_lines = []
    reward_lines = []
    action_markers = []

    num_agents = len(states_per_step[0]['emotions'])

    for i in range(num_agents):
        line, = ax.plot([], [], label=f'Agent {i} Emotion')
        emotion_lines.append(line)
        reward_line, = ax.plot([], [], label=f'Agent {i} Reward', linestyle='--')
        reward_lines.append(reward_line)

    ax.set_xlim(0, len(states_per_step))
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Resource, Agent Emotions, Rewards, and Actions Over Time")
    ax.legend(loc='upper right')

    x_data = []
    resource_data = []
    emotion_data = [[] for _ in range(num_agents)]
    reward_data = [[] for _ in range(num_agents)]

    paused = [False]

    def update(frame):
        if paused[0]:
            return []
        step_data = states_per_step[frame]
        x_data.append(step_data['step'])
        resource_data.append(step_data['resource'] / env.initial_resources)

        for i in range(num_agents):
            emotion_data[i].append(step_data['emotions'][i])
            reward_data[i].append(step_data['rewards'][i])

        resource_line.set_data(x_data, resource_data)
        for i in range(num_agents):
            emotion_lines[i].set_data(x_data, emotion_data[i])
            reward_lines[i].set_data(x_data, reward_data[i])

        for marker in action_markers:
            marker.remove()
        action_markers.clear()
        for i, action in enumerate(step_data['actions']):
            color = 'red' if action == 1 else 'blue'
            marker_style = '^' if action == 1 else 'o'
            marker = ax.plot(step_data['step'], 1.05 + i * 0.05, marker_style, color=color)[0]
            action_markers.append(marker)

        return [resource_line] + emotion_lines + reward_lines + action_markers

    def toggle_pause(event):
        paused[0] = not paused[0]

    pause_ax = plt.axes([0.45, 0.05, 0.1, 0.075])
    pause_button = Button(pause_ax, 'Pause/Resume')
    pause_button.on_clicked(toggle_pause)

    ani = animation.FuncAnimation(fig, update, frames=len(states_per_step), interval=300, blit=True)
    plt.show()

if __name__ == '__main__':
    states, env = run_simulation()
    export_to_csv(states)
    animate_simulation(states, env)
