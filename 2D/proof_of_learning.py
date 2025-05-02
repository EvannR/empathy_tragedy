# proof_of_learning.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import random

from agents_policies import QAgent, DQNAgent, SocialRewardCalculator
from env import RandomizedGridMaze

# === CONFIGURATION ===
EPISODES = 100
STEPS = 30
NB_TESTS = 5

# === ENV CONFIG ===
env_config = {
    'size': 6,
    'nb_agents': 3,
    'agent_configs': [{'memory_size': 10} for _ in range(3)],
    'reward_density': 0.2,
    'respawn_prob': 0.1,
    'simple_mode': True,
    'auto_consume': True,
    'exploit_only': False
}

empathy_config = {"alpha": 0.5, "beta": 0.5}

def run_test(agent_type):
    all_rewards = []
    for test in range(NB_TESTS):
        env = RandomizedGridMaze(**env_config)
        if agent_type == "QLearning":
            params = {
                "learning_rate": 0.1, "gamma": 0.99,
                "epsilon": 1.0, "epsilon_decay": 0.995,
                "epsilon_min": 0.01
            }
            agent_class = QAgent
        elif agent_type == "DQN":
            params = {
                "learning_rate": 0.001, "gamma": 0.99,
                "epsilon": 1.0, "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "batch_size": 16, "hidden_size": 64,
                "update_target_every": 5
            }
            agent_class = DQNAgent
        else:  # Random
            params = None
            agent_class = None

        agents = []
        for i in range(env_config['nb_agents']):
            if agent_type == "Random":
                agents.append(None)  # On utilisera np.random.choice()
            else:
                agents.append(agent_class(state_size=10, action_size=env.number_actions, agent_id=i, **params))

        reward_calc = SocialRewardCalculator(env_config['nb_agents'], **empathy_config)
        episode_rewards = []

        for episode in range(EPISODES):
            env.new_episode()
            for i, agent in enumerate(agents):
                if agent_type != "Random":
                    agent.start_episode(env.agents[i].get_state(env))
            episode_reward = 0

            for step in range(STEPS):
                for i in range(env_config['nb_agents']):
                    s = env.agents[i].get_state(env)
                    if agent_type == "Random":
                        a = np.random.choice(env.actions)
                    else:
                        a = agents[i].select_action(s)
                    r, _ = env.make_step(i, a)
                    episode_reward += r
                    s2 = env.agents[i].get_state(env)
                    if agent_type != "Random":
                        agents[i].step(s2, r, False)
                        if agent_type == "QLearning":
                            key = agents[i].get_state_key(s2)
                            q_vals = agents[i].q_table.get(key, np.zeros(env.number_actions))
                            #print(f"Agent {i} | Épisode {episode} Step {step} | Q-values: {q_vals.round(2)}")
                env.update_environment()

            rewards_social = reward_calc.calculate_rewards(env.agents)
            for i in range(env_config['nb_agents']):
                if agent_type != "Random":
                    s_final = env.agents[i].get_state(env)
                    agents[i].step(s_final, rewards_social[i], True)

            episode_rewards.append(episode_reward)
        all_rewards.append(episode_rewards)
    return np.array(all_rewards)

# === RUN AND PLOT ===
results = {}
for model in ["Random", "QLearning"]:
    print(f"\n>>> Test de {model}...")
    rewards = run_test(model)
    mean_r = np.mean(rewards, axis=0)
    std_r = np.std(rewards, axis=0)
    results[model] = (mean_r, std_r)

plt.figure(figsize=(12, 6))
for model, (mean_r, std_r) in results.items():
    plt.plot(mean_r, label=f"{model}")
    plt.fill_between(range(EPISODES), mean_r - std_r, mean_r + std_r, alpha=0.2)

plt.title("Comparaison des stratégies : QLearning vs DQN vs Random")
plt.xlabel("Épisode")
plt.ylabel("Récompense moyenne")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
