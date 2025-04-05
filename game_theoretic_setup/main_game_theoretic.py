import argparse
import numpy as np
import matplotlib.pyplot as plt
from env_game_theoretic import GameTheoretic
from agent_policies_game_theoretic import DQNAgent


def run_simulation(
    num_agents=2,
    num_episodes=100,
    max_steps=100,
    epsilon=0.1,
    alpha=0.5,
    gamma=0.9,
):
    env = GameTheoretic(num_agents=num_agents)
    agents = [
        DQNAgent(
            id=i,
            action_space=env.action_space,
            state_shape=env.state_shape,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
        ) for i in range(num_agents)
    ]

    episode_rewards = np.zeros((num_agents, num_episodes))

    for ep in range(num_episodes):
        states = env.reset()
        total_rewards = [0 for _ in range(num_agents)]

        for _ in range(max_steps):
            actions = [agent.select_action(states[agent.id]) for agent in agents]
            next_states, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.learn(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=done,
                )
                total_rewards[i] += rewards[i]

            states = next_states
            if done:
                break

        for i in range(num_agents):
            episode_rewards[i, ep] = total_rewards[i]

    return episode_rewards


def plot_rewards(episode_rewards):
    num_agents = episode_rewards.shape[0]
    for i in range(num_agents):
        plt.plot(episode_rewards[i], label=f"Agent {i}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards per Agent")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--size", type=int, nargs=2, default=[5, 5], help="Grid size")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    rewards = run_simulation(
        grid_size=tuple(args.size),
        num_agents=args.agents,
        num_episodes=args.episodes,
        max_steps=args.steps,
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma,
    )
    plot_rewards(rewards)


if __name__ == "__main__":
    main()
