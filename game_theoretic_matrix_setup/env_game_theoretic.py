"""
game-theoretic multi-agent environment for the tragedy-of-the-commons scenario.
2x2 matrix version: see_emotions and alpha are per-agent attributes read from
agent_configs, allowing independent control of the observation and reward dimensions.

this module defines GameTheoreticEnv, where N agents share a finite resource pool
and independently decide whether to exploit (action=1) or abstain (action=0) at
each time step.  the resource decreases with exploitation and regenerates at a
configurable rate.

observations are now gated per-agent: each agent's see_emotions flag determines
whether it receives emotion signals of others or a zero vector.
reward shaping uses per-agent alpha values stored on each agent instance:
  combined_reward_i = (1 - alpha_i) * personal_satisfaction_i + alpha_i * empathic_reward_i

supports both deterministic and stochastic exploitation modes.
"""

import numpy as np
from agent_policies_game_theoretic import QAgent, DQNAgent, SocialRewardCalculator


class GameTheoreticEnv:
    """
    multi-agent environment where each agent can exploit a shared resource or not.

    each agent has its own see_emotions and alpha attributes (set via agent_configs),
    enabling a 2x2 factorial design:
      - see_emotions=False/True : observation dimension
      - alpha=0.0/0.5          : reward dimension
    """
    def __init__(self, nb_agents,
                 initial_resources=100,
                 regen_rate=1.0,
                 env_type="deterministic",
                 emotion_type="average",
                 see_emotions=True,      # env-level default (overridden by per-agent config)
                 alpha=0.5,              # env-level default (overridden by per-agent config)
                 beta=0.5,
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.7,
                 smoothing='linear',
                 sigmoid_gain=10.0,
                 round_emotions=None):

        # environment settings
        self.nb_agents = nb_agents
        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector"
        self.see_emotions = see_emotions      # env-level default (used as fallback in _init_agents)
        self._alpha = alpha                   # env-level default (used as fallback in _init_agents)
        self.beta = beta                      # weight of last vs history
        self.n_actions = 2                    # 0: abstain, 1: exploit
        self.actions = np.arange(self.n_actions)
        self.round_emotions = round_emotions
        self._threshold = threshold
        self._smoothing = smoothing
        self._sigmoid_gain = sigmoid_gain

        # agent setup
        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        # init agents FIRST so per-agent alphas can be read for SocialRewardCalculator
        self._init_agents()

        # build reward calculator with per-agent alphas extracted from instantiated agents
        alphas = [a.alpha for a in self.agents]
        self.reward_calculator = SocialRewardCalculator(
            nb_agents,
            alpha=alphas,
            beta=beta,
            threshold=threshold,
            smoothing=smoothing,
            sigmoid_gain=sigmoid_gain
        )

        self.reset()
        self._alpha_initial = alpha

    @property
    def alpha(self):
        return self._alpha

    def _init_agents(self):
        """
        instantiate agents with per-agent see_emotions and alpha from agent_configs.
        falls back to env-level defaults when not specified in a config dict.
        stores per-agent state sizes in self.agent_state_sizes.
        """
        self.agents = []
        self.agent_state_sizes = []

        for idx, config in enumerate(self.agent_configs):
            # read per-agent flags; fall back to env-level defaults
            agent_see_emotions = config.get('see_emotions', self.see_emotions)
            agent_alpha = config.get('alpha', self._alpha)

            # compute state_size for this specific agent
            if not agent_see_emotions:
                agent_state_size = 1  # dummy zero observation
            elif self.emotion_type == "average":
                agent_state_size = 1  # single averaged emotion scalar
            elif self.emotion_type == "vector":
                agent_state_size = self.nb_agents - 1  # one value per other agent
            else:
                raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

            self.agent_state_sizes.append(agent_state_size)

            # pass all other config keys through, but exclude see_emotions/alpha
            # since they are passed as explicit named arguments
            agent_init_kwargs = {
                k: v for k, v in config.items()
                if k not in ('see_emotions', 'alpha')
            }

            agent = self.agent_class(
                state_size=agent_state_size,
                action_size=self.n_actions,
                agent_id=idx,
                see_emotions=agent_see_emotions,
                alpha=agent_alpha,
                **agent_init_kwargs
            )
            self.agents.append(agent)

        # backward compat: expose a single state_size (valid for homogeneous populations)
        self.state_size = self.agent_state_sizes[0] if self.agent_state_sizes else 1

    def reset(self):
        self.resource = self.initial_resources
        self.time_step = 0

        obs = self.get_observations()

        for idx, agent in enumerate(self.agents):
            agent.reset(observation=obs[idx])

            if hasattr(agent, 'memory'):
                agent.memory.buffer.clear()

        return obs

    def get_observations(self):
        """
        return list of observations, one per agent.
        each agent's observation is gated by its own see_emotions flag:
          - see_emotions=False : zero vector of size agent_state_sizes[i]
          - see_emotions=True  : average or vector of other agents' emotions

        emotions are always computed for all agents regardless of see_emotions,
        since they feed into calculate_rewards for agents with alpha > 0.
        """
        # always compute emotions (needed for reward calculation even if not observed)
        emotions = self.reward_calculator.calculate_emotions(self.agents)
        if self.round_emotions is not None:
            emotions = np.round(emotions, self.round_emotions)

        observations = []
        for i, agent in enumerate(self.agents):
            if not agent.see_emotions:
                # agent cannot observe others' emotions: return a zero vector
                obs = np.zeros(self.agent_state_sizes[i], dtype=float)
            else:
                other_emotions = [e for j, e in enumerate(emotions) if j != i]
                if self.emotion_type == "average":
                    obs = np.array([np.mean(other_emotions)], dtype=float)
                elif self.emotion_type == "vector":
                    obs = np.array(other_emotions, dtype=float)
                else:
                    raise ValueError(f"Unknown emotion_type: {self.emotion_type}")
            observations.append(obs)

        return observations

    def step(self, actions):
        """
        execute one timestep: agents choose actions, environment updates.
        actions: list of 0/1 for each agent.
        returns: next_observations, rewards, done, info.
        """
        consumed = 0
        immediate_rewards = []

        for idx, act in enumerate(actions):
            reward = 0.0
            success = False
            if act == 1 and self.resource > 0:
                if self.env_type == "stochastic":
                    # success probability decreases as resources are depleted
                    prob = self.resource / self.initial_resources
                    success = np.random.rand() < prob
                else:
                    # deterministic: exploitation always succeeds if resources remain
                    success = True
                if success:
                    reward = 1.0
                    consumed += 1

            if hasattr(self.agents[idx], 'record_meal'):
                self.agents[idx].record_meal(success, reward)
            else:
                raise ValueError('problem in the record meal')
            immediate_rewards.append(reward)

        emotions, personal_reward, empathic_reward, total_reward = self.reward_calculator.calculate_rewards(self.agents)

        # update resource pool and step counter
        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)
        self.time_step += 1
        next_obs = self.get_observations()
        # episode ends when the resource pool is fully depleted
        done = self.resource <= 0

        info = {
            'emotions': emotions,
            'personal_reward': personal_reward,
            'empathic_reward': empathic_reward,
            'combined_reward': total_reward
        }

        return next_obs, total_reward, done, info

    def get_agent_meal_stats(self, agent_idx):
        """return recent and total meals for one agent."""
        a = self.agents[agent_idx]
        return {
            'recent_meals': a.get_recent_meals(),
            'total_meals': a.total_meals,
            'meal_history': list(a.meal_history)
        }

    def get_all_agents_meal_stats(self):
        """return meal stats for all agents."""
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
