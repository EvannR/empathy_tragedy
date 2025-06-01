import numpy as np
from agent_policies_maze import QAgentMaze, DQNAgent, SocialRewardCalculator


class Maze2DEnv:
    """
    Multi-agent environment where agents move randomly in a 2D maze and must decide
    when to consume resources they encounter. Agents can optionally observe emotional
    states of others (average or full vector).
    """
    def __init__(self, nb_agents,
                 maze_size=(10, 10),  # Size of the maze (width, height)
                 initial_resources=100,
                 resource_regen_rate=0.1,  # Percentage of resources that regenerate per step
                 resource_distribution="random",  # How resources are distributed in the maze
                 resource_density=0.3,  # Percentage of cells that have resources
                 env_type="deterministic",
                 emotion_type="average",
                 see_emotions=True,
                 alpha=0.5,
                 beta=0.5,
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.7,
                 smoothing='linear',
                 sigmoid_gain=10.0,
                 round_emotions=None):

        # Environment settings
        self.nb_agents = nb_agents
        self.maze_size = maze_size
        self.initial_resources = initial_resources
        self.resource_regen_rate = resource_regen_rate
        self.resource_distribution = resource_distribution
        self.resource_density = resource_density
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector" (ignored if see_emotions=False)
        self.see_emotions = see_emotions      # If False, agents receive zero observations
        self.alpha = alpha                    # empathic weight on others
        self.beta = beta                      # weight of last vs history
        self.number_actions = 2               # 0: Not Consume, 1: Consume
        self.actions = np.arange(self.number_actions)
        self.round_emotions = round_emotions
        
        # Directions for movement (up, right, down, left)
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Agent setup
        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        # Social reward calculator
        self.reward_calculator = SocialRewardCalculator(nb_agents,
                                                        alpha=alpha,
                                                        beta=beta,
                                                        threshold=threshold,
                                                        smoothing=smoothing,
                                                        sigmoid_gain=sigmoid_gain
                                                        )

        # Initialize agents and reset environment state
        self._init_agents()
        self.reset()

    def _init_agents(self):
        # Determine input dimension for agents
        # Position coordinates (x,y) + resource at location + emotional inputs
        base_state_size = 3  # (x, y, resource_at_location)
        
        if not self.see_emotions:
            self.state_size = base_state_size
        else:
            if self.emotion_type == "average":
                self.state_size = base_state_size + 1
            elif self.emotion_type == "vector":
                self.state_size = base_state_size + self.nb_agents - 1
            else:
                raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

        # Instantiate agent objects
        self.agents = []
        for idx, config in enumerate(self.agent_configs):
            agent = self.agent_class(
                state_size=self.state_size,
                action_size=self.number_actions,
                agent_id=idx,
                **config
            )
            self.agents.append(agent)

    def _generate_maze(self):
        """
        Generate a 2D maze with resources.
        """
        # Initialize an empty maze
        self.maze = np.zeros(self.maze_size)
        
        # Place resources based on distribution type
        if self.resource_distribution == "random":
            # Randomly distribute resources
            num_resource_cells = self.initial_resources
            flat_indices = np.random.choice(
                self.maze_size[0] * self.maze_size[1],
                size=num_resource_cells,
                replace=False
            )
            
            for idx in flat_indices:
                x = idx % self.maze_size[0]
                y = idx // self.maze_size[0]
                
                # Each resource cell has a random amount between 1-5 units
                self.maze[x, y] = 1
        
        elif self.resource_distribution == "clustered":
            # Create a few cluster centers
            num_clusters = max(1, int(self.maze_size[0] * self.maze_size[1] * 0.05))
            centers = np.random.randint(0, min(self.maze_size), size=(num_clusters, 2))
            
            # For each cell, check distance to nearest cluster
            for x in range(self.maze_size[0]):
                for y in range(self.maze_size[1]):
                    # Distance to closest cluster center
                    min_dist = min(np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in centers)
                    # Probability of resource based on distance
                    prob = np.exp(-min_dist / 3)  # Decay factor of 3
                    if np.random.random() < prob * self.resource_density * 3:
                        self.maze[x, y] = np.random.randint(1, 6)
        
        self.initial_maze = self.maze.copy()  # Store initial state for statistics
        
    def reset(self):
        """
        Reset the environment to initial state.
        """
        self._generate_maze()
        self.time_step = 0
        
        # Randomly place agents in the maze
        self.agent_positions = []
        for _ in range(self.nb_agents):
            x = np.random.randint(0, self.maze_size[0])
            y = np.random.randint(0, self.maze_size[1])
            self.agent_positions.append((x, y))
        
        obs = self.get_observation()
        
        for idx, agent in enumerate(self.agents):
            agent.reset(observation=obs[idx])
            
            if hasattr(agent, 'memory'):
                agent.memory.buffer.clear()
                
        return obs
        
    def get_observation(self):
        """
        Return list of observations for each agent.
        Each observation includes:
        - Agent's position (x, y)
        - Resource amount at current location
        - Emotional information (if enabled)
        """
        # Calculate emotions for all agents
        emotions = self.reward_calculator.calculate_emotions(self.agents)
        
        if self.round_emotions is not None:
            emotions = np.round(emotions, self.round_emotions)
            
        observations = []
        
        for i, pos in enumerate(self.agent_positions):
            x, y = pos
            
            # Base observation: position and resource at location
            base_obs = [x / self.maze_size[0], y / self.maze_size[1], self.maze[x, y] / 5.0]
            
            # Add emotional component if enabled
            if not self.see_emotions:
                obs = np.array(base_obs, dtype=float)
            else:
                other_emotions = [e for j, e in enumerate(emotions) if j != i]
                
                if self.emotion_type == "average":
                    emotion_obs = [np.mean(other_emotions)] if other_emotions else [0.0]
                    obs = np.array(base_obs + emotion_obs, dtype=float)
                elif self.emotion_type == "vector":
                    obs = np.array(base_obs + other_emotions, dtype=float)
                else:
                    raise ValueError(f"Unknown emotion_type: {self.emotion_type}")
                    
            observations.append(obs)
            
        return observations
    
    def _move_agents_randomly(self):
        """
        Move all agents randomly to adjacent cells.
        """
        new_positions = []
        
        for x, y in self.agent_positions:
            # Choose a random direction
            dx, dy = self.directions[np.random.randint(0, 4)]
            
            # Calculate new position with boundary checks
            new_x = max(0, min(self.maze_size[0] - 1, x + dx))
            new_y = max(0, min(self.maze_size[1] - 1, y + dy))
            
            new_positions.append((new_x, new_y))
            
        self.agent_positions = new_positions
    
    def _regenerate_resources(self):
        """
        Regenerate a portion of consumed resources.
        """
        # Find cells where resources were consumed (comparing with initial state)
        for x in range(self.maze_size[0]):
            for y in range(self.maze_size[1]):
                if self.maze[x, y] < self.initial_maze[x, y]:
                    # Regenerate with probability based on regen_rate
                    if np.random.random() < self.resource_regen_rate:
                        self.maze[x, y] = min(self.maze[x, y] + 1, self.initial_maze[x, y])
    
    def make_step(self, actions):
        """
        Execute one timestep: 
        - First move all agents randomly
        - Then let agents decide whether to consume resources
        - Update environment state
        
        actions: list of 0/1 for each agent (not consume/consume).
        Returns: next_observations, rewards, done, info
        """
        # Move agents randomly first
        self._move_agents_randomly()
        
        consumed = 0
        immediate_rewards = []
        
        for idx, (act, pos) in enumerate(zip(actions, self.agent_positions)):
            x, y = pos
            reward = 0.0
            success = False
            
            # Check if agent chose to consume and there's a resource
            if act == 1 and self.maze[x, y] > 0:
                if self.env_type == "stochastic":
                    # Probability of success proportional to resource amount
                    max_resource = self.initial_maze[x, y] if self.initial_maze[x, y] > 0 else 5
                    prob = self.maze[x, y] / max_resource
                    success = np.random.random() < prob
                else:
                    success = True
                    
                if success:
                    # Consume one unit of resource
                    self.maze[x, y] -= 1
                    reward = 1.0
                    consumed += 1
            
            # Record whether agent got a meal
            if hasattr(self.agents[idx], 'record_meal'):
                self.agents[idx].record_meal(success, reward)
            else:
                raise ValueError('Problem in the record meal method')
                
            immediate_rewards.append(reward)
        
        # Calculate social rewards
        emotions, personal_reward, empathic_reward, total_reward = self.reward_calculator.calculate_rewards(self.agents)
        
        # Regenerate resources
        self._regenerate_resources()
        
        # Update environment state
        self.time_step += 1
        next_obs = self.get_observation()
        
        # Count remaining resources in the whole maze
        remaining_resources = np.sum(self.maze)
        done = remaining_resources <= 0
        
        info = {
            'emotions': emotions,
            'personal_satisfaction': personal_reward,
            'empathic_reward': empathic_reward,
            'exploitation_reward': immediate_rewards,
            'combined_reward': total_reward,
            'remaining_resources': remaining_resources,
            'agent_positions': self.agent_positions
        }
        
        return next_obs, total_reward, done, info
    
    def get_agent_meal_stats(self, agent_idx):
        """Return recent and total meals for one agent."""
        a = self.agents[agent_idx]
        return {
            'recent_meals': a.get_recent_meals(),
            'total_meals': a.total_meals,
            'meal_history': list(a.meal_history)
        }
    
    def get_all_agents_meal_stats(self):
        """Return meal stats for all agents."""
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
    
    def get_maze_visualization(self):
        """
        Return a string representation of the maze with agents and resources.
        For debugging and visualization purposes.
        """
        # Create a copy of the maze for visualization
        viz_maze = np.zeros(self.maze_size, dtype=object)
        
        # Fill with resource values
        for x in range(self.maze_size[0]):
            for y in range(self.maze_size[1]):
                viz_maze[x, y] = str(int(self.maze[x, y])) if self.maze[x, y] > 0 else '.'
        
        # Add agents (using their agent_id)
        for i, (x, y) in enumerate(self.agent_positions):
            viz_maze[x, y] = f"A{i}" if viz_maze[x, y] == '.' else f"{viz_maze[x, y]}+A{i}"
        
        # Convert to string representation
        rows = []
        for y in range(self.maze_size[1]):
            row = ' '.join(viz_maze[:, y])
            rows.append(row)
        
        return '\n'.join(rows)