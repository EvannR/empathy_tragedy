# simple_env.py
import numpy as np

class SimpleGridWorld:
    """
    Simplified grid environment where agents learn when to consume resources
    based on their own emotional state and others' emotional states.
    """
    def __init__(self, size=5, num_agents=3, resource_density=0.3, respawn_prob=0.1):
        """
        Initialize the environment.
        
        Args:
            size: Size of the grid (size x size)
            num_agents: Number of agents in the environment
            resource_density: Percentage of grid cells with resources
            respawn_prob: Probability of resource respawning each step
        """
        self.size = size
        self.num_agents = num_agents
        self.resource_density = resource_density
        self.respawn_prob = respawn_prob
        
        # Grid to track resources (values represent resource value)
        self.resources = np.zeros((size, size))
        
        # Define actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=CONSUME
        self.num_actions = 5
        self.action_names = {
            0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "CONSUME"
        }
        
        # Agent positions and emotional states
        self.agent_positions = []
        self.emotional_states = np.zeros(num_agents)  # Range -1 to 1
        self.satisfaction_history = [[] for _ in range(num_agents)]
        
        # Initialize environment
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode."""
        # Place agents randomly
        self.agent_positions = []
        positions = set()
        while len(positions) < self.num_agents:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in positions:
                positions.add(pos)
                self.agent_positions.append(pos)
        
        # Initialize resources (1 = present, 0 = absent)
        self.resources = np.zeros((self.size, self.size))
        num_resources = int(self.size * self.size * self.resource_density)
        flat_indices = np.random.choice(self.size * self.size, size=num_resources, replace=False)
        for idx in flat_indices:
            i, j = idx // self.size, idx % self.size
            self.resources[i, j] = 1.0
        
        # Reset emotional states
        self.emotional_states = np.zeros(self.num_agents)
        self.satisfaction_history = [[] for _ in range(self.num_agents)]
        
        # Return initial observations
        return self._get_observations()
    
    def step(self, actions):
        """
        Take a step in the environment based on agents' actions.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        
        # Process actions and collect rewards
        rewards = np.zeros(self.num_agents)
        consumed = [False] * self.num_agents
        
        for i, action in enumerate(actions):
            # Get current position
            curr_pos = self.agent_positions[i]
            
            # Calculate new position based on action
            if action == 0:  # UP
                new_pos = (max(0, curr_pos[0] - 1), curr_pos[1])
            elif action == 1:  # DOWN
                new_pos = (min(self.size - 1, curr_pos[0] + 1), curr_pos[1])
            elif action == 2:  # LEFT
                new_pos = (curr_pos[0], max(0, curr_pos[1] - 1))
            elif action == 3:  # RIGHT
                new_pos = (curr_pos[0], min(self.size - 1, curr_pos[1] + 1))
            elif action == 4:  # CONSUME
                new_pos = curr_pos  # Stay in place for consume action
                
                # Check if resource exists at current position
                if self.resources[curr_pos] > 0:
                    # Resource value is always 1.0 when present
                    resource_value = 1.0
                    # Record consumption
                    consumed[i] = True
                    # Remove the resource
                    self.resources[curr_pos] = 0
                    # Base reward for consumption
                    rewards[i] = resource_value
            else:
                raise ValueError(f"Invalid action: {action}")
            
            # Update position
            self.agent_positions[i] = new_pos
        
        # Update emotional states based on consumption
        for i in range(self.num_agents):
            # Update satisfaction history (1 if consumed, 0 if not)
            self.satisfaction_history[i].append(1 if consumed[i] else 0)
            
            # Keep only last 5 steps
            if len(self.satisfaction_history[i]) > 5:
                self.satisfaction_history[i] = self.satisfaction_history[i][-5:]
            
            # Calculate emotional state (-1 to 1)
            # Higher values mean more satiated/less hungry
            recent_consumption = sum(self.satisfaction_history[i]) / max(1, len(self.satisfaction_history[i]))
            self.emotional_states[i] = 2 * recent_consumption - 1
        
        # Compute social reward adjustment based on emotional states
        avg_emotional_state = np.mean(self.emotional_states)
        
        for i in range(self.num_agents):
            if consumed[i]:
                # If agent consumed while already above average emotional state,
                # apply negative reward adjustment
                if self.emotional_states[i] > avg_emotional_state:
                    social_penalty = -0.5 * (self.emotional_states[i] - avg_emotional_state)
                    rewards[i] += social_penalty
                    
                # If agent consumed while below average emotional state,
                # this is socially beneficial, so provide a bonus
                else:
                    social_bonus = 0.2 * (avg_emotional_state - self.emotional_states[i])
                    rewards[i] += social_bonus
        
        # Resource respawning
        empty_cells = np.where(self.resources == 0)
        respawn_mask = np.random.random(len(empty_cells[0])) < self.respawn_prob
        respawn_indices = (empty_cells[0][respawn_mask], empty_cells[1][respawn_mask])
        
        if len(respawn_indices[0]) > 0:
            self.resources[respawn_indices] = 1.0
        
        # Get observations for the next step
        observations = self._get_observations()
        
        # For simplicity, episodes don't terminate
        done = False
        
        # Collect info
        info = {
            'emotional_states': self.emotional_states,
            'average_emotional_state': avg_emotional_state,
            'consumed': consumed
        }
        
        return observations, rewards, done, info
    
    def _get_observations(self):
        """
        Get observations for all agents.
        Each observation includes:
        - Agent's position
        - Nearby resources
        - Agent's own emotional state
        - Average emotional state of all agents
        """
        observations = []
        avg_emotional_state = np.mean(self.emotional_states)
        
        for i in range(self.num_agents):
            pos = self.agent_positions[i]
            
            # Create observation vector
            obs = np.zeros(7)  # 2 for position, 4 for nearby resources, 1 for own emotional state
            
            # Position (normalized)
            obs[0] = pos[0] / (self.size - 1)
            obs[1] = pos[1] / (self.size - 1)
            
            # Check resources in 4 adjacent cells
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
            for d, (di, dj) in enumerate(directions):
                ni, nj = pos[0] + di, pos[1] + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    obs[2 + d] = self.resources[ni, nj]
            
            # Own emotional state
            obs[6] = self.emotional_states[i]
            
            # Add average emotional state of all agents
            obs = np.append(obs, avg_emotional_state)
            
            observations.append(obs)
        
        return observations
    
    def render(self):
        """Simple text-based rendering of the environment."""
        grid = np.zeros((self.size, self.size), dtype=object)
        
        # Fill grid with resource indicators (R for resource, . for empty)
        for i in range(self.size):
            for j in range(self.size):
                if self.resources[i, j] > 0:
                    grid[i, j] = "R"
                else:
                    grid[i, j] = "."
        
        # Add agents
        for i, pos in enumerate(self.agent_positions):
            grid[pos] = f"A{i}"
        
        # Print grid
        print("\n" + "-" * (self.size * 4 + 1))
        for i in range(self.size):
            row = "|"
            for j in range(self.size):
                cell = str(grid[i, j]).center(3)
                row += f" {cell} |"
            print(row)
            print("-" * (self.size * 4 + 1))
        
        # Print agent emotional states
        print("\nAgent Emotional States:")
        for i, emotion in enumerate(self.emotional_states):
            print(f"Agent {i}: {emotion:.2f}")
        print(f"Average: {np.mean(self.emotional_states):.2f}")
        print()