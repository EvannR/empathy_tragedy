import numpy as np
from agents_policies import Agent
import time 
import os


class GridMaze:
    """
    Base class for grid-based environments where agents can move around
    and interact with resources.
    """
    def __init__(self, size=4, nb_agents=1, agent_configs=None):
        """
        Initialize the grid environment.
        
        
            size: Size of the grid (size x size)
            nb_agents: Number of agents in the environment
            agent_configs: Configuration parameters for each agent
        """
        self.size = size
        self.nb_agents = nb_agents
        self.number_actions = 5  # UP, DOWN, LEFT, RIGHT, EXPLOIT
        self.actions = np.arange(self.number_actions)

        # Define action names for clarity
        self.action_names = {
            0: "UP",
            1: "DOWN", 
            2: "LEFT", 
            3: "RIGHT", 
            4: "EXPLOIT"
        }
        
        # Initialize agent positions
        self.agents_positions = self.initialize_positions()
        
        # # Create agents with their position
        if agent_configs is None:
            agent_configs = [{'memory_size': 10} for _ in range(nb_agents)]
        
        self.agents = []
        for i in range(nb_agents):
            config = agent_configs[i] if i < len(agent_configs) else {'memory_size': 10}
            memory_size = config.get('memory_size', 10)
            self.agents.append(Agent(i, self.agents_positions[i], memory_size))
        
       
        self.rewards = np.zeros((size, size))  # Initialize reward grid
        self.init_transitions()                # Initialize action transition map
        self.time_step = 0                     # Time step counter

    def initialize_positions(self):
        """
        Randomly place agents on the grid ensuring no overlap.
        
        Returns:
            List of (row, col) positions for each agent
        """

        positions = set()
        while len(positions) < self.nb_agents:
            i, j = np.random.randint(0, self.size, size=2)
            positions.add((i, j))
        return list(positions)

    def init_transitions(self):
        """
        Initialize the state transition function for each action.
        Creates a mapping from (position, action) to new position.
        """

        UP, DOWN, LEFT, RIGHT, EXPLOIT = self.action_names.keys()
        self.P = {}
        
        for i in range(self.size):
            for j in range(self.size):
                self.P[(i, j)] = {}
                
                # UP action
                if i == 0:
                    self.P[(i, j)][UP] = (i, j)
                else:
                    self.P[(i, j)][UP] = (i - 1, j)
                
                # DOWN action
                if i == self.size - 1:
                    self.P[(i, j)][DOWN] = (i, j)
                else:
                    self.P[(i, j)][DOWN] = (i + 1, j)
                
                # LEFT action
                if j == 0:
                    self.P[(i, j)][LEFT] = (i, j)
                else:
                    self.P[(i, j)][LEFT] = (i, j - 1)
                
                # RIGHT action
                if j == self.size - 1:
                    self.P[(i, j)][RIGHT] = (i, j)
                else:
                    self.P[(i, j)][RIGHT] = (i, j + 1)
                
                # EXPLOIT action
                self.P[(i, j)][EXPLOIT] = (i, j)

    def new_episode(self):
        """ Reset the environment for a new episode."""
        self.time_step = 0
        self.agents_positions = self.initialize_positions()
        
        # update of position of the agents
        for i, agent in enumerate(self.agents):
            agent.update_position(self.agents_positions[i])

    def update_environment(self):
        self.time_step += 1
        

    def make_step(self, agent_idx, action):
        """Met à jour l'état d'un agent spécifique et enregistre les repas"""
        agent = self.agents[agent_idx]
        current_pos = agent.position
        new_pos = self.P[current_pos][action]
        
        # update the position of the agent
        agent.update_position(new_pos)
        self.agents_positions[agent_idx] = new_pos
        
        # Check if there's a reward at this position
        reward = self.rewards[new_pos]
        has_eaten = reward > 0
        
        
        agent.record_meal(has_eaten, reward) # Record if the agent has eaten
        
        return reward, new_pos
    
    def get_agent_meal_stats(self, agent_idx):
        """
        Return meal statistics for a specific agent.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            Dict with meal statistics
        """
        agent = self.agents[agent_idx]
        return {
            'recent_meals': agent.get_recent_meals(),
            'total_meals': agent.total_meals,
            'meal_history': list(agent.meal_history)
        }
    
    def get_all_agents_meal_stats(self):
        """
        Return meal statistics for all agents.
        
        Returns:
            List of meal statistics for each agent
        """
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
    
    def get_state(self):
        """
        Get the current state of the entire environment.
        
        Returns:
            Dict containing grid state, agent positions, and rewards
        """
        return {
            'grid_size': self.size,
            'time_step': self.time_step,
            'agent_positions': self.agents_positions,
            'rewards': self.rewards.copy()
        }


class RandomizedGridMaze(GridMaze):
    """
    Extended GridMaze with randomized resource generation and consumption.
    """
    def __init__(self, size=4, nb_agents=1, agent_configs=None, reward_density=0.4, 
                 respawn_prob=0.1, simple_mode=False, auto_consume=False, 
                 exploit_only=True):
        super().__init__(size, nb_agents, agent_configs)
        self.reward_density = reward_density
        self.respawn_prob = respawn_prob
        self.simple_mode = simple_mode   # Simple vs. complex resource dynamics
        self.auto_consume = auto_consume  # Auto consumption of resources
        self.exploit_only = exploit_only  # Resources only consumed with EXPLOIT action
        self.initialize_rewards()
    
    def initialize_rewards(self):
        """Generate random initial rewards on the grid based on reward density."""
        self.rewards = np.zeros((self.size, self.size))
        num_rewards = int(self.size * self.size * self.reward_density)
        reward_positions = np.random.choice(self.size * self.size, num_rewards, replace=False)
        for pos in reward_positions:
            i, j = divmod(pos, self.size)
            self.rewards[i, j] = np.random.uniform(0.1, 1.0)
    
    def update_environment(self):
        """Update the environment by potentially adding or removing resources."""
        super().update_environment()
        
        
        if self.simple_mode:
            if np.random.rand() < self.respawn_prob: 
                empty_cells = np.argwhere(self.rewards == 0)
                if empty_cells.size > 0:
                    i, j = empty_cells[np.random.choice(len(empty_cells))]
                    self.rewards[i, j] = np.random.uniform(0.1, 1.0)

        elif not self.simple_mode:
            for i in range(self.size):
                for j in range(self.size):
                    p_new_ressources_next_state = self.respawn_prob * self.reward_density
                    if np.random.rand() < p_new_ressources_next_state:
                        if self.rewards[i, j] == 0:
                            self.rewards[i, j] = np.random.uniform(0.1, 1.0)
                        else:
                            self.rewards[i, j] = 0 
    
    def make_step(self, agent_idx, action):
        """
        Update state for a specific agent, handling resource consumption.
        
        Args:
            agent_idx: Index of the agent
            action: Action ID (0-4)
            
        Returns:
            Tuple of (reward, new_position)
        """
        agent = self.agents[agent_idx]
        current_pos = agent.position
        new_pos = self.P[current_pos][action]
        
       
        agent.update_position(new_pos)    # Update agent position
        self.agents_positions[agent_idx] = new_pos
        
        # Record if the agent has eaten
        can_consume = True
        if self.exploit_only and action != 4:  #If only EXPLOIT allows consumption
            can_consume = False
        
        reward = 0
        has_eaten = False
        
        if can_consume:
            reward = self.rewards[new_pos]
            has_eaten = reward > 0
            
            #  # If agent ate and auto-consume is enabled, remove the resource
            if has_eaten and self.auto_consume:
                self.rewards[new_pos] = 0
        
        # Enregistrer si l'agent a mangé
        agent.record_meal(has_eaten, reward)
        
        return reward, new_pos






