# empathy_tragedy

## Multi-Agent Simulation with an Empathy Component

**!! Warning !!** The files in this folder that are not listed here are either test files or outdated files that are no longer used to run the simulations.

### Overview

This project implements a multi-agent reinforcement learning (MARL) system in which several agents interact on a grid to collect resources. The key feature is that each agent’s reward depends not only on its own consumption (personal satisfaction) but also on the consumption of other agents (empathy).

The architecture is modular and includes several files:

1. `env.py` – Defines the environment  
2. `agents.py` – Defines agent structures  
3. `policies.py` – Implements learning algorithms (Q-Learning and DQN)  
4. `marl_simulation.py` – Orchestrates the simulation  
5. `analyze_results.py` – Analyzes and visualizes results  

---

### 1. Environment (`env.py`)

The environment is a 2D grid containing agents and resources:

- **`GridMaze`** (base class):  
  - Defines a grid with configurable size  
  - Manages agent movement (UP, DOWN, LEFT, RIGHT, EXPLOIT)  
  - Tracks agent positions and rewards  

- **`RandomizedGridMaze`** (derived class):  
  - Randomly places resources in the grid  
  - Configures resource behavior (spawning, consumption)  
  - Allows choosing whether resources are automatically consumed or only via the EXPLOIT action  

The environment maintains the state of the system at each timestep and can be reset for a new episode.

---

### 2. Agents (`agents.py`)

The `Agent` class encapsulates fundamental features and behaviors of the agents:

- Stores the agent’s current position  
- Keeps a circular buffer (`deque`) for a meal history  
- Tracks the total number of meals  
- Provides methods for recording new meals and computing statistics  

Each agent can have a different memory capacity, affecting the length of its meal history.

---

### 3. Learning Policies (`policies.py`)

This file contains the reinforcement learning algorithms:

- **`QAgent`**: Classic Q-Learning implementation  
  - Uses a table to store Q-values  
  - Employs an epsilon-greedy policy for exploration and exploitation  
  - Updates Q-values based on the Bellman equation  

- **`DQNAgent`**: Deep Q-Network implementation  
  - Uses neural networks to approximate the Q-function  
  - Utilizes a replay buffer for batch learning  
  - Employs a target network to stabilize learning  

- **`EmotionalModel`** and **`SocialRewardCalculator`**:  
  - Compute rewards based on empathy  
  - Balance personal satisfaction with that of other agents  
  - Factor in meal history and the most recent meal  

- **`ReplayBuffer`**:  
  - Stores experiences (state, action, reward, next state, terminal)  
  - Samples mini-batches of experiences for learning  

---

### 4. Main Simulation (`marl_simulation.py`)

This file orchestrates the entire learning process:

1. Initializes the environment and agents  
2. Creates RL agents (Q-Learning or DQN) according to the chosen algorithm  
3. Runs complete episodes by:  
   - Resetting the environment  
   - Having agents select actions  
   - Executing actions in the environment  
   - Computing social rewards at the end of each episode  
   - Training the agents  
4. Collects statistics for analysis  
5. Visualizes the environment and agent performance  

The method `get_state_representation` converts the grid state into a format suitable for reinforcement learning—normalized agent positions and nearby resource information.

---

### 5. Results Analysis (`analyze_results.py`)

This file provides tools to:

- Run experiments with various configurations (algorithms, parameters)  
- Visualize learning curves  
- Analyze the convergence of different algorithms  
- Compare performance in terms of social welfare and individual rewards  

---

## Key Technical Aspects

### 1. State Representation

States are represented as feature vectors that include:
- The agent’s normalized position (i and j divided by the grid size)  
- Resource values in the eight directions around the agent  

### 2. Empathy Model

The empathy model is parameterized by two coefficients:
- `alpha`: Balances personal satisfaction (1.0) and empathy (0.0)  
- `beta`: Balances the most recent meal against the full meal history  

### 3. Reward System

At the end of each episode, the final reward includes:
- **Personal satisfaction**:  
  \[
  \beta \times (\text{last meal}) + (1 - \beta) \times (\text{average over the meal history})
  \]
- **Emotional reward**:  
  \[
  \alpha \times (\text{personal satisfaction}) + (1 - \alpha) \times (\text{average satisfaction of other agents})
  \]

### 4. Error Handling and Debugging

The code features a robust error-handling system:
- Traces issues related to data types and conversions  
- Displays detailed information about states, actions, and rewards  
- Continues learning even if an error occurs during an episode  

---

## Parameterization and Flexibility

The system is highly configurable:
- Grid size  
- Number of agents  
- Resource density and dynamics  
- Learning algorithm (Q-Learning vs. DQN)  
- Empathy parameters (`alpha`, `beta`)  
- Learning hyperparameters (learning rate, discount factor, exploration)  

This flexibility allows for testing different hypotheses and configurations to find the optimal setups for MARL research.

---

## Conclusion
