# Architecture of Blended RL Agent (`cap_rl_blended_agent.py`)

## 1. Overview
The `cap_rl_blended_agent.py` implements a **Blended Reinforcement Learning Agent** that combines two distinct learning methods to solve the Credit Assignment Problem (CAP) in the `MountainCar-v0` environment:
*   **Temporal Difference (TD) Learning (Q-Learning):** Provides fast, step-by-step updates.
*   **Monte Carlo (MC) Learning:** Provides unbiased, stable corrections at the end of each episode.

This hybrid approach aims to leverage the speed of TD learning while using MC updates to prevent the agent from converging to poor local optima.

## 2. System Components

### 2.1 Dependencies
*   **Gymnasium (`gym`):** RL environment (`MountainCar-v0`).
*   **NumPy (`np`):** Matrix operations and state discretization.
*   **Pickle:** Model persistence (save/load).

### 2.2 Class: `BlendedAgent`
The core logic is encapsulated in the `BlendedAgent` class.

#### **State Representation (Discretization)**
The MountainCar environment gives continuous state values (position, velocity). The agent discretizes these into bins to use a tabular Q-learning approach.
*   **Position:** Discretized into `x_bin` (default 20) bins.
*   **Velocity:** Discretized into `vel_bin` (default 20) bins.
*   **Q-Table:** A 3D NumPy array of shape `(x_bin, vel_bin, action_space_size)`.

#### **Learning Algorithms**
The agent maintains two sets of memory for learning:
1.  **Q-Table:** Scores for `(state, action)` pairs, updated by both TD and MC.
2.  **Returns Memory (`returns_sum`, `returns_count`):** used for calculating the Monte Carlo average return for state-action pairs.

### 3. Training Workflow (`train` method)

The training process follows a dual-update cycle for each episode:

#### **Phase 1: Intra-Episode (TD Update)**
During the episode, at *every time step*:
1.  **Action Selection:** Epsilon-greedy policy (`choose_action`).
2.  **Environment Step:** Execute action, observe `next_state` and `reward`.
3.  **Reward Shaping:** A bonus `complete_reward` (100) is added if the car reaches the goal (`position >= 0.5`).
4.  **TD Update:** The Q-table is updated immediately using the Q-Learning formula:
    $$ Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s', a') - Q(s,a)] $$

#### **Phase 2: End-of-Episode (MC Update)**
After the episode finishes:
1.  **Backpropagation:** The agent iterates backwards through the episode history.
2.  **Return Calculation:** Computes the return `G` (cumulative discounted reward) for each step.
3.  **MC Update:** The Q-table is updated ("blended") effectively averaging the current Q-value with the observed Monte Carlo return:
    $$ Q(s,a) \leftarrow (1 - \alpha) Q(s,a) + \alpha \times \text{Average}(G) $$
    *Note: The code actually updates using `self.alpha` to blend the current Q-value with the `mc_value` (average return).*

### 4. Persistence
*   **`save(filename)`:** Serializes the entire agent object using `pickle`.
*   **`load(filename)`:** Deserializes the agent object.

### 5. Execution Flow (`__main__`)
1.  Checks if a pre-trained model (`mountaincar_blended_agent_50k.pkl`) exists.
2.  **If found:** Loads the agent.
3.  **If not found:**
    *   Initializes a new `BlendedAgent`.
    *   Trains for 10,000 episodes.
    *   Saves the model and reward logs.
4.  **Visualization:** Runs the `solve` method to demonstrate the learned policy with rendering enabled.
