# Hybrid-Approach-for-CAP-in-RL

## 1. PROJECT TITLE & CORE HYPOTHESIS

**Title**: Hybrid-Approach-for-CAP-in-RL (Credit Assignment Problem in Reinforcement Learning)

**Core Hypothesis**: 
Pure 1-step Temporal Difference (TD) learning provides rapid online updates but suffers from slow credit assignment propagation in sparse-reward environments. Pure Monte Carlo (MC) updates provide unbiased full-episode return signals but have high variance. This project implements a **"Blended Agent"** that executes step-wise TD Q-learning during the episode AND applies post-episode MC return corrections to blend empirical averages back into the Q-table, significantly accelerating convergence and avoiding local optima.

---

## 2. ENVIRONMENT & LIBRARIES USED

**Environment**: Gymnasium `MountainCar-v0`
- **State Space**: Continuous 2D observation [position $x$, velocity $v$]
  - Position ($x$): range `[-1.2, 0.6]`
  - Velocity ($v$): range `[-0.07, 0.07]`
- **Action Space**: Discrete set of 3 actions:
  - `0`: Push Left
  - `1`: No Push
  - `2`: Push Right
- **Reward Signal**: Base reward of `-1.0` per step until reaching the goal at $x \geq 0.5$.

**Libraries & Modules Used**:
- `gymnasium`: Environment simulation and state-action stepping.
- `numpy`: State discretization (`np.linspace`, `np.digitize`) and 3D Q-table matrix representation.
- `tqdm`: Visual progress bar for 50,000 training iterations.
- `pickle`: Model serialization (`mountaincar_blended_agent_50k.pkl`) and reward log storage (`blended_rewards_50k.pkl`).
- `os`: Checkpoint checking before training/loading.
- `collections.defaultdict`: Dynamically tracking state-action return sums and visit counts for MC updates.
- `time.sleep`: Delaying frames during visual evaluation.

---

## 3. DISCRETIZATION & STATE-ACTION STRUCTURE

**Discretization Technique**:
- **Position Bins ($x\_bin = 20$)**: Cutoffs generated via `np.linspace(-1.2, 0.6, 19)`.
- **Velocity Bins ($vel\_bin = 20$)**: Cutoffs generated via `np.linspace(-0.07, 0.07, 19)`.
- Index mapping is handled via `np.digitize(x, pos_bins)` and `np.digitize(v, vel_bins)`, yielding a discrete index tuple `(pos_idx, vel_idx)`.

**Q-Table Dimensions**:
- **Array Shape**: `(20, 20, 3)` initialized to zeros.
- **Discrete States**: 20 $\times$ 20 = 400 state combinations.
- **Total State-Action Values**: 400 $\times$ 3 = 1,200 trainable parameters.

---

## 4. MATHEMATICAL FORMULATIONS

### 1. Epsilon-Greedy Action Selection & Exponential Decay
Action selection follows an $\epsilon$-greedy policy, where $\epsilon$ decays exponentially over 50,000 episodes from $\epsilon_0 = 1.0$ to $\epsilon_{\text{min}} = 0.01$.

$$
\pi(s) = 
\begin{cases} 
\text{random action} & \text{with probability } \epsilon_t \\ 
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon_t 
\end{cases}
$$

$$ 
\epsilon_t = \max(\epsilon_{\text{min}}, \epsilon_0 \cdot e^{-\lambda t}) 
$$
*(where $\lambda$ is the decay rate determined by the target minimum epsilon and total episodes)*

### 2. Reward Shaping
To guide the agent, a completion bonus is injected when the agent reaches the terminal flag.

$$ 
R_t = 
\begin{cases} 
+100 & \text{if } x \geq 0.5 \text{ (Goal reached)} \\ 
-1 & \text{otherwise (Step cost)} 
\end{cases} 
$$

### 3. 1-Step Online TD Update
Standard step-by-step Q-learning update applied during the episode trajectory.

$$ 
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right] 
$$

### 4. Post-Episode MC Correction Phase
After the episode terminates, the trajectory is traversed backward to compute actual returns.

**Return Calculation:**
$$ G_t = R_{t+1} + \gamma G_{t+1} $$

**Empirical Return Average:**
$$ 
V_{\text{MC}}(s, a) = \frac{\text{returns sum}(s, a)}{\text{returns count}(s, a)} 
$$

**Blended Q-Value Update:**
The empirical MC return is blended back into the Q-table using the learning rate $\alpha$.

$$ 
Q(s, a) \leftarrow (1 - \alpha) \cdot Q(s, a) + \alpha \cdot V_{\text{MC}}(s, a) 
$$

---

## 5. EXECUTION FLOW & PROCESS

```text
+-------------------------------------------------------------+
| 1. Episode Initialization                                   |
|    - Reset env & discretize initial observation (x, v)      |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 2. Online TD Step Loop                                      |
|    - Select action via Epsilon-Greedy                       |
|    - Step env, compute shaped reward                        |
|    - Apply 1-step TD update                                 |
|    - Log trajectory step: (state, action, reward)           |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 3. Post-Episode MC Loop                                     |
|    - Loop backwards over episode trajectory                 |
|    - Compute return G = R_{t+1} + gamma * G_{t+1}           |
|    - Update visit counts/sums                               |
|    - Blend empirical MC return into Q-table                 |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 4. Decay Epsilon & Save Checkpoint                          |
|    - Exponentially decay epsilon                            |
|    - Pickle model/logs upon completion                      |
+-------------------------------------------------------------+
```

---

## 6. CONFIGURATION & RUNNING INSTRUCTIONS

### Hyperparameter Summary Table

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `x_bin` | 20 | Number of position discretization bins |
| `vel_bin` | 20 | Number of velocity discretization bins |
| `learning_rate` ($\alpha$) | 0.1 | Step size for TD and MC blending updates |
| `discount_factor` ($\gamma$) | 0.99 | Importance of future rewards |
| `initial_epsilon` ($\epsilon_0$) | 1.0 | Initial exploration rate |
| `min_epsilon` ($\epsilon_{\text{min}}$) | 0.01 | Minimum exploration rate |
| `complete_reward` | 100 | Reward shaping bonus for reaching $x \geq 0.5$ |
| `TRAIN_EPISODES` | 50,000 | Total iterations for the training loop |

### Setup & Execution Commands

1. **Install Dependencies**:
   ```bash
   pip install gymnasium numpy tqdm
   ```

2. **Run the Project**:
   ```bash
   python  cap_rl_blended_agent.py
   ```
   *Note on Execution logic*: The script will automatically check for the existence of `mountaincar_blended_agent_50k.pkl`. If found, it will bypass the 50,000-episode training loop and immediately load the model for visual evaluation. If no checkpoint is detected, it will commence the full training loop from scratch.
