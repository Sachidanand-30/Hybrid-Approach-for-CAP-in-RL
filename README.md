# Hybrid-Approach-for-CAP-in-RL
A Hybrid Approach of Combining Temporal Difference and Monte Carlo Based Learning for Better Credit Assignment in MountainCar-v0  Environment
# Blended TD-MC Q-Learning for MountainCar-v0

## Overview

This repository implements a **Blended Reinforcement Learning Agent** that combines **Temporal Difference (TD) Q-Learning** with **Monte Carlo (MC) Return Correction** to solve the `MountainCar-v0` environment.

### Core Hypothesis

Standard 1-step TD learning provides rapid online updates but can suffer from severe bias and slow credit assignment propagation across long horizons in sparse reward environments. Every-episode Monte Carlo updates supply an unbiased, long-term reward signal. By blending step-wise TD updates with post-episode MC corrections, the agent gains fast convergence while preventing local optima entrapment.

---

## 1. Environment & Dependencies

### Environment

* **Environment Name:** `MountainCar-v0` (Gymnasium)
* **Goal:** Drive an underpowered car up a steep hill to reach the goal post at position $x \ge 0.5$.
* **State Space:** Continuous 2D vector $\mathbf{s} = [x, v]$
* **Position ($x$):** $[-1.2, 0.6]$
* **Velocity ($v$):** $[-0.07, 0.07]$


* **Action Space:** Discrete 1D set of 3 actions:
* `0`: Accelerate to the left
* `1`: Don't accelerate
* `2`: Accelerate to the right


* **Base Reward:** $-1.0$ for every step until the goal state is reached.

### Required Python Libraries

* **`gymnasium`**: Simulation environment interface.
* **`numpy`**: Fast matrix operations and multi-dimensional state table management.
* **`tqdm`**: Progress tracking for training loops.
* **`pickle`**: Model serialization and reward logging.
* **`os`**: File system path verification.
* **`collections.defaultdict`**: Dynamically tracking MC state-action return sums and visit counts.

---

## 2. State Discretization & Action Space Structure

Because $Q$-Learning requires discrete tabular representations, the continuous 2D observation space is discretized into uniform bins.

### Bin Configuration

* **Position Bins ($x\_bin = 20$):** Divided into 19 threshold cutoffs generated via `np.linspace(-1.2, 0.6, 19)`.
* **Velocity Bins ($vel\_bin = 20$):** Divided into 19 threshold cutoffs generated via `np.linspace(-0.07, 0.07, 19)`.

### Discrete Mapping

Using `np.digitize`, continuous continuous states $(x, v)$ map into discrete state indices $(s_x, s_v)$:

$$s_x = \text{digitize}(x, \text{pos\_bins})$$

$$s_v = \text{digitize}(v, \text{vel\_bins})$$

where $s_x \in \{0, 1, \dots, 19\}$ and $s_v \in \{0, 1, \dots, 19\}$.

### State-Action Q-Table

The $Q$-table is stored as a 3D NumPy array of shape $(20, 20, 3)$, initialized to all zeros:

$$\mathbf{Q} \in \mathbb{R}^{20 \times 20 \times 3}$$

* Total Discrete States: $20 \times 20 = 400$
* Total State-Action Pairs: $400 \times 3 = 1200$

---

## 3. Mathematical Formulations & Techniques

### Exploration Strategy ($\epsilon$-Greedy Decay)

Actions are selected according to an $\epsilon$-greedy policy:

$$a_t = \begin{cases} \text{random choice from } \{0, 1, 2\}, & \text{with probability } \epsilon \\ \arg\max_{a} Q(s_t, a), & \text{with probability } 1 - \epsilon \end{cases}$$

The exploration rate decays exponentially from $\epsilon_0 = 1.0$ to $\epsilon_{\text{min}} = 0.01$ over $N = 50,000$ episodes:

$$\text{decay\_rate} = \left( \frac{\epsilon_{\text{min}}}{\epsilon_0} \right)^{\frac{1}{N}} = (0.01)^{\frac{1}{50000}} \approx 0.9999079$$

$$\epsilon_{t+1} = \max\left( \epsilon_{\text{min}}, \epsilon_t \cdot \text{decay\_rate} \right)$$

### Reward Shaping

To resolve sparse reward feedback in Mountain Car, a terminal reward bonus ($R_{\text{complete}} = +100$) is added when the target state $x \ge 0.5$ is achieved:

$$R_{\text{shaped}} = \begin{cases} +100, & \text{if } x \ge 0.5 \text{ and episode terminal} \\ -1, & \text{otherwise} \end{cases}$$

---

### Step-Wise Temporal Difference (TD) Update

At each environment step $t$, the agent applies standard 1-step Q-learning:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{\text{shaped}} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

If the step terminates directly at the goal ($x \ge 0.5$), the terminal state-action value is explicitly set to the completion reward:

$$Q(s_t, a_t) \leftarrow R_{\text{complete}}$$

---

### Post-Episode Monte Carlo (MC) Correction Phase

At the end of every episode, the trajectory history $[(s_0, a_0, r_1), (s_1, a_1, r_2), \dots, (s_{T-1}, a_{T-1}, r_T)]$ is iterated in reverse to compute empirical returns $G_t$:

$$G_t = r_{t+1} + \gamma G_{t+1}$$

For every visited state-action pair $(s_t, a_t)$, cumulative returns and visit counts are maintained:

$$\text{returns\_sum}(s_t, a_t) \leftarrow \text{returns\_sum}(s_t, a_t) + G_t$$

$$\text{returns\_count}(s_t, a_t) \leftarrow \text{returns\_count}(s_t, a_t) + 1$$

The empirical mean Monte Carlo value $\bar{G}(s_t, a_t)$ is computed:

$$\bar{G}(s_t, a_t) = \frac{\text{returns\_sum}(s_t, a_t)}{\text{returns\_count}(s_t, a_t)}$$

Finally, the existing $Q$-value is blended with the empirical Monte Carlo return:

$$Q(s_t, a_t) \leftarrow (1 - \alpha) Q(s_t, a_t) + \alpha \bar{G}(s_t, a_t)$$

---

## 4. End-to-End Execution Flow

```
+-------------------------------------------------------------------+
|                        Start Episode                              |
|   Reset Gym Env -> Get continuous observation (x, v)              |
|   Discretize -> s_0 = (pos_idx, vel_idx)                          |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                      Step-Wise Loop (TD Phase)                    |
| 1. Select action via Epsilon-Greedy policy                        |
| 2. Step environment -> get next (x', v'), base reward, done       |
| 3. Discretize next observation -> s_{t+1}                         |
| 4. Compute shaped reward (+100 if reached x >= 0.5)               |
| 5. Update Q-table immediately using TD target formulation         |
| 6. Append (s_t, a_t, reward) to trajectory log                    |
+-------------------------------------------------------------------+
                                  |
                   (Episode Ends / Goal Reached)
                                  v
+-------------------------------------------------------------------+
|                   Post-Episode Loop (MC Phase)                    |
| 1. Traverse trajectory backwards: compute Return G_t              |
| 2. Accumulate G_t and increment visit count for (s_t, a_t)        |
| 3. Compute empirical mean MC return V_MC = sum / count            |
| 4. Blend: Q(s_t, a_t) = (1 - alpha) * Q(s_t, a_t) + alpha * V_MC   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Update Hyperparameters                        |
| Decay Epsilon: epsilon = max(min_epsilon, epsilon * decay_rate)   |
+-------------------------------------------------------------------+

```

---

## 5. Agent Parameters Reference

| Parameter | Default Value | Description |
| --- | --- | --- |
| `x_bin` | `20` | Number of discrete grid divisions along the position axis |
| `vel_bin` | `20` | Number of discrete grid divisions along the velocity axis |
| `learning_rate` ($\alpha$) | `0.1` | Step size multiplier for both TD and MC blended updates |
| `discount_factor` ($\gamma$) | `0.99` | Future reward discount factor |
| `epsilon` ($\epsilon$) | `1.0` | Initial exploration rate |
| `min_epsilon` | `0.01` | Exploration floor limit |
| `complete_reward` | `100` | Terminal bonus for reaching target position ($x \ge 0.5$) |
| `TRAIN_EPISODES` | `50,000` | Total training episode count |

---

## 6. How to Run

### Installation

Ensure all dependent packages are installed:

```bash
pip install gymnasium numpy tqdm

```

### Training & Evaluation

Execute the primary script:

```bash
python main.py

```

* **If no trained model exists:** The script trains the `BlendedAgent` over 50,000 episodes, saves the trained agent to `mountaincar_blended_agent_50k.pkl`, and writes reward logs to `blended_rewards_50k.pkl`.
* **If a trained model exists:** The script loads the pickle file directly and invokes `.solve(render=True)` to render the trained car reaching the goal in human visual mode.
