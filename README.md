# Hybrid-Approach-for-CAP-in-RL
A Hybrid Approach of Combining Temporal Difference and Monte Carlo Based Learning for Better Credit Assignment in MountainCar-v0  Environment

This repository explores credit assignment techniques for the `MountainCar-v0` environment by comparing and combining Temporal Difference (TD) learning, Monte Carlo (MC) learning, eligibility traces, and deep reinforcement learning (SAC).

## Project Overview

The main goal is to study how different reinforcement learning strategies handle the Credit Assignment Problem (CAP) in MountainCar. The repository includes:

- Pure TD learning with Q-Learning and SARSA(λ)
- Monte Carlo-based models and MDP simulations
- Blended TD+MC and hybrid λ-MC approaches
- A reward-shaped Soft Actor-Critic (SAC) implementation for continuous control
- Training logs, saved models, and learning-curve plots

## Key Files

### Core RL agent implementations

- `cap_rl_td.py`
  - Q-Learning agent for `MountainCar-v0`.
  - Uses discretized position and velocity states.
  - Applies reward shaping when the car reaches the goal.

- `cap_rl_td_sarsa.py`
  - Sarsa(λ) agent using eligibility traces.
  - Combines stepwise TD learning with credit propagation through traces.

- `cap_rl_mc.py`
  - Monte Carlo / MDP-style model builder for MountainCar.
  - Estimates transition probabilities and reward expectations from simulated trials.
  - Includes value iteration to compute an optimal policy.

- `cap_rl_blended_agent.py`
  - Blended RL agent combining TD updates during episodes with MC updates at episode end.
  - Uses a tabular Q-table and returns averaging for Monte Carlo corrections.

- `cap_rl_adaptive_agent.py`
  - Adaptive agent implementation that likely blends TD and MC updates based on performance.
  - Useful for comparing adaptive strategies against pure and blended algorithms.

- `cap_rl_hybrid.py`
  - Hybrid-λ-MC agent combining TD(λ) eligibility traces with Monte Carlo corrections.
  - Applies a trace reset at the end of every episode.

- `cap_rl_sac.py`
  - Soft Actor-Critic (SAC) implementation for the continuous `MountainCarContinuous-v0` environment.
  - Includes reward shaping wrapper and Stable-Baselines3 callback logging.

### Experiment logging and analysis

- `cap_rl_mc_log.py`
  - Logging support for Monte Carlo experiments.
  - Stores reward histories or training performance data.

- `cap_rl_td_log.py`
  - Logging support for TD experiments.
  - Captures episode rewards and training progress.

- `cap_td_vs_mc_adaptive_blend.py`
  - Compares learning curves for TD, MC, and blended approaches.
  - Loads reward logs and plots performance over episodes.

- `cap_td_vs_mc_train_plot.py`
  - Generates training plots for the experimental results.
  - Saves comparison charts for different agent families.

### Models, logs, and results

The repository stores many generated model and reward files.

- `*.pkl` files
  - Saved model checkpoints and reward logs for the agents.
  - Examples: `mountaincar_blended_agent_50k.pkl`, `hybrid_lambda_mc.pkl`, `sac_agent_shaped_v2_50k.pkl`

- `*_rewards_*.pkl`
  - Episode reward histories used to plot and compare learning curves.
  - Examples: `blended_rewards_50k.pkl`, `mc_rewards_50k.pkl`, `q_learning_rewards_50k.pkl`

- `*.pki` files
  - Tabular Q-table snapshots and intermediate model data.
  - Examples: `qtable_0.pki`, `qtable_final.pki`, `mountaincar_adaptive_agent_50k.pkl`

- `*.png` files
  - Generated visualizations of training progress and comparisons.
  - Examples: `learning_curves_1000k.png`, `learning_curves_TD_MC_Blended_50000k.png`

### Documentation and misc files

- `architecture_cap_rl_blended_agent.md`
  - Design and architecture documentation for `cap_rl_blended_agent.py`.
  - Explains the blended TD+MC system, training flow, and algorithm design.

- `qtable_display.py`
  - Appears to be a binary Q-table snapshot file rather than a Python source module.
  - Likely contains serialized Q-table data for visual inspection or later loading.

- `.gitconfig`
  - Repository-local Git identity configuration.
  - Should generally stay out of version control; it is now ignored by `.gitignore`.

- `.gitignore`
  - Ignores local configuration files such as `.gitconfig`.

## Getting Started

### Dependencies

Install the main Python dependencies with:

```bash
pip install gymnasium numpy tqdm matplotlib pickle5
```

For SAC training, also install:

```bash
pip install stable-baselines3[torch] torch
```

### Running the scripts

- Train the Q-Learning agent:
  ```bash
  python cap_rl_td.py
  ```

- Train the blended TD+MC agent:
  ```bash
  python cap_rl_blended_agent.py
  ```

- Train the hybrid λ-MC agent:
  ```bash
  python cap_rl_hybrid.py
  ```

- Train SAC with reward shaping:
  ```bash
  python cap_rl_sac.py
  ```

- Plot comparison learning curves:
  ```bash
  python cap_td_vs_mc_adaptive_blend.py
  ```

## Notes

- The repository contains many saved results and logs. If you want a fresh run, remove or rename the `.pkl` and `.pki` files first.
- The `MountainCar` experiments use state discretization for tabular agents, while the SAC implementation uses a continuous environment.
- The blended/hybrid files are the core experiments for testing credit assignment strategies.
