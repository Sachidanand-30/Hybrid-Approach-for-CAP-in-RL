import gymnasium as gym
import numpy as np
import pickle
import os
from time import sleep

# --- IMPORTANT ---
# This code requires Stable-Baselines3 and PyTorch
# Run this in your terminal first:
# pip install stable-baselines3[torch] torch gymnasium
# ---
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Error: stable-baselines3 or torch not found.")
    print("Please run: pip install stable-baselines3[torch] torch gymnasium")
    exit()

# --- NEW: REWARD SHAPING WRAPPER ---
class RewardShapingWrapper(gym.Wrapper):
    """
    A wrapper to "shape" the reward for MountainCarContinuous.
    This gives the agent a "dense" reward to help it learn.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_position = -np.inf

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 1. The original "goal" reward
        if done and obs[0] >= 0.45: # Goal position
            reward = 100.0
        
        # 2. The original "action penalty"
        reward -= (action[0]**2) * 0.1

        # 3. --- The "Dense" Reward (SCALED DOWN) ---
        # We reward the agent for its "potential energy" (height)
        # This is 100x smaller than before. It's now just a "nudge."
        reward += (obs[0] + 0.5) # obs[0] is position
        
        # 4. Bonus for new max position (SCALED DOWN)
        # This is 20x smaller than before.
        if obs[0] > self.max_position:
            reward += 1.0  # Bonus for setting a new high score
            self.max_position = obs[0]

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.max_position = -np.inf
        # Call reset on the unwrapped environment
        # This was buggy before. This is the correct way.
        obs, info = self.env.reset(**kwargs)
        return obs, info

class RewardLoggingCallback(BaseCallback):
    """
    A custom callback to log the total reward of each episode.
    This is the standard way to log data in Stable-Baselines3.
    """
    def __init__(self, check_freq: int, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # 'self.training_env' is the environment
        # 'self.n_calls' is the total number of steps
        
        # Add the reward from this step
        # In SB3, 'self.locals["rewards"]' is an array (one for each env)
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if the episode is done
        # 'self.locals["dones"]' is also an array
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        
        return True # Continue training

def train_sac_agent(train_timesteps=50000):
    """
    Train a Soft Actor-Critic (SAC) agent.
    """
    print("Initializing SAC agent...")
    
    # --- 1. KEY DIFFERENCE: The Continuous Environment ---
    # We must use 'MountainCarContinuous-v0' because SAC is
    # designed for continuous action spaces (e.g., force from -1.0 to 1.0).
    env = gym.make('MountainCarContinuous-v0')
    
    # --- APPLY THE WRAPPER ---
    env = RewardShapingWrapper(env) # <-- NEW
    
    # We wrap it in DummyVecEnv, which is standard for SB3.
    env = DummyVecEnv([lambda: env])

    # --- 2. Setup Logging ---
    # --- CHANGED FILENAME to be different ---
    log_file = "sac_rewards_shaped_v2_50k.pkl" 
    reward_callback = RewardLoggingCallback(check_freq=1)
    
    # --- 3. Instantiate the SAC Model ---
    # "MlpPolicy" means a standard Multi-Layer Perceptron (neural network).
    # 'learning_starts=1000' means it will take 1000 random steps
    # to fill its "replay buffer" before it starts learning.
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_starts=1000
        # tensorboard_log=None  (Removed to fix the error)
    )

    # --- 4. Train the Model ---
    # We pass the callback to the .learn() method
    print(f"Training SAC agent for {train_timesteps} timesteps...")
    model.learn(total_timesteps=train_timesteps, callback=reward_callback)
    
    # --- 5. Save the Model and Reward Log ---
    # --- CHANGED FILENAME to be different ---
    model_file = "sac_agent_shaped_v2_50k.pkl"
    model.save(model_file)
    print(f"✅ SAC Agent saved to {model_file}")

    with open(log_file, "wb") as f:
        pickle.dump(reward_callback.episode_rewards, f)
    print(f"✅ Reward log saved to {log_file}")
    
    env.close()
    return model_file, log_file

def solve(model_file="sac_agent_shaped_v2_50k.pkl", episodes=3): # <-- CHANGED DEFAULT
    """
    Watch the trained SAC agent perform.
    """
    print("Loading SAC model for visualization...")
    
    # We need to re-create the env, this time for rendering
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    # --- IMPORTANT: We DO NOT use the wrapper for solving ---
    # The wrapper is only for training. We want to see how the
    # agent performs on the *original* environment.
    
    model = SAC.load(model_file, env)
    
    print("Running trained SAC agent...")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # --- 1. Get Action ---
            # We ask the model (the "Actor") to predict the best action
            # for the current observation.
            # 'deterministic=True' means we're not exploring, just exploiting.
            action, _states = model.predict(obs, deterministic=True)
            
            # --- 2. Take Action ---
            obs, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            sleep(0.01)

        print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")
    env.close()

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    
    # We can use a shorter timestep now because the dense
    # reward makes learning much, much faster.
    TOTAL_TIMESTEPS = 50000 
    
    # --- CHANGED FILENAMES ---
    model_file = "sac_agent_shaped_v2_50k.pkl"
    log_file = "sac_rewards_shaped_v2_50k.pkl"

    if os.path.exists(model_file):
        print(f"Found pre-trained model at {model_file}")
    else:
        print("No model file found. Training a new SHAPED REWARD (v2) SAC agent...")
        model_file, log_file = train_sac_agent(TOTAL_TIMESTEPS)
    
    # Visualize learned policy
    solve(model_file, episodes=3)
    