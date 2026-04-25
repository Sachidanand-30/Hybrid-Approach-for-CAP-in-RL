import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle
import os
import collections

class MonteCarloAgent:
    """
    A class to implement Monte Carlo Control (model-free)
    for the Mountain Car environment.
    """

    def __init__(self, 
                 x_bin=20, 
                 vel_bin=20,
                 discount_factor=0.99,
                 epsilon=1.0, 
                 epsilon_decay=0.9999, 
                 min_epsilon=0.01,
                 complete_reward=100):
        
        self.x_bin_count = x_bin
        self.vel_bin_count = vel_bin
        
        # --- Discretization ---
        self.pos_bins = np.linspace(-1.2, 0.6, self.x_bin_count - 1)
        self.vel_bins = np.linspace(-0.07, 0.07, self.vel_bin_count - 1)

        # --- Q-Table and Returns Initialization ---
        self.action_space_size = 3
        # Optimistic initialization (0.0) helps exploration
        self.q_table = np.zeros((self.x_bin_count, self.vel_bin_count, self.action_space_size))
        
        # Returns storage
        self.returns_sum = collections.defaultdict(float)
        self.returns_count = collections.defaultdict(float)

        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.complete_reward = complete_reward
        
        self.reward_log = []

    def get_state(self, x, v):
        """Discretize the continuous state."""
        pos_idx = np.digitize(x, self.pos_bins)
        vel_idx = np.digitize(v, self.vel_bins)
        return (pos_idx, vel_idx)

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2]) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit

    def update_epsilon(self):
        """Decay the epsilon value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes):
        """Run the agent in the environment to learn."""
        env = gym.make('MountainCar-v0')
        self.reward_log = []
        
        print(f"Training Monte Carlo agent for {episodes} episodes...")
        for ep in tqdm(range(episodes)):
            episode_history = []
            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_obs, reward, done, _, _ = env.step(action)
                next_state = self.get_state(next_obs[0], next_obs[1])
                
                if done and next_obs[0] >= 0.5:
                    reward = self.complete_reward
                
                total_reward += reward
                episode_history.append((state, action, reward))
                state = next_state
            
            self.reward_log.append(total_reward)

            # --- MC Update ---
            G = 0
            for state, action, reward in reversed(episode_history):
                G = reward + self.gamma * G
                self.returns_sum[(state, action)] += G
                self.returns_count[(state, action)] += 1
                self.q_table[state + (action,)] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

            self.update_epsilon()

        env.close()
        print("Training complete.\n")

    def solve(self, max_steps=1000, render=False, episodes=1):
        """Play MountainCar using the learned policy."""
        render_mode = 'human' if render else None
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.epsilon = 0 

        for ep in range(episodes):
            (x, v), _ = env.reset()
            state = self.get_state(x, v)
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.choose_action(state)
                (x, v), r, done, _, _ = env.step(action)
                state = self.get_state(x, v)
                if done and x >= 0.5:
                    r = self.complete_reward
                total_reward += r
                steps += 1
                if render:
                    sleep(0.01)

            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")
        env.close()
    
    def save(self, filename="mountaincar_mc_agent.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Monte Carlo Agent saved to {filename}")

    @staticmethod
    def load(filename="mountaincar_mc_agent.pkl"):
        with open(filename, "rb") as f:
            agent = pickle.load(f)
        print(f"✅ Monte Carlo Agent loaded from {filename}")
        return agent

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Ensure this matches the TD and Blended scripts for a fair comparison
    TRAIN_EPISODES = 50000 
    
    # FILENAMES
    model_file = "mountaincar_mc_agent_50k.pkl"
    log_file = "mc_rewards_50k.pkl"

    # --- IMPORTANT: Decay Calculation ---
    # Calculates decay so epsilon reaches 0.01 exactly at the LAST episode
    decay_rate = (0.01 / 1.0)**(1 / TRAIN_EPISODES) 

    if os.path.exists(model_file):
        agent = MonteCarloAgent.load(model_file)
    else:
        agent = MonteCarloAgent(
            x_bin=20, 
            vel_bin=20,
            epsilon_decay=decay_rate 
        )
        agent.train(TRAIN_EPISODES)
        agent.save(model_file)

        with open(log_file, "wb") as f:
            pickle.dump(agent.reward_log, f)
        print(f"✅ Reward log saved to {log_file}")

    agent.solve(episodes=5, render=True)
