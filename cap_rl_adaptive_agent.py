import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle
import os
import collections

class AdaptiveAgent:
    """
    An agent that adaptively mixes Monte Carlo (MC) and
    Temporal Difference (TD) updates to solve the credit assignment problem.

    Hypothesis:
    1. Early in training (epsilon is high), TD estimates are bad.
       The agent should rely on unbiased, full-episode MC returns.
    2. Late in training (epsilon is low), TD estimates are reliable.
       The agent should rely on fast, single-step TD updates.
    """

    def __init__(self, 
                 x_bin=20, 
                 vel_bin=20,
                 learning_rate=0.1, 
                 discount_factor=0.99,
                 epsilon=1.0, 
                 min_epsilon=0.01,
                 complete_reward=100,
                 adaptive_threshold=0.1): # When epsilon < this, stop MC
        
        self.x_bin_count = x_bin
        self.vel_bin_count = vel_bin
        
        self.pos_bins = np.linspace(-1.2, 0.6, self.x_bin_count - 1)
        self.vel_bins = np.linspace(-0.07, 0.07, self.vel_bin_count - 1)

        self.action_space_size = 3
        
        # We use TWO Q-Tables to compare them.
        # q_table_td learns *only* from TD updates
        self.q_table = np.zeros((self.x_bin_count, self.vel_bin_count, self.action_space_size))
        
        # MC-specific tables
        self.returns_sum = collections.defaultdict(float)
        self.returns_count = collections.defaultdict(float)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.complete_reward = complete_reward
        self.adaptive_threshold = adaptive_threshold
        
        self.reward_log = []

    def get_state(self, x, v):
        """Discretize the continuous state (x, v) into a state tuple (pos_idx, vel_idx)."""
        pos_idx = np.digitize(x, self.pos_bins)
        vel_idx = np.digitize(v, self.vel_bins)
        return (pos_idx, vel_idx)

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2]) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit

    def update_epsilon(self, decay_rate):
        """Decay the epsilon value."""
        self.epsilon = max(self.min_epsilon, self.epsilon * decay_rate)

    def update_q_table_td(self, state, action, reward, next_state):
        """
        The standard 1-step Q-Learning (TD) update.
        """
        current_q = self.q_table[state + (action,)]
        max_future_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table[state + (action,)] = new_q

    def train(self, episodes, decay_rate):
        """
        Run the agent in the environment.
        It learns using TD *every step*.
        It *also* learns using MC *at the end of the episode* if epsilon is high.
        """
        env = gym.make('MountainCar-v0')
        self.reward_log = []
        
        print("Training Adaptive (TD+MC) agent...")
        for ep in tqdm(range(episodes)):
            episode_history = [] # Store (state, action, reward)
            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            done = False
            total_reward = 0

            # --- 1. TD LEARNING PHASE (DURING EPISODE) ---
            while not done:
                action = self.choose_action(state)
                next_obs, reward, done, _, _ = env.step(action)
                next_state = self.get_state(next_obs[0], next_obs[1])
                
                # Reward shaping
                shaped_reward = reward
                if done and next_obs[0] >= 0.5:
                    shaped_reward = self.complete_reward
                
                total_reward += reward # Use original reward for logging
                if done and next_obs[0] >= 0.5:
                    total_reward += self.complete_reward # Add bonus to log
                
                # --- THIS IS THE TD UPDATE ---
                # This happens *every step*, regardless of our adaptive logic
                if done and next_obs[0] >= 0.5:
                    self.q_table[state + (action,)] = self.complete_reward
                else:
                    self.update_q_table_td(state, action, shaped_reward, next_state)

                episode_history.append((state, action, shaped_reward))
                state = next_state
            
            self.reward_log.append(total_reward)

            # --- 2. ADAPTIVE MC LEARNING PHASE (AFTER EPISODE) ---
            #
            # We check if we are "early in training" (epsilon > threshold)
            # If so, we "correct" our Q-table with an unbiased MC update.
            # If we are "late in training" (epsilon < threshold),
            # we skip this and trust the TD updates we already did.
            #
            if self.epsilon > self.adaptive_threshold:
                G = 0 # G is the "Return"
                # Loop backwards through the episode to assign MC credit
                for state, action, reward in reversed(episode_history):
                    G = reward + self.gamma * G
                    
                    self.returns_sum[(state, action)] += G
                    self.returns_count[(state, action)] += 1
                    
                    # --- THIS IS THE MC UPDATE ---
                    # We *overwrite* the Q-value with the (more reliable)
                    # MC average. We use a "blending" alpha for stability.
                    mc_value = self.returns_sum[(state, action)] / self.returns_count[(state, action)]
                    self.q_table[state + (action,)] = (1.0 - self.alpha) * self.q_table[state + (action,)] + self.alpha * mc_value

            # 3. Decay Epsilon
            self.update_epsilon(decay_rate)

        env.close()
        print("Training complete.\n")

    def solve(self, max_steps=1000, render=False, episodes=1):
        """
        Play MountainCar using the learned Q-table (optimal policy).
        """
        render_mode = 'human' if render else None
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.epsilon = 0 # No exploration

        for ep in range(episodes):
            (x, v), _ = env.reset()
            state = self.get_state(x, v)
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.choose_action(state) # Will be optimal
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

    def save(self, filename="mountaincar_adaptive_agent.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Adaptive Agent saved to {filename}")

    @staticmethod
    def load(filename="mountaincar_adaptive_agent.pkl"):
        with open(filename, "rb") as f:
            agent = pickle.load(f)
        print(f"✅ Adaptive Agent loaded from {filename}")
        return agent

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    # --- FILENAMES FOR THIS EXPERIMENT ---
    # Using 1000 episodes to match your uploaded files
    model_file = "mountaincar_adaptive_agent_50k.pkl"
    log_file = "adaptive_rewards_50k.pkl"
    
    TRAIN_EPISODES = 50000 
    
    # Calculate decay rate to match your other files
    decay_rate = (0.01 / 1.0)**(1 / TRAIN_EPISODES)

    if os.path.exists(model_file):
        agent = AdaptiveAgent.load(model_file)
    else:
        agent = AdaptiveAgent(
            x_bin=20, 
            vel_bin=20,
            learning_rate=0.1,
            # We will stop MC updates when epsilon is 10%
            adaptive_threshold=0.1 
        )
        agent.train(TRAIN_EPISODES, decay_rate)
        agent.save(model_file)

        # Save the reward log after training
        with open(log_file, "wb") as f:
            pickle.dump(agent.reward_log, f)
        print(f"✅ Reward log saved to {log_file}")

    # Visualize learned policy
    agent.solve(episodes=3, render=True)