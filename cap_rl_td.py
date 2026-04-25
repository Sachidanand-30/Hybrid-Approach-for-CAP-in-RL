import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle
import os
# from gymnasium.wrappers import RecordVideo  <-- REMOVED

class QLearningAgent:
    """
    A class to implement the Q-Learning (Temporal Difference) algorithm
    for the Mountain Car environment.
    """

    def __init__(self, 
                 x_bin=20, 
                 vel_bin=20,
                 learning_rate=0.1, 
                 discount_factor=0.99,
                 epsilon=1.0, 
                 epsilon_decay=0.9999, 
                 min_epsilon=0.01,
                 complete_reward=100):
        """
        Initialize the agent and its Q-table.
        """
        
        self.x_bin_count = x_bin
        self.vel_bin_count = vel_bin
        
        # --- Discretization ---
        self.pos_bins = np.linspace(-1.2, 0.6, self.x_bin_count - 1)
        self.vel_bins = np.linspace(-0.07, 0.07, self.vel_bin_count - 1)

        # --- Q-Table Initialization ---
        self.action_space_size = 3 # (0: left, 1: none, 2: right)
        self.q_table = np.zeros((self.x_bin_count, self.vel_bin_count, self.action_space_size))
        
        # --- Hyperparameters ---
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # --- Reward Shaping ---
        self.complete_reward = complete_reward

    def get_state(self, x, v):
        """
        Discretize the continuous state (x, v) into a state tuple (pos_idx, vel_idx).
        """
        pos_idx = np.digitize(x, self.pos_bins)
        vel_idx = np.digitize(v, self.vel_bins)
        return (pos_idx, vel_idx)

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return np.random.choice([0, 1, 2])
        else:
            # Exploit: choose the best action from Q-table
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-Learning (Temporal Difference) update rule.
        """
        
        # 1. Get the current Q-value for (s, a)
        current_q = self.q_table[state + (action,)]
        
        # 2. Get the *maximum* Q-value for the *next* state (s')
        max_future_q = np.max(self.q_table[next_state])
        
        # 3. Calculate the "Temporal Difference (TD) Target"
        td_target = reward + self.gamma * max_future_q
        
        # 4. Calculate the "TD Error"
        td_error = td_target - current_q
        
        # 5. The Q-Learning Update Rule
        new_q = current_q + self.alpha * td_error
        self.q_table[state + (action,)] = new_q

    def update_epsilon(self):
        """
        Decay the epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes):
        """
        Run the agent in the environment to learn the Q-table.
        """
        env = gym.make('MountainCar-v0')
        
        print("Training Q-Learning agent...")
        for ep in tqdm(range(episodes)):
            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            
            done = False
            total_reward = 0

            while not done:
                # 1. Choose action (epsilon-greedy)
                action = self.choose_action(state)
                
                # 2. Take action and get (s', r, done)
                next_obs, reward, done, _, _ = env.step(action)
                next_state = self.get_state(next_obs[0], next_obs[1])
                
                # 3. Update Q-Table (The "learning" step)
                if done and next_obs[0] >= 0.5:
                    self.q_table[state + (action,)] = self.complete_reward
                    total_reward += self.complete_reward
                else:
                    self.update_q_table(state, action, reward, next_state)
                    total_reward += reward

                # 4. Move to the next state
                state = next_state
            
            # 5. After the episode, decay epsilon
            self.update_epsilon()

        env.close()
        print("Training complete.\n")

    def solve(self, max_steps=1000, render=False, episodes=1):
        """
        Play MountainCar using the learned Q-table (optimal policy).
        'render=True' will open a window to watch.
        """
        
        # --- UPDATED ---
        # Set render_mode to 'human' if render=True, otherwise None
        render_mode = 'human' if render else None
        env = gym.make('MountainCar-v0', render_mode=render_mode)

        # --- REMOVED 'if record:' block ---

        # We are *solving*, not *training*. Set epsilon to 0 for pure exploitation.
        self.epsilon = 0  

        for ep in range(episodes):
            (x, v), _ = env.reset()
            state = self.get_state(x, v)
            
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                # Get optimal action (since epsilon=0)
                action = self.choose_action(state)
                
                (x, v), r, done, _, _ = env.step(action)
                state = self.get_state(x, v)

                if done and x >= 0.5:
                    r = self.complete_reward
                
                total_reward += r
                steps += 1
                
                # --- UPDATED ---
                # Only sleep if we are rendering in 'human' mode
                if render:
                    sleep(0.01)

            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")
        env.close()

    def save(self, filename="mountaincar_q_agent.pkl"):
        """Save the entire agent object using pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Q-Learning Agent saved to {filename}")

    @staticmethod
    def load(filename="mountaincar_q_agent.pkl"):
        """Load a saved agent object from a pickle file."""
        with open(filename, "rb") as f:
            agent = pickle.load(f)
        print(f"✅ Q-Learning Agent loaded from {filename}")
        return agent

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    model_file = "mountaincar_q_agent.pkl"
    
    TRAIN_EPISODES = 10000 

    if os.path.exists(model_file):
        agent = QLearningAgent.load(model_file)
    else:
        agent = QLearningAgent(
            x_bin=20, 
            vel_bin=20,
            learning_rate=0.1,
            epsilon_decay=0.99995 
        )
        agent.train(TRAIN_EPISODES)
        agent.save(model_file)

    # Visualize learned policy
    # Set render=True to watch the solution
    agent.solve(episodes=3, render=True)
