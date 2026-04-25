import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle
import os

class SarsaLambdaAgent:
    """
    Implements Sarsa(λ) using Eligibility Traces.
    This combines the step-by-step learning of TD with the
    deep credit assignment of Monte Carlo.
    """

    def __init__(self, 
                 x_bin=20, 
                 vel_bin=20,
                 learning_rate=0.1, 
                 discount_factor=0.99,
                 lambda_trace=0.9, # <-- NEW PARAMETER (λ)
                 epsilon=1.0, 
                 min_epsilon=0.01,
                 complete_reward=100):
        
        self.x_bin_count = x_bin
        self.vel_bin_count = vel_bin
        
        self.pos_bins = np.linspace(-1.2, 0.6, self.x_bin_count - 1)
        self.vel_bins = np.linspace(-0.07, 0.07, self.vel_bin_count - 1)

        self.action_space_size = 3
        self.q_table = np.zeros((self.x_bin_count, self.vel_bin_count, self.action_space_size))
        
        # --- ELIGIBILITY TRACE TABLE ---
        # This is the "memory" of which (s, a) pairs
        # are eligible for credit.
        self.e_table = np.zeros((self.x_bin_count, self.vel_bin_count, self.action_space_size))

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.lambda_ = lambda_trace # (λ)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.complete_reward = complete_reward
        
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

    def update_epsilon(self, episode, total_episodes):
        """Linearly decay epsilon from 1.0 to min_epsilon."""
        self.epsilon = 1.0 - (1.0 - self.min_epsilon) * (episode / total_episodes)
        
    def train(self, episodes):
        """
        Run the agent in the environment to learn the Q-table using Sarsa(λ).
        """
        env = gym.make('MountainCar-v0')
        self.reward_log = []
        
        print(f"Training Sarsa(λ) agent with λ={self.lambda_}...")
        for ep in tqdm(range(episodes)):
            
            # --- 1. RESET ELIGIBILITY TRACE ---
            # At the start of each episode, clear the "memory".
            self.e_table.fill(0.0)

            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            action = self.choose_action(state) # S-A-r-s-a
            
            done = False
            total_reward = 0

            while not done:
                # --- 2. TAKE ACTION ---
                next_obs, reward, done, _, _ = env.step(action)
                next_state = self.get_state(next_obs[0], next_obs[1])
                
                # Reward shaping
                if done and next_obs[0] >= 0.5:
                    reward = self.complete_reward
                total_reward += reward

                # --- 3. CHOOSE NEXT ACTION ---
                # This is the "S-A-R-S-A" part. We choose the *next* action
                # *before* we update the Q-table.
                next_action = self.choose_action(next_state)
                
                # --- 4. CALCULATE TD ERROR (δ) ---
                if done:
                    td_target = reward # The value of the terminal state is 0
                else:
                    td_target = reward + self.gamma * self.q_table[next_state + (next_action,)]
                
                td_error = td_target - self.q_table[state + (action,)]

                # --- 5. UPDATE TRACE ---
                # Mark the current (s, a) pair as "visited" (eligible for credit)
                # We use "replacing traces" for stability.
                self.e_table[state + (action,)] = 1.0 

                # --- 6. UPDATE ALL Q-VALUES AND TRACES ---
                # This is the "magic" step.
                # All eligible states get credit, decayed by lambda.
                
                # Q(s, a) <- Q(s, a) + α * δ * e(s, a)   (for all s, a)
                self.q_table += self.alpha * td_error * self.e_table
                
                # e(s, a) <- γ * λ * e(s, a)   (for all s, a)
                self.e_table *= self.gamma * self.lambda_

                # --- 7. MOVE TO NEXT STATE ---
                state = next_state
                action = next_action
            
            self.reward_log.append(total_reward)
            self.update_epsilon(ep, episodes)

        env.close()
        print("Training complete.\n")

    def solve(self, max_steps=1000, render=False, episodes=1):
        """Identical to the other agents: run the learned policy."""
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
                action = self.choose_action(state) # Optimal
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

    def save(self, filename="mountaincar_sarsa_lambda_agent.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Sarsa(λ) Agent saved to {filename}")

    @staticmethod
    def load(filename="mountaincar_sarsa_lambda_agent.pkl"):
        with open(filename, "rb") as f:
            agent = pickle.load(f)
        print(f"✅ Sarsa(λ) Agent loaded from {filename}")
        return agent

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    # --- FILENAMES FOR THIS EXPERIMENT ---
    # Using 50k to match your previous long run
    model_file = "sarsa_lambda_agent_1k.pkl"
    log_file = "sarsa_lambda_rewards_1k.pkl"
    
    TRAIN_EPISODES = 1000 

    if os.path.exists(model_file):
        agent = SarsaLambdaAgent.load(model_file)
    else:
        agent = SarsaLambdaAgent(
            x_bin=20, 
            vel_bin=20,
            learning_rate=0.1,  # Sarsa(λ) can often use a slightly lower alpha
            lambda_trace=0.9,   # The "sweet spot"
            min_epsilon=0.01
        )
        agent.train(TRAIN_EPISODES)
        agent.save(model_file)

        # Save the reward log after training
        with open(log_file, "wb") as f:
            pickle.dump(agent.reward_log, f)
        print(f"✅ Reward log saved to {log_file}")

    # Visualize learned policy
    agent.solve(episodes=3, render=True)
