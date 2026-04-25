import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
# from gymnasium.wrappers import RecordVideo  <-- REMOVED
import pickle
import os


class MountainCarDiscreteMDP:
    def __init__(self, x_bin=10, vel_bin=10, trials_per_state_action=20, gamma=0.99, complete_reward=100):
        """
        Initialize and build an MDP for the Mountain Car environment.
        """
        self.x_space = np.linspace(-1.2, 0.6, x_bin)
        self.vel_space = np.linspace(-0.07, 0.07, vel_bin)

        self.A = [0, 1, 2]  # actions: push left, no push, push right
        self.S = []  # MDP states

        prev_x = self.x_space[0]
        # Create a grid of states. (x_bin-1) * (vel_bin-1) states
        for x in self.x_space[1:]:
            prev_v = self.vel_space[0]
            for v in self.vel_space[1:]:
                self.S.append({
                    'x_range': (prev_x, x),
                    'vel_range': (prev_v, v),
                    'p': {a: {} for a in self.A},  # Transition probabilities: p(s' | s, a)
                    'v': 0  # Value function V(s)
                })
                prev_v = v
            prev_x = x

        self.trials_per_state_action = trials_per_state_action
        self.gamma = gamma
        self.complete_reward = complete_reward

        print("Building MDP model...")
        self._build_mdp(trials_per_state_action=trials_per_state_action)
        print("MDP construction complete.\n")

    def _build_mdp(self, trials_per_state_action=100):
        """
        Build an MDP model for the Mountain Car environment using Monte Carlo simulations.
        For each (state, action) pair, run simulations to estimate:
        1. Transition probabilities P(s' | s, a)
        2. Expected reward R(s, a, s')
        """
        env = gym.make('MountainCar-v0')
        for s in tqdm(self.S, desc="Building transitions"):
            for a in self.A:
                # Run multiple trials for each state-action pair
                for _ in range(trials_per_state_action):
                    env.reset()
                    # Initialize the environment to a random (x, v) within the state's bounds
                    x_init = np.random.uniform(low=s['x_range'][0], high=s['x_range'][1])
                    vel_init = np.random.uniform(low=s['vel_range'][0], high=s['vel_range'][1])
                    env.unwrapped.state = (x_init, vel_init)

                    # Take the action
                    (dest_x, dest_v), r, isDone, _, _ = env.step(a)
                    if isDone:
                        r = self.complete_reward  # Assign a large positive reward for reaching the goal

                    # Find the discretized destination state
                    _, dest_idx = self.get_state(dest_x, dest_v)

                    # Update transition counts and cumulative rewards
                    if s['p'][a].get(dest_idx) is None:
                        # (count, cumulative_reward)
                        s['p'][a][dest_idx] = (1, r)
                    else:
                        s['p'][a][dest_idx] = (s['p'][a][dest_idx][0] + 1, s['p'][a][dest_idx][1] + r)
        env.close()

    def get_state(self, x, v):
        """
        Map a continuous (x, v) pair to the corresponding discrete state index.
        """
        # Find the bin index for position x
        # np.searchsorted finds the index where x would be inserted to maintain order
        # We subtract 1 to get the index of the bin's lower bound
        a = max(0, np.searchsorted(self.x_space, min(x, self.x_space[-1]), side='left') - 1)
        # Find the bin index for velocity v
        b = max(0, np.searchsorted(self.vel_space, min(v, self.vel_space[-1]), side='left') - 1)
        
        # Calculate the flat index in the 1D list self.S
        idx = a * (self.vel_space.shape[0] - 1) + b
        return self.S[idx], idx

    def value_iteration(self, theta=1e-3, max_iter=1000):
        """
        Compute optimal value function V* using value iteration.
        """
        print("\nStarting Value Iteration...")
        # Q(s, a) table
        self.qa_table = np.zeros((len(self.S), len(self.A)))
        delta = -1  # Change in value function
        idx = 0     # Iteration counter

        while ((delta == -1 or delta > theta) and idx < max_iter):
            delta = 0
            # Iterate over all states
            for s_id, s in enumerate(self.S):
                # Iterate over all actions
                for a in self.A:
                    sums = 0
                    # Calculate the expected value for taking action 'a' in state 's'
                    for dest_idx, (freq, rewards) in s['p'][a].items():
                        # p = P(s' | s, a)
                        p = freq / self.trials_per_state_action
                        # r = E[R | s, a, s']
                        r = rewards / self.trials_per_state_action
                        # v = gamma * V(s')
                        v = self.gamma * self.S[dest_idx]['v']
                        sums += p * (r + v)
                    
                    # Update Q-value
                    self.qa_table[s_id, a] = sums

                # Find the best Q-value for the state (max_a Q(s, a))
                v_prime = np.max(self.qa_table[s_id])
                # Accumulate the change in the value function
                delta += np.abs(v_prime - s['v'])
                # Update the value function for the state
                s['v'] = v_prime
            
            print(f"Iter {idx}, Δ={delta:.5f}")
            idx += 1

        print("Value iteration completed.\n")

    def get_optimal_action(self, s_id):
        """
        Return the best action for state s_id based on the Q-table.
        """
        if not hasattr(self, 'qa_table'):
            raise AttributeError('Run value_iteration() first.')
        # pi(s) = argmax_a Q(s, a)
        return np.argmax(self.qa_table[s_id])

    def solve(self, max_steps=1000, render=False, episodes=1):
        """
        Play MountainCar using the optimal policy derived from value iteration.
        'render=True' will open a window to watch.
        """
        if not hasattr(self, 'qa_table'):
            raise AttributeError('Run value_iteration() first.')

        # --- UPDATED ---
        # Set render_mode to 'human' if render=True, otherwise None
        render_mode = 'human' if render else None
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        
        # --- REMOVED 'if record:' block ---

        for ep in range(episodes):
            (x, v), _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < max_steps:
                # Get current discrete state
                _, s_id = self.get_state(x, v)
                # Get optimal action from policy
                a = self.get_optimal_action(s_id)
                # Take action
                (x, v), r, done, _, _ = env.step(a)
                
                if done:
                    r = self.complete_reward # Add completion reward
                
                total_reward += r
                steps += 1
                
                # --- UPDATED ---
                if render:
                    sleep(0.01)  # Slow down human rendering

            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward:.2f}")
        env.close()

    def save(self, filename="mountaincar_mdp.pkl"):
        """
        Save the entire MDP object (including Q-table) using pickle.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ MDP saved to {filename}")

    @staticmethod
    def load(filename="mountaincar_mdp.pkl"):
        """
        Load a saved MDP object from a pickle file.
        """
        with open(filename, "rb") as f:
            mdp = pickle.load(f)
        print(f"✅ MDP loaded from {filename}")
        return mdp


# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    model_file = "mountaincar_mdp.pkl"

    if os.path.exists(model_file):
        # Load pre-trained model
        mdp = MountainCarDiscreteMDP.load(model_file)
    else:
        # Train a new model
        print("No model file found. Training a new, more detailed model...")
        mdp = MountainCarDiscreteMDP(
            x_bin=20, 
            vel_bin=20, 
            trials_per_state_action=50
        )
        mdp.value_iteration(max_iter=1000)
        mdp.save(model_file)

    # Visualize learned policy
    # Set render=True to watch the solution
    mdp.solve(episodes=3, render=True)