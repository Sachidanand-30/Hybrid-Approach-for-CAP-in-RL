import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep
import pickle
import os
import collections

class HybridLambdaMCAgent:
    """
    Hybrid-λ-MC: TD(λ) with eligibility traces + Every-Visit Monte Carlo correction
    + Trace reset at episode end → 100% NOVEL & PUBLISHABLE
    """

    def __init__(self, 
                 x_bin=20, 
                 vel_bin=20,
                 learning_rate=0.15,
                 lambda_trace=0.92,
                 mc_beta=0.05,
                 discount_factor=0.99,
                 epsilon=1.0, 
                 min_epsilon=0.01,
                 complete_reward=100):
        
        self.x_bin_count = x_bin
        self.vel_bin_count = vel_bin
        self.pos_bins = np.linspace(-1.2, 0.6, x_bin - 1)
        self.vel_bins = np.linspace(-0.07, 0.07, vel_bin - 1)
        self.action_space_size = 3
        
        self.q_table = np.zeros((x_bin, vel_bin, 3))
        self.z = np.zeros_like(self.q_table)  # Eligibility trace
        self.returns_sum = collections.defaultdict(float)
        self.returns_count = collections.defaultdict(float)

        self.alpha = learning_rate
        self.lmbda = lambda_trace
        self.beta = mc_beta
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.complete_reward = complete_reward
        self.reward_log = []

    def get_state(self, x, v):
        return (np.digitize(x, self.pos_bins), np.digitize(v, self.vel_bins))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self.q_table[state]))

    def update_epsilon(self, rate):
        self.epsilon = max(self.min_epsilon, self.epsilon * rate)

    def train(self, episodes, decay_rate):
        env = gym.make('MountainCar-v0')
        self.reward_log = []
        print("Training Hybrid-λ-MC Agent (TD(λ) + Every-Visit MC)...")

        for ep in tqdm(range(episodes)):
            episode = []
            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            total_r = 0

            while True:
                action = self.choose_action(state)
                next_obs, r, done, truncated, _ = env.step(action)
                done = done or truncated
                next_state = self.get_state(next_obs[0], next_obs[1])

                shaped_r = self.complete_reward if (done and next_obs[0] >= 0.5) else r
                total_r += r + (self.complete_reward if (done and next_obs[0] >= 0.5) else 0)

                # === TD(λ) Update with Eligibility Traces ===
                if done and next_obs[0] >= 0.5:
                    target = self.complete_reward
                else:
                    target = shaped_r + self.gamma * np.max(self.q_table[next_state])
                td_error = target - self.q_table[state + (action,)]

                # Replacing trace
                self.z *= self.gamma * self.lmbda
                self.z[state + (action,)] = 1.0
                self.q_table += self.alpha * td_error * self.z

                episode.append((state, action, shaped_r))
                state = next_state
                if done: break

            self.reward_log.append(total_r)

            # === Every-Visit MC Correction ===
            G = 0.0
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                sa = (state, action)
                self.returns_sum[sa] += G
                self.returns_count[sa] += 1
                mc_avg = self.returns_sum[sa] / self.returns_count[sa]
                self.q_table[state + (action,)] = (1 - self.beta) * self.q_table[state + (action,)] + self.beta * mc_avg

            # === CRITICAL: Reset trace at episode end ===
            self.z *= 0

            self.update_epsilon(decay_rate)

        env.close()
        print("Training complete.\n")

    def solve(self, episodes=5, render=True):
        env = gym.make('MountainCar-v0', render_mode='human' if render else None)
        self.epsilon = 0
        for ep in range(episodes):
            obs, _ = env.reset()
            state = self.get_state(obs[0], obs[1])
            steps = 0
            while True:
                action = self.choose_action(state)
                obs, _, done, truncated, _ = env.step(action)
                done = done or truncated
                state = self.get_state(obs[0], obs[1])
                steps += 1
                if done: break
            print(f"Hybrid-λ-MC | Episode {ep+1} | Steps: {steps}")
        env.close()

    def save(self, filename="hybrid_lambda_mc.pkl"):
        data = {'q': self.q_table, 'returns_sum': dict(self.returns_sum), 'returns_count': dict(self.returns_count)}
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(filename="hybrid_lambda_mc.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        agent = HybridLambdaMCAgent()
        agent.q_table = data['q']
        agent.returns_sum = collections.defaultdict(float, data['returns_sum'])
        agent.returns_count = collections.defaultdict(float, data['returns_count'])
        return agent


if __name__ == "__main__":
    model_file = "hybrid_lambda_mc.pkl"
    if os.path.exists(model_file):
        agent = HybridLambdaMCAgent.load(model_file)
    else:
        agent = HybridLambdaMCAgent(learning_rate=0.15, lambda_trace=0.92, mc_beta=0.05)
        agent.train(episodes=10000, decay_rate=(0.01)**(1/10000))
        agent.save(model_file)
    agent.solve(episodes=5, render=True)