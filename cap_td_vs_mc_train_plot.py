import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filename):
    """Loads a pickled reward log file."""
    if not os.path.exists(filename):
        print(f"Error: Log file not found at {filename}")
        print(f"Please make sure you have run the training script for {filename}")
        return None
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def smooth_rewards(rewards, window_size=100, filename=""):
    """
    Calculates the moving average of a list of rewards.
    """
    if not rewards or len(rewards) < window_size:
        print(f"Warning: Not enough data in {filename} to smooth with window {window_size}.")
        return np.array([])
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(all_rewards, window_size=100, train_episodes=10000):
    """
    Generates and saves a plot comparing the learning curves.
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', "#1ab329"] # Blue (TD), Orange (MC), Red (Blended)
    styles = ['-', '-', '-.'] # Solid, Solid, Dash-Dot

    for i, (label, rewards) in enumerate(all_rewards.items()):
        if rewards:
            smoothed_rewards = smooth_rewards(rewards, window_size, filename=label)
            x_axis = np.arange(window_size, len(rewards) + 1)
            
            plot_len = min(len(x_axis), len(smoothed_rewards))
            if plot_len > 0:
                plt.plot(x_axis[:plot_len], smoothed_rewards[:plot_len], 
                         label=f"{label} - Avg. over {window_size} ep.", 
                         alpha=0.9, linewidth=2, linestyle=styles[i], color=colors[i])

    plt.title(f'Algorithm Comparison: Credit Assignment (First {train_episodes} Episodes)', fontsize=18)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Total Reward per Episode', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Clamp Y-axis to focus on relevant reward range
    plt.ylim(-250, 120) 
    plt.xlim(0, train_episodes)

    output_filename = f"learning_curves_TD_MC_Blended_{train_episodes}k.png"
    plt.savefig(output_filename)
    print(f"\n✅ Final comparison plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    
    # --- FILENAMES TO LOAD (10k versions) ---
    log_files = {
        "TD(0) (Q-Learning)": "q_learning_rewards_10k.pkl",
        "Monte Carlo": "mc_rewards_10k.pkl",
        "Blended (TD+MC)": "blended_rewards_10k.pkl"
    }
    
    # --- PLOTTING PARAMETERS ---
    SMOOTHING_WINDOW = 100 # Smaller window for shorter run
    TOTAL_EPISODES = 10000

    print("Loading reward logs...")
    
    all_reward_data = {}
    all_logs_found = True
    
    for label, filename in log_files.items():
        rewards = load_data(filename)
        if rewards is not None:
            all_reward_data[label] = rewards
            print(f"Loaded {len(rewards)} episodes for {label}.")
        else:
            all_logs_found = False

    if all_logs_found and len(all_reward_data) == 3:
        print("\nAll logs found. Generating plot...")
        plot_learning_curves(all_reward_data, SMOOTHING_WINDOW, TOTAL_EPISODES)
    else:
        print("\nCould not generate plot. Please check for missing log files.")
        print("Expected files:")
        for f in log_files.values():
            if not os.path.exists(f):
                print(f"- {f} (MISSING)")