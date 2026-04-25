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
    This helps to visualize the learning trend.
    """
    if not rewards or len(rewards) < window_size:
        print(f"Warning: Not enough data in {filename} to smooth with window {window_size}.")
        return np.array([]) # Return empty array
    
    # The 'valid' mode means it will only output once it has a full window
    # of data.
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(all_rewards, window_size=100, train_episodes=50000):
    """
    Generates and saves a plot comparing the learning curves of
    all experimental agents.
    """
    plt.figure(figsize=(14, 8))
    
    # Define colors and styles for the three agents
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] # Blue (TD), Orange (MC), Red (Blended)
    styles = ['-', '-', '-.'] # Solid, Solid, Dash-Dot

    for i, (label, rewards) in enumerate(all_rewards.items()):
        if rewards:
            smoothed_rewards = smooth_rewards(rewards, window_size, filename=label) # Pass label for warning
            
            # Create an x-axis that starts *after* the first window
            x_axis = np.arange(window_size, len(rewards) + 1)
            
            # Ensure smoothed data and x-axis match length
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
    
    # Clamp the Y-axis to zoom in on the important -250 to +100 range
    plt.ylim(-250, 120) 
    
    # Ensure x-axis goes up to the total number of episodes
    plt.xlim(0, train_episodes)

    # Save the plot
    output_filename = f"learning_curves_TD_MC_Blended_{train_episodes}k.png"
    plt.savefig(output_filename)
    print(f"\n✅ Final comparison plot saved to {output_filename}")
    
    # Show the plot
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    # --- FILENAMES TO LOAD ---
    # We will use the 50k episode logs, as per your last successful plot
    log_files = {
        "TD(0) (Q-Learning)": "q_learning_rewards_50k.pkl",
        "Monte Carlo": "mc_rewards_50k.pkl",
        # "Adaptive (TD->TD)": "adaptive_rewards_50k.pkl", # <-- REMOVED as requested
        "Blended (TD+MC)": "blended_rewards_50k.pkl"
    }
    
    # --- PLOTTING PARAMETERS ---
    SMOOTHING_WINDOW = 250 # 250 is a good window for 50k episodes
    TOTAL_EPISODES = 50000

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
            # We will try to load the 10k or 1k logs as a fallback
            fallback_10k = filename.replace("50k", "10k")
            fallback_1k = filename.replace("50k", "1k")
            if os.path.exists(fallback_10k):
                print(f"Warning: Could not find {filename}, but found {fallback_10k}.")
                print(f"         Please update 'log_files' in the script if you want to use this.")
            elif os.path.exists(fallback_1k):
                print(f"Warning: Could not find {filename}, but found {fallback_1k}.")
                print(f"         Please update 'log_files' in the script if you want to use this.")


    if all_logs_found and len(all_reward_data) == 3:
        print("\nAll logs found. Generating plot...")
        plot_learning_curves(all_reward_data, SMOOTHING_WINDOW, TOTAL_EPISODES)
    else:
        print("\nCould not generate plot. Please check for missing log files.")
        print("Expected files:")
        for f in log_files.values():
            if not os.path.exists(f):
                print(f"- {f} (MISSING)")