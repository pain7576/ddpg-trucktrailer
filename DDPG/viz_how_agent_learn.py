import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def read_pickle(file_path: str):
    """Reads a pickle file and returns the loaded object."""
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def compute_metrics(data):
    """Compute efficiency, smoothness, precision, safety from reward_info."""
    reward_info = data['info']

    progress_reward = sum([step['progress_reward'] for step in reward_info])
    staged_success = sum([step['staged_success'] for step in reward_info])
    exploration_bonus = sum([step['exploration_bonus'] for step in reward_info])
    final_success_bonus = sum([step['final_success_bonus'] for step in reward_info])
    backward_penalty = sum([step['backward_penalty'] for step in reward_info])
    smoothness_penalty = sum([step['smoothness_penalty'] for step in reward_info])
    heading_reward = sum([step['heading_reward'] for step in reward_info])
    orientation_reward = sum([step['orientation_reward'] for step in reward_info])
    safety_penalty = sum([step['safety_penalty'] for step in reward_info])

    efficiency = progress_reward + staged_success + exploration_bonus + final_success_bonus + backward_penalty
    smoothness = smoothness_penalty
    precision = heading_reward + orientation_reward
    safety = safety_penalty

    return efficiency, smoothness, precision, safety


def plot_running_average(folder_path = 'episode_replays'):
    # Buffers for last 100 values
    eff_hist, smooth_hist, prec_hist, safe_hist = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

    # Store averages for plotting
    eff_avg, smooth_avg, prec_avg, safe_avg = [], [], [], []

    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])

    plt.ion()  # interactive mode on

    # --- CHANGE: Create two subplots, one for positive and one for negative metrics ---
    # We create a figure with 2 rows and 1 column of subplots.
    # `sharex=True` links the x-axis of both plots, so zooming/panning is synchronized.
    # `figsize` is adjusted to better accommodate two plots.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Running Averages of Metrics", fontsize=16)

    for i, filename in enumerate(file_list, start=1):
        file_path = os.path.join(folder_path, filename)
        data = read_pickle(file_path)
        eff, smooth, prec, safe = compute_metrics(data)

        # Update buffers
        eff_hist.append(eff)
        smooth_hist.append(smooth)
        prec_hist.append(prec)
        safe_hist.append(safe)

        # Running averages
        eff_avg.append(np.mean(eff_hist))
        smooth_avg.append(np.mean(smooth_hist))
        prec_avg.append(np.mean(prec_hist))
        safe_avg.append(np.mean(safe_hist))

        # --- CHANGE: Clear and redraw on separate subplots ---
        ax1.clear()
        ax2.clear()

        # Plot 1: Positive Metrics
        ax1.plot(eff_avg, label="Efficiency", color="blue")
        ax1.plot(prec_avg, label="Precision", color="green")
        ax1.set_ylabel("Running Average (last 100)")
        ax1.set_title("Positive Metrics")
        ax1.legend()
        ax1.grid(True) # Add grid for better readability

        # Plot 2: Negative Metrics
        ax2.plot(smooth_avg, label="Smoothness", color="orange")
        ax2.plot(safe_avg, label="Safety", color="red")
        ax2.set_ylabel("Running Average (last 100)")
        ax2.set_title("Negative Metrics")
        ax2.legend()
        ax2.grid(True) # Add grid for better readability

        # Set the shared x-axis label only on the bottom plot
        ax2.set_xlabel("Number of files read")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.pause(0.1)  # update graph in real-time

    plt.ioff()
    plt.show()


# Run on your folder

plot_running_average("episode_replays")