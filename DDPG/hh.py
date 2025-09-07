import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter # Import the formatter

folder_path = r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\plots'  # replace with the path to your folder

# Dictionary to store data with filename (without .pkl) as keys
data_dict = {}

# Load all pickle files
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                var_name = os.path.splitext(filename)[0]  # remove '.pkl'
                data_dict[var_name] = data
        except Exception as e:
            print(f"Could not load or read {filename}: {e}")

# Extract score_history and step_history from all files
score_histories = {}
step_histories = {}

for name, data in data_dict.items():
    # Ensure both score and step history exist to make a valid plot
    if 'score_history' in data and 'step_history' in data:
        # Also ensure they have the same number of entries
        if len(data['score_history']) == len(data['step_history']):
            score_histories[name] = np.array(data['success_history'])
            step_histories[name] = np.array(data['step_history'])
        else:
            print(f"Warning: Mismatch in length between score_history and step_history in {name}.pkl. Skipping this file.")
    else:
        print(f"Warning: 'score_history' or 'step_history' not found in {name}.pkl. Skipping this file.")

# Check if we have any valid data to plot
if not score_histories:
    print("No valid data found to plot!")
    exit()

# --- Plotting Section ---
plt.figure(figsize=(12, 8))

# Define the window for the running average
window_size = 100

# Iterate through each file's data to calculate the running average and plot it
for name in score_histories.keys():
    scores = score_histories[name]
    cumulative_steps = step_histories[name]

    num_episodes = len(scores)
    running_avg = np.zeros(num_episodes)

    # Calculate the running average over the specified window
    for i in range(num_episodes):
        start_index = max(0, i - window_size + 1)
        running_avg[i] = np.mean(scores[start_index : i+1])

    # Plot the running average vs cumulative timesteps
    plt.plot(cumulative_steps, running_avg, label=f'({name})')

plt.xlabel('Cumulative Timesteps')
plt.ylabel(f'Running Average Score (Window={window_size})')
plt.title('Running Average Score vs. Cumulative Timesteps')
plt.legend()
plt.grid(True)

# --- NEW CODE TO FORMAT THE X-AXIS ---
# Get the current axes
ax = plt.gca()

# Option 1: Simple method to disable scientific notation
ax.ticklabel_format(style='plain', axis='x')

# Option 2 (Bonus): Format with comma separators for even better readability (e.g., 1,000,000)
# Uncomment the line below to use this instead of Option 1.
# ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))


plt.tight_layout()
plt.show()

# --- Summary Information Section ---
print(f"\nSummary:")
print(f"Total valid files processed for plotting: {len(score_histories)}")
for name in score_histories.keys():
    total_episodes = len(score_histories[name])

    if total_episodes > 0:
        total_steps = step_histories[name][-1]
        final_raw_score = score_histories[name][-1]
        print(f"- {name}: {total_episodes} episodes, {total_steps} total timesteps. (Final raw score: {final_raw_score:.2f})")
    else:
        print(f"- {name}: 0 episodes, 0 total timesteps.")