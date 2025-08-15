import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

folder_path = r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\plots'  # replace with the path to your folder

# Dictionary to store data with filename (without .pkl) as keys
data_dict = {}

# Load all pickle files
for filename in os.listdir(folder_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            var_name = os.path.splitext(filename)[0]  # remove '.pkl'
            data_dict[var_name] = data

# Extract score_history from all files
score_histories = {}
for name, data in data_dict.items():
    if 'score_history' in data:
        score_histories[name] = data['score_history']
    else:
        print(f"Warning: 'score_history' not found in {name}.pkl")

# Find the maximum length among all score histories
max_length = max(len(scores) for scores in score_histories.values()) if score_histories else 0

if max_length == 0:
    print("No valid score histories found!")
    exit()

# Pad all arrays to have the same length
padded_scores = {}
for name, scores in score_histories.items():
    if len(scores) < max_length:
        padded_scores[name] = np.pad(scores, (0, max_length - len(scores)), 'constant')
    else:
        padded_scores[name] = np.array(scores)

# Create x-axis (episode numbers)
x = [i + 1 for i in range(max_length)]

# Calculate running averages and plot
plt.figure(figsize=(12, 8))

for name, scores in padded_scores.items():
    running_avg = np.zeros(max_length)
    for i in range(max_length):
        # Calculate running average over last 100 episodes (or all available if less than 100)
        running_avg[i] = np.mean(scores[max(0, i-99):i+1])

    plt.plot(x, running_avg, label=f'Running Average {name}')

plt.xlabel('Episode')
plt.ylabel('Running Average Reward')
plt.title('Running Averages of All Score Histories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary information
print(f"\nSummary:")
print(f"Total files processed: {len(score_histories)}")
print(f"Maximum episode count: {max_length}")
for name, scores in score_histories.items():
    print(f"{name}: {len(scores)} episodes")