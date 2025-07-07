import pickle
import numpy as np

# Load the transitions from pickle file
filename = "replay_buffer/transitions_episode_51_replay_buffer.pkl"  # Change to your actual filename

with open(filename, 'rb') as f:
    transitions_data = pickle.load(f)

print(f"Loaded transitions from {filename}")
print(f"Number of episodes: {len(transitions_data)}")

# Show info about each episode
for i, episode in enumerate(transitions_data):
    print(f"Episode {i}: {len(episode)} transitions")

# Access specific data
if len(transitions_data) > 0:
    first_episode = transitions_data[0]
    first_transition = first_episode[0]

    observation, action, reward, observation_, done = first_transition

    print("\nFirst transition of first episode:")
    print(f"Observation shape: {observation.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Reward: {reward}")
    print(f"Next observation shape: {observation_.shape}")
    print(f"Done: {done}")

    print(f"\nObservation values: {observation}")
    print(f"Action values: {action}")

# Count total transitions
total_transitions = sum(len(episode) for episode in transitions_data)
print(f"\nTotal transitions across all episodes: {total_transitions}")

# Example: Access transition 5 from episode 2 (if exists)
if len(transitions_data) > 2 and len(transitions_data[2]) > 5:
    obs, act, rew, obs_, done = transitions_data[2][5]
    print(f"\nEpisode 2, Transition 5:")
    print(f"Action: {act}, Reward: {rew}, Done: {done}")