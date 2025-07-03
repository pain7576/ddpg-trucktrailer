import pickle
import os
import numpy as np


class EpisodeReplaySystem:
    def __init__(self, env, save_dir='episode_replays'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.env = env

    def save_episode(self, episode_num, states, actions, env_data=None):
        """Save episode data to file"""
        episode_data = {
            'states': states,
            'actions': actions,
            'episode_num': episode_num,
            'env_data': env_data  # Add environment data
        }
        filepath = os.path.join(self.save_dir, f'episode_{episode_num}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)

    def replay_episode(self, episode_num):
        """Load and replay a specific episode"""
        filepath = os.path.join(self.save_dir, f'episode_{episode_num}.pkl')

        if not os.path.exists(filepath):
            print(f"Episode {episode_num} not found!")
            return

        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)

        states = episode_data['states']
        actions = episode_data['actions']
        env_data = episode_data.get('env_data', None)

        print(f"\nReplaying Episode {episode_num}")
        print(f"Total steps: {len(states) - 1}")

        # Set initial state
        self.env.state = states[0]

        # Restore environment data if available
        if env_data:
            self.env.startx = env_data['startx']
            self.env.starty = env_data['starty']
            self.env.startyaw = env_data['startyaw']
            self.env.goalx = env_data['goalx']
            self.env.goaly = env_data['goaly']
            self.env.goalyaw = env_data['goalyaw']
            self.env.path_x = env_data['path_x']
            self.env.path_y = env_data['path_y']
            self.env.path_yaw = env_data['path_yaw']
        else:
            # If no env_data available, call reset to generate new random scenario
            print("No environment data found, generating new random scenario...")
            self.env.reset()
            self.env.state = states[0]  # Override with saved initial state

        # Render initial state
        self.env.render()

        # Step through the episode
        for i in range(len(actions)):
            # Apply the recorded action
            observation, reward, done, info = self.env.step(actions[i])
            self.env.render()

            if done:
                print(f"Episode ended at step {i + 1}")
                break

        print("Replay complete!")