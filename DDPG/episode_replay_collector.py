import pickle
import os
import numpy as np



class EpisodeReplaySystem:
    def __init__(self, env, save_dir='episode_replays'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.env = env

    def save_episode(self, episode_num, states, actions):
        """Save episode data to file"""
        episode_data = {
            'states': states,
            'actions': actions,
            'episode_num': episode_num

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

        print(f"\nReplaying Episode {episode_num}")
        print(f"Total steps: {len(states) - 1}")

        # Set initial state
        self.env.state = states[0]

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