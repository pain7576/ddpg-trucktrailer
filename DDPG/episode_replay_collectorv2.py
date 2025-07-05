import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class EpisodeReplaySystem:
    def __init__(self, env, save_dir='episode_replays'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.env = env

    def save_episode(self, episode_num, states, actions, info, env_data=None):
        """Save episode data to file"""
        episode_data = {
            'states': states,
            'actions': actions,
            'episode_num': episode_num,
            'env_data': env_data,  # Add environment data
            'info' : info
        }
        total_episode_reward = sum(step_info['total_reward'] for step_info in info)
        reward_for_filename = int(total_episode_reward)
        filepath = os.path.join(self.save_dir, f'episode_{episode_num}_reward_{reward_for_filename}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)

    def plot_reward_components(self, step_num):
        """Plot bar chart of reward components in the same window"""
        # Clear the previous plot
        self.ax.clear()

        # Prepare data for plotting
        component_names = [
            'Distance\nReward',
            'Progress\nReward',
            'Heading\nReward',
            'Orientation\nReward',
            'Staged\nSuccess',
            'Safety\nPenalty',
            'Exploration\nBonus',
            'Final Success\nBonus',
            'Backward\nPenalty'
        ]
        graph_data = self.reward_info[step_num]


        component_values = [
            graph_data['distance_reward'],
            graph_data['progress_reward'],
            graph_data['heading_reward'],
            graph_data['orientation_reward'],
            graph_data['staged_success'],
            graph_data['safety_penalty'],
            graph_data['exploration_bonus'],
            graph_data['final_success_bonus'],
            graph_data['backward_penalty']
        ]

        # Create color map (positive rewards in green, penalties in red)
        colors = []
        for value in component_values:
            if value > 0:
                colors.append('lightgreen')
            elif value < 0:
                colors.append('lightcoral')
            else:
                colors.append('lightgray')

        # Create bar chart
        bars = self.ax.bar(component_names, component_values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on top of bars
        for bar, value in zip(bars, component_values):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2.,
                         height + (0.01 * max(abs(min(component_values)), max(component_values))),
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Customize plot
        title = f'Reward Components'
        if step_num is not None:
            title += f' - step {step_num}'
            title += f' (Total: {graph_data["total_reward"]:.3f})'

        if graph_data['final_success_bonus'] > 0:
            status = "SUCCESS"
            title += f' - {status}'
        else:
            status = "FAILURE"
            title += f' - {status}'

        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Reward Value', fontsize=12)
        self.ax.set_xlabel('Reward Components', fontsize=12)

        # Add horizontal line at y=0
        self.ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Rotate x-axis labels for better readability
        self.ax.tick_params(axis='x', rotation=45)

        # Adjust layout to prevent label cutoff
        self.fig.tight_layout()

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def replay_episode(self, episode_num, reward_for_filename):
        """Load and replay a specific episode"""
        filepath = os.path.join(self.save_dir, f'episode_{episode_num}_reward_{reward_for_filename}.pkl')

        if not os.path.exists(filepath):
            print(f"Episode {episode_num} not found!")
            return

        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)

        states = episode_data['states']
        actions = episode_data['actions']
        env_data = episode_data.get('env_data', None)
        self.reward_info = episode_data['info']
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

        else:
            # If no env_data available, call reset to generate new random scenario
            print("No environment data found, generating new random scenario...")
            self.env.reset()
            self.env.state = states[0]  # Override with saved initial state


        self.fig, self.ax = plt.subplots(figsize=(12, 6))  # Create attributes
        plt.ion()  # Turn on interactive mode
        self.fig.show()

        # Render initial state
        self.env.render()

        # Step through the episode
        for i in range(len(actions)):
            # Apply the recorded action
            observation, reward, done, info = self.env.step(actions[i])
            self.env.render()
            self.plot_reward_components(i)

            if done:
                print(f"Episode ended at step {i + 1}")
                break

        print("Replay complete!")