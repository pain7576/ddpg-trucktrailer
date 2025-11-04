import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import pygame
from PIL import Image


class EpisodeReplaySystem:
    def __init__(self, env, save_dir='episode_replays'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.env = env
        # Initialize tracking variables for the twin plot
        self.cumulative_rewards = []
        self.step_actions = []

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

    def plot_twin_graph(self, step_num):
        """Plot twin x-axis graph showing cumulative reward and actions with modern styling"""
        # Clear the previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Set modern style
        plt.style.use('default')  # Reset to default then customize

        # Prepare x-axis data (step numbers from 0 to current step)
        steps = list(range(len(self.cumulative_rewards)))

        # Convert actions from radians to degrees
        actions_degrees = []
        for action in self.step_actions:
            action_deg = np.degrees(action)
            # Convert to scalar if it's a numpy array
            if hasattr(action_deg, 'item'):
                action_deg = action_deg.item()
            elif hasattr(action_deg, '__len__') and len(action_deg) == 1:
                action_deg = action_deg[0]
            actions_degrees.append(action_deg)

        # Modern color palette
        reward_color = '#2E86C1'  # Modern blue
        action_color = '#E74C3C'  # Modern red

        # Plot cumulative reward on left y-axis with modern styling
        line1 = self.ax1.plot(steps, self.cumulative_rewards, color=reward_color, linewidth=3,
                              marker='o', markersize=6, markerfacecolor=reward_color,
                              markeredgecolor='white', markeredgewidth=2, alpha=0.8,
                              label='Cumulative Reward')

        self.ax1.set_xlabel('Step', fontsize=14, fontweight='bold', color='#2C3E50')
        self.ax1.set_ylabel('Cumulative Reward', color=reward_color, fontsize=14, fontweight='bold')
        self.ax1.tick_params(axis='y', labelcolor=reward_color, labelsize=11)
        self.ax1.tick_params(axis='x', labelsize=11, color='#2C3E50')

        # Modern grid styling
        self.ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#BDC3C7')
        self.ax1.set_facecolor('#FAFAFA')  # Light background

        # Plot actions on right y-axis with modern styling
        line2 = self.ax2.plot(steps, actions_degrees, color=action_color, linewidth=3,
                              marker='s', markersize=6, markerfacecolor=action_color,
                              markeredgecolor='white', markeredgewidth=2, alpha=0.8,
                              label='Action')

        self.ax2.set_ylabel('Action (degrees)', color=action_color, fontsize=14, fontweight='bold')
        self.ax2.tick_params(axis='y', labelcolor=action_color, labelsize=11)

        # Set title with modern styling
        current_reward = self.cumulative_rewards[-1] if self.cumulative_rewards else 0
        current_action_degrees = actions_degrees[-1] if actions_degrees else 0
        # Convert to scalar if it's a numpy array
        if hasattr(current_action_degrees, 'item'):
            current_action_degrees = current_action_degrees.item()
        elif hasattr(current_action_degrees, '__len__') and len(current_action_degrees) == 1:
            current_action_degrees = current_action_degrees[0]

        title = f'Step {step_num}: Cumulative Reward = {current_reward:.3f}, Action = {current_action_degrees:.1f}°'
        self.ax1.set_title(title, fontsize=16, fontweight='bold', color='#2C3E50', pad=20)

        # Add subtle border
        for spine in self.ax1.spines.values():
            spine.set_color('#BDC3C7')
            spine.set_linewidth(1)
        for spine in self.ax2.spines.values():
            spine.set_color('#BDC3C7')
            spine.set_linewidth(1)

        # Add legend with modern styling
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        legend = self.ax1.legend(lines1 + lines2, labels1 + labels2,
                                 loc='upper left', frameon=True, fancybox=True,
                                 shadow=True, framealpha=0.9, fontsize=11)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#BDC3C7')

        # Update the plot
        self.fig_twin.tight_layout()
        self.fig_twin.canvas.draw()
        self.fig_twin.canvas.flush_events()

    def plot_reward_components(self, step_num):
        """Plot bar chart of reward components with modern styling"""
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
            'Backward\nPenalty',
            'smoothness\nPenalty',
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
            graph_data['backward_penalty'],
            graph_data['smoothness_penalty'],
        ]

        # Modern color scheme
        colors = []
        for value in component_values:
            if value > 0:
                colors.append('#27AE60')  # Modern green
            elif value < 0:
                colors.append('#E74C3C')  # Modern red
            else:
                colors.append('#95A5A6')  # Modern gray

        # Create modern bar chart
        bars = self.ax.bar(component_names, component_values, color=colors, alpha=0.8,
                           edgecolor='white', linewidth=2, width=0.8)

        # Add modern value labels on bars
        for bar, value in zip(bars, component_values):
            height = bar.get_height()
            label_y = height + (0.02 * max(abs(min(component_values)), max(component_values))) if height >= 0 else height - (0.05 * max(abs(min(component_values)), max(component_values)))
            self.ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                         f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=10, fontweight='bold', color='#2C3E50')

        # Modern title styling
        title = f'Reward Components'
        if step_num is not None:
            title += f' - Step {step_num}'
            title += f' (Total: {graph_data["total_reward"]:.3f})'

        if graph_data['final_success_bonus'] > 0:
            status = "SUCCESS ✓"
            title += f' - {status}'
            title_color = '#27AE60'
        else:
            status = "FAILURE ✗"
            title += f' - {status}'
            title_color = '#E74C3C'

        self.ax.set_title(title, fontsize=16, fontweight='bold', color=title_color, pad=20)
        self.ax.set_ylabel('Reward Value', fontsize=14, fontweight='bold', color='#2C3E50')
        self.ax.set_xlabel('Reward Components', fontsize=14, fontweight='bold', color='#2C3E50')

        # Modern horizontal line at y=0
        self.ax.axhline(y=0, color='#34495E', linestyle='-', alpha=0.8, linewidth=2)

        # Modern styling
        self.ax.tick_params(axis='x', rotation=45, labelsize=10, color='#2C3E50')
        self.ax.tick_params(axis='y', labelsize=11, color='#2C3E50')
        self.ax.set_facecolor('#FAFAFA')  # Light background

        # Modern grid
        self.ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#BDC3C7', axis='y')

        # Modern border styling
        for spine in self.ax.spines.values():
            spine.set_color('#BDC3C7')
            spine.set_linewidth(1)

        # Adjust layout to prevent label cutoff
        self.fig.tight_layout()

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_fig_as_image(self, fig):
        """Converts a matplotlib figure (any backend) to a PIL Image safely."""
        # Create an Agg canvas and draw the figure onto it
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_np = np.asarray(buf, dtype=np.uint8)[..., :3]  # Drop alpha channel
        return Image.fromarray(img_np)

    def _get_pygame_as_image(self):
        """Not used: env uses Matplotlib, not Pygame."""
        return Image.new('RGB', (800, 600), (255, 255, 255))

    def replay_episode(self, episode_num, reward_for_filename, save_gif=True, gif_fps=10):
        """Load and replay a specific episode, saving two GIFs:
           (1) env only, (2) plots only (reward + twin graphs)."""
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

        gif_frames_env = []   # 🆕 for environment-only GIF
        gif_frames_plots = [] # 🆕 for plots-only GIF

        # Reset tracking variables for new episode
        self.cumulative_rewards = []
        self.step_actions = []

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
            self.env.max_episode_steps = self.env.compute_max_steps()
            self.env.episode_steps = 0
        else:
            print("No environment data found, generating new random scenario...")
            self.env.reset()
            self.env.state = states[0]

        # Create two separate figure windows
        self.fig, self.ax = plt.subplots(figsize=(12, 6))       # Reward components plot
        self.fig_twin, self.ax1 = plt.subplots(figsize=(12, 6)) # Twin plot
        self.ax2 = self.ax1.twinx()                             # Twin y-axis

        plt.ion()
        self.fig.show()
        self.fig_twin.show()

        # Render initial state
        self.env.render()

        cumulative_reward = 0

        for i in range(len(actions)):
            observation, reward, done, info = self.env.step(actions[i])

            step_reward = self.reward_info[i]['total_reward']
            cumulative_reward += step_reward
            self.cumulative_rewards.append(cumulative_reward)
            self.step_actions.append(actions[i])

            # Render and update plots
            self.env.render()
            self.plot_reward_components(i)
            self.plot_twin_graph(i)
            plt.pause(0.001)

            if save_gif:
                # 🆕 Capture each figure separately
                env_img = self._get_fig_as_image(self.env.fig)
                reward_plot_img = self._get_fig_as_image(self.fig)
                twin_plot_img = self._get_fig_as_image(self.fig_twin)

                # 🆕 Combine the two plots side by side
                plot_w, plot_h = reward_plot_img.size
                twin_plot_resized = twin_plot_img.resize((plot_w, plot_h), Image.Resampling.LANCZOS)
                combined_plots = Image.new('RGB', (plot_w * 2, plot_h), (255, 255, 255))
                combined_plots.paste(reward_plot_img, (0, 0))
                combined_plots.paste(twin_plot_resized, (plot_w, 0))

                # 🆕 Save both GIF frames separately
                gif_frames_env.append(np.array(env_img))
                gif_frames_plots.append(np.array(combined_plots))

            if done:
                print(f"Episode ended at step {i + 1}")
                break

        if save_gif:
            # 🆕 Save environment-only GIF
            env_gif_filename = f'replay_episode_{episode_num}_reward_{reward_for_filename}_env.gif'
            env_gif_filepath = os.path.join(self.save_dir, env_gif_filename)
            print(f"\nSaving environment GIF to {env_gif_filepath}...")
            imageio.mimsave(env_gif_filepath, gif_frames_env, fps=gif_fps, loop=0)

            # 🆕 Save plots-only GIF
            plots_gif_filename = f'replay_episode_{episode_num}_reward_{reward_for_filename}_plots.gif'
            plots_gif_filepath = os.path.join(self.save_dir, plots_gif_filename)
            print(f"Saving plots GIF to {plots_gif_filepath}...")
            imageio.mimsave(plots_gif_filepath, gif_frames_plots, fps=gif_fps, loop=0)

            print("Both GIFs saved successfully!")

        print("Replay complete!")
        plt.show(block=True)