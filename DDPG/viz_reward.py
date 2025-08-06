import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ==============================================================================
# --- START OF CODE FROM reb2.py ---
# Description: Visualizes a 3D reward landscape for an agent approaching a target.
# ==============================================================================

# --- 1. Define the Reward Function and Parameters ---
# This function remains unchanged.

# Define the parameters that were part of the 'self' object
WEIGHTS_REB2 = (10, 20, 50) # Rewards for stage 1, 2, 3
POSITION_THRESHOLD_FINAL_REB2 = 0.5  # meters
ORIENTATION_THRESHOLD_FINAL_RAD_REB2 = np.deg2rad(15) # radians

def calculate_staged_reward_landscape(distance, orientation_error_rad):
    """
    Calculates the reward for a grid of distance and orientation values.
    This is a vectorized version for efficient plotting.
    """
    # Initialize rewards to zero
    rewards = np.zeros_like(distance)

    # Stage 1: Getting close (within 5 meters)
    rewards[distance <= 5.0] += WEIGHTS_REB2[0]

    # Stage 2: Very close with decent orientation
    stage2_mask = (distance <= 2.0) & (orientation_error_rad <= np.deg2rad(45))
    rewards[stage2_mask] += WEIGHTS_REB2[1]

    # Stage 3: Final docking position
    stage3_mask = (distance <= POSITION_THRESHOLD_FINAL_REB2) & \
                  (orientation_error_rad <= ORIENTATION_THRESHOLD_FINAL_RAD_REB2)
    rewards[stage3_mask] += WEIGHTS_REB2[2]

    return rewards

def run_reb2_visualization():
    """
    Generates data and creates the 3D interactive plot for the staged reward landscape.
    """
    print("--- Running Visualization from reb2.py ---")
    print("Generating interactive 3D reward landscape. Please close the browser/plot window to continue.")

    # --- 2. Generate Input Data for the Plot ---
    # We need a grid of (x, y) coordinates and a range of orientation errors.

    # Define the spatial grid for the agent's position over the new, larger range.
    # We increase the number of points to maintain a reasonable resolution.
    x_coords = np.linspace(-40, 40, 320)
    y_coords = np.linspace(-40, 40, 320)

    # Define a set of discrete orientation errors for the color dimension.
    orientation_errors_deg = np.array([0, 10, 20, 40, 50, 70, 90])
    orientation_errors_rad = np.deg2rad(orientation_errors_deg)

    # Create a 3D meshgrid: one dimension for x, one for y, and one for orientation
    xx, yy, oo_rad = np.meshgrid(x_coords, y_coords, orientation_errors_rad)
    oo_deg = np.rad2deg(oo_rad)

    # --- 3. Calculate the Rewards and Create a DataFrame ---

    # Set the goal position at the origin for distance calculation
    goal_position = np.array([-10, 15])

    # Calculate the distance from each (x, y) point in the grid to the goal
    current_distances = np.sqrt((xx - goal_position[0])**2 + (yy - goal_position[1])**2)

    # Now, calculate the final Z-value (reward) for each point in our 3D grid.
    reward_values = calculate_staged_reward_landscape(current_distances, oo_rad)

    # Flatten all 3D arrays into 1D arrays to create the DataFrame
    df_rewards = pd.DataFrame({
        'x': xx.flatten(),
        'y': yy.flatten(),
        'reward': reward_values.flatten(),
        'orientation_error_deg': oo_deg.flatten()
    })

    # IMPORTANT: This filtering step is now even more critical.
    # Without it, we would plot tens of thousands of points at reward=0,
    # obscuring the view and slowing down the plot.
    df_rewards = df_rewards[df_rewards['reward'] > 0]

    print("Sample of the generated reward data (only showing points with non-zero reward):")
    print(df_rewards.sample(min(10, len(df_rewards))))

    # --- 4. Create the 3D Interactive Plot with Plotly ---

    fig = px.scatter_3d(
        df_rewards,
        x='x',
        y='y',
        z='reward',
        color='orientation_error_deg',
        color_continuous_scale=px.colors.sequential.Plasma_r, # _r reverses the scale
        title="Reward Landscape in a Large (x, y) Space",
        labels={
            'x': 'X Position (m)',
            'y': 'Y Position (m)',
            'reward': 'Staged Reward',
            'orientation_error_deg': 'Orientation Error (°)'
        },
        color_continuous_midpoint=45,
        # Explicitly set the axis ranges to match our input
        range_x=[-40, 40],
        range_y=[-40, 40]
    )

    # Improve the plot's appearance
    fig.update_traces(marker=dict(size=2.5, opacity=0.9))
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Staged Reward',
            # Set the aspect ratio to make the plot look proportional to the space
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2) # Make the Z-axis less tall relative to X/Y
        ),
        margin=dict(r=10, b=10, l=10, t=40)
    )

    fig.show()

# ==============================================================================
# --- END OF CODE FROM reb2.py ---
# ==============================================================================


# ==============================================================================
# --- START OF CODE FROM reb1.py ---
# Description: Visualizes a safety penalty function on a polar plot.
# ==============================================================================

# 1. Define the penalty logic
def get_safety_penalty(theta_diff_rad, weights):
    if abs(theta_diff_rad) > np.deg2rad(85):
        return weights['safety_major']
    elif abs(theta_diff_rad) > np.deg2rad(70):
        return weights['safety_minor']
    else:
        return 0

def run_reb1_visualization():
    """
    Generates data and creates the polar plot for the safety penalty.
    """
    print("\n--- Running Visualization from reb1.py ---")
    print("Generating polar plot for safety penalty. Please close the plot window to continue.")

    # 2. Set up data in degrees, then convert to radians
    weights = {'safety_minor': -50, 'safety_major': -500}
    theta_diffs_deg = np.linspace(-90, 90, 400)
    theta_diffs_rad = np.deg2rad(theta_diffs_deg)
    penalties = [get_safety_penalty(t_rad, weights) for t_rad in theta_diffs_rad]

    # 3. Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the data using RADIANS
    ax.plot(theta_diffs_rad, penalties, color='blue', linewidth=1)

    # 4. Customize the plot layout and view
    ax.set_title("Safety Penalty vs. Angle Difference", pad=20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction('clockwise')

    # Force the angular view to be a semi-circle from -90° (left) to +90° (right).
    ax.set_thetalim(-np.pi/2, np.pi/2)

    # Set the visible angle labels using DEGREES
    degree_ticks = [-90, -70, -45, 0, 45, 70, 90]
    ax.set_thetagrids(degree_ticks, labels=[f'{d}°' for d in degree_ticks])

    # Set the radial (penalty) label
    ax.set_ylim(0, -500)

    plt.show()

# ==============================================================================
# --- END OF CODE FROM reb1.py ---
# ==============================================================================


# ==============================================================================
# --- START OF CODE FROM reb.py ---
# Description: Visualizes a steering smoothness penalty as a 3D surface plot.
# ==============================================================================

class SteeringSmoothnessPenalty:
    def __init__(self):
        self.previous_steering = 0  # Initialize previous steering

    def calculate_steering_smoothness_penalty(self, current_steering, previous_steering=None):
        """
        Calculate steering smoothness penalty
        Args:
            current_steering: Current steering angle in radians
            previous_steering: Previous steering angle in radians (optional, uses self.previous_steering if None)
        """
        if previous_steering is None:
            previous_steering = self.previous_steering

        steering_change = current_steering - previous_steering
        normalized_steering_change = abs(steering_change) / np.deg2rad(90)
        smoothness_penalty = normalized_steering_change

        return smoothness_penalty

def run_reb_visualization():
    """
    Creates and displays the 3D surface plot for steering smoothness penalty.
    """
    print("\n--- Running Visualization from reb.py ---")
    print("Generating 3D surface plot for steering penalty. Please close the plot window to continue.")

    # Create figure
    fig = plt.figure(figsize=(10, 8))

    # Define steering range (-45 to +45 degrees)
    steering_range = np.deg2rad(np.linspace(-45, 45, 100))

    # 3D Surface Plot
    ax = fig.add_subplot(111, projection='3d')
    current_grid, previous_grid = np.meshgrid(steering_range, steering_range)
    penalty_grid = np.abs(current_grid - previous_grid) / np.deg2rad(90)

    surf = ax.plot_surface(np.rad2deg(current_grid), np.rad2deg(previous_grid),
                           penalty_grid, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Current Steering (degrees)')
    ax.set_ylabel('Previous Steering (degrees)')
    ax.set_zlabel('Smoothness Penalty')
    ax.set_title('3D Surface: Steering Smoothness Penalty')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Penalty')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# --- END OF CODE FROM reb.py ---
# ==============================================================================


# ==============================================================================
# --- START OF CODE FROM reward_viz.py ---
# Description: Creates an animated polar plot showing dynamic reward weighting.
# ==============================================================================

class RewardSystem:
    def __init__(self):
        self.goal_orientation = np.deg2rad(120)  # Final desired trailer orientation

    def compute_dynamic_weights(self, journey_progress):
        """Compute distance-dependent weights"""
        sharpness = 7
        transition_val = np.tanh(sharpness * (journey_progress - 0.3))
        trailer_final_orientation_priority = (transition_val + 1) / 2.0
        trailer_heading_priority = 1.0 - trailer_final_orientation_priority
        return {
            'trailer_heading': trailer_heading_priority,
            'trailer_final_orientation': trailer_final_orientation_priority,
            'journey_progress': journey_progress
        }

    def calculate_orientation_alignment_reward(self, current_orientation, desired_heading):
        """Calculate heading reward using cosine similarity"""
        angle_difference = desired_heading - (current_orientation + np.pi)
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
        return np.cos(angle_difference)

    def calculate_trailer_goal_orientation_reward(self, current_orientation):
        """Reward for trailer orientation matching the final goal orientation"""
        angle_difference = self.goal_orientation - current_orientation
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
        return np.cos(angle_difference)

def run_reward_viz_animation():
    """
    Sets up and runs the animated polar plot visualization.
    """
    print("\n--- Running Visualization from reward_viz.py ---")
    print("Generating animated polar plot for reward components. Please close the plot window to finish.")

    # Initialize
    reward_system = RewardSystem()
    desired_heading = np.deg2rad(45)  # Desired heading
    angles_deg = np.arange(0, 360)
    angles_rad = np.deg2rad(angles_deg)

    # Setup plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    heading_line, = ax.plot([], [], label='Heading Reward', linewidth=2)
    orientation_line, = ax.plot([], [], label='Goal Orientation Reward', linewidth=2)

    # Add goal orientation marker
    goal_orientation_line, = ax.plot(
        [reward_system.goal_orientation, reward_system.goal_orientation],
        [-15, 15],
        linestyle='--',
        color='red',
        linewidth=1.5,
        label='Goal Orientation'
    )

    goal_heading_line, = ax.plot(
        [desired_heading, desired_heading],
        [-15, 15],
        linestyle='--',
        color='blue',
        linewidth=1.5,
        label='Goal Orientation'
    )

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title("Reward Components vs Trailer Orientation")
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.2))
    plt.tight_layout()

    # Animation function
    def update(frame):
        journey_progress = frame
        dynamic_weights = reward_system.compute_dynamic_weights(journey_progress)

        heading_rewards = []
        orientation_rewards = []

        for angle in angles_rad:
            heading = reward_system.calculate_orientation_alignment_reward(angle, desired_heading)*15
            orientation = reward_system.calculate_trailer_goal_orientation_reward(angle)*15

            heading *= dynamic_weights['trailer_heading']
            orientation *= dynamic_weights['trailer_final_orientation']

            heading_rewards.append(heading)
            orientation_rewards.append(orientation)

        heading_line.set_data(angles_rad, heading_rewards)
        orientation_line.set_data(angles_rad, orientation_rewards)

        # Update title with current progress
        ax.set_title(f"Journey Progress: {journey_progress:.2f}\n"
                     f"Heading Weight: {dynamic_weights['trailer_heading']:.2f}, "
                     f"Goal Weight: {dynamic_weights['trailer_final_orientation']:.2f}")

        return heading_line, orientation_line

    # Create animation - removed blit=True to allow title updates
    # The 'ani' variable must be kept in scope for the animation to run.
    frames = np.linspace(0, 1, 100)
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    plt.show()

# ==============================================================================
# --- END OF CODE FROM reward_viz.py ---
# ==============================================================================


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# Runs each visualization sequentially.
# ==============================================================================

if __name__ == "__main__":
    # Run the first visualization
    run_reb2_visualization()

    # Run the second visualization
    run_reb1_visualization()

    # Run the third visualization
    run_reb_visualization()

    # Run the fourth and final visualization
    run_reward_viz_animation()

    print("\nAll visualizations complete!")