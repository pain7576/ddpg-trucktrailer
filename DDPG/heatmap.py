import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DDPG_agent import Agent
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from seed_utils import set_seed

# --- Configuration ---
# --- IMPORTANT: Running this script can take a very long time! ---
# To reduce time, you can:
# 1. Increase GRID_RESOLUTION (e.g., to 2.0 or 4.0)
# 2. Decrease the area to test by changing MAP_X_RANGE or MAP_Y_RANGE
# 3. Decrease TRIALS_PER_CELL


SEED = 66
# --- Grid and Test Configuration ---
GRID_RESOLUTION = 2.0  # The size of each grid cell (e.g., 2.0 means 2x2 meter cells).
TRIALS_PER_CELL = 5    # Number of episodes to run for each grid cell.
MAP_X_RANGE = (-40, 40) # Range of x-coordinates to test.
MAP_Y_RANGE = (-30, 40) # Range of y-coordinates to test.
# Range of starting orientations to test for each point (in degrees).
START_ORIENTATION_RANGE_DEG = (45, 120)

# --- Fixed Goal Pose ---
# All tests will try to reach this single, consistent goal.
GOAL_POSE = {'x': 0, 'y': -30, 'yaw_deg': 90}

# --- Output ---
OUTPUT_DIR = 'heatmaps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_heatmap_data(agent, env):
    """
    Iterates through a grid of starting positions, runs episodes, and collects data.
    """
    print("--- Starting Heatmap Data Generation ---")
    print(f"Grid Resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}m")
    print(f"Test Area (X): {MAP_X_RANGE}")
    print(f"Test Area (Y): {MAP_Y_RANGE}")
    print(f"Trials per Cell: {TRIALS_PER_CELL}")
    print("This may take several hours. Progress will be shown below.")

    # Define the grid coordinates
    x_coords = np.arange(MAP_X_RANGE[0], MAP_X_RANGE[1], GRID_RESOLUTION)
    y_coords = np.arange(MAP_Y_RANGE[0], MAP_Y_RANGE[1], GRID_RESOLUTION)

    # Initialize grids to store results. Use np.nan for cells that are not tested.
    reward_grid = np.full((len(y_coords), len(x_coords)), np.nan)
    success_grid = np.full((len(y_coords), len(x_coords)), np.nan)

    # Store exact orientations for each trial - list of lists
    orientations_data = []

    # Use tqdm for a progress bar
    for iy, start_y in enumerate(tqdm(y_coords, desc="Testing Grid Rows")):
        for ix, start_x in enumerate(x_coords):
            cell_scores = []
            cell_successes = []
            cell_orientations = []

            # Run multiple trials for the current grid cell
            for _ in range(TRIALS_PER_CELL):
                # --- Set up the environment for a specific start pose ---
                env.reset() # Reset to clear any old state

                # Set the fixed goal pose
                goal_yaw_rad = np.deg2rad(GOAL_POSE['yaw_deg'])
                env.goalx, env.goaly, env.goalyaw = GOAL_POSE['x'], GOAL_POSE['y'], goal_yaw_rad

                # Set the specific starting pose for this trial
                start_yaw_deg = np.random.uniform(START_ORIENTATION_RANGE_DEG[0], START_ORIENTATION_RANGE_DEG[1])
                start_yaw_rad = np.deg2rad(start_yaw_deg)
                env.startx, env.starty, env.startyaw = start_x, start_y, start_yaw_rad

                # Store the exact orientation used
                cell_orientations.append({
                    'x': start_x,
                    'y': start_y,
                    'yaw_deg': start_yaw_deg,
                    'yaw_rad': start_yaw_rad
                })

                # Recompute max steps based on this new configuration
                env.max_episode_steps = env.compute_max_steps()

                # Manually calculate and set the initial state vector based on the trailer's pose
                psi_2, x2, y2 = start_yaw_rad, start_x, start_y
                psi_1 = start_yaw_rad
                x1 = start_x + env.L2 * np.cos(start_yaw_rad)
                y1 = start_y + env.L2 * np.sin(start_yaw_rad)
                env.state = np.array([psi_1, psi_2, x1, y1, x2, y2], dtype=np.float32)

                # Get the initial observation for the agent
                observation = env.compute_observation(env.state, steering_angle=0.0)

                # --- Run the episode ---
                done = False
                score = 0
                while not done:
                    # env.render() # DO NOT RENDER for performance
                    action = agent.choose_action(observation, evaluate=True) # evaluate=True turns off noise
                    scaled_action = np.clip(action, -1, 1) * env.action_space.high
                    observation_, reward, done, info = env.step(scaled_action)
                    score += reward
                    observation = observation_

                # Record the results for this trial
                is_success = info.get('success', False)
                cell_scores.append(score)
                cell_successes.append(1 if is_success else 0)

            # Aggregate results for the cell
            if cell_scores:
                reward_grid[iy, ix] = np.mean(cell_scores)
                success_grid[iy, ix] = np.mean(cell_successes)
                orientations_data.extend(cell_orientations)

    return reward_grid, success_grid, x_coords, y_coords, orientations_data

def plot_success_distribution(grid_data, title, filename):
    """
    Creates a bar graph showing the distribution of success rates.
    Specifically designed for success rates: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    Args:
        grid_data: 2D numpy array with success rate values (may contain NaN)
        title: Title for the plot
        filename: Output filename for the plot
    """
    # Remove NaN values
    valid_data = grid_data[~np.isnan(grid_data)]

    if len(valid_data) == 0:
        print(f"Warning: No valid data for '{title}' distribution. Cannot generate bar graph.")
        return

    print(f"Generating success rate distribution bar graph: {title}")

    # Define the exact success rate values (based on TRIALS_PER_CELL = 5)
    success_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    success_labels = ['0%', '20%', '40%', '60%', '80%', '100%']

    # Count occurrences of each success rate
    counts = []
    for rate in success_rates:
        count = np.sum(np.abs(valid_data - rate) < 0.01)  # Small tolerance for floating point comparison
        counts.append(count)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bar chart with colors from plasma colormap
    colors = plt.cm.plasma(np.array(success_rates))
    bars = ax.bar(success_labels, counts, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)

    # Add value labels on top of bars
    max_count = max(counts) if counts else 1
    for i, (label, count) in enumerate(zip(success_labels, counts)):
        if count > 0:  # Only label non-zero bars
            ax.text(i, count + max_count * 0.01,
                    str(int(count)), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_title(f'Success Rate Distribution\n({len(valid_data)} grid cells total)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('Number of Grid Cells', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    # Add statistics text box
    stats_text = f'Total Cells: {len(valid_data)}\n'
    stats_text += f'Mean Success: {np.mean(valid_data):.1%}\n'
    stats_text += f'Cells with 100% Success: {counts[5]}\n'
    stats_text += f'Cells with 0% Success: {counts[0]}\n'
    stats_text += f'Cells with Partial Success: {sum(counts[1:5])}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Success rate distribution bar graph saved to {output_path}")

def plot_heatmap(grid_data, x_coords, y_coords, orientations_data, title, cmap, filename):
    """
    Generates and saves a heatmap plot from the grid data with exact orientations.
    """
    if np.all(np.isnan(grid_data)):
        print(f"Warning: All data for '{title}' is NaN. Cannot generate heatmap.")
        return

    print(f"Generating heatmap: {title}")
    fig, ax = plt.subplots(figsize=(12, 12))

    # We use pcolormesh. Remove the transpose (.T) - grid_data already has the correct shape
    # We add the resolution to the coords to define the corners of the last cells.
    x_mesh, y_mesh = np.append(x_coords, x_coords[-1] + GRID_RESOLUTION), np.append(y_coords, y_coords[-1] + GRID_RESOLUTION)
    mesh = ax.pcolormesh(x_mesh, y_mesh, grid_data, cmap=cmap, shading='auto', vmin=np.nanmin(grid_data), vmax=np.nanmax(grid_data))

    # Add a color bar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(title.split('(')[0].strip())

    # Plot all exact orientations that were tested
    if orientations_data:
        # Arrow length scaled to grid resolution
        arrow_length = GRID_RESOLUTION * 0.25

        # Draw arrows for each exact orientation tested
        for orientation in orientations_data:
            x_pos = orientation['x']
            y_pos = orientation['y']
            yaw_rad = orientation['yaw_rad']

            # Arrow components
            dx = arrow_length * np.cos(yaw_rad)
            dy = arrow_length * np.sin(yaw_rad)

            ax.arrow(x_pos, y_pos, dx, dy,
                     head_width=GRID_RESOLUTION*0.08, head_length=GRID_RESOLUTION*0.08,
                     fc='yellow', ec='black', linewidth=0.8, alpha=0.8)

        # Add small dots at grid centers for reference
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        valid_mask = ~np.isnan(grid_data)
        if np.any(valid_mask):
            ax.scatter(X_grid[valid_mask], Y_grid[valid_mask],
                       c='white', s=6, marker='o', alpha=0.9,
                       edgecolors='black', linewidths=0.5, label='Grid Centers')

    # Plot the fixed goal pose for context
    ax.plot(GOAL_POSE['x'], GOAL_POSE['y'], 'w*', markersize=18, markeredgecolor='black', linewidth=2, label='Goal Pose')
    goal_arrow_len = 6
    ax.arrow(GOAL_POSE['x'], GOAL_POSE['y'],
             goal_arrow_len * np.cos(np.deg2rad(GOAL_POSE['yaw_deg'])),
             goal_arrow_len * np.sin(np.deg2rad(GOAL_POSE['yaw_deg'])),
             head_width=2, head_length=2, fc='white', ec='black', linewidth=2)


    # Formatting
    ax.set_title(f'Agent Performance Heatmap: {title}\n(Yellow arrows show exact start orientations)', fontsize=14)
    ax.set_xlabel('Start X-coordinate (m)')
    ax.set_ylabel('Start Y-coordinate (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('gray') # Color for untested areas
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white', alpha=0.3)

    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")

def main():
    """
    Main function to initialize agent, generate data, and plot heatmaps.
    """

    if SEED is not None:
        set_seed(SEED)

    # --- Initialize Environment and Agent ---
    env = Truck_trailer_Env_2()
    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=env.action_space.shape[0])

    # --- Load Saved Models ---
    try:
        print("🔄 Loading saved models from 'tmp/ddpg/'...")
        agent.load_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please ensure you have trained models in 'tmp/ddpg/'.")
        return

    # --- Generate Data ---
    start_time = time.time()
    reward_data, success_data, x_coords, y_coords, orientations_data = generate_heatmap_data(agent, env)
    end_time = time.time()
    print(f"\nData generation completed in { (end_time - start_time) / 3600:.2f} hours.")
    print(f"Total orientations tested: {len(orientations_data)}")

    # --- Plot Heatmaps ---
    plot_heatmap(
        grid_data=reward_data,
        x_coords=x_coords,
        y_coords=y_coords,
        orientations_data=orientations_data,
        title='Average Episode Reward',
        cmap='viridis',
        filename='heatmap_average_reward.png'
    )

    plot_heatmap(
        grid_data=success_data,
        x_coords=x_coords,
        y_coords=y_coords,
        orientations_data=orientations_data,
        title='Success Rate (0.0 to 1.0)',
        cmap='plasma',
        filename='heatmap_success_rate.png'
    )

    # --- Plot Value Distribution Bar Graphs ---

    plot_success_distribution(
        grid_data=success_data,
        title='Success Rate Distribution',
        filename='distribution_success_rate.png'
    )


    print("\n✅ Heatmap and distribution analysis complete!")

if __name__ == '__main__':
    main()