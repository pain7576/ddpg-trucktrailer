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
    Now also collects trajectory end points and violation types.
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

    # Store trajectory end points and violation types
    trajectory_endpoints = []

    # Store full trajectories - one per cell
    trajectories_data = []

    # Use tqdm for a progress bar
    for iy, start_y in enumerate(tqdm(y_coords, desc="Testing Grid Rows")):
        for ix, start_x in enumerate(x_coords):
            cell_scores = []
            cell_successes = []
            cell_orientations = []
            cell_trajectory = None  # Will store one trajectory per cell

            # Run multiple trials for the current grid cell
            for trial_idx in range(TRIALS_PER_CELL):
                # --- Set up the environment for a specific start pose ---
                env.reset() # Reset to clear any old state

                # Set the fixed goal pose
                goal_yaw_rad = np.deg2rad(GOAL_POSE['yaw_deg'])
                env.goalx, env.goaly, env.goalyaw = GOAL_POSE['x'], GOAL_POSE['y'], goal_yaw_rad

                # Set the specific starting pose for this trial
                start_yaw_deg = np.random.uniform(START_ORIENTATION_RANGE_DEG[0], START_ORIENTATION_RANGE_DEG[1])
                start_yaw_rad = np.deg2rad(start_yaw_deg)
                env.startx, env.starty, env.startyaw = start_x, start_y, start_yaw_rad
                env.L2 = np.random.uniform(5,7)

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
                # # pick psi_1 randomly in [0, 180], but within ±40° of startyaw
                # psi_candidates = []
                # for _ in range(1000):  # try multiple random picks
                #     candidate = np.random.uniform(np.deg2rad(0), np.deg2rad(180))  # random angle in degrees
                #     if abs(candidate - start_yaw_rad) < np.deg2rad(40):
                #         psi_candidates.append(candidate)
                #
                # # if no candidate found (e.g. startyaw near 0 or 180), just use startyaw
                # if psi_candidates:
                #     psi_1 = np.random.choice(psi_candidates)
                # else:
                #     psi_1 = start_yaw_rad
                psi_1 = start_yaw_rad
                x1 = start_x + env.L2 * np.cos(start_yaw_rad)
                y1 = start_y + env.L2 * np.sin(start_yaw_rad)
                env.state = np.array([psi_1, psi_2, x1, y1, x2, y2], dtype=np.float32)

                # Get the initial observation for the agent
                observation = env.compute_observation(env.state, steering_angle=0.0)

                # Initialize trajectory recording for the first trial of each cell
                if trial_idx == 0:
                    cell_trajectory = {
                        'trailer_x': [env.state[4]],  # x2 - trailer x position
                        'trailer_y': [env.state[5]],  # y2 - trailer y position
                        'start_x': start_x,
                        'start_y': start_y,
                        'start_yaw_deg': start_yaw_deg
                    }

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

                    # Record trajectory for the first trial
                    if trial_idx == 0:
                        cell_trajectory['trailer_x'].append(env.state[4])
                        cell_trajectory['trailer_y'].append(env.state[5])

                # Record the results for this trial
                is_success = info.get('success', False)
                cell_scores.append(score)
                cell_successes.append(1 if is_success else 0)

                # Finalize trajectory for the first trial
                if trial_idx == 0:
                    cell_trajectory['success'] = is_success or env.goal_reached

                # Determine violation type and record trajectory endpoint
                if is_success or env.goal_reached:
                    violation_type = 'success'
                elif env.jackknife:
                    violation_type = 'jackknife'
                elif env.out_of_map:
                    violation_type = 'out_of_map'
                elif env.goal_passed:
                    violation_type = 'goal_passed'
                elif env.max_steps_reached:
                    violation_type = 'max_steps'
                else:
                    # Check if it was excessive backward movement or other failure
                    violation_type = 'other_failure'

                trajectory_endpoints.append({
                    'end_x': env.state[4],  # trailer x position
                    'end_y': env.state[5],  # trailer y position
                    'start_x': start_x,
                    'start_y': start_y,
                    'violation_type': violation_type,
                    'score': score
                })

            # Aggregate results for the cell
            if cell_scores:
                reward_grid[iy, ix] = np.mean(cell_scores)
                success_grid[iy, ix] = np.mean(cell_successes)
                orientations_data.extend(cell_orientations)

                # Store the trajectory for this cell (from first trial)
                if cell_trajectory is not None:
                    trajectories_data.append(cell_trajectory)

    return reward_grid, success_grid, x_coords, y_coords, orientations_data, trajectory_endpoints, trajectories_data

def plot_trajectories(trajectories_data, filename):
    """
    Plot trajectories - one per cell, colored by success/failure.
    """
    if not trajectories_data:
        print("No trajectory data available.")
        return

    print("Generating trajectories plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    successful_count = 0
    failed_count = 0

    # Plot each trajectory
    for traj in trajectories_data:
        if traj['success']:
            color = 'green'
            alpha = 0.7
            linewidth = 1.5
            successful_count += 1
        else:
            color = 'red'
            alpha = 0.6
            linewidth = 1.0
            failed_count += 1

        # Plot trajectory path
        ax.plot(traj['trailer_x'], traj['trailer_y'],
                color=color, alpha=alpha, linewidth=linewidth)

        # Mark start point
        ax.plot(traj['trailer_x'][0], traj['trailer_y'][0],
                'o', color=color, markersize=4, markeredgecolor='black',
                markeredgewidth=0.5, alpha=alpha)

    # Plot the goal pose
    ax.plot(GOAL_POSE['x'], GOAL_POSE['y'], 'gold', marker='*', markersize=15,
            markeredgecolor='black', linewidth=2, label='Goal')

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, alpha=0.7,
               label=f'Successful ({successful_count})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.6,
               label=f'Failed ({failed_count})'),
        Line2D([0], [0], marker='o', color='black', markersize=4,
               linewidth=0, label='Start points')
    ]
    ax.legend(handles=legend_elements)

    # Formatting
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.set_title('Trajectories - One per Grid Cell')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectories plot saved to {output_path}")

def plot_trajectory_endpoints(trajectory_endpoints, filename):
    """
    Plot scatter plot of trajectory end points colored by violation type.
    """
    if not trajectory_endpoints:
        print("No trajectory endpoint data available.")
        return

    print("Generating trajectory endpoints plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for each violation type
    violation_colors = {
        'success': 'green',
        'jackknife': 'red',
        'out_of_map': 'orange',
        'max_steps': 'blue',
        'goal_passed': 'purple',
        'other_failure': 'brown'
    }

    # Group endpoints by violation type
    violation_groups = {}
    for endpoint in trajectory_endpoints:
        vtype = endpoint['violation_type']
        if vtype not in violation_groups:
            violation_groups[vtype] = {'x': [], 'y': []}
        violation_groups[vtype]['x'].append(endpoint['end_x'])
        violation_groups[vtype]['y'].append(endpoint['end_y'])

    # Plot each violation type
    for vtype, points in violation_groups.items():
        color = violation_colors.get(vtype, 'gray')
        ax.scatter(points['x'], points['y'], c=color, alpha=0.6, s=20,
                   label=f'{vtype} ({len(points["x"])})')

    # Plot the goal pose
    ax.plot(GOAL_POSE['x'], GOAL_POSE['y'], 'gold', marker='*', markersize=15,
            markeredgecolor='black', linewidth=2, label='Goal')

    # Formatting
    ax.set_xlabel('End X-coordinate (m)')
    ax.set_ylabel('End Y-coordinate (m)')
    ax.set_title('Trajectory End Points by Violation Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectory endpoints plot saved to {output_path}")

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
        print("📄 Loading saved models from 'tmp/ddpg/'...")
        agent.load_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please ensure you have trained models in 'tmp/ddpg/'.")
        return

    # --- Generate Data ---
    start_time = time.time()
    reward_data, success_data, x_coords, y_coords, orientations_data, trajectory_endpoints, trajectories_data = generate_heatmap_data(agent, env)
    end_time = time.time()
    print(f"\nData generation completed in { (end_time - start_time) / 3600:.2f} hours.")
    print(f"Total orientations tested: {len(orientations_data)}")
    print(f"Total trajectory endpoints recorded: {len(trajectory_endpoints)}")
    print(f"Total trajectories recorded: {len(trajectories_data)}")

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

    # --- Plot Trajectory End Points ---
    plot_trajectory_endpoints(
        trajectory_endpoints=trajectory_endpoints,
        filename='trajectory_endpoints_by_violation.png'
    )

    # --- Plot Trajectories ---
    plot_trajectories(
        trajectories_data=trajectories_data,
        filename='trajectories_per_cell.png'
    )

    print("\n✅ Heatmap, trajectory endpoint, and trajectory analysis complete!")

if __name__ == '__main__':
    main()