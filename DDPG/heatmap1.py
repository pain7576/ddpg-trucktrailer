# --- START OF FILE heatmap.py (MODIFIED) ---

import sys
import os
import time
import argparse # Import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- DDPG and Environment Imports ---
from DDPG_agent import Agent
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from seed_utils import set_seed

# --- NPC Solver Import ---
from npc_solver import solve_with_npc

# --- Configuration ---
# --- IMPORTANT: Running this script can take a very long time! ---
# To reduce time, you can:
# 1. Increase GRID_RESOLUTION (e.g., to 2.0 or 4.0)
# 2. Decrease the area to test by changing MAP_X_RANGE or MAP_Y_RANGE
# 3. Decrease TRIALS_PER_CELL

SEED = 66
# --- Grid and Test Configuration ---
GRID_RESOLUTION = 10.0  # The size of each grid cell (e.g., 2.0 means 2x2 meter cells).
TRIALS_PER_CELL = 3    # Number of episodes to run for each grid cell.
MAP_X_RANGE = (-40, 40) # Range of x-coordinates to test.
MAP_Y_RANGE = (-30, 40) # Range of y-coordinates to test.
# Range of starting orientations to test for each point (in degrees).
START_ORIENTATION_RANGE_DEG = (45, 120)

# --- NPC Specific Configuration ---
NPC_PARAMS = {
    'p': 100,       # Prediction horizon
    'Ts': 0.2,      # Sample time
    'velocity': -5.0 # m/s (negative for reverse)
}


# --- Fixed Goal Pose ---
# All tests will try to reach this single, consistent goal.
GOAL_POSE = {'x': 0, 'y': -30, 'yaw_deg': 90}

# --- Output ---
OUTPUT_DIR = 'heatmaps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_heatmap_data(method, agent, env):
    """
    Iterates through a grid of starting positions, runs episodes/solvers, and collects data.
    Works for both 'ddpg' and 'npc' methods.
    """
    print(f"--- Starting Heatmap Data Generation for method: {method.upper()} ---")
    print(f"Grid Resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}m")
    print(f"Test Area (X): {MAP_X_RANGE}")
    print(f"Test Area (Y): {MAP_Y_RANGE}")
    print(f"Trials per Cell: {TRIALS_PER_CELL}")
    print("This may take several hours. Progress will be shown below.")

    # Define the grid coordinates
    x_coords = np.arange(MAP_X_RANGE[0], MAP_X_RANGE[1], GRID_RESOLUTION)
    y_coords = np.arange(MAP_Y_RANGE[0], MAP_Y_RANGE[1], GRID_RESOLUTION)

    # Initialize grids
    reward_grid = np.full((len(y_coords), len(x_coords)), np.nan)
    success_grid = np.full((len(y_coords), len(x_coords)), np.nan)
    orientations_data = []
    trajectory_endpoints = []
    trajectories_data = []

    # Use tqdm for a progress bar
    ii = 0
    for iy, start_y in enumerate(tqdm(y_coords, desc="Testing Grid Rows")):
        for ix, start_x in enumerate(x_coords):
            cell_scores = []
            cell_successes = []
            cell_orientations = []
            cell_trajectory = None

            for trial_idx in range(TRIALS_PER_CELL):
                # Randomize starting orientation for this trial
                start_yaw_deg = np.random.uniform(START_ORIENTATION_RANGE_DEG[0], START_ORIENTATION_RANGE_DEG[1])
                start_yaw_rad = np.deg2rad(start_yaw_deg)

                # Store orientation
                cell_orientations.append({'x': start_x, 'y': start_y, 'yaw_deg': start_yaw_deg, 'yaw_rad': start_yaw_rad})

                # --- METHOD-SPECIFIC LOGIC ---
                is_success = False
                score = 0
                final_state_x, final_state_y = None, None
                violation_type = 'unknown'

                if method == 'ddpg':
                    # --- DDPG AGENT EXECUTION ---
                    env.reset()
                    goal_yaw_rad = np.deg2rad(GOAL_POSE['yaw_deg'])
                    env.goalx, env.goaly, env.goalyaw = GOAL_POSE['x'], GOAL_POSE['y'], goal_yaw_rad
                    env.startx, env.starty, env.startyaw = start_x, start_y, start_yaw_rad
                    env.L2 = np.random.uniform(5,7)
                    env.max_episode_steps = env.compute_max_steps()

                    # Manually set initial state
                    psi_1 = psi_2 = start_yaw_rad
                    x1 = start_x + env.L2 * np.cos(start_yaw_rad)
                    y1 = start_y + env.L2 * np.sin(start_yaw_rad)
                    env.state = np.array([psi_1, psi_2, x1, y1, start_x, start_y], dtype=np.float32)

                    observation = env.compute_observation(env.state, steering_angle=0.0)

                    if trial_idx == 0:
                        cell_trajectory = {'trailer_x': [env.state[4]], 'trailer_y': [env.state[5]],
                                           'start_x': start_x, 'start_y': start_y, 'start_yaw_deg': start_yaw_deg}

                    done = False
                    while not done:
                        action = agent.choose_action(observation, evaluate=True)
                        scaled_action = np.clip(action, -1, 1) * env.action_space.high
                        observation_, reward, done, info = env.step(scaled_action)
                        score += reward
                        observation = observation_

                        if trial_idx == 0:
                            cell_trajectory['trailer_x'].append(env.state[4])
                            cell_trajectory['trailer_y'].append(env.state[5])

                    is_success = info.get('success', False) or env.goal_reached
                    final_state_x, final_state_y = env.state[4], env.state[5]

                    # Determine DDPG violation type
                    if is_success: violation_type = 'success'
                    elif env.jackknife: violation_type = 'jackknife'
                    elif env.out_of_map: violation_type = 'out_of_map'
                    elif env.goal_passed: violation_type = 'goal_passed'
                    elif env.max_steps_reached: violation_type = 'max_steps'
                    else: violation_type = 'other_failure'

                elif method == 'npc':
                    # --- NPC SOLVER EXECUTION ---
                    initial_pose_npc = np.array([start_x, start_y, start_yaw_rad, 0]) # x,y,theta,beta
                    target_pose_npc = np.array([GOAL_POSE['x'], GOAL_POSE['y'], np.deg2rad(GOAL_POSE['yaw_deg']), 0])
                    map_boundaries = (MAP_X_RANGE, MAP_Y_RANGE)

                    npc_success, X_opt, _ = solve_with_npc(initial_pose_npc, target_pose_npc, map_boundaries, NPC_PARAMS)

                    is_success = npc_success
                    score = 1.0 if is_success else 0.0 # Binary score for NPC

                    if is_success:
                        violation_type = 'success'
                        final_state_x, final_state_y = X_opt[0, -1], X_opt[1, -1]
                        if trial_idx == 0:
                            cell_trajectory = {'trailer_x': X_opt[0, :].tolist(), 'trailer_y': X_opt[1, :].tolist(),
                                               'start_x': start_x, 'start_y': start_y, 'start_yaw_deg': start_yaw_deg}
                    else:
                        violation_type = 'npc_failed_to_solve'
                        final_state_x, final_state_y = start_x, start_y # Endpoint is the start point on failure

                # --- Record results for this trial ---
                cell_scores.append(score)
                cell_successes.append(1 if is_success else 0)
                ii += 1
                print({ii})

                if trial_idx == 0 and cell_trajectory is not None:
                    cell_trajectory['success'] = is_success

                trajectory_endpoints.append({
                    'end_x': final_state_x, 'end_y': final_state_y,
                    'start_x': start_x, 'start_y': start_y,
                    'violation_type': violation_type, 'score': score
                })

            # Aggregate results for the cell
            if cell_scores:
                reward_grid[iy, ix] = np.mean(cell_scores)
                success_grid[iy, ix] = np.mean(cell_successes)
                if trial_idx == (TRIALS_PER_CELL - 1): # Extend after all trials for a cell
                    orientations_data.extend(cell_orientations)
                if cell_trajectory is not None:
                    trajectories_data.append(cell_trajectory)

    return reward_grid, success_grid, x_coords, y_coords, orientations_data, trajectory_endpoints, trajectories_data

def plot_trajectories(trajectories_data, filename):
    # (This function remains unchanged)
    if not trajectories_data:
        print("No trajectory data available.")
        return
    print("Generating trajectories plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    successful_count = 0
    failed_count = 0
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
        ax.plot(traj['trailer_x'], traj['trailer_y'],
                color=color, alpha=alpha, linewidth=linewidth)
        ax.plot(traj['trailer_x'][0], traj['trailer_y'][0],
                'o', color=color, markersize=4, markeredgecolor='black',
                markeredgewidth=0.5, alpha=alpha)
    ax.plot(GOAL_POSE['x'], GOAL_POSE['y'], 'gold', marker='*', markersize=15,
            markeredgecolor='black', linewidth=2, label='Goal')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, alpha=0.7, label=f'Successful ({successful_count})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.6, label=f'Failed ({failed_count})'),
        Line2D([0], [0], marker='o', color='black', markersize=4, linewidth=0, label='Start points')]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.set_title('Trajectories - One per Grid Cell')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
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
        'other_failure': 'brown',
        'npc_failed_to_solve': 'cyan' # Added color for NPC failure
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
    # (This function can remain mostly unchanged, but check for TRIALS_PER_CELL)
    valid_data = grid_data[~np.isnan(grid_data)]
    if len(valid_data) == 0: return

    print(f"Generating success rate distribution bar graph: {title}")

    # Make bins dynamic based on TRIALS_PER_CELL
    num_bins = TRIALS_PER_CELL + 1
    bins = np.linspace(-0.01, 1.01, num_bins + 1)
    counts, _ = np.histogram(valid_data, bins=bins)

    success_labels = [f'{i/TRIALS_PER_CELL:.0%}' for i in range(num_bins)]
    success_rates = np.linspace(0, 1, num_bins)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(success_rates)
    bars = ax.bar(success_labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    max_count = max(counts) if len(counts) > 0 else 1
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, count + max_count * 0.01, str(int(count)), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title(f'Success Rate Distribution\n({len(valid_data)} grid cells total)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('Number of Grid Cells', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    stats_text = f'Total Cells: {len(valid_data)}\n' \
                 f'Mean Success: {np.mean(valid_data):.1%}\n' \
                 f'Cells with 100% Success: {counts[-1]}\n' \
                 f'Cells with 0% Success: {counts[0]}\n' \
                 f'Cells with Partial Success: {sum(counts[1:-1])}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Success rate distribution bar graph saved to {output_path}")


def plot_heatmap(grid_data, x_coords, y_coords, orientations_data, title, cmap, filename):
    # (This function remains unchanged)
    if np.all(np.isnan(grid_data)):
        print(f"Warning: All data for '{title}' is NaN. Cannot generate heatmap.")
        return
    print(f"Generating heatmap: {title}")
    fig, ax = plt.subplots(figsize=(12, 12))
    x_mesh, y_mesh = np.append(x_coords, x_coords[-1] + GRID_RESOLUTION), np.append(y_coords, y_coords[-1] + GRID_RESOLUTION)
    mesh = ax.pcolormesh(x_mesh, y_mesh, grid_data, cmap=cmap, shading='auto', vmin=np.nanmin(grid_data), vmax=np.nanmax(grid_data))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(title.split('(')[0].strip())
    if orientations_data:
        arrow_length = GRID_RESOLUTION * 0.25
        for orientation in orientations_data:
            x_pos, y_pos, yaw_rad = orientation['x'], orientation['y'], orientation['yaw_rad']
            dx, dy = arrow_length * np.cos(yaw_rad), arrow_length * np.sin(yaw_rad)
            ax.arrow(x_pos, y_pos, dx, dy, head_width=GRID_RESOLUTION*0.08, head_length=GRID_RESOLUTION*0.08, fc='yellow', ec='black', linewidth=0.8, alpha=0.8)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        valid_mask = ~np.isnan(grid_data)
        if np.any(valid_mask):
            ax.scatter(X_grid[valid_mask], Y_grid[valid_mask], c='white', s=6, marker='o', alpha=0.9, edgecolors='black', linewidths=0.5, label='Grid Centers')
    ax.plot(GOAL_POSE['x'], GOAL_POSE['y'], 'w*', markersize=18, markeredgecolor='black', linewidth=2, label='Goal Pose')
    goal_arrow_len = 6
    ax.arrow(GOAL_POSE['x'], GOAL_POSE['y'],
             goal_arrow_len * np.cos(np.deg2rad(GOAL_POSE['yaw_deg'])),
             goal_arrow_len * np.sin(np.deg2rad(GOAL_POSE['yaw_deg'])),
             head_width=2, head_length=2, fc='white', ec='black', linewidth=2)
    ax.set_title(f'Agent Performance Heatmap: {title}\n(Yellow arrows show exact start orientations)', fontsize=14)
    ax.set_xlabel('Start X-coordinate (m)')
    ax.set_ylabel('Start Y-coordinate (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('gray')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white', alpha=0.3)
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


def main():
    """
    Main function to initialize, generate data, and plot heatmaps for either DDPG or NPC.
    """
    parser = argparse.ArgumentParser(description="Generate heatmaps for truck-trailer controllers.")
    parser.add_argument('--method', type=str, default='ddpg', choices=['ddpg', 'npc'],
                        help="The control method to test ('ddpg' or 'npc').")
    args = parser.parse_args()
    method = args.method

    if SEED is not None:
        set_seed(SEED)

    # --- Initialize based on method ---
    agent = None
    env = None
    if method == 'ddpg':
        print("--- Initializing DDPG Agent and Environment ---")
        env = Truck_trailer_Env_2()
        agent = Agent(alpha=0.0001, beta=0.001,
                      input_dims=env.observation_space.shape, tau=0.001,
                      batch_size=64, fc1_dims=400, fc2_dims=300,
                      n_actions=env.action_space.shape[0])
        try:
            print("📄 Loading saved models from 'tmp/ddpg/'...")
            agent.load_models()
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return
    else: # method == 'npc'
        print("--- Using NPC Solver (no DDPG agent needed) ---")
        # Env is still useful for some parameters if needed, but not essential for NPC logic
        env = Truck_trailer_Env_2()

    # --- Generate Data ---
    start_time = time.time()
    data = generate_heatmap_data(method, agent, env)
    reward_data, success_data, x_coords, y_coords, orientations_data, trajectory_endpoints, trajectories_data = data
    end_time = time.time()
    print(f"\nData generation completed in {(end_time - start_time) / 3600:.2f} hours.")
    print(f"Total endpoints/trials recorded: {len(trajectory_endpoints)}")

    # --- Plot Results (with method-specific filenames) ---
    suffix = f"_{method}.png"

    plot_heatmap(reward_data, x_coords, y_coords, orientations_data,
                 'Average Episode Reward' if method == 'ddpg' else 'Success Score (1=success, 0=fail)',
                 'viridis', 'heatmap_reward' + suffix)

    plot_heatmap(success_data, x_coords, y_coords, orientations_data,
                 'Success Rate (0.0 to 1.0)',
                 'plasma', 'heatmap_success_rate' + suffix)

    plot_success_distribution(success_data,
                              'Success Rate Distribution',
                              'distribution_success_rate' + suffix)

    plot_trajectory_endpoints(trajectory_endpoints,
                              'trajectory_endpoints_by_violation' + suffix)

    plot_trajectories(trajectories_data,
                      'trajectories_per_cell' + suffix)

    print(f"\n✅ Analysis for method '{method.upper()}' complete!")

if __name__ == '__main__':
    main()