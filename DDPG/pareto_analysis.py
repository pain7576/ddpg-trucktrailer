# FILE: pareto_analysis.py

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from rich.console import Console
from rich.progress import Progress
from scipy.stats import gaussian_kde

# --- Setup Environment Path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Your Custom Classes ---
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from truck_trailer_sim.reward_functionv1 import RewardFunction

# Initialize Rich Console for better output
console = Console()

# --- Step 1: Conceptual Framework & Objective Identification ---

def group_objectives(reward_info):
    """
    Groups the detailed reward components from reward_info into
    high-level, coherent objectives. Converts all objectives to benefits
    (higher is better) for consistent analysis.
    """
    safety = reward_info.get('safety_penalty', 0)
    precision = (reward_info.get('heading_reward', 0) +
                 reward_info.get('orientation_reward', 0))
    smoothness = reward_info.get('smoothness_penalty', 0)
    efficiency = (reward_info.get('progress_reward', 0) +
                  reward_info.get('staged_success', 0) +
                  reward_info.get('exploration_bonus', 0) +
                  reward_info.get('final_success_bonus', 0) +
                  reward_info.get('backward_penalty', 0))

    return {
        'Safety': safety,
        'Precision': precision,
        'Smoothness': smoothness,
        'Efficiency': efficiency
    }

# --- Step 2: Computing the Pareto Frontier ---

def find_pareto_frontier(points):
    """
    Finds the Pareto frontier from a set of points.
    Assumes all objectives are benefits (higher is better).
    Returns a boolean mask of the same size as points.
    """
    console.print("🧠 Computing the Pareto frontier...")
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)

    with Progress() as progress:
        task = progress.add_task("[cyan]Finding dominant points...", total=num_points)
        for i in range(num_points):
            if not is_pareto[i]:
                progress.update(task, advance=1)
                continue
            for j in range(num_points):
                if i == j:
                    continue
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_pareto[i] = False
                    break
            progress.update(task, advance=1)

    num_frontier_points = np.sum(is_pareto)
    console.print(f"✅ Found [bold green]{num_frontier_points}[/] Pareto-efficient points out of {num_points}.")
    return is_pareto

# --- Step 3: Visualization and Pattern Recognition ---

def plot_pareto_analysis(objective_vectors, pareto_mask, objective_names):
    """
    Generates 2D scatter plots for pairs of objectives, highlighting the Pareto frontier
    and coloring other points by density.
    """
    console.print("📊 Generating visualizations...")
    plot_pairs = list(combinations(enumerate(objective_names), 2))
    num_plots = len(plot_pairs)
    console.print(f"   - Creating plots for all {num_plots} objective pairs...")

    for (idx1, obj1_name), (idx2, obj2_name) in plot_pairs:

        if 'Efficiency' in (obj1_name, obj2_name):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), sharex=False, sharey=False)

            if obj1_name == 'Efficiency':
                eff_idx, x_is_eff = idx1, True
            else:
                eff_idx, x_is_eff = idx2, False

            # --- Panel 1: Low Efficiency ---
            mask_low = (objective_vectors[:, eff_idx] >= 0) & (objective_vectors[:, eff_idx] <= 80)
            all_low = objective_vectors[mask_low]
            pareto_low = objective_vectors[pareto_mask & mask_low]
            x_all_low, y_all_low = all_low[:, idx1], all_low[:, idx2]
            x_pareto_low, y_pareto_low = pareto_low[:, idx1], pareto_low[:, idx2]

            if len(x_all_low) > 1:
                try:
                    points = np.vstack([x_all_low, y_all_low])
                    density = gaussian_kde(points)(points)
                    sort_idx = density.argsort()
                    x_all_low, y_all_low, density = x_all_low[sort_idx], y_all_low[sort_idx], density[sort_idx]
                    scatter = ax1.scatter(x_all_low, y_all_low, c=density, cmap='viridis', s=50, alpha=0.7, label=f'Solutions ({len(x_all_low)})')
                    cbar = fig.colorbar(scatter, ax=ax1)
                    cbar.set_label('Point Density')
                except np.linalg.LinAlgError:
                    console.print(f"⚠️  [yellow]Warning: Could not compute density for '{obj1_name} vs {obj2_name}' (low-efficiency panel). Data may be co-linear. Plotting without density coloring.[/yellow]")
                    ax1.scatter(x_all_low, y_all_low, c='dodgerblue', s=50, alpha=0.7, label=f'Solutions ({len(x_all_low)})')

            ax1.scatter(x_pareto_low, y_pareto_low, alpha=0.9, s=80, c='red', marker='o', edgecolors='k', label=f'Pareto Frontier ({len(x_pareto_low)})')
            ax1.set_title("Low-Efficiency Regime", fontsize=16, fontweight='bold')
            ax1.set_xlabel(f"{obj1_name} Objective"); ax1.set_ylabel(f"{obj2_name} Objective")
            ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend(fontsize=12)

            # --- Panel 2: High Efficiency ---
            mask_high = (objective_vectors[:, eff_idx] > 80) & (objective_vectors[:, eff_idx] <= 350)
            all_high = objective_vectors[mask_high]
            pareto_high = objective_vectors[pareto_mask & mask_high]
            x_all_high, y_all_high = all_high[:, idx1], all_high[:, idx2]
            x_pareto_high, y_pareto_high = pareto_high[:, idx1], pareto_high[:, idx2]

            if len(x_all_high) > 1:
                try:
                    points = np.vstack([x_all_high, y_all_high])
                    density = gaussian_kde(points)(points)
                    sort_idx = density.argsort()
                    x_all_high, y_all_high, density = x_all_high[sort_idx], y_all_high[sort_idx], density[sort_idx]
                    scatter = ax2.scatter(x_all_high, y_all_high, c=density, cmap='viridis', s=50, alpha=0.7, label=f'Solutions ({len(x_all_high)})')
                    cbar = fig.colorbar(scatter, ax=ax2)
                    cbar.set_label('Point Density')
                except np.linalg.LinAlgError:
                    console.print(f"⚠️  [yellow]Warning: Could not compute density for '{obj1_name} vs {obj2_name}' (high-efficiency panel). Data may be co-linear. Plotting without density coloring.[/yellow]")
                    ax2.scatter(x_all_high, y_all_high, c='dodgerblue', s=50, alpha=0.7, label=f'Solutions ({len(x_all_high)})')

            ax2.scatter(x_pareto_high, y_pareto_high, alpha=0.9, s=80, c='red', marker='o', edgecolors='k', label=f'Pareto Frontier ({len(x_pareto_high)})')
            ax2.set_title("High-Efficiency (Successful) Regime", fontsize=16, fontweight='bold')
            ax2.set_xlabel(f"{obj1_name} Objective")
            ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend(fontsize=12)

            if x_is_eff:
                ax1.set_xlim(0, 80); ax2.set_xlim(80, 350)
            else:
                ax1.set_ylim(0, 80); ax2.set_ylim(80, 350)

            fig.suptitle(f"Pareto Frontier: {obj1_name} vs. {obj2_name}", fontsize=18, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96])

        else:
            # --- Standard plots for non-Efficiency pairs ---
            fig, ax = plt.subplots(figsize=(10, 8))
            x_all, y_all = objective_vectors[:, idx1], objective_vectors[:, idx2]
            x_pareto, y_pareto = objective_vectors[pareto_mask, idx1], objective_vectors[pareto_mask, idx2]

            if len(x_all) > 1:
                try:
                    points = np.vstack([x_all, y_all])
                    density = gaussian_kde(points)(points)
                    sort_idx = density.argsort()
                    x_all, y_all, density = x_all[sort_idx], y_all[sort_idx], density[sort_idx]
                    scatter = ax.scatter(x_all, y_all, c=density, cmap='viridis', s=50, alpha=0.7, label=f'Solutions ({len(x_all)})')
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Point Density')
                except np.linalg.LinAlgError:
                    console.print(f"⚠️  [yellow]Warning: Could not compute density for '{obj1_name} vs {obj2_name}'. Data may be co-linear. Plotting without density coloring.[/yellow]")
                    ax.scatter(x_all, y_all, c='dodgerblue', s=50, alpha=0.7, label=f'Solutions ({len(x_all)})')

            ax.scatter(x_pareto, y_pareto, alpha=0.9, s=80, c='red', marker='o', edgecolors='k', label=f'Pareto Frontier ({len(x_pareto)})')
            ax.set_xlabel(f"{obj1_name} Objective (Higher is Better)"); ax.set_ylabel(f"{obj2_name} Objective (Higher is Better)")
            ax.set_title(f"Pareto Frontier: {obj1_name} vs. {obj2_name}", fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize=12)
            plt.tight_layout()

    plt.show()

def main():
    """Main execution function."""
    console.print("=====================================================", style="bold cyan")
    console.print("== 🚛 TRUCK-TRAILER PARETO FRONTIER ANALYSIS TOOL 🚛 ==", style="bold cyan")
    console.print("=====================================================\n", style="bold cyan")

    env = Truck_trailer_Env_2()
    objective_vectors = []
    objective_names = list(group_objectives({}).keys())

    replay_dir = 'episode_replays'
    if not os.path.exists(replay_dir) or not os.listdir(replay_dir):
        console.print(f"❌ [bold red]Error: No replay files found in the '{replay_dir}' directory.[/]")
        console.print("Please run the training script (`trainv2.py`) to generate replay data.")
        return

    all_files_in_dir = os.listdir(replay_dir)
    parsed_files = []

    console.print(f"🔍 Scanning [bold cyan]{len(all_files_in_dir)}[/] items in '{replay_dir}' directory...")
    for filename in all_files_in_dir:
        if filename.endswith('.pkl'):
            try:
                parts = filename.split('_')
                episode_num = int(parts[1])
                full_path = os.path.join(replay_dir, filename)
                parsed_files.append((episode_num, full_path))
            except (ValueError, IndexError):
                console.print(f"⚠️  [yellow]Skipping malformed file:[/yellow] {filename}")
                continue

    parsed_files.sort(key=lambda item: item[0], reverse=True)
    limit = 50 # Reduced for faster debugging, you can set it back to 400
    replay_files_to_process = [item[1] for item in parsed_files[:limit]]

    if not replay_files_to_process:
        console.print("❌ [bold red]No valid replay files found after parsing. Exiting.[/]")
        return

    if len(replay_files_to_process) < limit:
        console.print(f"Found fewer than {limit} valid replay files. Analyzing all [bold green]{len(replay_files_to_process)}[/] available episodes.")
    else:
        console.print(f"Selected the [bold green]last {len(replay_files_to_process)}[/] episodes for analysis.")

    with Progress() as progress:
        task = progress.add_task("[magenta]Processing replay files...", total=len(replay_files_to_process))

        for file_path in replay_files_to_process:
            try:
                with open(file_path, 'rb') as f:
                    episode_data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                console.print(f"⚠️ [yellow]Warning: Could not read file {os.path.basename(file_path)}. Skipping.[/]")
                progress.update(task, advance=1)
                continue

            env_data = episode_data.get('env_data')
            if env_data:
                env.startx, env.starty, env.startyaw = env_data['startx'], env_data['starty'], env_data['startyaw']
                env.goalx, env.goaly, env.goalyaw = env_data['goalx'], env_data['goaly'], env_data['goalyaw']
            else:
                env.reset()

            states = episode_data['states']
            actions = episode_data['actions']
            reward_state = None

            for i in range(len(actions)):
                current_state = states[i + 1]
                current_action = actions[i][0] if isinstance(actions[i], (list, np.ndarray)) else actions[i]
                observation = env.compute_observation(current_state, current_action)

                reward_function = RewardFunction(
                    observation, current_state, episode_steps=i + 1,
                    position_threshold=env.position_threshold, orientation_threshold=env.orientation_threshold,
                    goalx=env.goalx, goaly=env.goaly, startx=env.startx, starty=env.starty,
                    **(reward_state or {})
                )
                _, reward_info = reward_function.compute_reward()

                reward_state = reward_function.get_persistent_state()
                grouped_obj = group_objectives(reward_info)
                objective_vectors.append(list(grouped_obj.values()))

            progress.update(task, advance=1)

    if not objective_vectors:
        console.print("\n❌ [bold red]No valid data points were generated for analysis. Exiting.[/]")
        return

    console.print(f"\n🔎 [bold]Total data points for analysis: {len(objective_vectors)}[/]")
    objective_vectors_np = np.array(objective_vectors)

    pareto_mask = find_pareto_frontier(objective_vectors_np)
    plot_pareto_analysis(objective_vectors_np, pareto_mask, objective_names)


if __name__ == '__main__':
    main()