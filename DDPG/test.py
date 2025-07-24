# test_agent.py (with interactive prompts and replay saving)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from DDPG_agent import Agent
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from seed_utils import set_seed
from episode_replay_collectorv2 import EpisodeReplaySystem

# Rich library imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

def main():
    """
    Main function to run the DDPG agent in test mode via interactive prompts.
    """
    console = Console()

    # --- Display Banner ---
    banner = Text("🚛 DDPG Agent Inference Test", style="bold magenta")
    panel = Panel(Align.center(banner), box=box.DOUBLE, style="cyan", padding=(1, 2))
    console.print(panel)

    # --- Initialize Environment, Agent, and Replay System ---
    env = Truck_trailer_Env_2()
    replay_system = EpisodeReplaySystem(env, save_dir='test_episode_replays') # <-- Save to a separate directory

    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=env.action_space.shape[0])

    # --- Load Saved Models ---
    try:
        console.print("🔄 [bold]Loading saved models from 'tmp/ddpg/'...[/bold]")
        agent.load_models()
        console.print("✅ [bold green]Models loaded successfully![/bold green]\n")
    except Exception as e:
        console.print(f"❌ [bold red]Error loading models: {e}[/bold red]")
        return

    # --- Interactive Mode Selection ---
    test_mode = Prompt.ask(
        "Choose a test mode",
        choices=["random", "specific"],
        default="random"
    )

    episodes_to_run = 1
    use_custom_pose = False
    custom_pose_data = {}
    seed = 42

    if test_mode == "random":
        console.print("\n[bold cyan]--- Random Pose Testing ---[/bold cyan]")
        episodes_to_run = IntPrompt.ask("Enter the number of episodes to run", default=5)
        seed = IntPrompt.ask("Enter the random seed for reproducibility", default=42)
        set_seed(seed)
        console.print(f"🚀 [bold]Running {episodes_to_run} test episodes with random start poses...[/bold]\n")

    elif test_mode == "specific":
        console.print("\n[bold yellow]--- Specific Start Pose Testing ---[/bold yellow]")
        use_custom_pose = True
        custom_pose_data['x'] = FloatPrompt.ask("Enter starting X-coordinate for the trailer")
        custom_pose_data['y'] = FloatPrompt.ask("Enter starting Y-coordinate for the trailer")
        custom_pose_data['yaw'] = FloatPrompt.ask("Enter starting yaw/orientation for the trailer (in degrees)")
        console.print(f"🚀 [bold]Running 1 test episode with the specified start pose...[/bold]\n")

    # --- Run Test Episodes ---
    scores = []
    success_count = 0

    for i in range(episodes_to_run):
        # --- Data collection for replay ---
        episode_states = []
        episode_actions = []
        episode_reward_info = []

        seed_for_reset = seed + i if test_mode == "random" else None
        observation, _ = env.reset(seed=seed_for_reset)

        # Save initial state and environment data for replay
        episode_states.append(env.state.copy())
        env_data = {'startx': env.startx, 'starty': env.starty, 'startyaw': env.startyaw,
                    'goalx': env.goalx, 'goaly': env.goaly, 'goalyaw': env.goalyaw}

        if use_custom_pose:
            start_x = custom_pose_data['x']
            start_y = custom_pose_data['y']
            start_yaw_rad = np.deg2rad(custom_pose_data['yaw'])

            env.startx, env.starty, env.startyaw = start_x, start_y, start_yaw_rad
            env_data['startx'], env_data['starty'], env_data['startyaw'] = start_x, start_y, start_yaw_rad

            env.max_episode_steps = env.compute_max_steps()

            psi_2, x2, y2 = start_yaw_rad, start_x, start_y
            psi_1 = start_yaw_rad
            x1 = start_x + env.L2 * np.cos(start_yaw_rad)
            y1 = start_y + env.L2 * np.sin(start_yaw_rad)

            env.state = np.array([psi_1, psi_2, x1, y1, x2, y2], dtype=np.float32)
            observation = env.compute_observation(env.state, steering_angle=0.0)

            # Replace the first state with the custom one
            episode_states[0] = env.state.copy()

        done = False
        score = 0
        step = 0

        while not done:
            env.render()
            action = agent.choose_action(observation, evaluate=True)
            scaled_action = np.clip(action, -1, 1) * env.action_space.high

            observation_, reward, done, info = env.step(scaled_action)

            # Record data for this step for potential replay
            episode_actions.append(scaled_action)
            episode_states.append(env.state.copy())
            episode_reward_info.append(info)

            score += reward
            observation = observation_
            step += 1
            time.sleep(0.05)

        scores.append(score)
        is_success = info.get('final_success_bonus', 0) > 0
        if is_success:
            success_count += 1

        console.print(
            f"🔹 [white]Episode {i+1:02d}[/white] | "
            f"Score: [yellow]{score:8.1f}[/yellow] | "
            f"Steps: [cyan]{step:03d}[/cyan] | "
            f"Success: {'[bold green]Yes[/bold green]' if is_success else '[bold red]No[/bold red]'}"
        )

        # --- Ask user if they want to save the replay ---
        if Confirm.ask("\n💾 Do you want to save this episode replay?", default=False):
            # Create a descriptive filename
            status_str = "SUCCESS" if is_success else "FAILURE"
            if use_custom_pose:
                # Filename for specific pose tests
                sx, sy, syaw = custom_pose_data['x'], custom_pose_data['y'], custom_pose_data['yaw']
                episode_num_str = f"test_pose_{int(sx)}_{int(sy)}_{int(syaw)}_{status_str}"
            else:
                # Filename for random pose tests
                episode_num_str = f"test_random_e{i+1}_{status_str}"

            # The replay system's save_episode needs an 'episode_num' and total reward
            # We use our descriptive string as the 'episode_num' and the score as the reward
            replay_system.save_episode(
                episode_num=episode_num_str,
                states=episode_states,
                actions=episode_actions,
                info=episode_reward_info,
                env_data=env_data
            )
            console.print(f"✅ Replay saved to '[cyan]{replay_system.save_dir}[/cyan]' directory.\n")

        time.sleep(1)

    env.close()

    # --- Display Final Results ---
    avg_score = np.mean(scores) if scores else 0
    success_rate = (success_count / episodes_to_run) * 100 if episodes_to_run > 0 else 0

    results_table = Table(title="Inference Test Summary", box=box.ROUNDED, show_header=True, header_style="bold blue")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    results_table.add_row("Total Episodes", str(episodes_to_run))
    results_table.add_row("Average Score", f"{avg_score:.2f}")
    results_table.add_row("Successes", f"{success_count}/{episodes_to_run}")
    results_table.add_row("Success Rate", f"{success_rate:.2f}%")

    console.print("\n")
    console.print(results_table)
    console.print("\n✅ [bold green]Testing complete![/bold green]")

if __name__ == '__main__':
    main()