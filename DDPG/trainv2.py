import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import gym
import os
import numpy as np
from DDPG_agent import Agent
from ploting_utils.plot_learning_curve import Plot_learning_curve
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from episode_replay_collector import EpisodeReplaySystem
from collections import deque
from seed_utils import set_seed

# Rich library imports
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.columns import Columns
import time
import keyboard
# Initialize rich console
console = Console()


class RichCLI:
    def __init__(self):
        self.console = Console()

    def display_banner(self):
        """Display a beautiful banner"""
        banner = Text("🚛 DDPG Truck-Trailer Training System", style="bold magenta")
        panel = Panel(
            Align.center(banner),
            box=box.DOUBLE,
            style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

    def display_parameter_table(self):
        """Display parameter descriptions"""
        param_table = Table(title="Parameter Descriptions", box=box.ROUNDED)
        param_table.add_column("Parameter", style="cyan", no_wrap=True)
        param_table.add_column("Description", style="white")
        param_table.add_column("Typical Range", style="yellow")

        param_table.add_row("alpha", "Actor learning rate", "0.0001 - 0.001")
        param_table.add_row("beta", "Critic learning rate", "0.001 - 0.01")
        param_table.add_row("tau", "Soft update parameter", "0.001 - 0.01")
        param_table.add_row("batch_size", "Training batch size", "32 - 128")
        param_table.add_row("fc1_dims", "First layer dimensions", "256 - 512")
        param_table.add_row("fc2_dims", "Second layer dimensions", "128 - 400")
        param_table.add_row("games", "Number of episodes", "100 - 2000")

        self.console.print(param_table)
        self.console.print()

    def display_default_parameters(self):
        """Display default parameters in a nice table"""
        self.console.print("📋 [bold green]Using Default Parameters[/bold green]")

        param_table = Table(title="Default Configuration", box=box.ROUNDED)
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="green")

        param_table.add_row("alpha", "0.0001")
        param_table.add_row("beta", "0.001")
        param_table.add_row("tau", "0.001")
        param_table.add_row("batch_size", "64")
        param_table.add_row("fc1_dims", "400")
        param_table.add_row("fc2_dims", "300")
        param_table.add_row("games", "200")

        self.console.print(param_table)

    def display_training_info(self, agent, n_games, filename):
        """Display training configuration"""
        self.console.print("🚀 [bold green]Training Configuration[/bold green]")

        config_table = Table(title="Current Setup", box=box.ROUNDED)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")

        config_table.add_row("Filename", filename)
        config_table.add_row("Episodes", str(n_games))
        config_table.add_row("Alpha", str(agent.alpha))
        config_table.add_row("Beta", str(agent.beta))
        config_table.add_row("Tau", str(agent.tau))
        config_table.add_row("Batch Size", str(agent.batch_size))

        self.console.print(config_table)

    def display_episode_result(self, episode, score, avg_score, best_score, best_success_rate, success, total_steps,is_best=False):
        """Display episode results with beautiful formatting"""
        # Colorize success: red if "FAILURE", else green
        success_display = (
            f"[red]{success}[/red]" if success == "FAILURE" else f"[green]{success}[/green]"
        )

        if is_best:
            self.console.print(
                f"🏆[bold green]Episode {episode:4d}[/bold green]|"
                f"Score:[yellow]{score:8.1f}[/yellow]|"
                f"Avg:[cyan]{avg_score:8.1f}[/cyan]|"
                f"goal:{success_display}|"
                f"rate:[green]{best_success_rate:8.3f}[/green]|"
                f"Best:[green]{best_score:8.1f}[/green]|"
                f"Steps:[magenta]{total_steps}[/magenta][bold red]NEW BEST![/bold red]|"
            )
        else:
            self.console.print(
                f"📈[bold white]Episode {episode:4d}[/bold white]|"
                f"Score:[yellow]{score:8.1f}[/yellow]|"
                f"Avg:[cyan]{avg_score:8.1f}[/cyan]|"
                f"goal:{success_display}|"
                f"rate:[green]{best_success_rate:8.3f}[/green]|"
                f"Best:[green]{best_score:8.1f}[/green]|"
                f"Steps:[magenta]{total_steps}[/magenta]"
            )


    def display_message(self, message, style="white"):
        """Display a styled message"""
        self.console.print(f"[{style}]{message}[/{style}]")

    def display_success(self, message):
        """Display success messages"""
        self.console.print(f"✅ [bold green]{message}[/bold green]")

    def display_error(self, message):
        """Display error messages"""
        self.console.print(f"❌ [bold red]{message}[/bold red]")

    def display_warning(self, message):
        """Display warning messages"""
        self.console.print(f"⚠️  [yellow]{message}[/yellow]")

    def display_info(self, message):
        """Display info messages"""
        self.console.print(f"ℹ️  [blue]{message}[/blue]")


# Initialize CLI - keeping this global for compatibility
cli = RichCLI()

def prompt_load_replaybuffer():
    cli.display_message("🔄 Checkpoint Loading Options", "bold cyan")
    load_checkpoint = input("Load replay buffer? (y/n): ").strip().lower()
    if load_checkpoint == 'y':
        return True
    elif load_checkpoint == 'n':
        return False
    else:
        cli.display_warning("Invalid input. Defaulting to 'n'.")
        return False



def prompt_load_checkpoint():
    """Original function with rich enhancement"""
    cli.display_message("🔄 Checkpoint Loading Options", "bold cyan")
    load_checkpoint = input("Load checkpoint? (y/n): ").strip().lower()
    if load_checkpoint == 'y':
        return True
    elif load_checkpoint == 'n':
        return False
    else:
        cli.display_warning("Invalid input. Defaulting to 'n'.")
        return False

def prompt_load_seed():
    """Original function with rich enhancement"""
    cli.display_message("🔄 seed loading option", "bold cyan")
    load_checkpoint = input("want to load seed to reproduce result (n will result in default seed=27) ? (y/n): ").strip().lower()
    if load_checkpoint == 'y':
        return True
    elif load_checkpoint == 'n':
        return False
    else:
        cli.display_warning("Invalid input. Defaulting to 'n'.")
        return False


def prompt_set_parameters():
    """Original function with rich enhancement"""
    cli.display_message("⚙️  Parameter Configuration", "bold cyan")
    cli.display_message("Default parameters will be used if you choose 'No'", "dim")
    set_parameters = input("set parameters no will result into default parameters ? (y/n): ").strip().lower()
    if set_parameters == 'y':
        return True
    elif set_parameters == 'n':
        return False
    else:
        cli.display_warning("Invalid input. Defaulting to 'n'.")
        return False


def save_training_state(episode_num, score_history, best_score, best_success_rate, success_history, total_steps, step_history, filename):
    """Save training state for resuming later - ORIGINAL LOGIC PRESERVED"""
    training_state = {
        'episode_num': episode_num,
        'score_history': score_history,
        'best_score': best_score,
        'best_success_rate': best_success_rate,
        'success_history': success_history,
        'total_steps': total_steps,
        'step_history': step_history,
        'filename': filename
    }

    os.makedirs('training_states', exist_ok=True)
    state_file = f'training_states/{filename}_training_state.pkl'

    with open(state_file, 'wb') as f:
        pickle.dump(training_state, f)

    cli.display_success(f"Training state saved to {state_file}")


def load_training_state(filename):
    """Load training state for resuming - ORIGINAL LOGIC PRESERVED"""
    state_file = f'training_states/{filename}_training_state.pkl'

    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            training_state = pickle.load(f)

        cli.display_success(f"Training state loaded from {state_file}")
        return training_state
    else:
        cli.display_warning(f"No training state found at {state_file}")
        return None

def list_replay_buffer():
    """List available training state files and return the selected one - ORIGINAL LOGIC PRESERVED"""
    training_dir = 'replay_buffer'
    if not os.path.exists(training_dir):
        cli.display_error("No replay_buffer directory found.")
        return None

    files = [f for f in os.listdir(training_dir) if f.endswith('_replay_buffer.pkl')]
    if not files:
        cli.display_error("No training state files found.")
        return None

    cli.display_message("💾 Available Replay Buffer", "bold cyan")

    # Create a table for training states
    state_table = Table(title="Replay Buffer", box=box.ROUNDED)
    state_table.add_column("ID", style="cyan", no_wrap=True)
    state_table.add_column("Replay Buffer", style="white")
    state_table.add_column("File Size", style="yellow")

    for i, file in enumerate(files, 1):
        base_name = file.replace('_replay_buffer.pkl', '')
        file_path = os.path.join(training_dir, file)
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB"
        state_table.add_row(str(i), base_name, size_str)

    console.print(state_table)

    try:
        choice = int(input("Select a training state by number: "))
        if 1 <= choice <= len(files):
            # Return just the base name without the suffix
            selected = files[choice - 1].replace('_replay_buffer.pkl', '')
            cli.display_success(f"Selected: {selected}")
            return selected
        else:
            cli.display_error("Invalid selection.")
            return None
    except ValueError:
        cli.display_error("Invalid input. Please enter a number.")
        return None

def list_training_states():
    """List available training state files and return the selected one - ORIGINAL LOGIC PRESERVED"""
    training_dir = 'training_states'
    if not os.path.exists(training_dir):
        cli.display_error("No training_states directory found.")
        return None

    files = [f for f in os.listdir(training_dir) if f.endswith('_training_state.pkl')]
    if not files:
        cli.display_error("No training state files found.")
        return None

    cli.display_message("💾 Available Training States", "bold cyan")

    # Create a table for training states
    state_table = Table(title="Training States", box=box.ROUNDED)
    state_table.add_column("ID", style="cyan", no_wrap=True)
    state_table.add_column("Training State", style="white")
    state_table.add_column("File Size", style="yellow")

    for i, file in enumerate(files, 1):
        base_name = file.replace('_training_state.pkl', '')
        file_path = os.path.join(training_dir, file)
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB"
        state_table.add_row(str(i), base_name, size_str)

    console.print(state_table)

    try:
        choice = int(input("Select a training state by number: "))
        if 1 <= choice <= len(files):
            # Return just the base name without the suffix
            selected = files[choice - 1].replace('_training_state.pkl', '')
            cli.display_success(f"Selected: {selected}")
            return selected
        else:
            cli.display_error("Invalid selection.")
            return None
    except ValueError:
        cli.display_error("Invalid input. Please enter a number.")
        return None


def save_transitions(episode_num, transitions_history):
    """Save transition history from last 200 episodes"""
    os.makedirs('replay_buffer', exist_ok=True)
    filename = f"replay_buffer/transitions_episode_{episode_num}_replay_buffer.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(list(transitions_history), f)
    print(f"Saved transitions for last {len(transitions_history)} episodes to {filename}")

def load_transitions(filename):
    """Load transition history from file"""
    replay_file = f'replay_buffer/{filename}_replay_buffer.pkl'
    if os.path.exists(replay_file):
        with open(replay_file, 'rb') as f:
            transitions = pickle.load(f)
        print(f"Loaded {len(transitions)} episodes of transitions from {replay_file}")
        return transitions
    else:
        print(f"No transitions file found at {replay_file}")
        return None

if __name__ == '__main__':
    # Display banner
    cli.display_banner()

    # ORIGINAL LOGIC PRESERVED - just with enhanced display
    should_load = prompt_load_checkpoint()
    if should_load:
        cli.display_info("Loading checkpoint...")
    else:
        cli.display_info("Starting fresh...")

    should_load_replaybuffer = prompt_load_replaybuffer()
    if should_load_replaybuffer:
        cli.display_info("Loading Replay Buffer...")
    else:
        cli.display_info("Starting fresh ono reply buffer (empty)...")

    should_load_seed = prompt_load_seed()
    if should_load_seed:
        cli.display_info("Loading seed...")
        seed = int(input("Enter seed for reproducibility: "))
    else:
        cli.display_info("Starting default seed...")
        seed = 27

    set_seed(seed)

    env = Truck_trailer_Env_2()
    replay_system = EpisodeReplaySystem(env)

    parameter_load = prompt_set_parameters()
    if parameter_load:
        # Show parameter descriptions before asking for input
        cli.display_parameter_table()

        # ORIGINAL INPUT LOGIC PRESERVED
        alpha = float(input("Enter alpha value: "))
        beta = float(input("Enter beta value: "))
        tau = float(input("Enter tau value: "))
        batch_size = int(input("Enter batch size: "))
        fc1_dims = int(input("Enter fc1_dims value: "))
        fc2_dims = int(input("Enter fc2_dims value: "))
        games = int(input("Enter number of games: "))

        agent = Agent(alpha=alpha, beta=beta,
                      input_dims=env.observation_space.shape, tau=tau,
                      batch_size=batch_size, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                      n_actions=env.action_space.shape[0])
        n_games = games
    else:
        cli.display_default_parameters()
        agent = Agent(alpha=0.0001, beta=0.001,
                      input_dims=env.observation_space.shape, tau=0.001,
                      batch_size=64, fc1_dims=400, fc2_dims=300,
                      n_actions=env.action_space.shape[0])
        n_games = 5000


    # ORIGINAL FILENAME GENERATION PRESERVED
    filename = 'truck_trailer_v1' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    # Display training configuration
    cli.display_training_info(agent, n_games, filename)

    # ORIGINAL LOGIC PRESERVED
    best_score = env.reward_range[0]
    best_success_rate = 0.0
    score_history = []
    success_history = []
    step_history = []
    start_episode = 0
    resume_episode = start_episode
    total_steps = 0

    if should_load:
        additional_episode = int(input("Enter number of additional episodes: "))
        try:
            agent.load_models()
            cli.display_success("Pre-trained model loaded successfully!")

            load_filename = list_training_states()

            training_state = load_training_state(load_filename)
            if training_state is not None:
                score_history = training_state['score_history']
                #best_score = training_state['best_score']
                resume_episode = training_state['episode_num']
                #best_success_rate = training_state['best_success_rate']
                success_history = training_state['success_history']
                #step_history = training_state['step_history']
                total_steps = training_state.get('total_steps', 0)
                cli.display_info(f"Resuming from episode {resume_episode}")
                #cli.display_info(f"Previous best score: {best_score:.2f}")
                #cli.display_info(f"Previous best success rate: {best_success_rate:.3f}")
                cli.display_info(f"Previous score history length: {len(score_history)}")
                cli.display_info(f"Previous total steps: {total_steps}")

        except Exception as e:
            cli.display_error(f"Error loading pre-trained model: {e}")
            cli.display_warning("Starting new fresh training run...")
            resume_episode = 0

    if should_load_replaybuffer:
        load_filename = list_replay_buffer()
        stored_transitions = load_transitions(load_filename)
        if stored_transitions:
            cli.display_info(f"Loading transitions")
            for episode_transitions in stored_transitions:
                for transition in episode_transitions:
                    obs, action, reward, obs_next, done = transition
                    agent.remember(obs, action, reward, obs_next, done)
            cli.display_success(f"Loaded {agent.memory.mem_cntr} transitions into replay buffer")
        else:
            cli.display_warning("No stored transition files found")

    # ORIGINAL EPISODE CALCULATION PRESERVED
    if should_load:
        # Total episodes to train = resume point + additional episodes
        start_episode = resume_episode
        end_episode = resume_episode + additional_episode
    else:
        # Start from scratch
        start_episode = 0
        end_episode = n_games

    progress = False

    # Enhanced training loop with progress tracking
    cli.display_message(f"🎯 Starting Training: Episodes {start_episode} to {end_episode - 1}", "bold cyan")
    cli.display_info("Press the 'ESC' key to stop training early and generate the plot.")

    episode_transitions_history = deque(maxlen=200)
    # ORIGINAL TRAINING LOOP LOGIC COMPLETELY PRESERVED
    for i in range(start_episode, end_episode):
        observation, info = env.reset(seed=seed + i)
        done = False
        score = 0
        agent.noise.reset()

        episode_states = []
        episode_actions = []
        episode_reward = []
        episode_transitions = []

        episode_states.append(env.state.copy())  # save initial state

        # Save environment data for replay
        env_data = {
            'startx': env.startx,
            'starty': env.starty,
            'startyaw': env.startyaw,
            'goalx': env.goalx,
            'goaly': env.goaly,
            'goalyaw': env.goalyaw
        }
        counter = 0
        while not done:
            action = agent.choose_action(observation)
            # The actor output is in [-1, 1]. Scale it to the environment's action space.
            # The action space is symmetric, so we can just multiply by the high bound.
            # We also clip the action in case the noise pushes it outside the [-1, 1] range.
            scaled_action = np.clip(action, -1, 1) * env.action_space.high

            episode_actions.append(scaled_action)  # save the actions

            observation_, reward, done, info = env.step(scaled_action)

            episode_states.append(env.state.copy())  # save the resulting state
            episode_reward.append(info.copy())

            agent.remember(observation, action, reward, observation_, done)
            transition = (observation.copy(), action.copy(), reward, observation_.copy(), done)
            episode_transitions.append(transition)
            agent.learn()
            score += reward
            observation = observation_
            total_steps += 1

        episode_transitions_history.append(episode_transitions)

        # Save the data of complete episode with environment data
        replay_system.save_episode(i, episode_states, episode_actions, episode_reward, env_data)

        score_history.append(score)
        step_history.append(total_steps)
        avg_score = np.mean(score_history[-100:])

        if episode_reward[-1]['final_success_bonus'] > 0:
            status = "SUCCESS"
            counter = counter + 1
        else:
            status = "FAILURE"


        success_history.append(1 if status == "SUCCESS" else 0)
        success_rate = np.mean(success_history[-100:])

        if progress:
            current_success_percent = int(success_rate * 100)
            if int(current_success_percent) in [25, 50, 75]:
                # Pass the integer percentage to the save function
                agent.save_models_progress(int(current_success_percent))


        is_better_success = success_rate > best_success_rate
        is_equal_success_better_score = success_rate == best_success_rate and avg_score > best_score
        is_best = (is_better_success or is_equal_success_better_score) and i > (start_episode + 100)




        # Check if this is a new best score
        if is_best:
            agent.save_models()
            save_training_state(i + 1, score_history, best_score, best_success_rate, success_history, total_steps, step_history, filename)
            save_transitions(i, episode_transitions_history)
            best_success_rate = success_rate
            best_score = avg_score


        # Enhanced display but original print logic preserved
        cli.display_episode_result(i, score, avg_score, best_score, best_success_rate, status, total_steps, is_best)

        try:
            if keyboard.is_pressed('escape'):
                cli.display_warning("\nEscape key detected. Halting training and proceeding to plot generation...")
                break # Exit the for loop
        except ImportError:
            # Handle case where keyboard library is not installed or has issues
            if i == start_episode: # Only show this warning once
                cli.display_warning("Could not check for ESC key. Please install the 'keyboard' library (`pip install keyboard`) to enable this feature.")
        except Exception as e:
            # Handle other potential errors, e.g., permissions on Linux
            if i == start_episode: # Only show this warning once
                cli.display_warning(f"Could not check for ESC key. The 'keyboard' library may require special permissions (e.g., 'sudo' on Linux).")


    # ORIGINAL PLOT GENERATION PRESERVED
    x = [i + 1 for i in range(len(score_history))]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    Plot_learning_curve(x, score_history, success_history, figure_file)

    # Final summary
    cli.display_success("Training Complete!")
    cli.display_info(f"Plot saved to {figure_file}")
    cli.display_info(f"Final best score: {best_score:.2f}")
    cli.display_info(f"Total steps taken: {total_steps}")