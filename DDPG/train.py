import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import gym
import os
import numpy as np
from DDPG_agent import Agent
from ploting_utils.plot_learning_curve import Plot_learning_curve
from truck_trailer_sim.simv1 import Truck_trailer_Env_1
from episode_replay_collector import EpisodeReplaySystem

if __name__ == '__main__':
    def prompt_load_checkpoint():
        load_checkpoint = input("Load checkpoint? (y/n): ").strip().lower()
        if load_checkpoint == 'y':
            return True
        elif load_checkpoint == 'n':
            return False
        else:
            print("Invalid input. Defaulting to 'n'.")
            return False

    def prompt_set_parameters():
        set_parameters = input("set parameters no will result into default parameters ? (y/n): ").strip().lower()
        if set_parameters == 'y':
            return True
        elif set_parameters == 'n':
            return False
        else:
            print("Invalid input. Defaulting to 'n'.")
            return False


    def save_training_state(episode_num, score_history, best_score, filename):
        """Save training state for resuming later"""
        training_state = {
            'episode_num': episode_num,
            'score_history': score_history,
            'best_score': best_score,
            'filename': filename
        }

        os.makedirs('training_states', exist_ok=True)
        state_file = f'training_states/{filename}_training_state.pkl'

        with open(state_file, 'wb') as f:
            pickle.dump(training_state, f)

        print(f"Training state saved to {state_file}")


    def load_training_state(filename):
        """Load training state for resuming"""
        state_file = f'training_states/{filename}_training_state.pkl'

        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                training_state = pickle.load(f)

            print(f"Training state loaded from {state_file}")
            return training_state
        else:
            print(f"No training state found at {state_file}")
            return None


    def list_training_states():
        """List available training state files and return the selected one"""
        training_dir = 'training_states'
        if not os.path.exists(training_dir):
            print("No training_states directory found.")
            return None

        files = [f for f in os.listdir(training_dir) if f.endswith('_training_state.pkl')]
        if not files:
            print("No training state files found.")
            return None

        print("\nAvailable training states:")
        for i, file in enumerate(files, 1):
            base_name = file.replace('_training_state.pkl', '')
            print(f"{i}. {base_name}")

        try:
            choice = int(input("Select a training state by number: "))
            if 1 <= choice <= len(files):
                # Return just the base name without the suffix
                return files[choice - 1].replace('_training_state.pkl', '')
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None


    should_load = prompt_load_checkpoint()
    if should_load:
        print("Loading checkpoint...")
    else:
        print("Starting fresh...")

    env = Truck_trailer_Env_1()
    replay_system = EpisodeReplaySystem(env)

    parameter_load = prompt_set_parameters()
    if parameter_load:
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
        agent = Agent(alpha=0.0001, beta=0.001,
                      input_dims=env.observation_space.shape, tau=0.001,
                      batch_size=64, fc1_dims=400, fc2_dims=300,
                      n_actions=env.action_space.shape[0])
        n_games = 200


    filename = 'truck_trailer_v1' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'


    best_score = env.reward_range[0]
    score_history = []
    start_episode = 0
    resume_episode = start_episode

    if should_load:
        additional_episode = int(input("Enter number of additional episodes: "))
        try:
            agent.load_models()
            print("Pre-trained model loaded successfully!")

            load_filename = list_training_states()

            training_state = load_training_state(load_filename)
            if training_state is not None:
                score_history = training_state['score_history']
                best_score = training_state['best_score']
                resume_episode = training_state['episode_num']
                print(f"Resuming from episode {resume_episode}")
                print(f"Previous best score: {best_score:.2f}")
                print(f"Previous score history length: {len(score_history)}")

        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Starting new fresh training run...")
            resume_episode = 0



    if should_load:
        # Total episodes to train = resume point + additional episodes
        start_episode = resume_episode
        end_episode = resume_episode + additional_episode
    else:
        # Start from scratch
        start_episode = 0
        end_episode = n_games

    for i in range(start_episode, end_episode):
        observation, info = env.reset()
        done = False
        score = 0
        agent.noise.reset()

        episode_states = []
        episode_actions = []

        episode_states.append(env.state.copy())  # save initial state

        # Save environment data for replay
        env_data = {
            'startx': env.startx,
            'starty': env.starty,
            'startyaw': env.startyaw,
            'goalx': env.goalx,
            'goaly': env.goaly,
            'goalyaw': env.goalyaw,
            'path_x': env.path_x,
            'path_y': env.path_y,
            'path_yaw': env.path_yaw
        }

        while not done:
            action = agent.choose_action(observation)
            # The actor output is in [-1, 1]. Scale it to the environment's action space.
            # The action space is symmetric, so we can just multiply by the high bound.
            # We also clip the action in case the noise pushes it outside the [-1, 1] range.
            scaled_action = np.clip(action, -1, 1) * env.action_space.high

            episode_actions.append(scaled_action)  # save the actions

            observation_, reward, done, info = env.step(scaled_action)

            episode_states.append(env.state.copy())  # save the resulting state

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        # Save the data of complete episode with environment data
        replay_system.save_episode(i, episode_states, episode_actions, env_data)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

        if (i + 1) % 10 == 0:
            save_training_state(i + 1, score_history, best_score, filename)
    x = [i + 1 for i in range(end_episode)]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    Plot_learning_curve(x, score_history, figure_file)