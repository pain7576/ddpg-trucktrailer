import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from npc_solver import solve_with_npc
from truck_trailer_sim.simv2 import Truck_trailer_Env_2
from seed_utils import set_seed
import random
from episode_replay_collectorv2 import EpisodeReplaySystem
from collections import deque
from trainv2 import save_transitions
MAP_X_RANGE = (-40, 40)
MAP_Y_RANGE = (-30, 40)

target_pose_npc = np.array([0, -30, np.pi/2, 0])
set_seed(27)
env = Truck_trailer_Env_2()
replay_system = EpisodeReplaySystem(env)
num = random.randint(45, 120)
map_boundaries = (MAP_X_RANGE, MAP_Y_RANGE)
NPC_PARAMS = {
    'p': 110,       # Prediction horizon
    'Ts': 0.08,      # Sample time
    'velocity': -5.012 # m/s (negative for reverse)
}

episode_transitions_history = deque(maxlen=200)
for i in range(200):
    num = random.randint(45, 120)
    initial_pose_npc = np.array([0, 10, np.deg2rad(num), 0])
    npc_success, X_opt, U_opt = solve_with_npc(initial_pose_npc, target_pose_npc, map_boundaries, NPC_PARAMS)
    if not npc_success or U_opt is None:
        print(f"Episode {i+1} skipped: NPC failed to find solution.")
        print(f"Initial pose: {initial_pose_npc}")
        continue
    # --- Data collection for replay ---
    episode_states = []
    episode_actions = []
    episode_reward_info = []
    episode_transitions = []

    observation, _ = env.reset()

    # Save initial state and environment data for replay
    episode_states.append(env.state.copy())
    env_data = {'startx': env.startx, 'starty': env.starty, 'startyaw': env.startyaw,
                'goalx': env.goalx, 'goaly': env.goaly, 'goalyaw': env.goalyaw}


    start_x = initial_pose_npc[0]
    start_y = initial_pose_npc[1]
    start_yaw_rad = initial_pose_npc[2]

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
    j = 0

    while not done:
        #env.render()
        scaled_action = U_opt[j]
        action = scaled_action/np.radians(45)

        observation_, reward, done, info = env.step(scaled_action)

        # Record data for this step for potential replay
        episode_actions.append(scaled_action)
        episode_states.append(env.state.copy())
        episode_reward_info.append(info)
        transitions = (observation.copy(), action.copy(), reward, observation_.copy(), done)
        episode_transitions.append(transitions)


        score += reward
        observation = observation_
        step += 1
        j +=1
    episode_transitions_history.append(episode_transitions)

    replay_system.save_episode(
        episode_num=i,
        states=episode_states,
        actions=episode_actions,
        info=episode_reward_info,
        env_data=env_data
    )

    print(f"Episode {i+1} completed.")


    if i == 50 or i == 100 or i == 150 or i>190 :
        save_transitions(i, episode_transitions_history)



