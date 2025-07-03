import gym
import os
import numpy as np
from DDPG_agent import Agent
from ploting_utils.plot_learning_curve import Plot_learning_curve
from truck_trailer_sim.simv1 import Truck_trailer_Env_1

if __name__ == '__main__':
    env = Truck_trailer_Env_1()
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300,
                    n_actions=env.action_space.shape[0])
    n_games = 7000
    filename = 'truck_trailer_v1' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            # The actor output is in [-1, 1]. Scale it to the environment's action space.
            # The action space is symmetric, so we can just multiply by the high bound.
            # We also clip the action in case the noise pushes it outside the [-1, 1] range.
            scaled_action = np.clip(action, -1, 1) * env.action_space.high
            observation_, reward, done, info = env.step(scaled_action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    Plot_learning_curve(x, score_history, figure_file)



