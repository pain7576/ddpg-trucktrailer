import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from simv1 import Truck_trailer_Env_1


class SimulationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super(SimulationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            self.episode_count += 1

            if self.episode_count % self.eval_freq == 0:
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0

                while not done :
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    total_reward += reward
                    self.eval_env.render()

                print(f"Episode {self.episode_count}: Simulation reward = {total_reward}")

        return True


env = Truck_trailer_Env_1()
eval_env = Truck_trailer_Env_1()

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

simulation_callback = SimulationCallback(eval_env, eval_freq=1000)

model.learn(total_timesteps=50000, log_interval=1000, callback=simulation_callback)
model.save("ddpg_truck_trailer_1")