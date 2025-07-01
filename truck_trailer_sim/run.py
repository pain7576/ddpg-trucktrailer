import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simv1 import Truck_trailer_Env_1  # Replace with actual file name
import numpy as np
env = Truck_trailer_Env_1()
obs, _ = env.reset()
fixed_action = np.radians(10)
for step in range(4500):
    #action = env.action_space.sample()
    obs, reward, done, info = env.step(fixed_action)
    env.render()
    if done:
        break