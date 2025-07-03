import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from episode_replay_collector import EpisodeReplaySystem
from truck_trailer_sim.simv1 import Truck_trailer_Env_1

def main():
    episode_number = input("Enter the episode number you want to see replay of: ")
    env = Truck_trailer_Env_1()
    replay = EpisodeReplaySystem(env)
    replay.replay_episode(episode_number)

if __name__ == "__main__":
    main()

