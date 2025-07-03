import numpy as np

class Rewardfunction :
    def __init__(self, observation, state, episode_steps, position_threshold, orientation_threshold, goalx, goaly):
        self.observation = observation
        self.state = state
        self.episode_steps = episode_steps
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.goalx = goalx
        self.goaly = goaly

        self.min_map_x = -40
        self.min_map_y = -40
        self.max_map_x = 40
        self.max_map_y = 40

        self.workspace_width = self.max_map_x - self.min_map_x
        self.workspace_height = self.max_map_y - self.min_map_y
        self.max_expected_distance = np.sqrt(self.workspace_width ** 2 + self.workspace_height ** 2)
        self.max_episode_steps = 4500

        self.compute_weights_heading_orientation()

        self.weights = {
            'postion_error': 2.0,
            'heading_error': self.alpha,
            'orientation_error': 1- self.alpha,
            'safety': -100.0,
            'exploration': 0.5,
            'success_bonus': 1000

        }
    def check_jackknife(self) :
        """Check for jackknife condition."""
        theta = self.state[0] - self.state[1]
        if abs(theta) > np.deg2rad(90):
            return True  # Jackknife detected
        return False

    def check_out_of_Map(self):
        """Check if vehicle is out of bounds."""
        out_of_bounds_truck = (
                self.state[2] < self.min_map_x or self.state[2] > self.max_map_x or
                self.state[3] < self.min_map_y or self.state[3] > self.max_map_y
        )

        out_of_bounds_trailer = (
                self.state[4] < self.min_map_x or self.state[4] > self.max_map_x or
                self.state[5] < self.min_map_y or  self.state[5] > self.max_map_y
        )
        return out_of_bounds_truck or out_of_bounds_trailer


    def check_max_steps_reached(self):
        return self.episode_steps >= self.max_episode_steps



    def calculate_postion_error_reward(self) :
        self.postion_error_normalized = self.observation[16]
        return self.postion_error_normalized

    def calculate_heading_error_reward(self) :
        self.heading_error_normalized = (np.arctan2(self.observation[21], self.observation[22])+np.pi)/ (2*np.pi)
        return self.heading_error_normalized

    def calculate_orientation_error_reward(self) :
        self.orientation_error_normalized = (np.arctan2(self.observation[19], self.observation[20])+np.pi)/ (2*np.pi)
        return self.orientation_error_normalized



    def calculate_safety_reward(self) :
        self.jackknife = self.check_jackknife()
        self.out_of_map = self.check_out_of_Map()
        self.max_steps_reached = self.check_max_steps_reached()

        done = self.jackknife or self.out_of_map or self.max_steps_reached

        return 1 if done else 0

    def calculate_penalty_over_exploration(self) :
        self.exploration_penalty_normalized = -(self.episode_steps / self.max_episode_steps)
        return self.exploration_penalty_normalized

    def compute_weights_heading_orientation(self) :
        self.alpha = np.exp(-self.observation[16])
        return self.alpha

    def calculate_success_reward(self) :
        position_error = np.sqrt((self.state[4] - self.goalx) ** 2 + (self.state[5] - self.goaly) ** 2)
        orientation_error = np.arctan2(self.observation[19], self.observation[20])
        if position_error <= self.position_threshold and abs(orientation_error) <= self.orientation_threshold :
            return 1
        else:
            return 0

    def compute_reward(self) :
        position_error_reward = self.calculate_postion_error_reward()
        heading_error_reward = self.calculate_heading_error_reward()
        orientation_error_reward = self.calculate_orientation_error_reward()
        safety_reward = self.calculate_safety_reward()
        exploration_penalty = self.calculate_penalty_over_exploration()
        success_reward = self.calculate_success_reward()

        reward = position_error_reward * self.weights['postion_error'] + \
                 heading_error_reward * self.weights['heading_error'] + \
                 orientation_error_reward * self.weights['orientation_error'] + \
                 safety_reward * self.weights['safety'] + \
                 exploration_penalty * self.weights['exploration'] + \
                 success_reward * self.weights['success_bonus']

        reward_info = {
            'position_error': position_error_reward,
            'heading_error': heading_error_reward,
            'orientation_error': orientation_error_reward,
            'safety': safety_reward,
            'exploration': exploration_penalty,
            'success': success_reward
        }

        return reward , reward_info










