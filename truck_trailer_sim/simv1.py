import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import error, spaces
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from PythonRobotics.PathPlanning.DubinsPath.dubins_path_backward_planner import plan_dubins_path_backward
from matplotlib.patches import FancyArrow
import time
import random
import math
from reward_function import Rewardfunction

class Truck_trailer_Env_1(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # Configuration space for the environment
        self.min_map_x = -40
        self.min_map_y = -40
        self.max_map_x = 40
        self.max_map_y = 40

        self.min_map_orientation = np.radians(0)
        self.max_map_orientation = np.radians(360)

        self.steering_angle = 0
        self.L1 = 5.74  # wheelbase of truck
        self.L2 = 10.192  # length of trailer
        self.hitch_offset = 0.00  # hitch offset from truck's rear axle
        self.v1x = -5.012  # longitudinal velocity of the truck (negative for backing)
        self.u = 0.0

        self.dt = 0.08  # time step
        self.time = 0  # starting time

        self.min_heading_truck = np.deg2rad(0)
        self.max_heading_truck = np.deg2rad(360)

        self.min_heading_trailer = np.deg2rad(0)
        self.max_heading_trailer = np.deg2rad(360)

        self.max_hitch_angle = np.radians(90)

        self.min_steering_angle = np.radians(-45)
        self.max_steering_angle = np.radians(45)

        # Calculate workspace dimensions for normalization
        self.workspace_width = self.max_map_x - self.min_map_x
        self.workspace_height = self.max_map_y - self.min_map_y
        self.max_expected_distance = np.sqrt(self.workspace_width ** 2 + self.workspace_height ** 2)

        # Enhanced observation space dimension
        # Components:

                # Basic absolute position
    #            truck (x, y, sin(θ), cos(θ)) = 4
    #            trailer (x, y, sin(θ), cos(θ)) = 4

                # system dynamics
    #            hitch (sin(angle), cos(angle)) = 2
    #            steering angle (sin(delta), cos(delta)) = 2

                # Goal
    #            goal (x, y, sin(θ), cos(θ)) = 4

                # Relative position
    #            relative (distance, dx_local, dy_local, sin(angle_error), cos(angle_error),sin(heading_error), cos(heading_error)) = 7
    # Total = 23 dimensions
        self.observation_dim = 23

        # Define observation space with normalized bounds
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        # Action space: steering angle
        self.action_space = spaces.Box(low=self.min_steering_angle, high=self.max_steering_angle)

        # Track episode information
        self.episode_steps = 0
        self.max_episode_steps = 4500  # Maximum steps per episode

        #Tolerances to reach the goal
        self.position_threshold = 0.5
        self.orientation_threshold = np.deg2rad(15)

    def compute_observation(self, state, steering_angle):
        """Compute the enhanced observation vector from the current state."""
        # Extract state components
        psi_1, psi_2, x1, y1, x2, y2 = state


        # Normalize positions to [-1, 1] based on workspace center
        truck_x_norm = (x1 - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        truck_y_norm = (y1 - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)
        trailer_x_norm = (x2 - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        trailer_y_norm = (y2 - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)
        goal_x_norm = (self.goalx - (self.max_map_x + self.min_map_x) / 2) / (self.workspace_width / 2)
        goal_y_norm = (self.goaly - (self.max_map_y + self.min_map_y) / 2) / (self.workspace_height / 2)

        # Compute hitch-angle
        hitch_angle = psi_1 - psi_2

        # Compute distance to goal
        distance_to_goal = np.sqrt((x2 - self.goalx) ** 2 + (y2 - self.goaly) ** 2)
        distance_to_goal_norm = np.clip(distance_to_goal / self.max_expected_distance, 0, 1)

        # Compute angle from trailer to goal
        angle_to_goal = np.arctan2(self.goaly - y2, self.goalx - x2)

        # Compute orientation error
        orientation_error = self.goalyaw - psi_2

        # Compute position error in trailer's local coordinate frame
        dx_global = self.goalx - x2
        dy_global = self.goaly - y2
        dx_local = dx_global * np.cos(psi_2) + dy_global * np.sin(psi_2)
        dy_local = -dx_global * np.sin(psi_2) + dy_global * np.cos(psi_2)

        # Compute heading error
        heading_error = angle_to_goal - (psi_2 + np.deg2rad(180))

        # Normalize local errors
        dx_local_norm = np.clip(dx_local / self.max_expected_distance, -1, 1)
        dy_local_norm = np.clip(dy_local / self.max_expected_distance, -1, 1)



        # Construct observation vector
        observation = np.array([
            # Truck state (normalized position + angle as sin/cos)
            truck_x_norm, #0
            truck_y_norm, #1
            np.sin(psi_1), #2
            np.cos(psi_1), #3

            # Trailer state (normalized position + angle as sin/cos)
            trailer_x_norm, #4
            trailer_y_norm, #5
            np.sin(psi_2), #6
            np.cos(psi_2), #7

            # system dynamics (hitch angle as sin/cos) and (steering angle as sin/cos)
            np.sin(hitch_angle), #8
            np.cos(hitch_angle), #9
            np.sin(steering_angle), #10
            np.cos(steering_angle), #11

            # Goal (normalized position + angle as sin/cos)
            goal_x_norm, #12
            goal_y_norm, #13
            np.sin(self.goalyaw), #14
            np.cos(self.goalyaw), #15

            # Relative measurements
            distance_to_goal_norm, #16
            dx_local_norm, #17
            dy_local_norm, #18
            np.sin(orientation_error), #19
            np.cos(orientation_error), #20
            np.sin(heading_error), #21
            np.cos(heading_error) #22
        ], dtype=np.float32)

        return observation
    def kinematic_model(self, t, x, u):
        """Kinematic model of truck-trailer system."""
        xd = np.zeros(len(x))
        # Extract state variables
        psi_1 = x[0]  # truck heading
        psi_2 = x[1]  # trailer heading
        x1 = x[2]  # truck x
        y1 = x[3]  # truck y
        x2 = x[4]  # trailer x
        y2 = x[5]  # trailer y

        # Hitch angle
        self.hitch_angle = psi_1 - psi_2

        # Truck angular rate
        dpsi_1 = (self.v1x / self.L1) * np.tan(self.steering_angle)

        # Trailer velocity
        self.v2x = self.v1x * np.cos(self.hitch_angle) + (self.hitch_offset * dpsi_1 * np.sin(self.hitch_angle))

        # Trailer angular rate
        dpsi_2 = (self.v1x / self.L2) * np.sin(self.hitch_angle) - (
                (self.hitch_offset / self.L2) * dpsi_1 * np.cos(self.hitch_angle))

        # Truck position derivatives
        dx1 = self.v1x * np.cos(psi_1)
        dy1 = self.v1x * np.sin(psi_1)

        # Trailer position derivatives
        dx2 = self.v2x * np.cos(psi_2)
        dy2 = self.v2x * np.sin(psi_2)

        # Combine into state derivative
        xd = np.array([dpsi_1, dpsi_2, dx1, dy1, dx2, dy2])
        return xd

    def check_jackknife(self, psi_1, psi_2):
        """Check for jackknife condition."""
        theta = psi_1 - psi_2
        if abs(theta) > np.deg2rad(90):
            return True  # Jackknife detected
        return False

    def check_max_steps_reached(self,current_step):
        return current_step >= self.max_episode_steps

    def check_out_of_Map(self, x1, y1, x2, y2):
        """Check if vehicle is out of bounds."""
        out_of_bounds_truck = (
                x1 < self.min_map_x or x1 > self.max_map_x or
                y1 < self.min_map_y or y1 > self.max_map_y
        )

        out_of_bounds_trailer = (
                x2 < self.min_map_x or x2 > self.max_map_x or
                y2 < self.min_map_y or y2 > self.max_map_y
        )
        return out_of_bounds_truck or out_of_bounds_trailer

    def check_path_out_of_Map(self, start_x, start_y, start_psi, goal_x, goal_y, goal_psi):
        """Check if Dubins path is within map bounds."""
        path_x, path_y, path_yaw = plan_dubins_path_backward(start_x,
                                                             start_y,
                                                             start_psi,
                                                             goal_x,
                                                             goal_y,
                                                             goal_psi,
                                                             curvature=1.0 / 6)

        for x, y in zip(path_x, path_y):
            if (x < self.min_map_x or x > self.max_map_x or
                    y < self.min_map_y or y > self.max_map_y):
                return True
        return False

    def generate_valid_random_poses(self, max_attempts=1000):
        """Generate random start and goal poses with valid path."""
        for attempt in range(max_attempts):
            # Generate random start pose
            start_x = random.uniform(self.min_map_x, self.max_map_x)
            start_y = random.uniform(self.min_map_y, self.max_map_y)
            start_psi2 = random.uniform(self.min_heading_trailer, self.max_heading_trailer)

            # Generate random goal pose
            goal_x = random.uniform(self.min_map_x, self.max_map_x)
            goal_y = random.uniform(self.min_map_y, self.max_map_y)
            goal_psi = random.uniform(self.min_heading_trailer, self.max_heading_trailer)

            # Check if start and goal are at least 15 meters apart
            distance = ((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2) ** 0.5
            if distance < 15:
                continue

            # Check if path is valid
            if not self.check_path_out_of_Map(start_x, start_y, start_psi2,
                                              goal_x, goal_y, goal_psi):
                return (start_x, start_y, start_psi2, goal_x, goal_y, goal_psi)

        # Return None if no valid poses found after max_attempts
        return None

    def plot_vehicle(self, ax, x, y, heading, length, width, label, color='blue', show_wheels=True, steering_angle=0.0):
        """Plot vehicle visualization."""
        dx = 0
        dy = -width / 2
        rect = Rectangle((dx, dy), length, width, linewidth=1.5, edgecolor='black',
                         facecolor=color, alpha=0.6)
        t = Affine2D().rotate_around(0, 0, heading).translate(x, y)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

        if show_wheels:
            rear_axle_pos = np.array([0, 0])
            front_axle_pos = np.array([length, 0])

            wheel_radius = width * 0.22
            wheel_offset_y = width / 2 * 0.9
            wheel_dir_len = wheel_radius * 2.0

            def transform_point(p):
                R = np.array([[np.cos(heading), -np.sin(heading)],
                              [np.sin(heading), np.cos(heading)]])
                return R @ p + np.array([x, y])

            # Rear wheels
            rear_left_wheel_center = transform_point(rear_axle_pos + np.array([0, wheel_offset_y]))
            rear_right_wheel_center = transform_point(rear_axle_pos + np.array([0, -wheel_offset_y]))

            for center in [rear_left_wheel_center, rear_right_wheel_center]:
                circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=1,
                                        zorder=5)
                ax.add_patch(circle_outline)

            for center in [rear_left_wheel_center, rear_right_wheel_center]:
                line_end = center + wheel_dir_len * np.array([np.cos(heading), np.sin(heading)])
                ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=1, zorder=6)

            if label.lower() == 'truck':
                front_left_wheel_center = transform_point(front_axle_pos + np.array([0, wheel_offset_y]))
                front_right_wheel_center = transform_point(front_axle_pos + np.array([0, -wheel_offset_y]))

                for center in [front_left_wheel_center, front_right_wheel_center]:
                    circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=1,
                                            zorder=5)
                    ax.add_patch(circle_outline)

                front_wheel_heading = heading + steering_angle
                for center in [front_left_wheel_center, front_right_wheel_center]:
                    line_end = center + wheel_dir_len * np.array(
                        [np.cos(front_wheel_heading), np.sin(front_wheel_heading)])
                    ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=1, zorder=7)

            rear_axle_line_start = transform_point(rear_axle_pos + np.array([0, -wheel_offset_y * 1.1]))
            rear_axle_line_end = transform_point(rear_axle_pos + np.array([0, wheel_offset_y * 1.1]))
            ax.plot([rear_axle_line_start[0], rear_axle_line_end[0]],
                    [rear_axle_line_start[1], rear_axle_line_end[1]], 'k-', linewidth=2.0, zorder=4)

            if label.lower() == 'truck':
                front_axle_line_start = transform_point(front_axle_pos + np.array([0, -wheel_offset_y * 1.1]))
                front_axle_line_end = transform_point(front_axle_pos + np.array([0, wheel_offset_y * 1.1]))
                ax.plot([front_axle_line_start[0], front_axle_line_end[0]],
                        [front_axle_line_start[1], front_axle_line_end[1]], 'k-', linewidth=2.0, zorder=4)

        arrow_color = 'blue' if label.lower() == 'trailer' else color
        arrow_length = width * 1.5
        arrow_dx = arrow_length * np.cos(heading)
        arrow_dy = arrow_length * np.sin(heading)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.4, head_length=0.6,
                 fc=arrow_color, ec=arrow_color, linewidth=2, zorder=10)

    def plot_pose(self, ax, x, y, yaw, color='blue', label=None, arrow_length=1.0):
        """Plot a pose as an arrow."""
        ax.add_patch(FancyArrow(x, y,
                                arrow_length * np.cos(yaw),
                                arrow_length * np.sin(yaw),
                                width=0.1,
                                head_width=0.4,
                                head_length=0.4,
                                length_includes_head=True,
                                color=color,
                                zorder=5))
        if label is not None:
            ax.plot(x, y, 'o', color=color, markersize=8, label=label, zorder=6)

    def reset(self, seed=None, options=None):

        # Generate valid random start and goal poses
        self.startx, self.starty, self.startyaw, self.goalx, self.goaly, self.goalyaw = self.generate_valid_random_poses()


        # Set initial-state from trailer configuration
        x1 = self.startx + self.L2 * np.cos(self.startyaw)
        y1 = self.starty + self.L2 * np.sin(self.startyaw)
        psi_1 = self.startyaw
        x2 = self.startx
        y2 = self.starty
        psi_2 = self.startyaw

        #Generate reference path
        self.path_x, self.path_y, self.path_yaw = plan_dubins_path_backward(self.startx,
                                                             self.starty,
                                                             self.startyaw,
                                                             self.goalx,
                                                             self.goaly,
                                                             self.goalyaw,

                                                            curvature=1.0 / 6)
        # Set initial-state      0       1    2   3   4   5
        self.state = np.array([psi_1, psi_2, x1, y1, x2, y2], dtype=np.float32)

        observation = self.compute_observation(self.state, np.deg2rad(0))

        self.episode_steps = 0

        return observation, {}
    def step(self, action):

        # Clip and apply steering action
        action = np.clip(action, self.min_steering_angle, self.max_steering_angle)
        self.steering_angle = float(action)

        state = self.state

        # Calculate new-state
        sol = spi.solve_ivp(
            fun=lambda t, y: self.kinematic_model(t, y, self.steering_angle),
            t_span=[self.time, self.dt],
            y0=state,
            method='RK45'
        )
        new_state = sol.y[:, -1]
        self.state = new_state

        observation = self.compute_observation(self.state, action)

        # Increment step counter
        self.episode_steps += 1

        # check episode end condition
        self.jackknife = self.check_jackknife(new_state[0], new_state[1])
        self.out_of_map = self.check_out_of_Map(new_state[2], new_state[3], new_state[4], new_state[5])
        self.max_steps_reached = self.check_max_steps_reached(self.episode_steps)

        done = self.jackknife or self.out_of_map or self.max_steps_reached

        # compute reward
        reward_class = Rewardfunction(observation, self.state, self.episode_steps, self.position_threshold, self.orientation_threshold, self.goalx, self.goaly)
        total_reward, reward_dict = reward_class.compute_reward()

        return observation, total_reward, done, reward_dict

    def render(self, mode='human'):
        """Render the environment."""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()

        self.ax.clear()

        psi1, psi2, x1, y1, x2, y2 = self.state
        x, y, yaw = self.path_x, self.path_y, self.path_yaw

        L1, W1 = self.L1, 2.0
        L2, W2 = self.L2, 2.0
        delta = self.steering_angle  # Use actual steering angle
        hitch_offset = 0.0
        default_arrow_length = 1.0

        # Draw vehicles
        self.plot_vehicle(self.ax, x1, y1, psi1, L1, W1, label='Truck', color='blue', show_wheels=True,
                          steering_angle=delta)
        self.plot_vehicle(self.ax, x2, y2, psi2, L2, W2, label='Trailer', color='green', show_wheels=True)

        # Draw hitch connection
        hitch_x = x1 - hitch_offset * np.cos(psi1)
        hitch_y = y1 - hitch_offset * np.sin(psi1)
        trailer_front_x = x2 + L2 * np.cos(psi2)
        trailer_front_y = y2 + L2 * np.sin(psi2)
        self.ax.plot([hitch_x, trailer_front_x], [hitch_y, trailer_front_y], 'r-', linewidth=2, label='Hitch Link')

        self.ax.set_xlim(self.min_map_x, self.max_map_x)
        self.ax.set_ylim(self.min_map_y, self.max_map_y)
        self.ax.set_aspect('equal')

        observation = self.compute_observation(self.state, self.steering_angle)
        reward_class = Rewardfunction(observation, self.state, self.episode_steps, self.position_threshold, self.orientation_threshold, self.goalx, self.goaly)
        total_reward, reward_dict = reward_class.compute_reward()

        info_text = (
            f"Truck: x={x1:.1f}, y={y1:.1f}, ψ={np.rad2deg(psi1):.0f}°, δ={np.rad2deg(delta):.0f}°\n"
            f"Trailer: x={x2:.1f}, y={y2:.1f}, ψ={np.rad2deg(psi2):.0f}°\n"
            f"Start: x={self.startx:.1f}, y={self.starty:.1f}, θ={np.rad2deg(self.startyaw):.0f}°\n"
            f"Goal: x={self.goalx:.1f}, y={self.goaly:.1f}, θ={np.rad2deg(self.goalyaw):.0f}°\n"
            f"Reward: {total_reward:.2f}\n"

        )
        self.ax.text(2.5, 21.0, info_text, fontsize=6, fontweight='bold', va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        # Draw reference path
        for i in range(len(self.path_x)):
            xi, yi, yawi = x[i], y[i], yaw[i]
            self.plot_pose(self.ax, xi, yi, yawi, color='blue', arrow_length=default_arrow_length)

        # Draw start and goal poses
        self.plot_pose(self.ax, self.startx, self.starty, self.startyaw, color='green', label='Start',
                       arrow_length=default_arrow_length * 2)
        self.plot_pose(self.ax, self.goalx, self.goaly, self.goalyaw, color='red', label='Goal',
                       arrow_length=default_arrow_length * 2)
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




















