import numpy as np


class RewardFunction:
    def __init__(self, observation, state, episode_steps, position_threshold, orientation_threshold, goalx, goaly, startx, starty,
                 previous_distance=None, cumulative_backward_movement=None, step_count_for_backward_tracking=None,
                 distance_history=None, stages_achieved=None, closest_distance_to_goal=None):
        self.observation = observation
        self.state = state
        self.episode_steps = episode_steps
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.goalx = goalx
        self.goaly = goaly
        self.startx = startx
        self.starty = starty

        # Environment boundaries
        self.min_map_x = -40
        self.min_map_y = -40
        self.max_map_x = 40
        self.max_map_y = 40

        self.trailer_length = 7.0

        # Calculate workspace dimensions for normalization
        self.workspace_width = self.max_map_x - self.min_map_x
        self.workspace_height = self.max_map_y - self.min_map_y
        self.max_expected_distance = np.sqrt(self.workspace_width ** 2 + self.workspace_height ** 2)
        self.max_episode_steps = 300

        # Initialize previous distance tracking for progress calculation
        # This is crucial for rewarding improvement rather than just proximity
        current_distance = self._calculate_raw_distance()
        self.initial_distance_to_goal = np.sqrt(
            (self.goalx - self.startx) ** 2 + (self.goaly - self.starty) ** 2) + 1e-6

        if previous_distance is not None:
            self.previous_distance = previous_distance
        else:
            self.previous_distance = current_distance  # First step of episode

        if cumulative_backward_movement is not None:
            self.cumulative_backward_movement = cumulative_backward_movement
        else:
            self.cumulative_backward_movement = 0.0

        if step_count_for_backward_tracking is not None:
            self.step_count_for_backward_tracking = step_count_for_backward_tracking
        else:
            self.step_count_for_backward_tracking = 0

        if distance_history is not None:
            self.distance_history = distance_history.copy()
        else:
            self.distance_history = [current_distance] * 5

        if 'stages_achieved' in locals() and stages_achieved is not None:
            self.stages_achieved = stages_achieved.copy()
        else:
            self.stages_achieved = [False, False, False]

        if closest_distance_to_goal is not None:
            self.closest_distance_to_goal = closest_distance_to_goal
            # Update closest distance if current is closer
            if current_distance < self.closest_distance_to_goal:
                self.closest_distance_to_goal = current_distance
        else:
            self.closest_distance_to_goal = current_distance

        # Mathematical parameters for reward shaping
        self.distance_decay_rate = 2.0  # β parameter for exponential distance reward
        self.progress_scale = 1.0  # Scaling factor for progress normalization
        self.orientation_weight_factor = 3.0  # How much orientation matters vs distance

        # Set up reward weights with proper hierarchy
        # Notice how goal-seeking rewards are now competitive with safety penalties
        self.weights = {
            'distance_exponential': 0,  # Strong pull toward goal
            'progress_reward': 15.0,  # Reward for getting closer
            'unified_orientation': 20.0,
            'staged_success': [10, 25, 100],  # Progressive success bonuses
            'safety_major': -500.0,  # Reduced from -1000 to allow risk-taking
            'safety_minor': -50.0,  # Minor safety violations
            'exploration_bonus': 2.0,  # Positive exploration encouragement
            'backward_movement': 1.0,  # Weight for anti-circling penalty
            'final_success': 500.0  # Big bonus for complete success
        }

    def get_persistent_state(self):
        """Return the state that needs to be passed to the next step"""
        return {
            'previous_distance': self.previous_distance,
            'cumulative_backward_movement': self.cumulative_backward_movement,
            'step_count_for_backward_tracking': self.step_count_for_backward_tracking,
            'distance_history': self.distance_history.copy(),
            'stages_achieved': self.stages_achieved.copy(),
            'closest_distance_to_goal': self.closest_distance_to_goal
        }

    def _calculate_raw_distance(self):
        """Calculate the actual distance from trailer to goal in meters."""
        # Using trailer position (state[4], state[5]) as this is what needs to dock
        return np.sqrt((self.state[4] - self.goalx) ** 2 + (self.state[5] - self.goaly) ** 2)

    def _normalize_distance(self, distance):
        """Normalize distance to [0, 1] range for consistent reward scaling."""
        return min(distance / self.max_expected_distance, 1.0)

    def check_excessive_backward_movement(self):
        """Check if trailer has moved too far away from goal."""
        current_distance = self._calculate_raw_distance()
        max_allowed_distance = self.closest_distance_to_goal + 6.0
        return current_distance > max_allowed_distance

    def calculate_exponential_distance_reward(self):
        """
        Exponential distance reward that creates a strong gradient toward the goal.

        Mathematical insight: Instead of linear penalty, we use exponential reward:
        R = weight * exp(-β * normalized_distance)

        This creates a 'funnel effect' where rewards grow exponentially as you approach
        the goal, providing strong directional guidance.
        """
        current_distance = self._calculate_raw_distance()
        normalized_distance = self._normalize_distance(current_distance)

        # Exponential reward function - gets dramatically larger near goal
        exponential_reward = np.exp(-self.distance_decay_rate * normalized_distance)

        return exponential_reward

    def calculate_progress_reward(self):
        """
        Anti-circling progress reward that prevents reward hacking.

        Mathematical insight: We now track both positive and negative progress,
        and use a momentum-based approach to prevent circular motion exploitation.
        """
        current_distance = self._calculate_raw_distance()

        # Calculate instantaneous progress (positive when getting closer)
        instant_progress = self.previous_distance - current_distance

        # SOLUTION 1: Bidirectional progress tracking
        # Reward getting closer, penalize moving away
        if instant_progress > 0:  # Moving closer
            progress_component = np.tanh(instant_progress / self.progress_scale)
        else:  # Moving away - apply penalty
            progress_component = np.tanh(instant_progress / self.progress_scale) * 0.5  # Half penalty

        # SOLUTION 2: Net progress tracking over multiple steps

        # Update distance history (rolling window)
        self.distance_history.append(current_distance)
        self.distance_history.pop(0)

        # Calculate net progress over the window
        net_progress = self.distance_history[0] - current_distance  # Total improvement over window
        net_progress_reward = np.tanh(net_progress / (self.progress_scale * 2)) * 0.5

        # SOLUTION 3: Path efficiency bonus
        # Reward straight-line approaches more than wandering paths
        if len(self.distance_history) >= 3:
            # Check if we're making consistent progress (monotonic improvement)
            recent_distances = self.distance_history[-3:]
            is_monotonic = all(recent_distances[i] >= recent_distances[i + 1] for i in range(len(recent_distances) - 1))
            efficiency_bonus = 0.2 if is_monotonic else 0
        else:
            efficiency_bonus = 0

        # Combine all progress components
        total_progress_reward = progress_component + net_progress_reward + efficiency_bonus


        return total_progress_reward


    def calculate_backward_movement_penalty(self):
        """
        Fixed version that works with new object creation every step
        """
        current_distance = self._calculate_raw_distance()

        # ✅ DEBUG: Print to verify it's working
        #print(f"Step {self.episode_steps}: current={current_distance:.3f}, previous={self.previous_distance:.3f}")

        # Calculate backward movement for this step
        backward_movement_this_step = max(0, current_distance - self.previous_distance)
        #print(f"  Backward movement this step: {backward_movement_this_step:.3f}")

        # Accumulate backward movement
        self.cumulative_backward_movement += backward_movement_this_step
        self.step_count_for_backward_tracking += 1

        #print(f"  Cumulative backward: {self.cumulative_backward_movement:.3f}")

        # Define the "movement budget"
        base_budget = 5.0
        time_factor = min(1.0, self.step_count_for_backward_tracking / 50)
        movement_budget = base_budget * time_factor

        # Calculate penalty based on excess backward movement
        excess_backward_movement = max(0, self.cumulative_backward_movement - movement_budget)

        # Apply escalating penalty function
        if excess_backward_movement > 0:
            penalty_magnitude = (excess_backward_movement ** 1.5) * 0.5
            backward_penalty = -penalty_magnitude
            #print(f"  🚨 PENALTY ACTIVATED: {backward_penalty:.3f}")
        else:
            backward_penalty = 0
            #print(f"  ✅ No penalty")

        penalty_info = {
            'cumulative_backward': self.cumulative_backward_movement,
            'movement_budget': movement_budget,
            'excess_movement': excess_backward_movement,
            'penalty': backward_penalty
        }

        return backward_penalty, penalty_info

    def calculate_unified_orientation_reward(self):
        """
        Calculates a single, unified orientation reward that guides both approach and final alignment.
        It works by creating a "look-behind" target point that the trailer should aim for.
        This naturally guides the trailer into the correct final orientation without conflict.
        """
        trailer_x, trailer_y = self.state[4], self.state[5]
        # The goal's orientation is encoded in observation 14, 15
        goal_yaw = np.arctan2(self.observation[14], self.observation[15])
        current_trailer_yaw = np.arctan2(self.observation[6], self.observation[7])

        # 1. Define the "look-behind" point. This is a virtual target behind the goal.
        #    The distance can be tuned, but half the trailer length is a good start.
        look_behind_dist = self.trailer_length * 0.75
        target_x = self.goalx - look_behind_dist * np.cos(goal_yaw)
        target_y = self.goaly - look_behind_dist * np.sin(goal_yaw)

        # 2. Calculate the desired heading from the trailer to this new virtual target.
        desired_heading_to_target = np.arctan2(target_y - trailer_y, target_x - trailer_x)

        # 3. Calculate the error between the trailer's current heading and the desired heading.
        #    We compare it to the trailer's backward-facing direction.
        trailer_backward_heading = current_trailer_yaw + np.pi
        angle_error = desired_heading_to_target - trailer_backward_heading
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize to [-pi, pi]

        # 4. Use cosine similarity to create a smooth reward.
        #    Reward is 1 for perfect alignment, -1 for opposite.
        #    We use (cos(error) + 1) / 2 to scale it to [0, 1] to prevent negative rewards here.
        orientation_reward = (np.cos(angle_error) + 1) / 2

        # 5. Additionally, reward matching the final orientation, but only when very close.
        #    This helps with the last bit of fine-tuning.
        current_distance = self._calculate_raw_distance()
        final_alignment_weight = np.exp(-0.5 * current_distance)  # Weight increases exponentially as distance -> 0

        final_orientation_error = goal_yaw - current_trailer_yaw
        final_orientation_error = np.arctan2(np.sin(final_orientation_error), np.cos(final_orientation_error))
        final_alignment_reward = (np.cos(final_orientation_error) + 1) / 2

        # Combine the two components. The look-behind target does the heavy lifting for the approach,
        # and the final alignment reward polishes the result at the end.
        total_orientation_reward = (
                                               1 - final_alignment_weight) * orientation_reward + final_alignment_weight * final_alignment_reward
        return total_orientation_reward

    def calculate_staged_success_rewards(self):
        """
        Multi-level success criteria that provide frequent positive feedback.

        Educational insight: Instead of binary success/failure, we create multiple
        achievement levels. This gives the agent more opportunities for positive
        reinforcement during learning.
        """
        current_distance = self._calculate_raw_distance()
        current_orientation_error = abs(np.arctan2(self.observation[19], self.observation[20]))

        staged_rewards = 0

        # Stage 1: Getting close (within 5 meters)
        if current_distance <= 5.0:
            staged_rewards += self.weights['staged_success'][0]
            self.stages_achieved[0] = True

        # Stage 2: Very close with decent orientation (within 2 meters, roughly aligned)
        if (current_distance <= 2.0 and current_orientation_error <= np.deg2rad(45) and not self.stages_achieved[1]):
            staged_rewards += self.weights['staged_success'][1]
            self.stages_achieved[1] = True

        # Stage 3: Final docking position
        if (current_distance <= self.position_threshold and
                current_orientation_error <= self.orientation_threshold and not self.stages_achieved[2]):
            staged_rewards += self.weights['staged_success'][2]
            self.stages_achieved[2] = True

        return staged_rewards

    def calculate_safety_penalties(self):
        """
        Proportional safety penalties that discourage dangerous behavior
        without completely overwhelming goal-seeking behavior.

        Key improvement: We now have major and minor safety violations
        instead of treating all safety issues as catastrophic.
        """
        safety_penalty = 0
        violation_type = "none"

        # Check for jackknife (major violation)
        theta_diff = self.state[0] - self.state[1]
        if abs(theta_diff) > np.deg2rad(85):  # Slightly more permissive than 90°
            safety_penalty += self.weights['safety_major']
            violation_type = "jackknife"
        elif abs(theta_diff) > np.deg2rad(70):  # Warning zone
            safety_penalty += self.weights['safety_minor']
            violation_type = "jackknife_warning"

        # Check boundaries with graduated penalties
        truck_x, truck_y = self.state[2], self.state[3]
        trailer_x, trailer_y = self.state[4], self.state[5]

        # Major boundary violation (completely outside)
        if (truck_x < self.min_map_x - 2 or truck_x > self.max_map_x + 2 or
                truck_y < self.min_map_y - 2 or truck_y > self.max_map_y + 2 or
                trailer_x < self.min_map_x - 2 or trailer_x > self.max_map_x + 2 or
                trailer_y < self.min_map_y - 2 or trailer_y > self.max_map_y + 2):
            safety_penalty += self.weights['safety_major']
            violation_type = "major_boundary"

        # Minor boundary violation (near edge)
        elif (truck_x < self.min_map_x or truck_x > self.max_map_x or
              truck_y < self.min_map_y or truck_y > self.max_map_y or
              trailer_x < self.min_map_x or trailer_x > self.max_map_x or
              trailer_y < self.min_map_y or trailer_y > self.max_map_y):
            safety_penalty += self.weights['safety_minor']
            violation_type = "minor_boundary"

        if self.goaly > trailer_y:
            safety_penalty += self.weights['safety_major']
            violation_type = "past_the_goal"

        return safety_penalty, violation_type

    def calculate_exploration_bonus(self):
        """
        Positive exploration encouragement instead of time penalty.

        Educational insight: Instead of penalizing time spent, we reward
        efficient solutions while still encouraging exploration early in training.
        """
        # Early in episode, reward exploration more
        exploration_factor = max(0, 1 - (self.episode_steps / (self.max_episode_steps * 0.7)))

        # Small bonus for reasonable exploration, larger bonus for efficiency
        if self.episode_steps < self.max_episode_steps * 0.5:  # Efficient solution
            return self.weights['exploration_bonus'] * 2
        elif self.episode_steps < self.max_episode_steps * 0.8:  # Reasonable time
            return self.weights['exploration_bonus']
        else:  # Taking too long
            return 0

    def check_episode_termination(self):
        """
        Check various termination conditions and determine if episode should end.
        """
        # Success condition
        current_distance = self._calculate_raw_distance()
        orientation_error = abs(np.arctan2(self.observation[19], self.observation[20]))

        success = (current_distance <= self.position_threshold and
                   orientation_error <= self.orientation_threshold)

        # Failure conditions
        theta_diff = self.state[0] - self.state[1]
        jackknife = abs(theta_diff) > np.deg2rad(90)

        # Strict boundary check for termination (more lenient than penalty check)
        truck_x, truck_y = self.state[2], self.state[3]
        trailer_x, trailer_y = self.state[4], self.state[5]
        out_of_bounds = (truck_x < self.min_map_x - 3 or truck_x > self.max_map_x + 3 or
                         truck_y < self.min_map_y - 3 or truck_y > self.max_map_y + 3 or
                         trailer_x < self.min_map_x - 3 or trailer_x > self.max_map_x + 3 or
                         trailer_y < self.min_map_y - 3 or trailer_y > self.max_map_y + 3)

        max_steps = self.episode_steps >= self.max_episode_steps

        done = success or jackknife or out_of_bounds or max_steps

        return done, success, jackknife, out_of_bounds, max_steps

    def compute_reward(self):
        """
        Main reward computation that combines all components with proper weighting.

        Educational insight: The total reward is now structured to create a strong
        gradient toward goal achievement while maintaining safety constraints.
        The mathematical balance ensures goal-seeking behavior dominates over
        excessive caution.
        """

        # Calculate all reward components
        distance_reward = self.calculate_exponential_distance_reward()
        progress_reward = self.calculate_progress_reward()
        unified_orientation_reward = self.calculate_unified_orientation_reward()
        staged_success = self.calculate_staged_success_rewards()
        safety_penalty, violation_type = self.calculate_safety_penalties()
        exploration_bonus = self.calculate_exploration_bonus()
        backward_penalty, backward_info = self.calculate_backward_movement_penalty()

        # Check termination conditions
        done, success, jackknife, out_of_bounds, max_steps = self.check_episode_termination()

        # Final success bonus
        final_success_bonus = self.weights['final_success'] if success else 0
        self.previous_distance = self._calculate_raw_distance()

        # Combine all components with their weights
        total_reward = (
                (distance_reward * self.weights['distance_exponential']) +
                (progress_reward * self.weights['progress_reward'] ) +
                (unified_orientation_reward * self.weights['unified_orientation']) +
                staged_success +  # Already weighted
                safety_penalty +  # Already weighted
                exploration_bonus +  # Already weighted
                (backward_penalty * self.weights['backward_movement']) +
                final_success_bonus
        )

        # Create detailed reward information for analysis and debugging
        reward_info = {
            'total_reward': total_reward,
            'distance_reward': distance_reward * self.weights['distance_exponential'] ,
            'progress_reward': progress_reward * self.weights['progress_reward'] ,
            'orientation_reward': unified_orientation_reward * self.weights['unified_orientation'],
            'staged_success': staged_success,
            'safety_penalty': safety_penalty,
            'exploration_bonus': exploration_bonus,
            'final_success_bonus': final_success_bonus,
            'violation_type': violation_type,
            'backward_penalty' : backward_penalty * self.weights['backward_movement'],
            'backward_movement_info': backward_info,
            'done': done,
            'success': success
        }

        return total_reward, reward_info
