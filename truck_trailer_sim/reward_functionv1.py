import numpy as np


class RewardFunction:
    def __init__(self, observation, state, episode_steps, position_threshold, orientation_threshold, goalx, goaly):
        self.observation = observation
        self.state = state
        self.episode_steps = episode_steps
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.goalx = goalx
        self.goaly = goaly

        # Environment boundaries
        self.min_map_x = -40
        self.min_map_y = -40
        self.max_map_x = 40
        self.max_map_y = 40

        # Calculate workspace dimensions for normalization
        self.workspace_width = self.max_map_x - self.min_map_x
        self.workspace_height = self.max_map_y - self.min_map_y
        self.max_expected_distance = np.sqrt(self.workspace_width ** 2 + self.workspace_height ** 2)
        self.max_episode_steps = 300

        # Initialize previous distance tracking for progress calculation
        # This is crucial for rewarding improvement rather than just proximity
        if not hasattr(self, 'previous_distance'):
            self.previous_distance = self._calculate_raw_distance()

        # Mathematical parameters for reward shaping
        self.distance_decay_rate = 2.0  # β parameter for exponential distance reward
        self.progress_scale = 1.0  # Scaling factor for progress normalization
        self.orientation_weight_factor = 3.0  # How much orientation matters vs distance

        # Set up reward weights with proper hierarchy
        # Notice how goal-seeking rewards are now competitive with safety penalties
        self.weights = {
            'distance_exponential': 15.0,  # Strong pull toward goal
            'progress_reward': 8.0,  # Reward for getting closer
            'Heading_alignment': 5.0,  # Reward for pointing correctly
            'Orientation_alignment': 5.0, #Reward for pointing to correctly with goal pose
            'staged_success': [10, 25, 100],  # Progressive success bonuses
            'safety_major': -500.0,  # Reduced from -1000 to allow risk-taking
            'safety_minor': -50.0,  # Minor safety violations
            'exploration_bonus': 2.0,  # Positive exploration encouragement
            'backward_movement': 1.0,  # Weight for anti-circling penalty
            'final_success': 200.0  # Big bonus for complete success
        }

    def _calculate_raw_distance(self):
        """Calculate the actual distance from trailer to goal in meters."""
        # Using trailer position (state[4], state[5]) as this is what needs to dock
        return np.sqrt((self.state[4] - self.goalx) ** 2 + (self.state[5] - self.goaly) ** 2)

    def _normalize_distance(self, distance):
        """Normalize distance to [0, 1] range for consistent reward scaling."""
        return min(distance / self.max_expected_distance, 1.0)

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
        # Track progress over last N steps to prevent short-term cycling
        if not hasattr(self, 'distance_history'):
            self.distance_history = [current_distance] * 5  # Initialize with current distance

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

        # Update previous distance for next step
        self.previous_distance = current_distance

        return total_progress_reward

    def compute_dynamic_weights(self):
        """
        Compute distance-dependent weights that shift priorities during approach.

        Educational insight: This implements adaptive curriculum learning where
        the agent focuses on different trailer-centric skills at different phases.

        Far from goal: Focus on trailer heading toward goal (approach trajectory)
        Close to goal: Focus on trailer orientation matching target pose (final alignment)

        Note: All orientation rewards are trailer-focused because the trailer
        is the end effector that needs to be precisely positioned, while the
        truck is just the actuator mechanism.
        """
        current_distance = self._calculate_raw_distance()
        normalized_distance = self._normalize_distance(current_distance)

        # Create a smooth transition function using sigmoidal curves
        # When normalized_distance = 1.0 (far), trailer_heading_priority ≈ 1.0
        # When normalized_distance = 0.0 (close), trailer_heading_priority ≈ 0.0
        distance_factor = normalized_distance

        # Trailer heading priority: High when far, low when close
        # During approach, we want trailer pointing toward goal
        trailer_heading_priority = np.exp(-3.0 * (1.0 - distance_factor))

        # Trailer final orientation priority: Low when far, high when close
        # During docking, we want trailer matching goal pose orientation
        trailer_final_orientation_priority = np.exp(-3.0 * distance_factor)

        # Distance priority: Always important but peaks at medium distances
        # Uses a bell curve to emphasize consistent progress
        distance_priority = 1.0 + 0.5 * np.exp(-8.0 * (distance_factor - 0.5) ** 2)

        # Progress priority: Consistent throughout but slightly higher when close
        # Encourages steady movement with extra emphasis on final approach
        progress_priority = 1.0 + 0.3 * (1.0 - distance_factor)

        return {
            'trailer_heading': trailer_heading_priority,
            'trailer_final_orientation': trailer_final_orientation_priority,
            'distance': distance_priority,
            'progress': progress_priority,
            'current_distance': current_distance,
            'normalized_distance': normalized_distance
        }

    def calculate_backward_movement_penalty(self):
        """
        Penalize excessive backward movement using cumulative tracking.

        Educational insight: This demonstrates elegant simplicity in reward design.
        Instead of complex pattern detection, we use a simple "movement budget"
        approach that allows tactical retreats while preventing exploitation.

        Mathematical framework: We track cumulative backward movement and apply
        penalties once it exceeds a reasonable threshold for legitimate maneuvering.
        """
        current_distance = self._calculate_raw_distance()

        # Initialize tracking variables if they don't exist
        if not hasattr(self, 'cumulative_backward_movement'):
            self.cumulative_backward_movement = 0.0
            self.step_count_for_backward_tracking = 0

        # Calculate backward movement for this step
        backward_movement_this_step = max(0, current_distance - self.previous_distance)

        # Accumulate backward movement
        self.cumulative_backward_movement += backward_movement_this_step
        self.step_count_for_backward_tracking += 1

        # Define the "movement budget" - how much backward movement is acceptable
        # This represents legitimate tactical maneuvering
        # We scale this with episode length to be more generous early in learning
        base_budget = 5.0  # meters of backward movement allowed
        time_factor = min(1.0, self.step_count_for_backward_tracking / 50)  # Ramp up over 50 steps
        movement_budget = base_budget * time_factor

        # Calculate penalty based on excess backward movement
        excess_backward_movement = max(0, self.cumulative_backward_movement - movement_budget)

        # Apply escalating penalty function
        # Mathematical insight: We use a quadratic penalty to make excessive
        # backward movement increasingly costly
        if excess_backward_movement > 0:
            # Quadratic penalty - gets more severe as excess increases
            penalty_magnitude = (excess_backward_movement ** 1.5) * 0.5
            backward_penalty = -penalty_magnitude
        else:
            backward_penalty = 0

        # Optional: Provide some feedback about the penalty for debugging
        penalty_info = {
            'cumulative_backward': self.cumulative_backward_movement,
            'movement_budget': movement_budget,
            'excess_movement': excess_backward_movement,
            'penalty': backward_penalty
        }

        return backward_penalty, penalty_info

    def calculate_orientation_alignment_reward(self):
        """
        Orientation reward using cosine similarity for natural alignment.

        Mathematical insight: Instead of just measuring orientation error,
        we calculate how well the trailer is aligned with the direction to goal.
        cos(0) = 1 (perfect alignment), cos(π) = -1 (opposite direction)
        """
        # Calculate desired heading (from trailer to goal)
        trailer_x, trailer_y = self.state[4], self.state[5]
        desired_heading = np.arctan2(self.goaly - trailer_y, self.goalx - trailer_x)

        # Current trailer orientation from observation
        current_orientation = np.arctan2(self.observation[6], self.observation[7])

        # Calculate alignment using cosine similarity
        angle_difference = desired_heading - (current_orientation + np.deg2rad(180))

        # Normalize angle difference to [-π, π] range
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))

        # Cosine gives us natural alignment reward: 1 for perfect, -1 for opposite
        heading_reward = np.cos(angle_difference)

        return heading_reward

    def calculate_trailer_goal_orientation_reward(self):
        """
        Reward for trailer orientation matching the final goal orientation.

        This becomes primary focus when close to goal, teaching precision
        docking behavior for the final positioning.
        """
        # Assuming the goal orientation is 0 radians (pointing east)
        # Modify this based on your specific goal orientation requirements

        # Cosine similarity for orientation matching
        trailer_goal_alignment = self.observation[20]

        return trailer_goal_alignment

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

        # Stage 2: Very close with decent orientation (within 2 meters, roughly aligned)
        if current_distance <= 2.0 and current_orientation_error <= np.deg2rad(45):
            staged_rewards += self.weights['staged_success'][1]

        # Stage 3: Final docking position
        if (current_distance <= self.position_threshold and
                current_orientation_error <= self.orientation_threshold):
            staged_rewards += self.weights['staged_success'][2]

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

        dynamic_weights = self.compute_dynamic_weights()
        # Calculate all reward components
        distance_reward = self.calculate_exponential_distance_reward()
        progress_reward = self.calculate_progress_reward()
        headings_reward = self.calculate_orientation_alignment_reward()
        goal_orientation_reward = self.calculate_trailer_goal_orientation_reward()
        staged_success = self.calculate_staged_success_rewards()
        safety_penalty, violation_type = self.calculate_safety_penalties()
        exploration_bonus = self.calculate_exploration_bonus()
        backward_penalty, backward_info = self.calculate_backward_movement_penalty()

        # Check termination conditions
        done, success, jackknife, out_of_bounds, max_steps = self.check_episode_termination()

        # Final success bonus
        final_success_bonus = self.weights['final_success'] if success else 0

        # Combine all components with their weights
        total_reward = (
                (distance_reward * self.weights['distance_exponential'] * dynamic_weights['distance']) +
                (progress_reward * self.weights['progress_reward'] * dynamic_weights['progress']) +
                (headings_reward * self.weights['Heading_alignment'] * dynamic_weights['trailer_heading']) +
                (goal_orientation_reward * self.weights['Orientation_alignment'] * dynamic_weights['trailer_final_orientation']) +
                staged_success +  # Already weighted
                safety_penalty +  # Already weighted
                exploration_bonus +  # Already weighted
                (backward_penalty * self.weights['backward_movement']) +
                final_success_bonus
        )

        # Create detailed reward information for analysis and debugging
        reward_info = {
            'total_reward': total_reward,
            'distance_reward': distance_reward * self.weights['distance_exponential'] * dynamic_weights['distance'],
            'progress_reward': progress_reward * self.weights['progress_reward'] * dynamic_weights['progress'],
            'heading_reward' : headings_reward * self.weights['Heading_alignment'] * dynamic_weights['trailer_heading'],
            'orientation_reward': goal_orientation_reward * self.weights['Orientation_alignment'] * dynamic_weights['trailer_final_orientation'],
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
