import numpy as np
import matplotlib.pyplot as plt


def reward(orientation_error, distance_error):
    if abs(orientation_error) <= np.deg2rad(11) and abs(distance_error) <= 0.5:
        return 200
    else:
        return 0

def reward_continuous(orientation_error, distance_error):
    orientation_error = abs(orientation_error)
    distance_error = abs(distance_error)
    orientation_reward = np.cos(orientation_error / np.deg2rad(11) * np.pi/2)
    orientation_reward = max(0, orientation_reward)  # Clamp to [0, 1]
    position_reward = np.exp(-distance_error / 0.5)

    # Combine position and orientation rewards
    # Use weighted geometric mean to encourage both good position AND orientation
    position_weight = 0.6  # Position is slightly more important
    orientation_weight = 0.4

    combined_reward = (position_reward ** position_weight) * (orientation_reward ** orientation_weight)

    # Scale by the maximum reward weight
    if (distance_error <= 0.5 and
            orientation_error <= np.deg2rad(11)):
        final_reward = (combined_reward * 200) + 200
    else:
        final_reward = combined_reward * 200

    return final_reward


def bell_shaped_reward(orientation_error, distance_error):
    """
    Calculates a reward based on a 2D Gaussian (bell-shaped) function.
    The reward is maximal at (0, 0) and decreases smoothly as errors increase.
    """
    # --- Parameters to tune the shape of the bell curve ---

    # The maximum reward at the peak (when both errors are zero)
    max_reward = 400.0

    # These 'spread' parameters control how wide the bell is along each axis.
    # A smaller value creates a sharper, narrower peak.
    # A larger value creates a wider, flatter peak.
    orientation_spread = 0.05  # Controls width along the orientation axis
    distance_spread = 0.25     # Controls width along the distance axis

    # --- Gaussian function calculation ---

    # Calculate the reward components for each error type
    orientation_reward = np.exp(- (orientation_error**2) / orientation_spread)
    distance_reward = np.exp(- (distance_error**2) / distance_spread)

    # The final reward is the product of the components, scaled by the max reward
    final_reward = max_reward * orientation_reward * distance_reward

    return final_reward

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def trapezoidal_reward(orientation_error, distance_error):
    """
    Calculates a reward based on a 2D trapezoidal function.
    This creates a flat plateau of high reward in the center, with linear slopes
    down to zero.
    """
    # --- Parameters to define the trapezoid's shape ---
    max_reward = 400.0

    # Define the inner rectangle (the plateau)
    # The reward will be max_reward inside this region.
    inner_orient_thresh = np.deg2rad(11)  # +/- 5 degrees
    inner_dist_thresh = 0.5              # +/- 0.2 meters

    # Define the outer rectangle (where the slopes end)
    # The reward will be 0 outside this region.
    outer_orient_thresh = np.deg2rad(22) # +/- 15 degrees
    outer_dist_thresh = 1              # +/- 0.7 meters

    # Ensure outer thresholds are greater than inner ones to prevent division by zero
    if not (outer_orient_thresh > inner_orient_thresh and outer_dist_thresh > inner_dist_thresh):
        raise ValueError("Outer thresholds must be greater than inner thresholds.")

    # Take the absolute value of errors for symmetrical reward
    orient_err = np.abs(orientation_error)
    dist_err = np.abs(distance_error)

    # --- Vectorized Calculation using np.select ---

    # Condition for being on the plateau
    cond_plateau = (orient_err <= inner_orient_thresh) & (dist_err <= inner_dist_thresh)

    # Condition for being on the slopes
    cond_slopes = (orient_err <= outer_orient_thresh) & (dist_err <= outer_dist_thresh)

    # --- Calculate the reward value for the slope region ---

    # Calculate the normalized "progress" down the slope for each dimension (0=at plateau, 1=at base)
    orient_progress = (orient_err - inner_orient_thresh) / (outer_orient_thresh - inner_orient_thresh)
    dist_progress = (dist_err - inner_dist_thresh) / (outer_dist_thresh - inner_dist_thresh)

    # We clip to ensure values inside the plateau don't become negative
    orient_progress = np.clip(orient_progress, 0, 1)
    dist_progress = np.clip(dist_progress, 0, 1)

    # The overall progress is the maximum of the two (the "worst" of the two errors)
    total_progress = np.maximum(orient_progress, dist_progress)

    # The reward on the slope is a linear interpolation from max_reward to 0
    slope_reward_values = max_reward * (1 - total_progress)

    # Use np.select to apply conditions across the whole grid
    # 1. If in plateau, reward is max_reward
    # 2. Else if in slopes, reward is the calculated slope value
    # 3. Else (default), reward is 0
    conditions = [cond_plateau, cond_slopes]
    choices = [max_reward, slope_reward_values]

    return np.select(conditions, choices, default=1)



# Create grid
orientation = np.linspace(np.deg2rad(-30), np.deg2rad(30), 200)   # 0 to 30 degrees in radians
distance = np.linspace(-1.5, 1.5, 200)                 # 0 to 1.5 meters
O, D = np.meshgrid(orientation, distance)

# Evaluate reward function on the grid
R = np.zeros_like(O)
for i in range(O.shape[0]):
    for j in range(O.shape[1]):
        R[i, j] = trapezoidal_reward(O[i, j], D[i, j])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(O, D, R, cmap="viridis", edgecolor="k", alpha=0.8)

ax.set_xlabel("Orientation Error (radians)")
ax.set_ylabel("Distance Error (m)")
ax.set_zlabel("Reward")
ax.set_title("Reward Function")

plt.show()


