from matplotlib.patches import FancyArrow
import matplotlib.pyplot as plt
import numpy as np
from PythonRobotics.PathPlanning.DubinsPath.dubins_path_planner import plan_dubins_path

def plot_pose(ax, x, y, yaw, color='blue', label=None, arrow_length=1.0):
    """
    Plot a single pose as an arrow on the given axes.
    """
    # Use FancyArrow to represent the pose with direction
    ax.add_patch(FancyArrow(x, y,
                            arrow_length * np.cos(yaw),
                            arrow_length * np.sin(yaw),
                            width=0.1,
                            head_width=0.4,
                            head_length=0.4,
                            length_includes_head=True,
                            color=color,
                            zorder=5))
    # Plot a marker at the base of the arrow for clarity
    if label is not None:
        ax.plot(x, y, 'o', color=color, markersize=8, label=label, zorder=6)


def plot_path_with_orientation(startx, starty, startyaw, goalx, goaly, goalyaw, pathx, pathy, pathyaw, default_arrow_length=1.0):
    """
    Plots a path with orientation arrows without connecting lines.
    """

    x, y, yaw = pathx, pathy, pathyaw

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(0, 60)
    ax.set_ylim(-30, 30)
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Path with Orientation Arrows (No Connecting Line)")

    # The line connecting the waypoints has been removed.
    # ax.plot(x, y, '-b', label="Path") # This line was removed.

    # Plot orientation arrows along the path
    for i in range(len(pathx)):
        xi, yi, yawi = x[i], y[i], yaw[i]
        # Using a fixed arrow length for all intermediate points
        plot_pose(ax, xi, yi, yawi, color='blue', arrow_length=default_arrow_length)

    # Plot start and goal poses with distinct colors and labels
    plot_pose(ax, startx, starty, startyaw, color='green', label='Start', arrow_length=default_arrow_length*2)
    plot_pose(ax, goalx, goaly, goalyaw, color='red', label='Goal', arrow_length=default_arrow_length*2)

    ax.legend()
    plt.show()


# Example usage
startx = 2.0
starty = 0.0
startyaw = 0.0
goalx = 15.0
goaly = 0.0
goalyaw = np.deg2rad(210)

# Calculate the Dubins path
# The last parameter is the turning radius (curvature = 1/turning_radius)
path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(startx,
                                                           starty,
                                                           startyaw,
                                                           goalx,
                                                           goaly,
                                                           goalyaw,
                                                           curvature=1.0/13.714) # Using a turning radius of 5.0

# Plot the resulting path as a series of arrows
plot_path_with_orientation(startx, starty, startyaw, goalx, goaly, goalyaw, path_x, path_y, path_yaw)