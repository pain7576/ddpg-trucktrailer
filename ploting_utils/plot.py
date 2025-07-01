import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from matplotlib.patches import FancyArrow
from PythonRobotics.PathPlanning.DubinsPath.dubins_path_planner import plan_dubins_path
from PythonRobotics.PathPlanning.DubinsPath.dubins_path_backward_planner import plan_dubins_path_backward

def plot_vehicle(ax, x, y, heading, length, width, label, color='blue', show_wheels=True, steering_angle=0.0):
    dx = 0
    dy = -width / 2
    rect = Rectangle((dx, dy), length, width, linewidth=1.5, edgecolor='black',
                     facecolor=color, alpha=0.6)
    t = Affine2D().rotate_around(0, 0, heading).translate(x, y)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)
    # Label text inside vehicle removed per request

    if show_wheels:
        rear_axle_pos = np.array([0, 0])
        front_axle_pos = np.array([length, 0])

        wheel_radius = width * 0.22  # wheel radius for the outlines
        wheel_offset_y = width / 2 * 0.9
        wheel_dir_len = wheel_radius * 2.0  # direction line length

        def transform_point(p):
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading),  np.cos(heading)]])
            return R @ p + np.array([x, y])

        # Rear wheels outlines only (no fill)
        rear_left_wheel_center = transform_point(rear_axle_pos + np.array([0, wheel_offset_y]))
        rear_right_wheel_center = transform_point(rear_axle_pos + np.array([0, -wheel_offset_y]))

        for center in [rear_left_wheel_center, rear_right_wheel_center]:
            circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=2, zorder=5)
            ax.add_patch(circle_outline)

        for center in [rear_left_wheel_center, rear_right_wheel_center]:
            line_end = center + wheel_dir_len * np.array([np.cos(heading), np.sin(heading)])
            ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=2.0, zorder=6)

        if label.lower() == 'truck':
            front_left_wheel_center = transform_point(front_axle_pos + np.array([0, wheel_offset_y]))
            front_right_wheel_center = transform_point(front_axle_pos + np.array([0, -wheel_offset_y]))

            for center in [front_left_wheel_center, front_right_wheel_center]:
                circle_outline = Circle(center, wheel_radius, edgecolor='black', facecolor='none', linewidth=2, zorder=5)
                ax.add_patch(circle_outline)

            front_wheel_heading = heading + steering_angle
            for center in [front_left_wheel_center, front_right_wheel_center]:
                line_end = center + wheel_dir_len * np.array([np.cos(front_wheel_heading), np.sin(front_wheel_heading)])
                ax.plot([center[0], line_end[0]], [center[1], line_end[1]], color='black', linewidth=2.5, zorder=7)

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


def plot_truck_trailer():
    x1, y1, psi1 = 10.0, 10.0, np.deg2rad(45)
    x2, y2, psi2 = 10.0, 6.0, np.deg2rad(90)
    delta = np.deg2rad(45)

    startx = 10.0
    starty = 6.0
    startyaw = np.deg2rad(90)

    goalx = 15.0
    goaly = 0.0
    goalyaw = np.deg2rad(180)
    default_arrow_length = 1.0

    path_x, path_y, path_yaw =       plan_dubins_path_backward(startx,
                                                               starty,
                                                               startyaw,
                                                               goalx,
                                                               goaly,
                                                               goalyaw,
                                                               curvature=1.0 / 13.714)
    x, y, yaw = path_x, path_y, path_yaw
    L1, W1 = 2.0, 2.0
    L2, W2 = 4.0, 2.0
    hitch_offset = 0.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-30, 20)
    ax.set_ylim(-35, 15)
    ax.grid(True)

    plot_vehicle(ax, x1, y1, psi1, L1, W1, label='Truck', color='blue', show_wheels=True, steering_angle=delta)
    plot_vehicle(ax, x2, y2, psi2, L2, W2, label='Trailer', color='green', show_wheels=True)

    hitch_x = x1 - hitch_offset * np.cos(psi1)
    hitch_y = y1 - hitch_offset * np.sin(psi1)

    trailer_front_x = x2 + L2 * np.cos(psi2)
    trailer_front_y = y2 + L2 * np.sin(psi2)

    ax.plot([hitch_x, trailer_front_x], [hitch_y, trailer_front_y], 'r-', linewidth=2, label='Hitch Link')
    ax.plot(goalx, goaly, 'ro', label='goal')
    ax.plot(trailer_front_x, trailer_front_y, 'go', label='start')

    info_text = (
        f"Truck: x={x1:.1f}, y={y1:.1f}, ψ={np.rad2deg(psi1):.0f}°, δ={np.rad2deg(delta):.0f}°\n"
        f"Trailer: x={x2:.1f}, y={y2:.1f}, ψ={np.rad2deg(psi2):.0f}°\n"
        f"start: x={startx:.1f},y={starty:.1f}, θ={np.rad2deg(startyaw)}°\n"
        f"goal: x={goalx:.1f},y={goaly:.1f}, θ={np.rad2deg(goalyaw)}°"
    )
    ax.text(0.5, 21.0, info_text, fontsize=9, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.legend()
    ax.set_title("Truck-Trailer")

    for i in range(len(path_x)):
        xi, yi, yawi = x[i], y[i], yaw[i]
        # Using a fixed arrow length for all intermediate points
        plot_pose(ax, xi, yi, yawi, color='blue', arrow_length=default_arrow_length)

    # Plot start and goal poses with distinct colors and labels
    plot_pose(ax, startx, starty, startyaw, color='green', label='Start', arrow_length=default_arrow_length*2)
    plot_pose(ax, goalx, goaly, goalyaw, color='red', label='Goal', arrow_length=default_arrow_length*2)

    plt.show()
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


plot_truck_trailer()
