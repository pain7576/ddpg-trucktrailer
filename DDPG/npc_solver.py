# --- START OF FILE npc_solver.py ---

import numpy as np
import casadi as ca

# ==============================================================================
# 1. System Parameters and Model
# ==============================================================================

def get_truck_trailer_params():
    """Returns a dictionary of truck and trailer parameters."""
    return {
        'M': 0.0,   # Hitch length
        'L1': 5.0,  # Truck length
        'W1': 2.0,  # Truck width
        'L2': 7.0, # Trailer length
        'W2': 2.0,  # Trailer width
    }

def truck_trailer_dynamics(x, u, params, const_v):
    """
    Nonlinear dynamics for the truck and trailer system with constant velocity.
    States: [x, y, theta, beta]
    Inputs: [alpha] (steering angle only)
    """
    theta = x[2]
    beta = x[3]
    alpha = u[0]
    v = const_v

    M, L1, L2 = params['M'], params['L1'], params['L2']

    dxdt = ca.vertcat(
        v * ca.cos(beta) * (1 + M/L1 * ca.tan(beta) * ca.tan(alpha)) * ca.cos(theta),
        v * ca.cos(beta) * (1 + M/L1 * ca.tan(beta) * ca.tan(alpha)) * ca.sin(theta),
        v * (ca.sin(beta)/L2 - M/(L1*L2) * ca.cos(beta) * ca.tan(alpha)),
        v * (ca.tan(alpha)/L1 - ca.sin(beta)/L2 + M/(L1*L2) * ca.cos(beta) * ca.tan(alpha))
    )
    return dxdt

def generate_initial_guess_const_v(initial_pose, target_pose, p, u0_alpha):
    """Generates an initial guess for the constant velocity problem."""
    x_guess = np.linspace(initial_pose[0], target_pose[0], p + 1)
    y_guess = np.linspace(initial_pose[1], target_pose[1], p + 1)
    theta_guess = np.linspace(initial_pose[2], target_pose[2], p + 1)
    beta_guess = np.linspace(initial_pose[3], target_pose[3], p + 1)

    X0 = np.vstack([x_guess, y_guess, theta_guess, beta_guess])
    U0 = np.tile(u0_alpha, (1, p))
    return X0, U0


# ==============================================================================
# 2. Main NPC Solver Function
# ==============================================================================

def solve_with_npc(initial_pose_npc, target_pose_npc, map_bounds, npc_params):
    """
    Solves the optimal control problem for a given start and end pose.

    Args:
        initial_pose_npc: np.array [x, y, theta, beta]
        target_pose_npc: np.array [x, y, theta, beta]
        map_bounds: Tuple of ((xmin, xmax), (ymin, ymax))
        npc_params: Dictionary with 'p', 'Ts', 'velocity'

    Returns:
        (success, trajectory_states, control_inputs)
        On failure, returns (False, None, None)
    """
    params = get_truck_trailer_params()
    p = npc_params['p']
    Ts = npc_params['Ts']
    CONSTANT_VELOCITY = npc_params['velocity']
    u0_alpha = np.array([0]) # Initial guess for steering

    opti = ca.Opti()

    # Define variables
    nx = 4
    nu = 1
    X = opti.variable(nx, p + 1) # State trajectory
    U = opti.variable(nu, p)   # Control trajectory

    # Cost function (minimize control effort)
    cost = ca.sumsqr(U)
    opti.minimize(cost)

    # System dynamics constraint (using RK4 integration)
    f = lambda x, u: truck_trailer_dynamics(x, u, params, CONSTANT_VELOCITY)
    dt = Ts
    for k in range(p):
        k1 = f(X[:,k], U[:,k])
        k2 = f(X[:,k] + dt/2 * k1, U[:,k])
        k3 = f(X[:,k] + dt/2 * k2, U[:,k])
        k4 = f(X[:,k] + dt * k3, U[:,k])
        x_next = X[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:, k+1] == x_next)

    # Path constraints
    opti.subject_to(opti.bounded(-np.pi/4, U[0, :], np.pi/4)) # Steering angle limits
    opti.subject_to(opti.bounded(-np.pi/2, X[3, :], np.pi/2)) # Jackknife angle limits

    # Map boundary constraints
    x_min, x_max = map_bounds[0]
    y_min, y_max = map_bounds[1]
    opti.subject_to(opti.bounded(x_min, X[0, :], x_max)) # Trailer X
    opti.subject_to(opti.bounded(y_min, X[1, :], y_max)) # Trailer Y

    # Truck boundary constraints
    L2, M = params['L2'], params['M']
    for k in range(p + 1):
        x_trailer_k, y_trailer_k, theta_trailer_k, beta_k = X[0,k], X[1,k], X[2,k], X[3,k]
        theta_truck_k = theta_trailer_k + beta_k
        x_truck_k = x_trailer_k + L2 * ca.cos(theta_trailer_k) + M * ca.cos(theta_truck_k)
        y_truck_k = y_trailer_k + L2 * ca.sin(theta_trailer_k) + M * ca.sin(theta_truck_k)
        opti.subject_to(opti.bounded(x_min, x_truck_k, x_max))
        opti.subject_to(opti.bounded(y_min, y_truck_k, y_max))

    # Boundary conditions
    opti.subject_to(X[:, 0] == initial_pose_npc)
    opti.subject_to(X[:, -1] == target_pose_npc)

    # Initial guess
    X_guess, U_guess = generate_initial_guess_const_v(initial_pose_npc, target_pose_npc, p, u0_alpha)
    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)

    # Solver options
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    try:
        sol = opti.solve()
        # On success, return the solution
        return True, sol.value(X), sol.value(U)
    except RuntimeError:
        # On failure, return failure status
        return False, None, None