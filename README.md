# DDPG Truck-Trailer Autonomous Backing System

A Deep Deterministic Policy Gradient (DDPG) reinforcement learning implementation for training autonomous truck-trailer backing maneuvers. This project uses deep reinforcement learning to solve the complex non-holonomic control problem of reversing a truck-trailer system to a goal pose.

## Overview

This project implements a DDPG agent that learns to control the steering angle of a truck to back a trailer into a target position and orientation. The task is challenging due to the kinematic constraints of the articulated vehicle system and the counter-intuitive control dynamics when moving in reverse.

## Features

### Core Components

- **DDPG Agent Implementation**: Actor-critic architecture with target networks and soft updates
- **Kinematic Simulation**: Physics-based truck-trailer simulation with realistic vehicle dynamics
- **Advanced Reward Function**: Multi-component reward system with distance-based rewards, orientation alignment, progress tracking, and safety penalties
- **Ornstein-Uhlenbeck Noise**: Exploration noise for continuous action spaces
- **Experience Replay Buffer**: Efficient memory management for off-policy learning

### Neural Network Architecture

The agent uses separate actor and critic networks with the following architecture:

- **Actor Network**: Maps observations to continuous steering actions with tanh activation
- **Critic Network**: Estimates Q-values for state-action pairs

### Observation Space

The environment provides a 23-dimensional observation vector including:

- Truck position and orientation (normalized, sin/cos encoded)
- Trailer position and orientation (normalized, sin/cos encoded)
- Hitch angle (sin/cos encoded)
- Steering angle (sin/cos encoded)
- Goal position and orientation
- Relative measurements (distance, local errors, heading errors)

### Training Features

- **Interactive CLI**: Rich terminal interface for training configuration
- **Checkpoint Management**: Save and resume training from checkpoints
- **Episode Replay System**: Record and replay training episodes for analysis

### Visualization Tools

The project includes multiple visualization utilities:

- Learning curve plotting
- Episode replay visualization
- Trajectory analysis
- Reward component analysis
- Heatmap generation
- Saliency map analysis

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym
- SciPy
- Rich (for CLI interface)

### Setup

```bash
# Clone the repository
git clone https://github.com/pain7576/ddpg-trucktrailer.git
cd ddpg-trucktrailer

# Install dependencies
pip install torch numpy matplotlib gym scipy rich
```

## Usage

### Training

To train a new agent:

```bash
cd DDPG
python train.py
```

The training script offers an interactive CLI where you can:

- Choose to load from checkpoint or start fresh
- Configure hyperparameters (learning rates, batch size, network dimensions)
- Set the number of training episodes

### Training with Custom Parameters

When prompted, you can set custom hyperparameters:

- `alpha`: Actor learning rate (default: 0.0001)
- `beta`: Critic learning rate (default: 0.001)
- `tau`: Soft update parameter (default: 0.001)
- `batch_size`: Training batch size (default: 64)
- `fc1_dims`: First layer dimensions (default: 400)
- `fc2_dims`: Second layer dimensions (default: 300)

### Testing

To test a trained agent, use the episode player utilities to visualize agent performance on saved episodes.

## Project Structure

```
ddpg-trucktrailer/
├── DDPG/
│   ├── DDPG_agent.py              # Main DDPG agent implementation
│   ├── networks.py                 # Actor and Critic neural networks
│   ├── noise.py                    # Ornstein-Uhlenbeck noise
│   ├── replay_buffer.py            # Experience replay buffer
│   ├── train.py                    # Training script with CLI
│   ├── trainv2.py                  # Alternative training script
│   ├── episode_replay_collector.py # Episode recording system
│   └── [various visualization tools]
├── truck_trailer_sim/
│   ├── simv1.py                    # Environment v1 (with Dubins path)
│   ├── simv2.py                    # Environment v2 (simplified)
│   ├── reward_functionv1.py        # Comprehensive reward function
│   └── reward_function.py          # Alternative reward function
├── ploting_utils/
│   ├── plot_learning_curve.py      # Training progress visualization
│   ├── plot_path.py                # Trajectory plotting
│   └── plot.py                     # General plotting utilities
└── test.py                         # Model testing utilities
```

## Algorithm Details

### DDPG Learning Process

The agent uses the DDPG algorithm with the following update rules:

1. **Critic Update**: Minimize MSE loss between predicted Q-values and target Q-values
2. **Actor Update**: Maximize expected Q-value through policy gradient
3. **Target Network Updates**: Soft updates using tau parameter

### Reward Function Components

The reward function includes:

- **Progress Reward**: Rewards getting closer to the goal with anti-circling mechanisms
- **Heading Alignment**: Rewards pointing the trailer toward the goal
- **Orientation Alignment**: Rewards matching the goal orientation
- **Staged Success Rewards**: Progressive bonuses at different achievement levels
- **Safety Penalties**: Penalizes jackknifing, boundary violations, and excessive movements
- **Smoothness Penalty**: Discourages jerky steering movements
- **Exploration Bonus**: Encourages efficient solutions

### Dynamic Weight Adjustment

The reward function uses curriculum learning with distance-dependent weights that shift priorities:

- **Far from goal**: Focus on heading toward the goal (approach trajectory)
- **Close to goal**: Focus on orientation matching (final alignment)

## Environment Details

### State Space

The truck-trailer system state consists of:

- `psi_1`: Truck heading angle
- `psi_2`: Trailer heading angle
- `x1, y1`: Truck position
- `x2, y2`: Trailer position

### Action Space

Single continuous action: steering angle ∈ [-45°, 45°]

### Episode Termination

Episodes terminate when:

- Goal is reached (success)
- Jackknife condition occurs (hitch angle > 90°)
- Vehicle goes out of bounds
- Maximum steps reached
- Excessive backward movement detected

## Training Progress Tracking

The system automatically:

- Saves training states every 10 episodes
- Records best performing models
- Tracks score history and success rates
- Generates learning curve plots

## Notes

- The truck moves backward (negative velocity) to simulate backing maneuvers
- The trailer position is the primary control objective, not the truck position
- The reward function is carefully balanced to encourage goal-seeking behavior while maintaining safety constraints
- Layer normalization is used instead of batch normalization for more stable training
