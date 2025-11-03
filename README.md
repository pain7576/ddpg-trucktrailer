# DDPG Truck-Trailer Autonomous Backing System

A Deep Deterministic Policy Gradient (DDPG) reinforcement learning implementation for training autonomous truck-trailer backing maneuvers.  
This project uses deep reinforcement learning to solve the complex non-holonomic control problem of reversing a truck-trailer system to a goal pose.

---

## 🧭 Overview

This project implements a DDPG agent that learns to control the steering angle of a truck to back a trailer into a target position and orientation.  
The task is challenging due to the kinematic constraints of the articulated vehicle system and the counter-intuitive control dynamics when moving in reverse.

---

## ⚙️ Features

### Core Components

- **DDPG Agent Implementation**: Actor-critic architecture with target networks and soft updates  
- **Kinematic Simulation**: Physics-based truck-trailer simulation with realistic vehicle dynamics  
- **Advanced Reward Function**: Multi-component reward system with distance-based rewards, orientation alignment, progress tracking, and safety penalties  
- **Ornstein-Uhlenbeck Noise**: Exploration noise for continuous action spaces  
- **Experience Replay Buffer**: Efficient memory management for off-policy learning  

---

### Neural Network Architecture

The agent uses separate actor and critic networks with the following architecture:

- **Actor Network**: Maps observations to continuous steering actions using `tanh` activation  
- **Critic Network**: Estimates Q-values for state-action pairs  

---

### Observation Space

The environment provides a 23-dimensional observation vector including:

- Truck position and orientation (normalized, sin/cos encoded)  
- Trailer position and orientation (normalized, sin/cos encoded)  
- Hitch angle (sin/cos encoded)  
- Steering angle (sin/cos encoded)  
- Goal position and orientation  
- Relative measurements (distance, local errors, heading errors)  

---

### Training Features

- **Interactive CLI**: Rich terminal interface for training configuration  
- **Checkpoint Management**: Save and resume training from checkpoints  
- **Episode Replay System**: Record and replay training episodes for analysis  

---

### Visualization Tools

The project includes multiple visualization utilities:

- Learning curve plotting  
- Episode replay visualization  
- Trajectory analysis  
- Reward component analysis  
- Heatmap generation  
- Saliency map analysis  

---

## 💻 Installation

### Prerequisites

- Python 3.7+  
- PyTorch  
- NumPy  
- Matplotlib  
- Gym  
- SciPy  
- Rich (for CLI interface)  

---

### Setup

```sh
# Clone the repository  
git clone https://github.com/pain7576/ddpg-trucktrailer.git  
cd ddpg-trucktrailer  

# Install dependencies  
pip install torch numpy matplotlib gym scipy rich
