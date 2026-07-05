# Reinforcement Learning-Based Obstacle Avoidance for Mobile Robots in Webots

## Overview

This repository presents a reinforcement learning framework for autonomous mobile robot navigation using the **Webots** simulation environment. An **e-puck differential-drive robot** is trained to navigate through a complex obstacle-filled environment while learning collision-free path planning using Deep Reinforcement Learning.

The project systematically evaluates reinforcement learning algorithms under identical simulation conditions to compare convergence behavior, navigation performance, and policy refinement. It is being developed as part of an academic research internship with the goal of publication.

---

## Objectives

- Design a robust **OpenAI Gym / Gymnasium-compatible** environment for Webots.
- Train an autonomous e-puck robot to navigate obstacle-rich environments.
- Compare Standard DQN and an improved DQN implementation under identical simulation conditions.
- Engineer an effective reward function that encourages efficient navigation while preventing reward-hacking behaviors.

---

# Technical Performance Configuration

| Hyperparameter | Value | Purpose |
|---------------|-------|---------|
| **Training Episodes** | 2000 | Primary training for both DQN variants |
| **Fine-Tuning Episodes** | 50 | Additional training using saved model weights |
| **Maximum Steps / Episode** | 1400 | Provides sufficient navigation horizon |
| **Episode Termination** | Capped termination on Collision | termination after 50 collisions |
| **Initial ε (Exploration)** | 1.0 | Maximum early exploration |
| **Final ε** | 0.01 | Near-greedy exploitation |
| **Epsilon Decay** | 0.99712 | Smooth decay over training |
| **Replay Buffer Size** | 10,00,000 | Prevents catastrophic forgetting |

---

# Reward Function

The reward structure is carefully engineered to discourage unsafe behaviors while encouraging smooth and efficient navigation.

| Component | Reward |
|----------|---------|
| Goal Reached | **+500.0** |
| Bonus Reward | **+250.0** |
| Collision | **-250.0** |
| Step Penalty | **-0.1** |
| Progress Reward | **+50 × Δ Distance** |
| Obstacle Proximity | **-1 to -5** |
| Velocity Reward | Positive reward for forward motion |

### Reward Design

- Large terminal reward for successfully reaching the goal.
- Bounus reward for reaching the goal without collision.
- Severe collision penalty discourages unsafe policies.
- Continuous progress shaping rewards movement toward the goal.
- Distance sensor penalties discourage close obstacle interactions.
- Velocity reward encourages stable forward motion instead of unnecessary rotations.

---

# Repository Structure

```text
rl-mobile-robot-obstacle-avoidance/
│
├── controllers/
│   └── rl_supervisor/
│       └── rl_supervisor.py          # Webots Supervisor controller
│   └── rl_supervisor_advanced/
|       └── rl_supervisor_advanced.py 
|   └── dqn_controller_test/
|       └── dqn_controller_test.py  
|       └── networks.py  
|
├── env/
│   └── webots_env.py              # Custom Gymnasium environment
│   └── reward_function.py         # Reward function for both the algorithms
|
├── training/
│   ├── train_dqn.py               # Standard DQN training
│   ├── train_advanced_dqn.py      # Advanced DQN training
│   ├── dqn_agent.py               # DQN agent implementation
│   ├── dqn_network.py             # Neural network architecture
│   ├── replay_buffer.py           # Experience replay memory
│   └── plot_results.py            # Training visualization
│
├── models/
│   ├── training_log.csv
│   ├── standard_dqn.pth
│   ├── advanced_dqn.pth
│   └── training_log_advanced.png
│
├── worlds/
│   └── oba.wbt                    # Webots simulation world
│   └── oba_advanced.wbt                    # Webots simulation world
|
└── README.md
```

---

# Implemented Algorithms

| Algorithm | Status |
|-----------|--------|
| ✅ Standard Deep Q-Network (DQN) | Implemented & Trained |
| ✅ Advanced Deep Q-Network | Implemented & Trained |
| ⬜ Proximal Policy Optimization (PPO) | Planned |
| ⬜ Soft Actor-Critic (SAC) | Planned |

---

## Standard Deep Q-Network (DQN)

The baseline implementation includes:

- Experience Replay
- Target Network
- ε-Greedy Exploration
- Feedforward Neural Network Function Approximation

The Standard DQN agent was trained for **2000 episodes** to learn collision-free navigation in the designed Webots environment.

---

## Advanced Deep Q-Network

An improved DQN implementation was developed to enhance training stability and navigation performance.

The Advanced DQN model was also trained for **2000 episodes** under identical simulation conditions. After convergence, the trained model weights were saved and loaded again for an additional **50 episodes** of fine-tuning to further refine the learned navigation policy.

This allows a direct comparison between the baseline and improved DQN implementations while maintaining the same environment and reward function.

---

# Software Requirements

- Python 3.10+
- Webots
- PyTorch (CUDA supported)
- NumPy
- Gym / Gymnasium
- Matplotlib

---

# Installation

```bash
git clone https://github.com/your-username/rl-mobile-robot-obstacle-avoidance.git

cd rl-mobile-robot-obstacle-avoidance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

### Train Standard DQN

```bash
python training/train_dqn.py
```

### Train Advanced DQN

```bash
python training/train_advanced_dqn.py
```

During training, the console displays:

- Episode number
- Episode reward
- Current epsilon
- Collision status
- Goal completion
- Rolling success rate

After the initial **2000 episodes**, the trained weights are saved and used for an additional **50 episodes** of fine-tuning.

---

# Plot Training Results

Generate training analytics:

```bash
python training/plot_results.py
```

This produces:

- Reward Curve
- Success Rate
- Episode Length
- Epsilon Decay
- Training Statistics

Output:

```text
models/
└── training_plots.png
```

---

# Simulation Environment

The agent is trained inside a **1.5 m × 1.5 m Webots world** containing approximately **12–15 static obstacles** arranged to create narrow corridors and constrained navigation paths.

The robot learns to:

- Avoid collisions
- Navigate efficiently
- Reach the target position
- Optimize travel distance

---

# Future Work

The current implementation serves as the foundation for several planned research extensions aimed at improving robustness and sim-to-real applicability.

### Environment Improvements

- Introduce randomized obstacle placement after a fixed number of episodes.
- Generate multiple obstacle configurations during training to improve policy generalization.
- Evaluate robustness in previously unseen environments.

### Extended Training

- Increase total training to approximately **15,000 episodes**.
- Save model checkpoints every **1,000 episodes** for performance comparison and convergence analysis.

### Additional Reinforcement Learning Algorithms

- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

### Simulation Upgrade

- Transition from **Webots** to **Gazebo** for more realistic physics and sensor noise modeling.
- Compare policy performance between Webots and Gazebo.

### Robot Platform Upgrade

Replace the e-puck robot with a **TurtleBot** equipped with:

- 2D LiDAR
- Wheel Encoders
- IMU
- Odometry

This enables experimentation with more realistic sensing and navigation methods commonly used in autonomous robotics.

### Cross-Simulator Validation

- Train the navigation policy in Webots.
- Test the trained policy in Gazebo without retraining.
- Analyze the transferability and robustness of learned policies.

### Long-Term Research Directions

- Curriculum Learning
- Domain Randomization
- Dynamic Obstacle Avoidance
- Sim-to-Real Transfer
- Multi-Robot Reinforcement Learning

---

# Author

**Surya Pratap Devineni**

B.Tech Mechanical Engineering

SRM University AP

---

# Acknowledgments

This work is being developed as part of an academic research internship focused on:

- Mobile Robotics
- Autonomous Navigation
- Deep Reinforcement Learning
- Robot Path Planning
- Intelligent Control Systems

The project aims to provide a reproducible benchmarking framework for reinforcement learning-based obstacle avoidance in Webots while serving as a foundation for future research publications.

---

## License

This project is intended for academic and research purposes.