# Reinforcement Learning-Based Obstacle Avoidance for Mobile Robots in Webots

## Overview

This repository presents a reinforcement learning framework for autonomous mobile robot navigation using the **Webots** simulation environment. An **e-puck differential-drive robot** is trained to navigate through a complex obstacle-filled environment while learning collision-free path planning using Deep Reinforcement Learning.

The project systematically evaluates reinforcement learning algorithms under identical simulation conditions to compare sample efficiency, convergence behavior, navigation performance, and path optimization. It is being developed as part of an academic research internship with the goal of publication.

---

## Objectives

- Design a robust **OpenAI Gym / Gymnasium-compatible** environment for Webots.
- Train an autonomous e-puck robot to navigate obstacle-rich environments.
- Benchmark multiple reinforcement learning algorithms using identical environments.
- Engineer an effective reward function that encourages efficient navigation while preventing reward-hacking behaviors.

---

# Technical Performance Configuration

| Hyperparameter | Value | Purpose |
|---------------|-------|---------|
| **Training Episodes** | 2500 | Allows stable convergence after exploration |
| **Maximum Steps / Episode** | 800 | Provides sufficient navigation horizon |
| **Episode Termination** | Immediate on Collision | Prevents inefficient looping |
| **Initial ε (Exploration)** | 1.0 | Maximum early exploration |
| **Final ε** | 0.01 | Near-greedy exploitation |
| **Epsilon Decay** | 0.99712 | Smooth decay over ~1600 episodes |
| **Replay Buffer Size** | 500,000 | Prevents catastrophic forgetting |

---

# Reward Function

The reward structure is carefully engineered to discourage "suicide policies" (intentional early collisions) while encouraging smooth forward navigation.

| Component | Reward |
|----------|---------|
| Goal Reached | **+500.0** |
| Collision | **-250.0** |
| Step Penalty | **-0.1** |
| Progress Reward | **+50 × Δ Distance** |
| Obstacle Proximity | **-1 to -5** |
| Velocity Reward | Positive reward for forward motion |

### Reward Design

- Large terminal reward for successfully reaching the goal.
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
│       └── supervisor.py          # Webots Supervisor controller
│
├── env/
│   └── webots_env.py              # Custom Gymnasium environment
│
├── training/
│   ├── reward_function.py         # Reward computation
│   ├── train_dqn.py               # Training entry point
│   ├── dqn_agent.py               # DQN agent implementation
│   ├── dqn_network.py             # Neural network architecture
│   ├── replay_buffer.py           # Experience replay memory
│   └── plot_results.py            # Training visualization
│
├── models/
│   ├── training_log.csv
│   └── training_plots.png
│
├── worlds/
│   └── oba.wbt                    # Webots simulation world
│
└── README.md
```

---

# Implemented and Planned Algorithms

| Algorithm | Status |
|-----------|--------|
| ✅ Deep Q-Network (DQN) | Implemented |
| ⬜ Double Deep Q-Network (DDQN) | Planned |
| ⬜ Dueling Deep Q-Network | Planned |
| ⬜ Proximal Policy Optimization (PPO) | Planned |
| ⬜ Soft Actor-Critic (SAC) | Planned |

### Deep Q-Network (DQN)

Baseline value-based reinforcement learning algorithm using:

- Experience Replay
- Target Network
- ε-Greedy Exploration
- Neural Network Function Approximation

### Planned Extensions

- **Double DQN** to reduce value overestimation.
- **Dueling DQN** for separate state-value and advantage estimation.
- **PPO** for stable policy-gradient optimization.
- **SAC** for entropy-regularized continuous control.

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

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run the DQN training pipeline:

```bash
python training/train_dqn.py
```

During training the console displays:

- Episode number
- Episode reward
- Current epsilon
- Collision status
- Goal completion
- Rolling success rate

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

```
models/
└── training_plots.png
```

---

# Simulation Environment

The agent is trained inside a **1.5 m × 1.5 m Webots world** containing approximately **12–15 static obstacles** arranged to create narrow corridors and constrained navigation paths.

The robot must learn to:

- Avoid collisions
- Navigate efficiently
- Reach the target position
- Optimize travel distance

---

# Future Work

- Double DQN
- Dueling DQN
- PPO
- SAC
- Curriculum Learning
- Domain Randomization
- Sim-to-Real Transfer
- Dynamic Obstacle Avoidance
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
