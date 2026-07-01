# Reinforcement Learning-Based Obstacle Avoidance for Mobile Robots in Webots

## Overview

This repository contains my research work on reinforcement learning-based obstacle avoidance for autonomous mobile robots using the Webots simulation environment.

The primary objective of this project is to develop a reinforcement learning framework and evaluate the performance of different reinforcement learning algorithms for autonomous navigation and obstacle avoidance. The work is being carried out as part of my research internship.

---

## Objectives

- Develop a reinforcement learning framework for mobile robot navigation.
- Train a mobile robot to autonomously avoid obstacles.
- Compare the performance of different reinforcement learning algorithms.
- Analyze the strengths and limitations of each algorithm under identical simulation conditions.

---

## Implemented Algorithm

- Deep Q-Network (DQN)

## Planned Algorithms

- Double Deep Q-Network (DDQN)
- Dueling Deep Q-Network
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

---

## Repository Structure

```
rl-mobile-robot-obstacle-avoidance/
│
├── controllers/
│   └── rl_supervisor/
│
├── env/
│   └── webots_env.py
│
├── training/
│   ├── train_dqn.py
│   ├── dqn_agent.py
│   ├── dqn_network.py
│   ├── replay_buffer.py
│   └── plot_results.py
│
├── models/
│   ├── trained models
│   ├── training logs
│   └── training plots
│
├── worlds/
│   └── oba.wbt
│
└── README.md
```

---

## Software and Tools

- Python
- Webots
- PyTorch
- NumPy
- Matplotlib

---

## Current Progress

- ✅ Webots simulation environment developed
- ✅ Reinforcement learning environment implemented
- ✅ DQN agent implemented
- ✅ Reward function designed
- ✅ Training and evaluation pipeline developed
- ✅ Model checkpoint saving

---

## Future Work

- Implement DDQN and Dueling DQN
- Extend the framework to PPO and SAC
- Compare the performance of different RL algorithms
- Improve reward function design
- Evaluate algorithms using common navigation performance metrics
- Prepare the work for research publication

---

## Author

**Surya Pratap Devineni**

B.Tech Mechanical Engineering

SRM University, AP

---

## Acknowledgement

This project is being developed as part of my research internship in the field of reinforcement learning and autonomous mobile robot navigation.
