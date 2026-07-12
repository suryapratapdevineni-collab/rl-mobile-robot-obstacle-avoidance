# Reinforcement Learning-Based Mobile Robot Obstacle Avoidance in Webots

<p align="center">

Deep Reinforcement Learning Benchmark for Autonomous Mobile Robot Navigation using Standard Deep Q-Network (DQN) and Advanced Dueling Double DQN in the Webots Simulation Environment.

</p>

---

## Overview

This repository presents a reinforcement learning framework for autonomous obstacle avoidance using an **e-puck differential-drive mobile robot** in the **Webots** robotics simulator.

The project investigates the performance of two value-based reinforcement learning algorithms under identical training and testing conditions:

- Standard Deep Q-Network (DQN)
- Advanced Dueling Double Deep Q-Network (Advanced DDQN)

Both agents are trained using a custom **Gymnasium-compatible Webots environment** with an identical reward function, allowing a fair comparison of convergence, navigation efficiency, stability, and goal-reaching performance.

The repository also includes automated evaluation tools that generate research-quality plots and statistical comparison tables for benchmarking the trained models.

---

# Key Features

- Custom Webots Gymnasium Environment
- Standard DQN Implementation
- Advanced Dueling Double DQN
- Experience Replay
- Target Network
- Reward Shaping
- Automated Testing Framework
- Statistical Performance Comparison
- Publication-quality Figures
- Automated CSV Comparison Tables

---

# Repository Structure

```text
rl-mobile-robot-obstacle-avoidance/

├── controllers/
│
│   ├── rl_supervisor/
│   │      rl_supervisor.py
│   │
│   ├── rl_supervisor_advanced/
│   │      rl_supervisor_advanced.py
│   │
│   └── dqn_controller_test/
│          dqn_controller_test.py
│          networks.py
│
├── env/
│      webots_env.py
│      reward_function.py
│
├── training/
│      train_dqn.py
│      train_advanced_dqn.py
│      dqn_agent.py
│      dqn_network.py
│      replay_buffer.py
│      plot_results.py
│      comparison_tables.py
│
├── models/
│      standard_dqn.pth
│      advanced_dqn.pth
│      standard_dqn_test.csv
│      test_advanceddqn_results.csv
│
├── results/
│      comparison_summary.csv
│      comparison_summary.png
│      figure1_reward_comparison.png
│      figure2_success_rate.png
│      figure3_episode_steps.png
│      figure4_cumulative_success.png
│      figure5_reward_distribution.png
│      figure6_steps_distribution.png
│      figure7_reward_vs_steps.png
│      figure8_performance_summary.png
│
├── worlds/
│      oba.wbt
│      oba_advanced.wbt
│
└── README.md
```

---

# Implemented Algorithms

| Algorithm | Status |
|------------|--------|
| Standard Deep Q-Network | ✅ Implemented |
| Advanced Dueling Double DQN | ✅ Implemented |
| PPO | Planned |
| SAC | Planned |

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Episodes | 2000 |
| Fine-Tuning Episodes | 50 |
| Maximum Steps / Episode | 1400 |
| Replay Buffer | 1,000,000 |
| Initial ε | 1.0 |
| Final ε | 0.01 |
| Epsilon Decay | 0.99712 |
| Collision Termination | 50 Collisions |

---

# Reward Function

| Component | Reward |
|-----------|--------|
| Goal Reached | +500 |
| Bonus Goal Reward | +250 |
| Collision | -250 |
| Progress Reward | +50 × ΔDistance |
| Step Penalty | -0.1 |
| Obstacle Proximity | -1 to -5 |
| Velocity Reward | Positive |

The reward function encourages:

- Efficient navigation
- Collision avoidance
- Continuous progress
- Smooth forward motion
- Reduced unnecessary rotations

---

# Model Training

## Standard DQN

The baseline implementation consists of:

- Experience Replay
- Target Network
- Feedforward Q-Network
- ε-Greedy Exploration

The model is trained for **2000 episodes**.

---

## Advanced Dueling Double DQN

The enhanced implementation extends the baseline by incorporating:

- Double Q-Learning
- Dueling Network Architecture
- Improved target estimation
- Better training stability

The model is trained for **2000 episodes**, followed by **50 fine-tuning episodes** using the previously saved weights.

---

# Model Evaluation

The trained models are evaluated using identical testing environments.

For every test episode the following metrics are recorded:

- Episode Reward
- Episode Length
- Goal Completion
- Navigation Efficiency
- Success Rate

The evaluation results are stored as:

```text
models/
    standard_dqn_test.csv
    test_advanceddqn_results.csv
```

---

# Research Analytics

The repository automatically generates publication-ready figures.

Generated plots include:

- Reward Comparison
- Success Rate
- Episode Length
- Cumulative Success Rate
- Reward Distribution
- Episode Length Distribution
- Reward vs Steps
- Performance Summary

Output:

```text
results/
    figure1_reward_comparison.png
    ...
    figure8_performance_summary.png
```

---

# Statistical Comparison

Automated comparison tables are also generated.

The generated report contains metrics including:

- Total Episodes
- Successful Episodes
- Success Rate
- Highest Reward
- Average Reward
- Reward Standard Deviation
- Average Steps
- Fastest Successful Episode
- Most Efficient Episode
- Reward per Step
- Goal Completion Statistics
- Better Performing Model

Output:

```text
results/

comparison_summary.csv

comparison_summary.png
```

---

# Installation

```bash
git clone https://github.com/your_username/rl-mobile-robot-obstacle-avoidance.git

cd rl-mobile-robot-obstacle-avoidance
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Train Standard DQN

```bash
python training/train_dqn.py
```

Train Advanced DDQN

```bash
python training/train_advanced_dqn.py
```

---

# Testing

Evaluate the trained models

```bash
python controllers/dqn_controller_test/dqn_controller_test.py
```

---

# Generate Research Results

Generate comparison plots

```bash
python training/plot_results.py
```

Generate statistical comparison tables

```bash
python training/comparison_tables.py
```

---

# Simulation Environment

The experiments are conducted in a custom Webots environment containing approximately **12–15 static obstacles** arranged to create constrained navigation corridors.

Robot objectives include:

- Collision-free navigation
- Efficient path planning
- Goal-reaching
- Stable policy execution

---

# Future Work

Future extensions include:

- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Dynamic Obstacle Avoidance
- Curriculum Learning
- Domain Randomization
- Gazebo Benchmarking
- TurtleBot Platform
- Sim-to-Real Transfer
- Multi-Robot Reinforcement Learning

---

# Author

**Surya Pratap Devineni**

B.Tech Mechanical Engineering

SRM University AP

---

# License

This repository is intended for academic research and educational purposes.