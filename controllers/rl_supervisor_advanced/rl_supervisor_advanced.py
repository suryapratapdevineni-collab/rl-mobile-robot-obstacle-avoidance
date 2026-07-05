import os
import sys

# 1. Calculate the explicit absolute project root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               # controllers/rl_supervisor_advanced
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))   # Root folder

# 2. Derive concrete folder destinations
ENV_DIR = os.path.join(PROJECT_ROOT, "env")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
ADVANCED_DQN_DIR = os.path.join(TRAINING_DIR, "advanced_dqn")

# 3. Force-inject into sys.path in strict hierarchical priority order
sys.path.insert(0, ADVANCED_DQN_DIR)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, ENV_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 4. Perform imports cleanly now that all shared roots are globally clear
from train_advanced import train_advanced_aligned

if __name__ == "__main__":
    agent, rewards = train_advanced_aligned(
        num_episodes=5000,
        max_episode_steps=1400,
        buffer_capacity=1000000,
    )