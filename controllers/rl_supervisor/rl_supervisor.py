import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))

from train_dqn import train

if __name__ == "__main__":
    agent, rewards = train(
        num_episodes=2500,
        max_episode_steps=1800,
        buffer_capacity=5000000,
    )