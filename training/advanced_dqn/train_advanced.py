import os
import sys
import csv
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from webots_env import WebotsEnv
from advanced_agent import AdvancedDQNAgent
from replay_buffer import ReplayBuffer

def train_advanced_aligned(
    num_episodes=5000,            
    batch_size=256,
    buffer_capacity=1000000,
    target_update_every=20,
    warmup_steps=20000,
    max_episode_steps=1400,
    train_every=4,
    log_path=None,
):
    env = WebotsEnv(max_episode_steps=max_episode_steps)
    agent = AdvancedDQNAgent(obs_dim=10, n_actions=6, gamma=0.985, loss_fn="huber", grad_clip_norm=1.0, epsilon_decay=0.999023)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    if log_path is None:
        log_path = os.path.join(model_dir, "advanced_aligned_training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Steps", "CollisionCount", "ReachedGoal", "Epsilon", "BufferCapacity"])

    reward_history = []
    goal_history = []
    best_reward = -float("inf")
    total_steps = 0

    print("Launching Advanced Double Dueling DQN Training Routine (Continuous Bumping Engine)...\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
            reached_goal = info["reached_goal"]
            collision_count = info["collision_count"]

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps % train_every == 0 and len(replay_buffer) >= warmup_steps:
                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)

        agent.decay_epsilon()

        if (episode + 1) % target_update_every == 0:
            agent.update_target_network()

        reward_history.append(episode_reward)
        goal_history.append(int(reached_goal))

        if episode_reward > best_reward and len(replay_buffer) >= warmup_steps:
            best_reward = episode_reward
            torch.save(agent.q_network.state_dict(), os.path.join(model_dir, "best_advanced_aligned_dqn.pt"))

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, round(episode_reward, 2), episode_steps, collision_count, int(reached_goal), round(agent.epsilon, 4), len(replay_buffer)])

        avg_window = min(len(reward_history), 50)
        avg_reward = np.mean(reward_history[-avg_window:])
        success_rate_window = np.mean(goal_history[-avg_window:]) * 100

        if reached_goal:
            status_str = "SUCCESS"
        elif episode_steps >= max_episode_steps:
            status_str = "TIMEOUT"
        else:
            status_str = "RUNNING"

        print(f"Ep {episode + 1:5d} | Status: {status_str:7s} | Steps: {episode_steps:4d} | Bumps: {collision_count:3d} | Reward: {episode_reward:7.1f} | AvgReward: {avg_reward:7.1f} | Success: {success_rate_window:5.1f}%")

    return agent, reward_history