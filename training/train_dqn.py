import os
import sys
import csv
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))

from webots_env import WebotsEnv
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

def train(
    num_episodes=1000,         # UPDATED: 1000 episodes
    batch_size=128,
    buffer_capacity=500000,
    target_update_every=15,
    warmup_steps=2000,
    max_episode_steps=800,     # UPDATED: Max steps 800
    train_every=2,
    log_path=None,
):
    env = WebotsEnv(max_episode_steps=max_episode_steps)

    # Calculate epsilon decay for a 1600 episode horizon:
    # Epsilon ends at 0.02. exp(log(0.02) / 1600) ≈ 0.9975
    agent = DQNAgent(
        obs_dim=10,
        n_actions=6,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9975   # UPDATED: Decays slowly over a 1600 episode horizon schedule
    )

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    if log_path is None:
        log_path = os.path.join(model_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Steps", "Collided", "Goal", "Epsilon", "Buffer"])

    total_steps = 0
    best_reward = -1e9
    reward_history = []
    goal_history = []

    print("=" * 70)
    print("Training Started (Dynamic Random Environments)")
    print(f"Episodes={num_episodes} | Max steps/ep={max_episode_steps}")
    print("=" * 70)

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        was_collision = 0
        reached_goal = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("collision", False):
                was_collision = 1 # Flagged termination by collision

            if info.get("goal_reached", False):
                reached_goal = True

            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps % train_every == 0 and len(replay_buffer) >= max(batch_size, warmup_steps):
                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)

        agent.decay_epsilon()

        if (episode + 1) % target_update_every == 0:
            agent.update_target_network()

        reward_history.append(episode_reward)
        goal_history.append(int(reached_goal))

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.q_network.state_dict(), os.path.join(model_dir, "best_dqn_epuck.pt"))

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                round(episode_reward, 2),
                episode_steps,
                was_collision,
                int(reached_goal),
                round(agent.epsilon, 4),
                len(replay_buffer)
            ])

        avg_reward = np.mean(reward_history[-20:])
        success_rate_20 = np.mean(goal_history[-20:]) * 100

        print(
            f"Ep {episode + 1:4d} | "
            f"Rew {episode_reward:7.1f} | "
            f"Avg20 {avg_reward:7.1f} | "
            f"Steps {episode_steps:3d} | "
            f"Collided? {was_collision} | "
            f"Succ%20 {success_rate_20:5.1f}% | "
            f"Eps {agent.epsilon:.3f}"
        )

    final_path = os.path.join(model_dir, "dqn_epuck.pt")
    torch.save(agent.q_network.state_dict(), final_path)
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return agent, reward_history

if __name__ == "__main__":
    train()