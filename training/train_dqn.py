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
    num_episodes=250,
    batch_size=128,
    buffer_capacity=300000,
    target_update_every=15,
    warmup_steps=1500,
    max_episode_steps=1800,
    train_every=2,
    log_path=None,
):
    """
    ======================================================================
    RETUNED FOR: 250-episode budget, target >= 80% goal-success rate.
    ======================================================================
    Changes from the original defaults, and why:

      num_episodes: 1000 -> 250
        Matches the actual training budget being targeted.

      max_episode_steps: 2000 -> 1800
        Matches WebotsEnv's default episode length. This number now
        matters for more than just a timeout: the reward function in
        webots_env.py (all-negative-except-goal design) computes the
        terminal goal reward as (max_episode_steps - collisions) * 2,
        so this value MUST match whatever WebotsEnv is actually using
        -- if you change one, change the other. See webots_env.py's
        docstring for the full reward-function reasoning; that file is
        the single source of truth for reward shaping, shared
        unmodified across all four algorithm variants.

      target_update_every: 25 -> 15
        With fewer total episodes, the target network needs to track
        the online network more closely so bootstrapped Q-value
        targets do not go stale for a large fraction of the (now much
        shorter) training run.

      warmup_steps: 3000 -> 1500
        At up to 400 steps/episode, a warmup of 3000 steps alone could
        burn through 7-8+ full episodes before any learning starts --
        a large chunk of a 250-episode budget. Lowered so learning
        begins sooner without sacrificing a reasonably-filled buffer.

      train_every: 4 -> 2
        Trains on twice as many minibatches per environment step,
        extracting more learning signal per episode -- needed since
        there are fewer episodes overall to learn from.

    See dqn_agent.py for the matching epsilon_decay / epsilon_end
    changes (0.997->0.97, 0.05->0.02), which are just as important as
    the changes here -- without faster epsilon decay, the agent would
    still be acting substantially randomly near episode 250.
    ======================================================================
    """

    env = WebotsEnv(max_episode_steps=max_episode_steps)

    agent = DQNAgent(
        obs_dim=10,
        n_actions=6
    )

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    if log_path is None:
        log_path = os.path.join(model_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode",
            "Reward",
            "Steps",
            "Collisions",
            "Goal",
            "Epsilon",
            "Buffer"
        ])

    total_steps = 0
    best_reward = -1e9
    reward_history = []
    goal_history = []

    print("=" * 70)
    print("Training Started")
    print(f"Episodes={num_episodes}  Max steps/ep={max_episode_steps}")
    print("=" * 70)

    for episode in range(num_episodes):

        state, info = env.reset()

        done = False
        episode_reward = 0
        episode_steps = 0
        collision_count = 0
        reached_goal = False

        while not done:

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            if info.get("collision", False):
                collision_count += 1

            if info.get("goal_reached", False):
                reached_goal = True

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done)
            )

            state = next_state

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            ###############################
            # Train every few steps
            ###############################
            if (
                total_steps % train_every == 0 and
                len(replay_buffer) >= max(batch_size, warmup_steps)
            ):

                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)

        ##################################
        # Decay epsilon
        ##################################
        agent.decay_epsilon()

        ##################################
        # Update target network
        ##################################
        if (episode + 1) % target_update_every == 0:
            agent.update_target_network()

        reward_history.append(episode_reward)
        goal_history.append(int(reached_goal))

        ##################################
        # Save best model
        ##################################
        if episode_reward > best_reward:

            best_reward = episode_reward

            torch.save(
                agent.q_network.state_dict(),
                os.path.join(model_dir, "best_dqn_epuck.pt")
            )

        ##################################
        # Log CSV
        ##################################
        with open(log_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                episode + 1,
                round(episode_reward, 2),
                episode_steps,
                collision_count,
                int(reached_goal),
                round(agent.epsilon, 4),
                len(replay_buffer)
            ])

        ##################################
        # Console Output
        ##################################
        avg_reward = np.mean(reward_history[-20:])
        success_rate_20 = np.mean(goal_history[-20:]) * 100

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward {episode_reward:8.2f} | "
            f"Avg20 {avg_reward:8.2f} | "
            f"Steps {episode_steps:4d} | "
            f"Collisions {collision_count:3d} | "
            f"Goal {reached_goal} | "
            f"Succ%20 {success_rate_20:5.1f} | "
            f"Epsilon {agent.epsilon:.3f}"
        )

    ##################################
    # Save final model
    ##################################
    final_path = os.path.join(model_dir, "dqn_epuck.pt")

    torch.save(
        agent.q_network.state_dict(),
        final_path
    )

    overall_success_rate = np.mean(goal_history) * 100
    last_50_success_rate = np.mean(goal_history[-50:]) * 100 if len(goal_history) >= 50 else overall_success_rate

    print("=" * 70)
    print("Training Finished")
    print(f"Final Model : {final_path}")
    print(f"Best Reward : {best_reward:.2f}")
    print(f"Overall Success Rate       : {overall_success_rate:.1f}%")
    print(f"Success Rate (last 50 eps) : {last_50_success_rate:.1f}%")
    print("=" * 70)

    return agent, reward_history


if __name__ == "__main__":

    agent, rewards = train(
        num_episodes=250,
        batch_size=128,
        buffer_capacity=500000,
        target_update_every=15,
        warmup_steps=2000,
        max_episode_steps=1800,
        train_every=2,
    )