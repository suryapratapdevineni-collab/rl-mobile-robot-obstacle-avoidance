import csv
import os
import matplotlib.pyplot as plt


def load_log(log_path):
    """
    FIX: the original version declared `goals = []` but never appended
    to it inside the loop (only `epsilons.append(...)` was called), so
    `goals` stayed permanently empty -- this would crash later in
    plot_all() at moving_average(goals) / sum(goals). The 'Goal' column
    IS written by train_dqn.py's CSV logger, it just wasn't being read
    here. Fixed by reading row['Goal'] each iteration.
    """
    episodes, rewards, steps, collisions, goals, epsilons = [], [], [], [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['Episode']))
            rewards.append(float(row['Reward']))
            steps.append(int(row['Steps']))
            collisions.append(int(row['Collisions']))
            goals.append(int(row['Goal']))
            epsilons.append(float(row['Epsilon']))
    return episodes, rewards, steps, collisions, goals, epsilons


def moving_average(data, window=20):
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def plot_all(log_path):
    episodes, rewards, steps, collisions, goals, epsilons = load_log(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Reward per episode
    axes[0, 0].plot(episodes, rewards, alpha=0.3, label='raw')
    axes[0, 0].plot(episodes, moving_average(rewards), label='moving avg (20)')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].axhline(0, color='gray', linewidth=0.8, linestyle='--')

    # 2. Collisions per episode
    axes[0, 1].plot(episodes, collisions, alpha=0.3, color='red', label='raw')
    axes[0, 1].plot(episodes, moving_average(collisions), color='darkred', label='moving avg (20)')
    axes[0, 1].set_title('Collisions per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Collision Count')
    axes[0, 1].legend()

    # 3. Goal success rate (rolling, since per-episode is just 0/1)
    success_rate = moving_average(goals, window=20)
    axes[1, 0].plot(episodes, success_rate, color='green')
    axes[1, 0].axhline(0.8, color='black', linewidth=1.0, linestyle='--', label='80% target')
    axes[1, 0].set_title('Goal Success Rate (rolling 20-episode avg)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()

    # 4. Steps per episode (time taken)
    axes[1, 1].plot(episodes, steps, alpha=0.3, color='purple', label='raw')
    axes[1, 1].plot(episodes, moving_average(steps), color='indigo', label='moving avg (20)')
    axes[1, 1].set_title('Steps per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].legend()

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(log_path), 'training_plots.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved plots to {out_path}")
    print(f"Total episodes that reached goal: {sum(goals)} / {len(goals)}")

    if len(goals) >= 50:
        last_50 = goals[-50:]
        print(f"Success rate (last 50 episodes): {sum(last_50)/len(last_50)*100:.1f}%")

    plt.show()


if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_log.csv')
    plot_all(log_path)