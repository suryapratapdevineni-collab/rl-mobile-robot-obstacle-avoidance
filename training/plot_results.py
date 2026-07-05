import csv
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_log(log_path):
    episodes, rewards, steps, collided, goals, epsilons = [], [], [], [], [], []
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fallback support for column key name variations
                episodes.append(int(row.get('Episode') or row.get('episode') or 0))
                rewards.append(float(row.get('Reward') or row.get('reward') or 0.0))
                steps.append(int(row.get('Steps') or row.get('steps') or 0))
                
                # Normalize collision and goal keys between log specifications
                col_val = row.get('CollisionCount') or row.get('Collided') or row.get('collision_count') or 0
                collided.append(int(col_val))
                
                goal_val = row.get('ReachedGoal') or row.get('Goal') or row.get('reached_goal') or 0
                goals.append(int(goal_val))
                
                epsilons.append(float(row.get('Epsilon') or row.get('epsilon') or 0.0))
    except Exception as e:
        # Avoid crashing if a parallel process is writing to the file at the exact same split-second
        return None
    
    if not episodes:
        return None
    return episodes, rewards, steps, collided, goals, epsilons

def moving_average(data, window=20):
    if len(data) == 0:
        return []
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

def main():
    # Setup Paths dynamically relative to execution location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'models'))
    
    # Try looking in local folder first, then fallback to relative models directory
    standard_log = "training_log.csv" if os.path.exists("training_log.csv") else os.path.join(models_dir, 'training_log.csv')
    advanced_log = "advanced_aligned_training_log.csv" if os.path.exists("advanced_aligned_training_log.csv") else os.path.join(models_dir, 'advanced_aligned_training_log.csv')

    print(f"Tracking Log 1: {standard_log}")
    print(f"Tracking Log 2: {advanced_log}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window_size = 20  # Smooth out variations over a 20-episode horizon

    def animate(i):
        std_data = load_log(standard_log)
        adv_data = load_log(advanced_log)

        # Clear axes for incoming frame redraw
        for row in axes:
            for ax in row:
                ax.clear()

        # Re-map standard baseline metrics
        if std_data:
            ep, rew, step, col, go, _ = std_data
            axes[0, 0].plot(ep, moving_average(rew, window_size), color='tab:blue', linewidth=2, label='Standard DQN')
            axes[0, 1].plot(ep, moving_average(col, window_size), color='tab:blue', linewidth=2)
            axes[1, 0].plot(ep, [v * 100 for v in moving_average(go, window_size)], color='tab:blue', linewidth=2)
            axes[1, 1].plot(ep, moving_average(step, window_size), color='tab:blue', linewidth=2)

        # Re-map advanced dueling-ddqn metrics
        if adv_data:
            ep, rew, step, col, go, _ = adv_data
            axes[0, 0].plot(ep, moving_average(rew, window_size), color='tab:orange', linewidth=2, label='Advanced Dueling DDQN')
            axes[0, 1].plot(ep, moving_average(col, window_size), color='tab:orange', linewidth=2)
            axes[1, 0].plot(ep, [v * 100 for v in moving_average(go, window_size)], color='tab:orange', linewidth=2)
            axes[1, 1].plot(ep, moving_average(step, window_size), color='tab:orange', linewidth=2)

        # Apply structural styling attributes
        axes[0, 0].set_title(f'Smoothed Rewards ({window_size}-Ep Moving Avg)')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right')

        axes[0, 1].set_title('Average Collisions per Episode')
        axes[0, 1].set_ylabel('Bumps')
        axes[0, 1].set_ylim(-2, 55) # Clear visual indicator for our 50 max collision limit

        axes[1, 0].set_title('Goal Success Rate (%)')
        axes[1, 0].set_ylabel('Success %')
        axes[1, 0].set_ylim(-5, 105)

        axes[1, 1].set_title('Episode Durations (Steps)')
        axes[1, 1].set_ylabel('Steps Taken')

        for row in axes:
            for ax in row:
                ax.set_xlabel('Episodes')

        plt.tight_layout()

    # Create live animation loop updating every 5000 milliseconds (5 seconds)
    ani = FuncAnimation(fig, animate, cache_frame_data=False, interval=5000)
    plt.show()

if __name__ == '__main__':
    main()