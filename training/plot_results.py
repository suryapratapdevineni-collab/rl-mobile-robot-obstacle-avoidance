import csv
import os
import matplotlib.pyplot as plt

def load_log(log_path):
    episodes, rewards, steps, collided, goals, epsilons = [], [], [], [], [], []
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['Episode']))
            rewards.append(float(row['Reward']))
            steps.append(int(row['Steps']))
            collided.append(int(row['Collided']))
            goals.append(int(row['Goal']))
            epsilons.append(float(row['Epsilon']))
    return episodes, rewards, steps, collided, goals, epsilons

def moving_average(data, window=200):
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

def plot_comparative():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.abspath(os.path.join(base_dir, '..', 'models'))
    
    standard_log = os.path.join(models_dir, 'training_log.csv')
    advanced_log = os.path.join(models_dir, 'advanced_aligned_training_log.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Performance Comparison Matrix: Vanilla DQN vs Double Dueling DQN", fontsize=14, fontweight='bold')
    
    std_data = load_log(standard_log)
    adv_data = load_log(advanced_log)
    
    if std_data:
        ep, rew, step, col, go, _ = std_data
        axes[0, 0].plot(ep, moving_average(rew), color='blue', label='Standard DQN')
        axes[0, 1].plot(ep, [v * 100 for v in moving_average(col)], color='blue')
        axes[1, 0].plot(ep, [v * 100 for v in moving_average(go)], color='blue')
        axes[1, 1].plot(ep, moving_average(step), color='blue')
        
    if adv_data:
        ep, rew, step, col, go, _ = adv_data
        axes[0, 0].plot(ep, moving_average(rew), color='crimson', label='Double Dueling DQN')
        axes[0, 1].plot(ep, [v * 100 for v in moving_average(col)], color='crimson')
        axes[1, 0].plot(ep, [v * 100 for v in moving_average(go)], color='crimson')
        axes[1, 1].plot(ep, moving_average(step), color='crimson')

    axes[0, 0].set_title('Smoothed Accumulative Rewards')
    axes[0, 0].legend()
    axes[0, 1].set_title('Collision Rates (Rolling %)')
    axes[1, 0].set_title('Goal Success Rates (%)')
    axes[1, 0].axhline(80, color='black', linestyle='--')
    axes[1, 1].set_title('Steps Expended per Episode')
    
    for ax in axes.flat:
        ax.set_xlabel('Episodes')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = os.path.join(models_dir, 'comparative_training_plots.png')
    plt.savefig(out_path, dpi=200)
    print(f"Metrics plot updated at: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparative()