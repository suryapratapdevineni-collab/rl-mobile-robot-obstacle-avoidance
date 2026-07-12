
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = 20

def moving_average(data, window=20):
    out=[]
    for i in range(len(data)):
        s=max(0,i-window+1)
        out.append(sum(data[s:i+1])/(i-s+1))
    return out

def load_csv(path):
    df=pd.read_csv(path)
    return {
        "Episode":df["Episode"],
        "Reward":df["Reward"],
        "Steps":df["Steps"],
        "ReachedGoal":df["ReachedGoal"]
    }

base=os.path.dirname(os.path.abspath(__file__))
models=os.path.join(os.path.dirname(base),"models")
results=os.path.join(os.path.dirname(base),"results")
os.makedirs(results,exist_ok=True)

std=load_csv(os.path.join(models,"standard_dqn_test.csv"))
adv=load_csv(os.path.join(models,"test_advanceddqn_results.csv"))

def save(fig,name):
    fig.tight_layout()
    fig.savefig(os.path.join(results,name),dpi=300,bbox_inches="tight")
    plt.close(fig)

# Figure1 Reward
fig=plt.figure(figsize=(8,5))
plt.plot(std["Episode"],moving_average(std["Reward"],WINDOW),label="Standard DQN",lw=2)
plt.plot(adv["Episode"],moving_average(adv["Reward"],WINDOW),label="Advanced Dueling DDQN",lw=2)
plt.title("Reward Comparison")
plt.xlabel("Episode"); plt.ylabel("Reward")
plt.grid(True); plt.legend()
save(fig,"figure1_reward_comparison.png")

# Figure2 Success
fig=plt.figure(figsize=(8,5))
plt.plot(std["Episode"],[x*100 for x in moving_average(std["ReachedGoal"],WINDOW)],label="Standard DQN",lw=2)
plt.plot(adv["Episode"],[x*100 for x in moving_average(adv["ReachedGoal"],WINDOW)],label="Advanced Dueling DDQN",lw=2)
plt.title("Success Rate")
plt.xlabel("Episode"); plt.ylabel("Success (%)")
plt.ylim(0,100); plt.grid(True); plt.legend()
save(fig,"figure2_success_rate.png")

# Figure3 Steps
fig=plt.figure(figsize=(8,5))
plt.plot(std["Episode"],moving_average(std["Steps"],WINDOW),label="Standard DQN",lw=2)
plt.plot(adv["Episode"],moving_average(adv["Steps"],WINDOW),label="Advanced Dueling DDQN",lw=2)
plt.title("Episode Length")
plt.xlabel("Episode"); plt.ylabel("Steps")
plt.grid(True); plt.legend()
save(fig,"figure3_episode_steps.png")

# Figure4 cumulative success
fig=plt.figure(figsize=(8,5))
plt.plot(std["Episode"],std["ReachedGoal"].cumsum()/(std["Episode"])*100,label="Standard DQN",lw=2)
plt.plot(adv["Episode"],adv["ReachedGoal"].cumsum()/(adv["Episode"])*100,label="Advanced Dueling DDQN",lw=2)
plt.title("Cumulative Success Rate")
plt.xlabel("Episode"); plt.ylabel("Success (%)")
plt.grid(True); plt.legend()
save(fig,"figure4_cumulative_success.png")

# Figure5 box rewards
fig=plt.figure(figsize=(7,5))
plt.boxplot([std["Reward"],adv["Reward"]],tick_labels=["Standard","Advanced"])
plt.title("Reward Distribution")
plt.ylabel("Reward")
save(fig,"figure5_reward_distribution.png")

# Figure6 histogram steps
fig=plt.figure(figsize=(8,5))
plt.hist(std["Steps"],bins=20,alpha=0.6,label="Standard")
plt.hist(adv["Steps"],bins=20,alpha=0.6,label="Advanced")
plt.title("Episode Length Distribution")
plt.xlabel("Steps"); plt.ylabel("Frequency")
plt.legend(); plt.grid(True)
save(fig,"figure6_steps_distribution.png")

# Figure7 scatter
fig=plt.figure(figsize=(8,5))
plt.scatter(std["Steps"],std["Reward"],alpha=0.5,label="Standard")
plt.scatter(adv["Steps"],adv["Reward"],alpha=0.5,label="Advanced")
plt.title("Reward vs Steps")
plt.xlabel("Steps"); plt.ylabel("Reward")
plt.legend(); plt.grid(True)
save(fig,"figure7_reward_vs_steps.png")

# Figure8 summary
metrics=["Mean Reward","Mean Steps","Success %"]
std_vals=[std["Reward"].mean(),std["Steps"].mean(),std["ReachedGoal"].mean()*100]
adv_vals=[adv["Reward"].mean(),adv["Steps"].mean(),adv["ReachedGoal"].mean()*100]
import numpy as np
x=np.arange(len(metrics)); w=0.35
fig=plt.figure(figsize=(8,5))
plt.bar(x-w/2,std_vals,w,label="Standard")
plt.bar(x+w/2,adv_vals,w,label="Advanced")
plt.xticks(x,metrics)
plt.title("Performance Summary")
plt.legend(); plt.grid(axis="y")
save(fig,"figure8_performance_summary.png")

summary=pd.DataFrame({
"Metric":["Episodes","Mean Reward","Max Reward","Mean Steps","Success Rate (%)"],
"Standard":[len(std["Episode"]),std["Reward"].mean(),std["Reward"].max(),std["Steps"].mean(),std["ReachedGoal"].mean()*100],
"Advanced":[len(adv["Episode"]),adv["Reward"].mean(),adv["Reward"].max(),adv["Steps"].mean(),adv["ReachedGoal"].mean()*100]
})
summary.to_csv(os.path.join(results,"summary_statistics.csv"),index=False)
print("Saved results to:",results)