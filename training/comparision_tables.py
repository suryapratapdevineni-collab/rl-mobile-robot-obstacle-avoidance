
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR=os.path.dirname(BASE_DIR)
MODELS_DIR=os.path.join(PROJECT_DIR,"models")
RESULTS_DIR=os.path.join(PROJECT_DIR,"results")
os.makedirs(RESULTS_DIR,exist_ok=True)

STD=os.path.join(MODELS_DIR,"standard_dqn_test.csv")
ADV=os.path.join(MODELS_DIR,"test_advanceddqn_results.csv")

std=pd.read_csv(STD)
adv=pd.read_csv(ADV)

def calc(df):
    suc=df[df["ReachedGoal"]==1]
    fail=df[df["ReachedGoal"]==0]
    eff=df["Reward"]/df["Steps"]
    best=df.loc[df["Reward"].idxmax()]
    fast=suc.loc[suc["Steps"].idxmin()] if len(suc)>0 else None
    return {
        "Total Episodes":len(df),
        "Successful Episodes":int(df["ReachedGoal"].sum()),
        "Success Rate (%)":round(df["ReachedGoal"].mean()*100,2),
        "Highest Reward":round(df["Reward"].max(),2),
        "Average Reward":round(df["Reward"].mean(),2),
        "Median Reward":round(df["Reward"].median(),2),
        "Reward Std Dev":round(df["Reward"].std(),2),
        "Lowest Reward":round(df["Reward"].min(),2),
        "Average Steps":round(df["Steps"].mean(),2),
        "Fastest Successful Episode":int(fast["Episode"]) if fast is not None else "-",
        "Fastest Successful Steps":int(fast["Steps"]) if fast is not None else "-",
        "Highest Reward Episode":int(best["Episode"]),
        "Highest Reward Episode Steps":int(best["Steps"]),
        "Average Reward (Success)":round(suc["Reward"].mean(),2),
        "Average Reward (Failure)":round(fail["Reward"].mean(),2),
        "Average Steps (Success)":round(suc["Steps"].mean(),2),
        "Average Steps (Failure)":round(fail["Steps"].mean(),2),
        "Best Efficiency (Reward/Step)":round(eff.max(),4),
        "Most Efficient Episode":int(df.loc[eff.idxmax(),"Episode"])
    }

S=calc(std);A=calc(adv)
lower={"Average Steps","Fastest Successful Steps","Average Steps (Success)","Average Steps (Failure)"}
rows=[]
for k in S:
    sv,av=S[k],A[k]
    if isinstance(sv,(int,float,np.integer,np.floating)) and isinstance(av,(int,float,np.integer,np.floating)):
        if k in lower:
            b="Standard DQN" if sv<av else "Advanced DDQN" if av<sv else "Tie"
        else:
            b="Standard DQN" if sv>av else "Advanced DDQN" if av>sv else "Tie"
    else:
        b="-"
    rows.append([k,sv,av,b])

summary=pd.DataFrame(rows,columns=["Metric","Standard DQN","Advanced DDQN","Better Model"])
csv_path=os.path.join(RESULTS_DIR,"comparison_summary.csv")
summary.to_csv(csv_path,index=False)

fig_h=max(6,len(summary)*0.42)
fig,ax=plt.subplots(figsize=(12,fig_h))
ax.axis("off")
tbl=ax.table(cellText=summary.values,colLabels=summary.columns,loc="center",cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3,1.6)
for (r,c),cell in tbl.get_celld().items():
    if r==0:
        cell.set_facecolor("#1f77b4")
        cell.set_text_props(color="white",weight="bold")
    else:
        cell.set_facecolor("#f8f9fa" if r%2==0 else "white")
        if c==3:
            txt=cell.get_text().get_text()
            if txt=="Advanced DDQN":
                cell.set_facecolor("#d4edda")
            elif txt=="Standard DQN":
                cell.set_facecolor("#fff3cd")
plt.title("Standard DQN vs Advanced DDQN Performance Comparison",fontsize=14,weight="bold",pad=20)
png_path=os.path.join(RESULTS_DIR,"comparison_summary.png")
plt.savefig(png_path,dpi=300,bbox_inches="tight")
plt.close()
print("Generated:",csv_path)
print("Generated:",png_path)