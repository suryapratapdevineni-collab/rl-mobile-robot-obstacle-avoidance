import os
import sys
import csv
import torch
import numpy as np

# Adjust runtime search directories to recognize root project modules
current_dir = os.path.dirname(os.path.abspath(__file__))               
controller_dir = os.path.dirname(current_dir)                          
root_project_dir = os.path.dirname(controller_dir)                      

if root_project_dir not in sys.path:
    sys.path.append(root_project_dir)

from networks import StandardDQN, DuelingDQN
from webots_env import WebotsEnv

def run_deterministic_test(model_filename, model_type='standard', num_episodes=20):
    """
    Main evaluation routine running directly inside Webots. 
    Loads weights, tracks supervisor positions, and writes results to a CSV log.
    """
    model_path = os.path.join(root_project_dir, 'models', model_filename)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Weight file not found at: {model_path}")
        return

    # 1. Instantiate the network blueprint
    if model_type == 'advanced':
        model = DuelingDQN(state_dim=10, action_dim=6)
        print(f"\n--> Instantiated Dueling DQN architecture for {model_filename}")
    else:
        model = StandardDQN(state_dim=10, action_dim=6)
        print(f"\n--> Instantiated Standard DQN architecture for {model_filename}")

    # 2. Extract weights to core execution nodes
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("--> Model weights loaded successfully into network nodes.")

    # 3. Setup CSV Logging Paths (Saves inside the models folder)
    csv_filename = f"evaluation_{model_filename.replace('.pt', '')}_log.csv"
    csv_path = os.path.join(root_project_dir, 'models', csv_filename)
    
    # Initialize the CSV file with column headers
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Episode', 'Reward', 'Steps', 'Status', 'ReachedGoal'])

    # 4. Initialize Webots Environment Mapping
    env = WebotsEnv()
    
    # Bind Supervisor tracking hooks straight onto world objects
    robot_node = env.getFromDef("E_PUCK")       
    goal_node = env.getFromDef("GOAL")         
    
    print(f"\n============ STARTING EVALUATION ROUTINE ({num_episodes} Episodes) ============")
    print(f"--> All data metrics will be saved to: models/{csv_filename}")
    
    rewards_log = []
    successful_episodes = 0

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_counter = 0
        was_successful = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Pure deterministic action decision (Exploration off)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            # Advance simulator environment step
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward
            step_counter += 1

            # Direct Supervisor coordinate check to solve status tracking bugs
            if robot_node is not None and goal_node is not None:
                r_pos = np.array(robot_node.getPosition()[:2]) 
                g_pos = np.array(goal_node.getPosition()[:2])  
                distance_to_goal = np.linalg.norm(r_pos - g_pos)
                
                # If robot gets within the environment's goal target radius (0.15 meters)
                if distance_to_goal <= 0.15:
                    was_successful = True

        # Determine terminal verdict status
        reached_goal_flag = info.get('ReachedGoal', 0) if isinstance(info, dict) else 0
        if was_successful or reached_goal_flag == 1 or episode_reward > 200:
            successful_episodes += 1
            verdict = "GOAL REACHED"
            saved_success_integer = 1
        else:
            verdict = "FAILED"
            saved_success_integer = 0

        rewards_log.append(episode_reward)
        print(f"Ep {ep:02d} | Reward: {episode_reward:8.2f} | Steps: {step_counter:4d} | Status: {verdict}")

        # APPEND EPISODE DATA LINE STRAIGHT TO CSV DISK MEMORY
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([ep, round(episode_reward, 2), step_counter, verdict, saved_success_integer])

    overall_success_rate = (successful_episodes / num_episodes) * 100
    mean_reward = np.mean(rewards_log)

    print("\n================== VERIFICATION MARKSHEET ==================")
    print(f"Evaluated Profile : models/{model_filename}")
    print(f"Saved Log Path    : models/{csv_filename}")
    print(f"Total Test Runs   : {num_episodes} episodes")
    print(f"Success Rate      : {overall_success_rate:.2f}%")
    print(f"Mean Reward       : {mean_reward:.2f}")
    print("============================================================\n")

if __name__ == "__main__":
    # OPTION A: To test your Standard DQN model (best_dqn_epuck.pt)
    #run_deterministic_test(model_filename='best_dqn_epuck.pt', model_type='standard', num_episodes=50)
    
    # OPTION B: To test your Advanced Aligned model (best_advanced_aligned_dqn.pt)
    run_deterministic_test(model_filename='best_advanced_aligned_dqn.pt', model_type='advanced', num_episodes=50)