import numpy as np

GOAL_REWARD = 500.0         # Increased to dominate alternative options
STEP_COST = 0.1             # Fixed step cost (-0.1)
COLLISION_COST = 250.0      # FIX: Must be much worse than maximum step penalties (-80)
SMOOTH_COST_MAX = 0.5 

CAUTION_START = 0.35
PS_MAX = 250.0

def compute_custom_reward(obs, action, previous_action, goal_reached, collided, left_vel, right_vel, best_distance):
    if goal_reached:
        return float(GOAL_REWARD)
        
    if collided:
        return float(-COLLISION_COST)

    reward = 0.0

    # 1. Step Cost
    reward -= STEP_COST

    # 2. FIX: Positive Progress Reward (Gives a reason to live and move toward the goal)
    current_distance = float(obs[8])
    if current_distance < best_distance:
        # Reward proportional to how much closer it got
        reward += 50.0 * (best_distance - current_distance)

    # 3. Obstacle Proximity Penalty (-1 to -5)
    prox = obs[:8]
    closest_reading = float(np.max(prox))
    obs_severity = np.clip(closest_reading / PS_MAX, 0.0, 1.0)
    if obs_severity > CAUTION_START:
        frac = (obs_severity - CAUTION_START) / (1.0 - CAUTION_START)
        reward -= (1.0 + 4.0 * frac)

    # 4. Velocity Bonus
    forward_speed = (left_vel + right_vel) / 2.0
    if forward_speed > 0:
        reward += 0.2 * (forward_speed / 6.28)

    return float(reward)