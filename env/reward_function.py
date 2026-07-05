import numpy as np

# Global configurations/constants for consistent evaluation across algorithms
GOAL_REWARD = 1000.0

STEP_COST = 0.15          # Standard penalty per step to encourage speed
COLLISION_COST = 4.0      # Heavy penalty per step spent in collision
SMOOTH_COST_MAX = 0.8     # Penalty for jerky oscillations

# Obstacle proximity thresholds
CAUTION_START = 0.35
CAUTION_MAX_PENALTY = 1.5
DANGER_START = 0.70
DANGER_MAX_PENALTY = 4.0
PS_MAX = 250.0

def action_change_severity(action, previous_action):
    if previous_action is None or action == previous_action:
        return 0.0
    opposite_pairs = {(1, 2), (2, 1), (3, 4), (4, 3)}
    if (previous_action, action) in opposite_pairs:
        return 1.0
    if action == 5 or previous_action == 5:
        return 0.8
    return 0.4

def obstacle_proximity_penalty(obs_severity):
    penalty = 0.0
    if obs_severity > CAUTION_START:
        span = max(DANGER_START - CAUTION_START, 1e-6)
        frac = min((obs_severity - CAUTION_START) / span, 1.0)
        penalty += CAUTION_MAX_PENALTY * frac

    if obs_severity > DANGER_START:
        span = max(1.0 - DANGER_START, 1e-6)
        frac = min((obs_severity - DANGER_START) / span, 1.0)
        penalty += DANGER_MAX_PENALTY * (frac ** 2)
    return penalty

def compute_custom_reward(obs, action, previous_action, goal_reached, collided, best_distance):
    """
    Pure Negative / Cost-Based Shaped Reward Function to prevent reward hacking.
    """
    if goal_reached:
        return float(GOAL_REWARD)

    prox = obs[:8]
    distance = float(obs[8])
    
    reward = 0.0

    # 1. Progress Cost (Strictly negative: penalizes lack of progress or regression)
    if distance >= best_distance:
        # Penalizes moving away or just sitting still compared to the best distance achieved
        regression = distance - best_distance
        reward -= (25.0 * regression + 0.1)  # Small static penalty if idling at best distance
    else:
        # No positive reward granted for making progress to avoid farming cycles!
        pass

    # 2. Obstacle Proximity Penalty
    closest_reading = float(np.max(prox))
    obs_severity = np.clip(closest_reading / PS_MAX, 0.0, 1.0)
    reward -= obstacle_proximity_penalty(obs_severity)

    # 3. Collision Penalty (Per-step cost)
    if collided:
        reward -= COLLISION_COST

    # 4. Action Smoothness Penalty (Discourages rapid oscillations)
    smooth_severity = action_change_severity(action, previous_action)
    reward -= SMOOTH_COST_MAX * (smooth_severity ** 2)

    # 5. Fixed Step Cost
    reward -= STEP_COST

    return float(reward)