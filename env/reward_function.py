import numpy as np

# Global configurations/constants for consistent evaluation across algorithms
GOAL_REWARD = 500.0
PERFECT_BONUS = 250.0

STEP_COST = 0.10          # Standard penalty per step to encourage speed
COLLISION_COST = 5.0      # Penalty for bumping into obstacles
SMOOTH_COST_MAX = 0.5     # Penalty for jerky oscillations

# Obstacle proximity thresholds
CAUTION_START = 0.35
CAUTION_MAX_PENALTY = 1.0
DANGER_START = 0.70
DANGER_MAX_PENALTY = 3.0
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

def compute_custom_reward(obs, action, previous_action, goal_reached, collided, current_distance, previous_distance, best_distance, total_collisions):
    """
    Distance Potential Shaping + Dynamic Anchor Reward Function to safely eliminate reward hacking.
    """
    if goal_reached:
        # Deliver pristine bonus if the robot reached the target without hitting obstacles
        if total_collisions == 0:
            return float(GOAL_REWARD + PERFECT_BONUS)
        return float(GOAL_REWARD)

    reward = 0.0

    # 1. Potential-Based Distance Shaping (Dynamic Anchor variant)
    # Scales the raw physical progress against historical milestones achieved in this episode
    shaping_coeff = 15.0
    
    # Differential shaping reward based on physical location change
    potential_delta = shaping_coeff * (previous_distance - current_distance)
    reward += potential_delta

    # Dynamic Anchor protection: severe regression penalty if drifting away from its historical best milestone
    if current_distance > best_distance:
        regression_distance = current_distance - best_distance
        reward -= (10.0 * regression_distance)

    # 2. Obstacle Proximity Penalty
    prox = obs[:8]
    closest_reading = float(np.max(prox))
    obs_severity = np.clip(closest_reading / PS_MAX, 0.0, 1.0)
    reward -= obstacle_proximity_penalty(obs_severity)

    # 3. Collision Penalty (Instantaneous penalty for the current timestep bump)
    if collided:
        reward -= COLLISION_COST

    # 4. Action Smoothness Penalty (Discourages jittery motor commands)
    smooth_severity = action_change_severity(action, previous_action)
    reward -= SMOOTH_COST_MAX * (smooth_severity ** 2)

    # 5. Step Cost
    reward -= STEP_COST

    return float(reward)