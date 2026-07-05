import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
from reward_function import compute_custom_reward

class WebotsEnv(gym.Env, Supervisor):
    PS_MAX = 250.0
    PS_COLLISION = 350.0

    def __init__(self, max_episode_steps=1400, goal_def="GOAL"):
        gym.Env.__init__(self)
        Supervisor.__init__(self)

        self.timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.initial_distance = None
        self.previous_distance = None
        self.best_distance = None  # Track historical minimum for reward function anchor
        self.goal_threshold = 0.15
        self.previous_action = None

        self.ps = []
        for i in range(8):
            sensor = self.getDevice(f"ps{i}")
            sensor.enable(self.timestep)
            self.ps.append(sensor)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.current_left_vel = 0.0
        self.current_right_vel = 0.0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.robot_node = self.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        self.init_translation = self.trans_field.getSFVec3f()
        self.init_rotation = self.rot_field.getSFRotation()

        self.goal_node = self.getFromDef(goal_def)
        if self.goal_node is None:
            raise RuntimeError(f"Cannot find goal node '{goal_def}'")

        goal = self.goal_node.getField("translation").getSFVec3f()
        self.goal_position = np.array([goal[0], goal[1]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        
        self.trans_field.setSFVec3f(self.init_translation)
        self.rot_field.setSFRotation(self.init_rotation)
        self.robot_node.resetPhysics()

        self.step_count = 0
        self.previous_action = None
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0

        Supervisor.step(self, self.timestep)
        obs = self._get_obs()

        distance = float(obs[8])
        self.initial_distance = max(distance, 1e-3)
        self.previous_distance = distance
        self.best_distance = distance  # Anchor initial best distance

        return obs, {}

    def step(self, action):
        self._apply_action(action)
        Supervisor.step(self, self.timestep)
        self.step_count += 1

        obs = self._get_obs()
        goal_reached, collided = self._check_status(obs)

        # Map matching parameters into compute_custom_reward
        reward = compute_custom_reward(
            obs=obs,
            action=action,
            previous_action=self.previous_action,
            goal_reached=goal_reached,
            collided=collided,
            best_distance=self.best_distance
        )

        current_distance = float(obs[8])
        self.previous_distance = current_distance
        
        # Track historical minimum across the episode for progress checking
        if current_distance < self.best_distance:
            self.best_distance = current_distance

        # End episode ONLY when the goal is reached or max steps hit.
        terminated = bool(goal_reached)
        truncated = (self.step_count >= self.max_episode_steps)

        # FIXED: Provided both key formats to support train_dqn.py and train_advanced.py seamlessly
        info = {
            "goal_reached": bool(goal_reached),
            "reached_goal": bool(goal_reached),
        }

        self.previous_action = action
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        prox = np.array([s.getValue() for s in self.ps], dtype=np.float32)
        gps = self.gps.getValues()
        robot = np.array([gps[0], gps[1]], dtype=np.float32)

        north = self.compass.getValues()
        heading = np.arctan2(north[0], north[1])

        vector = self.goal_position - robot
        distance = np.linalg.norm(vector)

        goal_angle = np.arctan2(vector[1], vector[0])
        relative = goal_angle - heading
        relative = np.arctan2(np.sin(relative), np.cos(relative))

        return np.concatenate((prox, [distance], [relative])).astype(np.float32)

    def _apply_action(self, action):
        MAX_SPEED = 6.28
        if action == 0:
            left, right = MAX_SPEED, MAX_SPEED
        elif action == 1:
            left, right = 4.8, 6.28
        elif action == 2:
            left, right = 6.28, 4.8
        elif action == 3:
            left, right = -3.0, 6.0
        elif action == 4:
            left, right = 6.0, -3.0
        else:
            left, right = -5.0, -5.0

        self.current_left_vel = left
        self.current_right_vel = right
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def _check_status(self, obs):
        distance = float(obs[8])
        prox = obs[:8]

        goal_reached = distance <= self.goal_threshold
        collided = bool(np.max(prox) >= self.PS_COLLISION)

        if goal_reached:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
        return goal_reached, collided