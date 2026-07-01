import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# Import the decoupled reward framework
from reward_function import compute_custom_reward

class WebotsEnv(gym.Env, Supervisor):
    PS_MAX = 250.0
    PS_COLLISION = 350.0

    def __init__(self, max_episode_steps=800, goal_def="GOAL"):
        gym.Env.__init__(self)
        Supervisor.__init__(self)

        self.timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.initial_distance = None
        self.best_distance = None
        self.goal_threshold = 0.15

        self.prev_position = None
        self.stuck_counter = 0
        self.is_stuck = False
        self.previous_action = None

        # Setup Sensors
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
        self.stuck_counter = 0
        self.previous_action = None
        self.is_stuck = False

        Supervisor.step(self, self.timestep)
        obs = self._get_obs()

        distance = float(obs[8])
        self.initial_distance = max(distance, 1e-3)
        self.best_distance = distance

        gps = self.gps.getValues()
        self.prev_position = np.array([gps[0], gps[1]])

        return obs, {}

    def step(self, action):
        self._apply_action(action)
        Supervisor.step(self, self.timestep)
        self.step_count += 1

        obs = self._get_obs()
        goal_reached, collided = self._check_status(obs)

        # Call the standalone centralized reward function
        reward = compute_custom_reward(
            obs=obs,
            action=action,
            previous_action=self.previous_action,
            goal_reached=goal_reached,
            collided=collided,
            best_distance=self.best_distance
        )

        # Internally track the distance progression milestone
        current_distance = float(obs[8])
        if current_distance < self.best_distance:
            self.best_distance = current_distance

        terminated = bool(goal_reached)
        
        # Stuck protection logic
        gps = self.gps.getValues()
        position = np.array([gps[0], gps[1]])
        movement = np.linalg.norm(position - self.prev_position)
        self.prev_position = position

        if movement < 0.002:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > 60:
            self.is_stuck = True

        truncated = (self.step_count >= self.max_episode_steps) or self.is_stuck

        info = {
            "goal_reached": bool(goal_reached),
            "collision": bool(collided),
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
            if self.previous_action == 1:
                left, right = -2.5, -5.5
            elif self.previous_action == 2:
                left, right = -5.5, -2.5
            else:
                left, right = -5.0, -5.0

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
            print("=" * 50 + "\nGOAL REACHED\n" + "=" * 50)

        return goal_reached, collided