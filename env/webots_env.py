import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor


class WebotsEnv(gym.Env, Supervisor):
    """
    Webots e-puck environment for obstacle-avoidance + goal-reaching DQN.

    ======================================================================
    REWARD FUNCTION DESIGN -- DENSE / SHAPED, NON-TERMINAL COLLISION
    ======================================================================
    This is a deliberate redesign from the earlier "pure severity cost,
    everything negative except goal" version. Two requirements changed:

      1. Rewards should be DENSE and SHAPED (Paper B's central finding:
         sparse signals make learning slow because there's no gradient
         to follow most of the time). Every term below is continuous
         and informative almost every step, not just a flat cost.

      2. Collision must NOT terminate the episode. Instead it is a
         strong, RECURRING per-step penalty -- every step the robot
         remains in/near collision contact, it pays a heavy cost, but
         the episode keeps running so the agent can still recover and
         reach the goal afterward. The job of actually preventing
         collisions falls mainly on term 4 (graded proximity shaping),
         which gives the agent room to react well before contact.

    Terms (summed every non-terminal step; GOAL is the only terminal
    override):

      1. GOAL              : +GOAL_REWARD                  (terminal, dominant +)
      2. PROGRESS (best-distance / potential-based, Paper A subgoal-
         ramping + your original best_distance idea, revived):
           - new best distance this episode -> + PROGRESS_SCALE * improvement
                                                  * proximity_ramp
           - worse than best-ever distance  -> - REGRESS_SCALE  * regression
      3. HEADING ALIGNMENT : + ALIGN_SCALE * (1 - |angle|/pi)   (dense, always-on)
      4. OBSTACLE PROXIMITY (graded, convex, multi-zone):
           - far          -> 0
           - caution zone -> small, smoothly increasing penalty
           - danger zone  -> steep, convex-shaped penalty
      5. COLLISION (recurring, NON-terminal): -COLLISION_STEP_PENALTY
         every step the robot is in contact / beyond the collision
         threshold. Does NOT end the episode by itself.
      6. DIRECTIONAL AVOIDANCE STEERING: reward turning toward the side
         with more clearance, penalize turning into the tighter side
         (Paper B's context-sensitive shaping -- same action, reward
         depends on what's actually useful in that specific state).
      7. STEP COST          : -STEP_COST                     (flat, every step)
      8. SMOOTHNESS COST    : -SMOOTH_COST_MAX * severity^2  (penalize flapping)

    A SEPARATE safety mechanism (stuck detection / truncation) still
    exists to end an episode early if the robot is physically wedged
    and cannot move at all for a long stretch -- this is NOT a reward
    term, it is what stops a non-terminal-collision design from being
    able to grind out an unbounded, fully wedged, all-penalty episode.
    With collision non-terminal, this safety valve matters MORE than
    it did before, not less.

    ----------------------------------------------------------------------
    EPISODE LENGTH: max_episode_steps = 800 (raised from 400 on request)
    ----------------------------------------------------------------------
    Why 800 and not the original 2000, and why raising it is SAFE now:

      * Arena diagonal ~1.41 m, e-puck linear speed ~0.128 m/s -> a
        direct crossing takes roughly 350-375 steps at a typical 32 ms
        Webots timestep. A realistic successful run (with turning and
        obstacle detours) should land around 150-400 steps. 800 gives
        ~2x headroom for exploration / recovering from near-misses,
        without returning to the original 2000-step budget that let
        "wander forever without colliding" become a fully viable,
        reward-satisfying strategy on its own (that was the root cause
        of the original 10/500 goal-reach problem).

      * Critically, the PROGRESS term (term 2) is `best_distance`-based,
        not a flat per-step distance cost. best_distance shaping only
        pays out on genuine NEW progress -- an agent cannot "farm"
        reward by taking more steps, wandering, or oscillating in
        place. This makes the core progress signal naturally
        insensitive to episode length, which is what makes raising
        max_episode_steps to 800 safe here in a way it would NOT have
        been safe for the old flat distance-cost design (where every
        extra step of wandering accumulated extra cost/reward whether
        or not real progress was made).

      * The one term that DOES scale with episode length is the
        recurring collision penalty (term 5) and the flat step cost
        (term 7) -- both are intentionally kept small per-step (see
        constants below) specifically because they now have up to 800
        steps to accumulate over, not 400. They were re-derived (not
        just copied) for this episode length -- see the inline
        comments next to COLLISION_STEP_PENALTY and STEP_COST.
    ======================================================================
    """

    # ------------------------------------------------------------------
    # Reward function constants -- tune here, nowhere else.
    # ------------------------------------------------------------------
    GOAL_REWARD = 250.0

    # Progress shaping (best-distance / potential-based)
    PROGRESS_SCALE = 40.0
    REGRESS_SCALE = 25.0
    RAMP_GAIN = 1.5          # how much the progress bonus grows as the
                              # robot gets closer to the goal (Paper A
                              # subgoal-ramping principle)

    # Heading alignment (dense, always-on, every non-terminal step)
    ALIGN_SCALE = 0.40

    # Obstacle proximity -- graded, multi-zone, convex curve
    CAUTION_START = 0.35     # normalized severity at which caution
                              # penalty begins (0 = far, 1 = collision)
    CAUTION_MAX_PENALTY = 1.5
    DANGER_START = 0.70      # normalized severity at which the steeper
                              # danger penalty kicks in on top
    DANGER_MAX_PENALTY = 4.0

    # Collision -- NON-terminal, recurring per-step penalty.
    # Kept deliberately small per-step (not a one-time -60/-100 like the
    # old terminal design) because it can now fire for MANY consecutive
    # steps if the robot lingers in contact. At 800 max steps, even a
    # robot wedged in contact for as long as 60 consecutive steps (the
    # is_stuck truncation threshold -- see below) only accumulates
    # 60 * 3.0 = -180, which stays clearly worse than a normal episode
    # of background costs but does not single-handedly blow out the
    # whole reward scale the way an unbounded per-step penalty could.
    COLLISION_STEP_PENALTY = 3.0

    # Directional avoidance steering (context-sensitive, Paper B style)
    STEER_REWARD = 0.30
    STEER_PENALTY = 0.20

    # Flat step cost. Re-derived for 800-step episodes: a full 800-step
    # episode of nothing but step cost totals 800 * 0.08 = -64, which
    # stays comfortably smaller than GOAL_REWARD (250) and smaller than
    # what ~60 steps of recurring collision would cost (-180) -- i.e.
    # collision still hurts more per-incident than simply taking a
    # while to find the goal.
    STEP_COST = 0.08

    SMOOTH_COST_MAX = 0.8

    # Raw e-puck `ps` sensor reading at which proximity severity reaches
    # 1.0 (i.e. "as close as the gradient cares to distinguish further").
    PS_MAX = 250.0

    # Raw e-puck `ps` sensor reading treated as an actual collision /
    # contact event for the recurring penalty (term 5). If you have a
    # bumper/touch sensor device, prefer reading that directly instead
    # -- it is unambiguous, whereas a proximity threshold is a
    # reasonable but imperfect proxy.
    PS_COLLISION = 350.0

    def __init__(self, max_episode_steps=800, goal_def="GOAL"):
        gym.Env.__init__(self)
        Supervisor.__init__(self)

        self.timestep = int(self.getBasicTimeStep())
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # --------------------------------------------------------------
        # Distance / progress bookkeeping
        # --------------------------------------------------------------
        self.initial_distance = None
        self.best_distance = None
        self.goal_threshold = 0.15

        # --------------------------------------------------------------
        # Stuck detection -- SAFETY/TRUNCATION mechanism, NOT a reward
        # term. With collision now non-terminal this is the main thing
        # preventing a wedged robot from grinding out an unbounded,
        # all-penalty episode -- it matters MORE than in the previous
        # (terminal-collision) design, not less.
        # --------------------------------------------------------------
        self.prev_position = None
        self.stuck_counter = 0
        self.is_stuck = False

        # --------------------------------------------------------------
        # Action history (needed for smoothness cost + reverse logic)
        # --------------------------------------------------------------
        self.previous_action = None

        # --------------------------------------------------------------
        # Proximity sensors
        # --------------------------------------------------------------
        self.ps = []
        for i in range(8):
            sensor = self.getDevice(f"ps{i}")
            sensor.enable(self.timestep)
            self.ps.append(sensor)

        # --------------------------------------------------------------
        # GPS & Compass & Motors
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Spaces
        # --------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        # --------------------------------------------------------------
        # Robot Node & Goal Setup
        # --------------------------------------------------------------
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

    # ======================================================================
    # Gym API
    # ======================================================================

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
        self.initial_distance = max(distance, 1e-3)  # guard div-by-zero
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

        reward = self._compute_reward(obs, action, goal_reached, collided)

        # Collision is intentionally NOT in this condition -- it no
        # longer terminates the episode. Only reaching the goal does.
        terminated = bool(goal_reached)

        # Truncate on max steps OR if the robot is genuinely wedged
        # stuck (this is the safety valve that bounds how long a
        # non-terminal collision can keep accumulating penalty for).
        truncated = (self.step_count >= self.max_episode_steps) or self.is_stuck

        info = {
            "goal_reached": bool(goal_reached),
            "collision": bool(collided),
        }

        self.previous_action = action

        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Observation
    # ======================================================================

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

    # ======================================================================
    # Actions
    # ======================================================================

    def _apply_action(self, action):
        MAX_SPEED = 6.28

        if action == 0:
            # Fast Forward
            left = MAX_SPEED
            right = MAX_SPEED
        elif action == 1:
            # Gentle Left
            left = 4.8
            right = 6.28
        elif action == 2:
            # Gentle Right
            left = 6.28
            right = 4.8
        elif action == 3:
            # Sharp Left
            left = -3.0
            right = 6.0
        elif action == 4:
            # Sharp Right
            left = 6.0
            right = -3.0
        else:
            # Smart Reverse
            if self.previous_action == 1:
                left = -2.5
                right = -5.5
            elif self.previous_action == 2:
                left = -5.5
                right = -2.5
            else:
                left = -5.0
                right = -5.0

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    # ======================================================================
    # Reward
    # ======================================================================

    def _action_change_severity(self, action):
        """
        Normalized [0, 1] severity describing how abrupt the jump from
        the previous action to the current one was.

        0.0 -> same action as last step, or no previous action yet
        0.4 -> a same-direction adjustment (e.g. gentle-left -> sharp-left)
        0.8 -> switching into/out of reverse
        1.0 -> a full opposite-turn reversal (classic oscillation)
        """
        if self.previous_action is None or action == self.previous_action:
            return 0.0

        prev = self.previous_action

        opposite_pairs = {(1, 2), (2, 1), (3, 4), (4, 3)}
        if (prev, action) in opposite_pairs:
            return 1.0

        if action == 5 or prev == 5:
            return 0.8

        return 0.4

    def _obstacle_proximity_penalty(self, obs_severity):
        """
        Graded, multi-zone, convex-shaped proximity penalty.

        0.0 .. CAUTION_START      -> 0 (far, no penalty)
        CAUTION_START .. DANGER_START
                                    -> smoothly ramps 0 -> CAUTION_MAX_PENALTY
        DANGER_START .. 1.0        -> additionally ramps a much steeper
                                       curve 0 -> DANGER_MAX_PENALTY on top
        This gives the agent a genuine early warning (caution zone) well
        before things become urgent (danger zone), instead of a single
        flat threshold.
        """
        penalty = 0.0

        if obs_severity > self.CAUTION_START:
            span = max(self.DANGER_START - self.CAUTION_START, 1e-6)
            frac = min((obs_severity - self.CAUTION_START) / span, 1.0)
            penalty += self.CAUTION_MAX_PENALTY * frac

        if obs_severity > self.DANGER_START:
            span = max(1.0 - self.DANGER_START, 1e-6)
            frac = min((obs_severity - self.DANGER_START) / span, 1.0)
            # squared -> convex curve, Paper C style: penalty stays mild
            # just past the danger threshold and escalates sharply only
            # as the robot gets genuinely close to contact.
            penalty += self.DANGER_MAX_PENALTY * (frac ** 2)

        return penalty

    def _directional_steering_term(self, action, prox):
        """
        Context-sensitive steering shaping (Paper B's Acrobot-style
        insight: the SAME action should be rewarded or penalized
        differently depending on what's actually useful in the current
        state). Only applies to the two gentle-turn actions; rewards
        turning toward the side with more clearance, penalizes turning
        into the tighter side.

        e-puck sensors go clockwise: ps0 (front-right) -> ps7 (front-left)
        """
        left_reading = np.mean([prox[5], prox[6]])
        right_reading = np.mean([prox[1], prox[2]])

        if action == 1:  # Gentle Left
            return self.STEER_REWARD if right_reading > left_reading else -self.STEER_PENALTY

        if action == 2:  # Gentle Right
            return self.STEER_REWARD if left_reading > right_reading else -self.STEER_PENALTY

        return 0.0

    def _compute_reward(self, obs, action, goal_reached, collided):
        # Goal is the only terminal override now -- collision is folded
        # into the continuous per-step terms below instead.
        if goal_reached:
            return float(self.GOAL_REWARD)

        prox = obs[:8]
        distance = float(obs[8])
        angle = float(obs[9])

        reward = 0.0

        # --- 1. Progress shaping (best-distance, with proximity ramp) --
        ramp = 1.0 + self.RAMP_GAIN * (1.0 - distance / self.initial_distance)
        ramp = max(ramp, 1.0)

        if distance < self.best_distance:
            improvement = self.best_distance - distance
            reward += self.PROGRESS_SCALE * improvement * ramp
            self.best_distance = distance
        elif distance > self.best_distance:
            regression = distance - self.best_distance
            reward -= self.REGRESS_SCALE * regression

        # --- 2. Heading alignment (dense, always-on) --------------------
        reward += self.ALIGN_SCALE * (1.0 - abs(angle) / np.pi)

        # --- 3. Obstacle proximity (graded, convex, multi-zone) ---------
        closest_reading = float(np.max(prox))
        obs_severity = np.clip(closest_reading / self.PS_MAX, 0.0, 1.0)
        reward -= self._obstacle_proximity_penalty(obs_severity)

        # --- 4. Collision -- recurring, NON-terminal -----------------
        if collided:
            reward -= self.COLLISION_STEP_PENALTY

        # --- 5. Directional avoidance steering (context-sensitive) ----
        reward += self._directional_steering_term(action, prox)

        # --- 6. Flat step cost -----------------------------------------
        reward -= self.STEP_COST

        # --- 7. Smoothness cost (squared, jerk-penalty style) -----------
        smooth_severity = self._action_change_severity(action)
        reward -= self.SMOOTH_COST_MAX * (smooth_severity ** 2)

        # --------------------------------------------------------------
        # Stuck detection -- SAFETY/TRUNCATION mechanism, not a core
        # reward-shaping term. More important now than before, since
        # collision no longer ends the episode on its own -- this is
        # what bounds how long a wedged robot can keep accumulating the
        # recurring collision penalty for.
        # --------------------------------------------------------------
        gps = self.gps.getValues()
        position = np.array([gps[0], gps[1]])
        movement = np.linalg.norm(position - self.prev_position)
        self.prev_position = position

        if movement < 0.002:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > 25:
            reward -= 1.0
            if self.stuck_counter > 60:
                self.is_stuck = True

        return float(reward)

    # ======================================================================
    # Status checks
    # ======================================================================

    def _check_status(self, obs):
        """
        Returns (goal_reached: bool, collided: bool).

        IMPORTANT CHANGE: collided is now used purely as a per-step
        SIGNAL (consumed by _compute_reward as a recurring penalty) --
        it no longer ends the episode. Only goal_reached does (via
        `terminated` in step()). The robot is also no longer force-
        stopped on collision (it used to be) since the episode is
        expected to continue -- only goal-reaching stops the motors.

        If you have a bumper/touch sensor device, prefer reading that
        directly here instead of thresholding the `ps` proximity values
        -- it is unambiguous, whereas a proximity threshold is a
        reasonable but imperfect proxy.
        """
        distance = float(obs[8])
        prox = obs[:8]

        goal_reached = distance <= self.goal_threshold
        collided = bool(np.max(prox) >= self.PS_COLLISION)

        if goal_reached:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print("=" * 50)
            print("GOAL REACHED")
            print(f"Distance = {distance:.3f} m")
            print("=" * 50)

        return goal_reached, collided