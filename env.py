"""The environment contract: what the policy sees, what it emits, when it ends.

Task: stand on `groundcontact`, and stay standing through pushes that arrive at
times the policy cannot predict. Fixed-length episodes, 50 Hz, 14 position
targets out, 48 numbers in.

WHY 48 AND NOT 61
-----------------
Step 1's README quoted a 61-dim observation. That was the wrong number, and the
reason it was wrong is the whole point of this file.

61 is everything the *simulator* can hand over about the robot's state. 48 is
everything the *robot* can measure about itself. The 13-dim difference is:

    imu_lin_vel     3   a velocimeter. The real duck has no state estimator,
                        so base linear velocity does not exist on hardware.
    trunk position  3   world-frame xyz. Same problem, plus there is no
                        external tracking system in the loop.
    orientation     4   the full quaternion. Yaw is not observable from a
                        gyro and an accelerometer alone -- no magnetometer is
                        declared in sensors.xml -- so absolute heading would
                        be a number the hardware can only integrate and drift.
    root_angmom     3   subtree angular momentum, a MuJoCo computation over
                        the whole body tree, not a sensor at all.

Every one of those is free in sim and impossible on the robot. A policy trained
on them learns to depend on them, and then has nothing to run on. Since the
entire claim this project can make is about transfer, the observation is
restricted to sensors the robot actually carries, up front, before any training
happens. This costs nothing now and cannot be retrofitted later.

What survives, and where it comes from on hardware:

    joint_pos    14   servo present-position, one per XL330
    joint_vel    14   servo present-velocity. Real and noisy; it is included
                      because the servos do report it, not because it is clean.
    prev_action  14   the policy's own last output. Free -- it is in RAM.
    gyro          3   IMU rate gyro, taken from the `angular-velocity` sensor
                      rather than its twin `imu_ang_vel`, because that is the
                      one upstream declared noise="0.005" on. Both read the
                      same site. MuJoCo only applies declared sensor noise when
                      mjENBL_SENSORNOISE is set, which it is not here -- so
                      step 4 turns the flag on and gets upstream's noise
                      magnitude instead of one invented for the occasion.
    proj_grav     3   world -Z expressed in the trunk frame: which way is down,
                      from the robot's point of view. This is the observable
                      part of attitude -- roll and pitch are gravity-referenced,
                      yaw is not -- and it is what every IMU with an onboard
                      complementary filter gives you. Derived here from the
                      `orientation` framequat so that the step-4 noise flag
                      perturbs it too.

WHAT ELSE IS DELIBERATE
-----------------------
Fixed-length episodes, no early termination on falling. The usual locomotion
setup ends the episode the moment the robot goes down, which is right when the
task is walking and wrong here: recovery is half the task, and terminating on a
fall makes falling unrecoverable by construction. It also keeps returns
comparable -- every episode is 250 steps, so the PD baseline that topples at
0.79 s has a well-defined return instead of a short episode that looks cheap.

Actions are residuals around the STAND pose, clipped to the joint limits. That
clip is not decoration: `ctrlrange` on all 14 actuators is [-10, 10] rad, while
the tightest joint range (hip roll) is +-0.384 rad. Handing a position actuator
a 10 rad target raises nothing -- it saturates against the joint stop and
spends the whole force range holding there.
"""
import mujoco
import numpy as np

import common

OBS_DIM = 48
ACT_DIM = common.N_ACT

# Layout is fixed and tested. Anything reading an observation slice imports
# these rather than writing the indices out again.
OBS_SLICES = {
    "joint_pos":   slice(0, 14),
    "joint_vel":   slice(14, 28),
    "prev_action": slice(28, 42),
    "gyro":        slice(42, 45),
    "proj_grav":   slice(45, 48),
}

# Per-group scaling, so no input dominates by unit choice alone. Joint residuals
# and projected gravity are already order 1. Joint velocity reaches ~20 rad/s in
# a fall and the gyro ~4 rad/s, so both are divided down to roughly unit range.
OBS_SCALE = np.concatenate([
    np.ones(14),          # joint_pos, rad, |.| <= ~1.5
    np.full(14, 0.05),    # joint_vel, rad/s
    np.ones(14),          # prev_action, already in [-1, 1]
    np.full(3, 0.25),     # gyro, rad/s
    np.ones(3),           # proj_grav, unit vector
]).astype(np.float32)

# A saturated action moves a joint 0.35 rad, about 22% of the median joint
# range, in one 20 ms decision. It is a hyperparameter, not a derived quantity;
# step 3 reports what happens at 0.2 and 0.5.
ACTION_SCALE = 0.35

CONTROL_DT = 1.0 / common.CONTROL_HZ
FALL_HEIGHT = 0.04     # m. Same threshold bench.py resets on.
STAND_HEIGHT = 0.12    # m. The STAND keyframe's trunk height, asserted at init.

# Per-step reward, two versions.
#
# v1 is the step-2 reward, kept byte-identical because the 108.7 +- 2.9 PD
# baseline in the README was measured against it. Positive terms are bounded
# by 1 each, so a perfectly held stand scores 2.0 per step and 500 over an
# episode. baseline.py measured three of its four penalties at under 0.25% of
# the return, and the fallen region nearly flat (0.12 +- 0.04 per step).
#
# v2 is the step-3 decision, made on those measurements and one more taken
# while making it: under the PD baseline and under random actions the fallen
# duck lies with trunk cos in 0.0-0.3 and height 3.1-6 cm; over 1,500 fallen
# steps the cos never went below -0.06. So:
#   upright   unchanged, max(0, cos). Its slope of 1 per unit cos IS live in
#             the band the robot actually falls into; the floor at 90 degrees
#             is never reached. (1 + cos)/2 was considered and rejected: it
#             would halve the one gradient the fallen region already has.
#   height    a linear ramp clip(h / STAND_HEIGHT, 0, 1) instead of a 3 cm
#             Gaussian about 0.12 m. The Gaussian is 1e-4 .. 0.02 across the
#             whole fallen height range, i.e. no gradient; the ramp gives
#             8.3 per metre everywhere below STAND. This is the term that
#             makes "lift the trunk while tilted" worth anything.
#   posture, effort   dropped: each under 0.25% of the return under both the
#             PD baseline and random actions, and actuator force is not a
#             quantity the real servos report
#   joint_vel -4.2e-2, set from baseline.py --reward v2 on 2026-09-22: at
#             -2e-3 the term was 0.24% of the random-action return, and the
#             script asked for 21x to reach the ~5% target. Re-baselined below.
#   action_rate unchanged; it was the one penalty measured live.
# The ceiling stays 2.0 per step / 500 per episode. The v1 baseline number
# does not transfer to v2; baseline.py --reward v2 re-measures it.
REWARD_V1 = {
    "upright":     1.0,
    "height":      1.0,
    "posture":    -0.10,
    "action_rate": -0.05,
    "joint_vel":  -2.0e-4,
    "effort":     -0.02,
}
REWARD_V2 = {
    "upright":     1.0,
    "height":      1.0,
    "action_rate": -0.05,
    "joint_vel":  -4.2e-2,
}
REWARDS = {"v1": REWARD_V1, "v2": REWARD_V2}
REWARD_WEIGHTS = REWARD_V1      # the default; kept under its step-2 name


class MicroduckEnv:
    """One Microduck. Single process, no framework, no gym dependency.

    Deliberately not a gymnasium.Env. The only consumer is this repo's PPO, the
    surface is five methods, and a dependency whose API has changed three times
    is not worth taking on for a `.unwrapped` attribute.
    """

    def __init__(self, variant="groundcontact", seed=0, episode_steps=250,
                 action_scale=ACTION_SCALE, n_pushes=3,
                 push_speed=(0.15, 0.45), init_noise=0.02, reward="v1",
                 latency=0, sensor_noise=False, dr=None):
        """
        reward        "v1" (step 2, the baseline's reward) or "v2" (step 3)
        latency       control steps the commanded action is delayed by
        sensor_noise  add Gaussian noise to the gyro and orientation reads with
                      the standard deviations upstream declared in sensors.xml
                      (model.sensor_noise: 0.005 rad/s gyro, 0.001 quaternion).
                      MuJoCo 3.12 dropped the mjENBL_SENSORNOISE flag that used
                      to apply these, so the environment applies them itself.
        dr            a dr.DomainRandomizer-compatible object: at every reset
                      `dr.apply(self, rng)` is called and may change the model,
                      self.latency and self.init_noise for that episode
        """
        self.variant = variant
        self.episode_steps = int(episode_steps)
        self.action_scale = float(action_scale)
        self.n_pushes = int(n_pushes)
        self.push_speed = tuple(push_speed)
        self.init_noise = float(init_noise)
        if reward not in REWARDS:
            raise KeyError(f"reward must be one of {sorted(REWARDS)}, got {reward!r}")
        self.reward_version = reward
        self.weights = REWARDS[reward]
        self.latency = int(latency)
        self.dr = dr

        self.model, self.data = common.load(variant)
        self.sensor_noise = bool(sensor_noise)

        # Index maps, resolved once. On walk_backlash these are strided, which
        # is the entire reason they exist -- see common.actuated_qpos_index.
        self._qpos_i = common.actuated_qpos_index(self.model)
        self._qvel_i = common.actuated_qvel_index(self.model)
        self._lo, self._hi = common.joint_limits(self.model)
        self._default = common.default_pose(self.model, "STAND")
        self._gyro = common.sensor_slice(self.model, "angular-velocity")
        self._quat = common.sensor_slice(self.model, "orientation")
        self._gyro_sd = float(self.model.sensor_noise[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")])
        self._quat_sd = float(self.model.sensor_noise[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")])

        # The free joint must be joint 0 at qvel 0:6, or the push below lands
        # on some other body's velocity and nothing complains.
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("joint 0 is not the free joint; push code assumes it is")

        self.rng = np.random.default_rng(seed)
        self._prev_action = np.zeros(ACT_DIM)
        self._t = 0
        self._push_at = np.zeros(0, dtype=int)
        self._push_vel = np.zeros((0, 2))
        self._pushes_done = 0
        self._episode_started = False

        common.reset_to(self.model, self.data, "STAND")
        h = common.trunk_height(self.model, self.data)
        if abs(h - STAND_HEIGHT) > 1e-3:
            raise RuntimeError(f"STAND trunk height is {h:.4f}, expected {STAND_HEIGHT}")

    # -- observation ------------------------------------------------------

    def _proj_gravity(self):
        """World -Z in the trunk frame. (0, 0, -1) when upright."""
        q = np.array(self.data.sensordata[self._quat], dtype=float)
        if self.sensor_noise and self._quat_sd > 0:
            q += self.rng.normal(0.0, self._quat_sd, 4)
        mujoco.mju_normalize4(q)          # noise denormalises it
        qinv = np.empty(4)
        mujoco.mju_negQuat(qinv, q)       # conjugate: world -> site
        g = np.empty(3)
        mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), qinv)
        return g

    def observe(self):
        gyro = np.array(self.data.sensordata[self._gyro], dtype=float)
        if self.sensor_noise and self._gyro_sd > 0:
            gyro += self.rng.normal(0.0, self._gyro_sd, 3)
        raw = np.concatenate([
            self.data.qpos[self._qpos_i] - self._default,
            self.data.qvel[self._qvel_i],
            self._prev_action,
            gyro,
            self._proj_gravity(),
        ])
        return (raw * OBS_SCALE).astype(np.float32)

    # -- episode ----------------------------------------------------------

    def _schedule_pushes(self):
        """Draw this episode's kicks. Seeded, so an episode replays exactly.

        Kept away from the first and last half-second: a push at t=0 is a
        different initial condition rather than a disturbance, and one at the
        buzzer is never recovered from within the episode, so neither teaches
        anything about recovery.
        """
        margin = int(0.5 * common.CONTROL_HZ)
        lo, hi = margin, self.episode_steps - margin
        if self.n_pushes <= 0 or hi <= lo:
            self._push_at = np.zeros(0, dtype=int)
            self._push_vel = np.zeros((0, 2))
            return
        self._push_at = np.sort(self.rng.choice(np.arange(lo, hi),
                                                size=min(self.n_pushes, hi - lo),
                                                replace=False))
        speed = self.rng.uniform(*self.push_speed, size=self._push_at.size)
        theta = self.rng.uniform(0.0, 2 * np.pi, size=self._push_at.size)
        self._push_vel = np.stack([speed * np.cos(theta),
                                   speed * np.sin(theta)], axis=1)

    def reset(self, seed=None):
        """Start a new episode. `seed` REPLACES the RNG, it does not advance it.

        So `reset(seed=S)` inside a training loop gives the same push schedule
        and the same initial noise every single episode, forever. Seed once at
        construction, or on a deliberate replay, and call `reset()` bare after
        that. `VecEnv` autoreset does the right thing already.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.dr is not None:
            self.dr.apply(self, self.rng)
        common.reset_to(self.model, self.data, "STAND")
        self._act_buf = [np.zeros(ACT_DIM) for _ in range(self.latency)]
        if self.init_noise > 0:
            n = self.init_noise
            self.data.qpos[self._qpos_i] += self.rng.uniform(-n, n, ACT_DIM)
            self.data.qvel[self._qvel_i] += self.rng.uniform(-5 * n, 5 * n, ACT_DIM)
            # Assignment, not `out=`. `self._qpos_i` is an integer array, so
            # `qpos[idx]` is a copy and `np.clip(..., out=qpos[idx])` writes
            # into a temporary that is then discarded -- the clamp silently did
            # nothing. Harmless at the default init_noise of 0.02, because the
            # tightest joint sits 0.297 rad clear of its limit at STAND, and
            # live the moment step 4 randomises the initial state wider than
            # that. Measured at init_noise=0.6: 2 of 14 joints started 0.086 rad
            # outside their range, MuJoCo applied a limit impulse at t=0, and
            # nothing reported it.
            self.data.qpos[self._qpos_i] = np.clip(
                self.data.qpos[self._qpos_i], self._lo, self._hi)
            mujoco.mj_forward(self.model, self.data)
        self._prev_action = np.zeros(ACT_DIM)
        self._t = 0
        self._pushes_done = 0
        self._episode_started = True
        self._schedule_pushes()
        return self.observe()

    def step(self, action):
        if not self._episode_started:
            raise RuntimeError(
                "call reset() before step(). A freshly constructed env has an "
                "empty push schedule, so stepping it produces a push-free "
                "episode and raises nothing.")
        action = np.clip(np.asarray(action, dtype=float).reshape(ACT_DIM), -1.0, 1.0)
        # Latency: the action applied now is the one the policy emitted
        # `latency` steps ago. prev_action in the observation stays the one
        # just emitted, because that is what the policy actually knows.
        applied = action
        if self.latency > 0:
            self._act_buf.append(action.copy())
            applied = self._act_buf.pop(0)
        target = np.clip(self._default + self.action_scale * applied, self._lo, self._hi)
        self.data.ctrl[:] = target

        pushed = False
        if self._pushes_done < self._push_at.size and self._t == self._push_at[self._pushes_done]:
            self.data.qvel[0:2] += self._push_vel[self._pushes_done]
            self._pushes_done += 1
            pushed = True

        for _ in range(common.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        # mj_step integrates qpos and qvel to t+1 and leaves everything derived
        # from them -- sensordata, xpos, actuator_force, contacts -- at t. Read
        # straight after the loop, the observation pairs joint angles from t+1
        # with a gyro reading from 2 ms earlier, and the reward scores a pose
        # the robot has already left. Measured over a 250-step episode: up to
        # 0.054 rad/s on the gyro, 0.011 on upright cosine, 1.4 mm on trunk
        # height against the height term's 30 mm sigma.
        #
        # It matters here more than it would elsewhere. The whole claim behind
        # the 48-dim observation is that every channel is one a real robot
        # could produce, and no real IMU reports an angular rate that disagrees
        # with its own encoders by a timestep. One forward pass, no integration,
        # costs about 8% of the control step and buys a consistent snapshot.
        mujoco.mj_forward(self.model, self.data)

        terms = self._reward_terms(action)
        reward = float(sum(terms.values()))
        self._prev_action = action
        self._t += 1
        done = self._t >= self.episode_steps

        h = common.trunk_height(self.model, self.data)
        info = {"terms": terms, "trunk_height": h,
                "upright_cos": common.upright_cos(self.model, self.data),
                "fallen": h < FALL_HEIGHT, "pushed": pushed, "t": self._t,
                # This task has no early termination by design, so `done` is
                # ALWAYS the step limit and never a terminal state. Both flags
                # are carried explicitly because the distinction is invisible
                # otherwise and a stock GAE loop gets it wrong by default:
                # zeroing the bootstrap at a truncation corrupts the value
                # target back about 100 steps at gamma=0.99, which is 40% of a
                # 250-step episode. Bootstrap V(terminal_obs) on every done.
                "truncated": bool(done), "terminated": False,
                "reward": self.reward_version}
        return self.observe(), reward, done, info

    def _reward_terms(self, action):
        w = self.weights
        dq = self.data.qvel[self._qvel_i]
        h = common.trunk_height(self.model, self.data)
        cos = common.upright_cos(self.model, self.data)
        if self.reward_version == "v1":
            q = self.data.qpos[self._qpos_i] - self._default
            dh = h - STAND_HEIGHT
            return {
                "upright":     w["upright"] * max(0.0, cos),
                "height":      w["height"] * float(np.exp(-(dh / 0.03) ** 2)),
                "posture":     w["posture"] * float(np.mean(q ** 2)),
                "action_rate": w["action_rate"] * float(np.mean((action - self._prev_action) ** 2)),
                "joint_vel":   w["joint_vel"] * float(np.mean(dq ** 2)),
                "effort":      w["effort"] * float(np.mean(self.data.actuator_force ** 2)),
            }
        return {
            "upright":     w["upright"] * max(0.0, cos),
            "height":      w["height"] * float(np.clip(h / STAND_HEIGHT, 0.0, 1.0)),
            "action_rate": w["action_rate"] * float(np.mean((action - self._prev_action) ** 2)),
            "joint_vel":   w["joint_vel"] * float(np.mean(dq ** 2)),
        }

    # -- baselines --------------------------------------------------------

    def zero_action(self):
        """The PD hold-pose baseline: command the STAND pose and nothing else.

        This is the thing the learned policy has to beat, and step 1 already
        measured it -- it topples at 0.79 s.
        """
        return np.zeros(ACT_DIM)
