"""Domain randomisation for step 4, and the held-out physics it is judged on.

Two things live here and are kept apart on purpose:

    DRConfig             the ranges a TRAINING run samples at every reset
    common.HELDOUT_VARIANTS   the models a trained policy is EVALUATED on and
                         never trained on: `groundcontact_backlash` (a passive
                         backlash joint in series with every actuator) and
                         `rollers` (passive roller geoms at the feet). Both are
                         shipped by the robot's own authors, not invented here.

The actuator ranges are not guesses. `joints_properties.xml` ships four fits
of the same XL330 servo by different people on different benches
(`common.ACTUATOR_CLASSES`). Damping, armature, gain and force limit agree to
within about 40%; friction loss disagrees by 6.7x. The randomiser draws each
of the five actuator parameters uniformly between the min and max of the four
fits -- so the spread is wide exactly where the hardware people disagree and
narrow where they agree -- and applies it to all 14 actuators together, the
way a batch of the same servo would vary.

On top of that: body mass +-15% on every body, floor friction 0.6-1.2x,
an action latency of 0-2 control steps (20 ms each), initial-state noise up to
0.3 rad, and sensor noise at the magnitudes upstream declared in sensors.xml
(0.005 rad/s on the gyro, 0.001 on the orientation quaternion). MuJoCo 3.12 no
longer has the mjENBL_SENSORNOISE flag that applied those, so the environment
applies them itself when `sensor_noise` is on.

Nominal values are captured from the model once at construction, so every
`apply` is relative to the vendored model and nothing compounds. `restore()`
puts the model back exactly; `test_dr.py` checks it bit for bit.
"""
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

import common

_CLS = np.array(list(common.ACTUATOR_CLASSES.values()))    # (4, 5)
ACT_PARAM_NAMES = ("damping", "frictionloss", "armature", "kp", "forcerange")
ACT_PARAM_LO = _CLS.min(axis=0)
ACT_PARAM_HI = _CLS.max(axis=0)
NOMINAL_CLASS = np.array(common.ACTUATOR_CLASSES["chosen_actuator"])


@dataclass
class DRConfig:
    actuator: bool = True            # draw the 5 servo parameters between the 4 fits
    mass_scale: tuple = (0.85, 1.15)
    floor_friction: tuple = (0.6, 1.2)
    latency: tuple = (0, 2)          # control steps, inclusive
    init_noise: tuple = (0.02, 0.3)  # rad, uniform per episode
    sensor_noise: bool = True

    @classmethod
    def none(cls):
        return cls(actuator=False, mass_scale=(1.0, 1.0), floor_friction=(1.0, 1.0),
                   latency=(0, 0), init_noise=(0.02, 0.02), sensor_noise=False)

    def sample(self, rng):
        if self.actuator:
            act = rng.uniform(ACT_PARAM_LO, ACT_PARAM_HI)
        else:
            act = NOMINAL_CLASS.copy()
        return {
            "actuator": {n: float(v) for n, v in zip(ACT_PARAM_NAMES, act)},
            "mass_scale": float(rng.uniform(*self.mass_scale)),
            "floor_friction": float(rng.uniform(*self.floor_friction)),
            "latency": int(rng.integers(self.latency[0], self.latency[1] + 1)),
            "init_noise": float(rng.uniform(*self.init_noise)),
            "sensor_noise": bool(self.sensor_noise),
        }

    def to_dict(self):
        return asdict(self)


class DomainRandomizer:
    """Plug into `MicroduckEnv(dr=...)`. Called by the env at every reset."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else DRConfig()
        self.current = None
        self._nominal = None
        self._env_defaults = None

    def _capture(self, env):
        m = env.model
        self._act_i = common.actuated_qvel_index(m)     # dof addresses of the 14 joints
        self._floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._nominal = {
            "dof_damping": m.dof_damping.copy(),
            "dof_frictionloss": m.dof_frictionloss.copy(),
            "dof_armature": m.dof_armature.copy(),
            "gainprm": m.actuator_gainprm.copy(),
            "biasprm": m.actuator_biasprm.copy(),
            "forcerange": m.actuator_forcerange.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "geom_friction": m.geom_friction.copy(),
        }
        self._env_defaults = {"latency": env.latency, "init_noise": env.init_noise,
                              "sensor_noise": env.sensor_noise}

    def apply(self, env, rng):
        if self._nominal is None:
            self._capture(env)
        f = self.cfg.sample(rng)
        self.apply_factors(env, f)
        return f

    def apply_factors(self, env, f):
        m, n = env.model, self._nominal
        a = f["actuator"]
        m.dof_damping[:] = n["dof_damping"]
        m.dof_frictionloss[:] = n["dof_frictionloss"]
        m.dof_armature[:] = n["dof_armature"]
        m.dof_damping[self._act_i] = a["damping"]
        m.dof_frictionloss[self._act_i] = a["frictionloss"]
        m.dof_armature[self._act_i] = a["armature"]
        # position actuator: gain = kp, bias = (0, -kp, -kv). Scale kp in both.
        kp_nom = n["gainprm"][:, 0]
        ratio = a["kp"] / kp_nom
        m.actuator_gainprm[:] = n["gainprm"]
        m.actuator_biasprm[:] = n["biasprm"]
        m.actuator_gainprm[:, 0] = a["kp"]
        m.actuator_biasprm[:, 1] = n["biasprm"][:, 1] * ratio
        m.actuator_forcerange[:, 0] = -a["forcerange"]
        m.actuator_forcerange[:, 1] = a["forcerange"]
        m.body_mass[:] = n["body_mass"] * f["mass_scale"]
        m.body_inertia[:] = n["body_inertia"] * f["mass_scale"]
        m.geom_friction[:] = n["geom_friction"]
        m.geom_friction[self._floor, 0] = n["geom_friction"][self._floor, 0] * f["floor_friction"]
        env.sensor_noise = bool(f["sensor_noise"])
        env.latency = int(f["latency"])
        env.init_noise = float(f["init_noise"])
        self.current = f

    def restore(self, env):
        """Model, latency and init_noise back to what they were at capture."""
        if self._nominal is None:
            return
        m, n = env.model, self._nominal
        m.dof_damping[:] = n["dof_damping"]
        m.dof_frictionloss[:] = n["dof_frictionloss"]
        m.dof_armature[:] = n["dof_armature"]
        m.actuator_gainprm[:] = n["gainprm"]
        m.actuator_biasprm[:] = n["biasprm"]
        m.actuator_forcerange[:] = n["forcerange"]
        m.body_mass[:] = n["body_mass"]
        m.body_inertia[:] = n["body_inertia"]
        m.geom_friction[:] = n["geom_friction"]
        env.latency = self._env_defaults["latency"]
        env.init_noise = self._env_defaults["init_noise"]
        env.sensor_noise = self._env_defaults["sensor_noise"]
        self.current = None
