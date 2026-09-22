"""Shared constants and model loading for the Microduck CPU project.

The robot is the Pollen Robotics / Hugging Face "Microduck": a 25 cm, ~740 g
open-source biped with 14 position-controlled servos. Upstream trains it with
mjlab + MuJoCo Warp, which requires CUDA. This box has no GPU, so everything
here runs the same MJCF in plain CPU MuJoCo.

Three model variants matter, and they are the whole reason this project can
make a sim-to-sim claim later:

    walk            the nominal model, 14 actuated joints, nothing else
    walk_backlash   the same robot with a passive backlash joint added in
                    series with every actuated one (nq 21 -> 35)
    rollers         nominal plus passive roller geoms at the ground contact

All three expose the SAME 14 actuators under the SAME names in the SAME order.
So a policy trained on `walk` can be evaluated on `walk_backlash` with no
remapping at all -- the observation layout changes, the action layout does not.
That is the held-out physics for the domain-randomisation step, and it is
supplied by the robot's own authors rather than invented here.
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

VARIANTS = {
    "walk": "scene_walk.xml",
    "walk_backlash": "scene_walk_backlash.xml",
    "rollers": "scene_rollers.xml",
    "groundcontact": "scene.xml",
    # Backlash joints on the ground-contact model (10 floor-collidable geoms).
    # `walk_backlash` has the backlash but only the feet can touch the floor,
    # so a fallen robot sinks through the world there; for a stand-and-recover
    # policy the held-out physics has to be this one.
    "groundcontact_backlash": "scene_backlash.xml",
}
# The variants a policy trained on `groundcontact` is evaluated on and never
# trained on. Both rest on the floor when toppled; both keep the 14 actuators.
HELDOUT_VARIANTS = ("groundcontact_backlash", "rollers")

# Order is fixed by the <actuator> block in robot_walk.xml. Written out rather
# than read from the model so that a test can catch an upstream reordering,
# which would silently permute every action a trained policy emits.
ACTUATOR_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
N_ACT = 14

# A free joint: 3 translation + 4 quaternion in qpos, 3 + 3 in qvel.
FREE_NQ, FREE_NV = 7, 6

# Control runs at 50 Hz over a 500 Hz physics step, matching upstream.
PHYSICS_DT = 0.002
CONTROL_HZ = 50
SUBSTEPS = int(round(1.0 / CONTROL_HZ / PHYSICS_DT))   # 10

# The shipped actuator class `chosen_actuator` in joints_properties.xml, and
# the three other measured classes beside it. These are system-identification
# results for the same XL330 servo fitted by different people on different
# benches -- upstream kept all four in the file. They are not guesses, so they
# are the honest place to get a domain-randomisation range from instead of
# inventing +-20% around a nominal.
ACTUATOR_CLASSES = {
    #                  damping  frictionloss  armature     kp   forcerange
    "chosen_actuator":  (0.053,       0.0048,   0.0018,  0.55,        0.96),
    "chosen_actuator_old":     (0.048, 0.006,   0.002,   0.52,        0.91),
    "chosen_actuator_new":     (0.041, 0.032,   0.002,   0.386,       0.67),
    "chosen_actuator_antoine": (0.044, 0.013,   0.0017,  0.43,        0.75),
}


def variant_path(variant):
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}, have {sorted(VARIANTS)}")
    p = os.path.join(ASSETS, VARIANTS[variant])
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing -- assets/ is gitignored. Run ./fetch_assets.sh first."
        )
    return p


def load(variant="walk"):
    m = mujoco.MjModel.from_xml_path(variant_path(variant))
    return m, mujoco.MjData(m)


def keyframe(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if i < 0:
        raise KeyError(f"no keyframe {name!r}")
    return i


def reset_to(model, data, name="STAND"):
    mujoco.mj_resetDataKeyframe(model, data, keyframe(model, name))
    mujoco.mj_forward(model, data)
    return data


def actuated_qpos_index(model):
    """qpos indices of the 14 actuated joints, in actuator order.

    On the backlash variant qpos is 35 long and the actuated joints are no
    longer a contiguous slice, so anything that reads joint angles has to go
    through this rather than assuming qpos[7:21].
    """
    idx = []
    for name in ACTUATOR_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"joint {name!r} missing from this variant")
        idx.append(model.jnt_qposadr[j])
    return np.array(idx, dtype=int)


def actuated_qvel_index(model):
    """qvel indices of the 14 actuated joints, in actuator order.

    The qpos twin of this exists because `walk_backlash` interleaves passive
    joints; qvel has exactly the same problem. On `groundcontact` these are the
    contiguous 6..19, on `walk_backlash` they are 6, 8, 10 ... 32. Anything
    reading joint velocity goes through here.
    """
    idx = []
    for name in ACTUATOR_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"joint {name!r} missing from this variant")
        idx.append(model.jnt_dofadr[j])
    return np.array(idx, dtype=int)


def joint_limits(model):
    """(lo, hi) arrays of the 14 actuated joint ranges, in actuator order.

    Worth stating plainly because it is a trap: `ctrlrange` on every actuator
    is [-10, 10] rad, which is not a joint limit and not even reachable. The
    tightest real limit is the hip roll at +-0.384 rad. A position actuator
    handed a target of 10 rad does not error -- it just saturates against the
    joint stop and burns force range. Clamping is the caller's job.
    """
    lo, hi = [], []
    for name in ACTUATOR_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo.append(model.jnt_range[j, 0])
        hi.append(model.jnt_range[j, 1])
    return np.array(lo), np.array(hi)


def default_pose(model, name="STAND"):
    """The 14 actuated joint angles at a keyframe, in actuator order.

    Actions are residuals around this, so it is also what `ctrl` is set to when
    the policy emits zero. On STAND the keyframe's own `ctrl` equals its `qpos`
    at these joints, so the robot starts commanded to where it already is.
    """
    return model.key_qpos[keyframe(model, name)][actuated_qpos_index(model)].copy()


def sensor_slice(model, name):
    """Where a named sensor's values live in `data.sensordata`."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if i < 0:
        raise KeyError(f"no sensor {name!r}")
    adr = int(model.sensor_adr[i])
    return slice(adr, adr + int(model.sensor_dim[i]))


def floor_contact_geoms(model):
    """Names of the geoms that can actually touch the floor.

    MuJoCo lets two geoms collide only if one's contype shares a bit with the
    other's conaffinity. In `robot_walk.xml` the floor is contype/conaffinity
    1, the feet are 1, and every body and limb geom is 2. So on the `walk`
    variant ONLY THE FEET can touch the ground -- once the robot topples, the
    trunk passes straight through the floor.

    That is deliberate upstream: for a walking task the episode ends the moment
    the robot falls, so what happens afterwards never has to be physical. It
    makes `walk` the wrong model for anything involving lying on the ground or
    getting back up, and the failure is silent -- nothing errors, the robot
    just sinks. `scene.xml` (the `groundcontact` variant) enables ten
    ground-collidable geoms and comes to rest correctly.
    """
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ct, ca = model.geom_contype[floor], model.geom_conaffinity[floor]
    out = []
    for i in range(model.ngeom):
        if i == floor:
            continue
        if (model.geom_contype[i] & ca) or (ct & model.geom_conaffinity[i]):
            out.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"<unnamed {i}>")
    return out


def trunk_height(model, data):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    return float(data.xpos[b, 2])


def upright_cos(model, data):
    """Cosine of the trunk's tilt from vertical. 1.0 = upright, 0 = on its side."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    return float(data.xmat[b].reshape(3, 3)[2, 2])
