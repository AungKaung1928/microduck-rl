"""The two reward versions. v1 must not have moved; v2 must do what the
step-3 decision says it does.

Run:  python test_reward.py
"""
import numpy as np

import vec_env            # noqa: F401  sets OMP_NUM_THREADS before mujoco
import common
import env

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("\n--- v1 is byte-identical to the step-2 reward ---")
# Both values were computed with the step-2 code before v2 existed, on the
# same seeds and the same action sequence. They are the regression guard for
# the 108.7 +- 2.9 baseline.
e = env.MicroduckEnv(seed=0, reward="v1")
e.reset(seed=0)
R = sum(e.step(e.zero_action())[1] for _ in range(e.episode_steps))
check("PD hold-pose, seed 0, v1 return unchanged", R == 115.64036289721238, f"{R!r}")
rng = np.random.default_rng(0)
A = rng.uniform(-1, 1, (300, env.ACT_DIM))
e = env.MicroduckEnv(seed=3, reward="v1")
e.reset(seed=3)
R = sum(e.step(A[k % 300])[1] for k in range(e.episode_steps))
check("random actions, seed 3, v1 return unchanged", R == 114.17871144791378, f"{R!r}")
check("default reward is v1, so existing scripts and numbers are untouched",
      env.MicroduckEnv(seed=0).reward_version == "v1" and env.REWARD_WEIGHTS is env.REWARD_V1)
check("v1 term names are the six step-2 terms",
      set(env.REWARD_V1) == {"upright", "height", "posture", "action_rate", "joint_vel", "effort"})

print("\n--- v2 terms ---")
check("v2 drops posture and effort, keeps upright, height, action_rate, joint_vel",
      set(env.REWARD_V2) == {"upright", "height", "action_rate", "joint_vel"})
check("v2 joint_vel is 10x v1", abs(env.REWARD_V2["joint_vel"] / env.REWARD_V1["joint_vel"] - 10) < 1e-9)
check("v2 ceiling is still 2.0 per step",
      env.REWARD_V2["upright"] + env.REWARD_V2["height"] == 2.0)
e = env.MicroduckEnv(seed=0, reward="v2")
e.reset(seed=0)
_, r, _, info = e.step(e.zero_action())
check("v2 reward equals the sum of its logged terms and names match REWARD_V2",
      abs(r - sum(info["terms"].values())) < 1e-12 and set(info["terms"]) == set(env.REWARD_V2))
try:
    env.MicroduckEnv(seed=0, reward="v3")
    check("unknown reward version is refused", False)
except KeyError:
    check("unknown reward version is refused", True)

print("\n--- v2 height has a gradient everywhere below STAND; v1 has none near the floor ---")
def height_term(e, h):
    w = e.weights
    if e.reward_version == "v1":
        return w["height"] * float(np.exp(-((h - env.STAND_HEIGHT) / 0.03) ** 2))
    return w["height"] * float(np.clip(h / env.STAND_HEIGHT, 0, 1))

e1, e2 = env.MicroduckEnv(seed=0, reward="v1"), env.MicroduckEnv(seed=0, reward="v2")
heights = np.linspace(0.0, 0.12, 25)
h1 = [height_term(e1, h) for h in heights]
h2 = [height_term(e2, h) for h in heights]
check("v2 height is strictly increasing from the floor to STAND; v1 is under 0.02 below 6 cm",
      all(b > a for a, b in zip(h2, h2[1:])) and max(h1[:12]) < 0.02, f"v1 at 5.5 cm: {h1[11]:.4f}")
check("v2 upright is the v1 upright, unchanged (its slope is live where the duck actually lies)",
      all(abs(e2.weights["upright"] * max(0.0, c) - e1.weights["upright"] * max(0.0, c)) < 1e-12
          for c in np.linspace(-1, 1, 21)))

print("\n--- measured on trajectories: where the fallen duck actually is, and what each reward says ---")
# Same trajectories under both rewards (the physics does not depend on the
# reward), so the comparison is exact. Fallen = cos < 0.3. Regress per-step
# reward on trunk height and on cos separately.
def fallen(reward):
    rs, cs, hs = [], [], []
    for s in range(6):
        e = env.MicroduckEnv(seed=s, reward=reward)
        e.reset(seed=s)
        rr = np.random.default_rng(10_000 + s)
        for _ in range(e.episode_steps):
            _, r, _, info = e.step(rr.uniform(-1, 1, env.ACT_DIM) if s % 2 else e.zero_action())
            if info["upright_cos"] < 0.3:
                rs.append(r); cs.append(info["upright_cos"]); hs.append(info["trunk_height"])
    rs, cs, hs = np.array(rs), np.array(cs), np.array(hs)
    return rs, cs, hs, np.polyfit(hs, rs, 1)[0], np.polyfit(cs, rs, 1)[0]

f1, c1, hh1, kh1, kc1 = fallen("v1")
f2, c2, hh2, kh2, kc2 = fallen("v2")
check("the fallen duck never rolls past 90 degrees under PD or random actions",
      c1.min() > -0.2, f"min cos {c1.min():+.3f} over {c1.size} fallen steps -- the (1+cos)/2 floor "
                       f"removal would never engage")
check("v2 reward rises with trunk height inside the fallen region; v1 does not",
      kh2 > 4.0 and abs(kh1) < 2.0, f"slope per metre: v1 {kh1:+.2f}, v2 {kh2:+.2f}  (ramp is 8.33)")
check("and the fallen region still scores well below a held stand under v2",
      f2.mean() < 1.0, f"v2 fallen mean {f2.mean():.3f} vs 2.0 for a held stand")

print("\n--- latency and the DR hook ---")
e = env.MicroduckEnv(seed=0, latency=2, init_noise=0.0)
e.reset(seed=0)
e.step(np.ones(env.ACT_DIM))
check("with latency 2 the first action is not applied yet",
      np.allclose(e.data.ctrl, common.default_pose(e.model)))
e.step(np.ones(env.ACT_DIM))
e.step(np.zeros(env.ACT_DIM))
check("and arrives two steps later", not np.allclose(e.data.ctrl, common.default_pose(e.model)))
o, *_ = e.step(np.full(env.ACT_DIM, 0.3))
check("prev_action in the observation is the action just emitted, not the delayed one",
      np.allclose(o[env.OBS_SLICES["prev_action"]], 0.3, atol=1e-6))
# MuJoCo 3.12 has no mjENBL_SENSORNOISE flag any more, so the env applies the
# model's declared sensor_noise itself. Same seed, noise on and off.
ea = env.MicroduckEnv(seed=0, init_noise=0.0)
eb = env.MicroduckEnv(seed=0, init_noise=0.0, sensor_noise=True)
oa, ob = ea.reset(seed=0), eb.reset(seed=0)
da, db = ea.step(np.zeros(14))[0], eb.step(np.zeros(14))[0]
g = env.OBS_SLICES["gyro"]
check("sensor_noise perturbs the gyro read at upstream's 0.005 rad/s scale (x0.25 obs scale)",
      not np.allclose(da[g], db[g]) and np.abs(da[g] - db[g]).max() < 0.25 * 0.005 * 6,
      f"max gyro diff {np.abs(da[g] - db[g]).max():.5f}")
check("sensor_noise leaves joint angles and prev_action untouched",
      np.allclose(da[:42], db[:42]))
pg = env.OBS_SLICES["proj_grav"]
check("projected gravity is still a unit vector under noise",
      abs(np.linalg.norm(db[pg]) - 1) < 1e-5)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all reward checks passed")
