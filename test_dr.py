"""Domain randomisation: ranges come from the measured servo fits, application
is exact, restore is exact, and the held-out variants are what they claim.

Run:  python test_dr.py
"""
import numpy as np

import vec_env            # noqa: F401  sets OMP_NUM_THREADS before mujoco
import mujoco
import common
import dr
import env

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("\n--- ranges are the four measured fits, not a +-20% guess ---")
cls = np.array(list(common.ACTUATOR_CLASSES.values()))
check("actuator lo/hi are the min/max over the four classes",
      np.allclose(dr.ACT_PARAM_LO, cls.min(0)) and np.allclose(dr.ACT_PARAM_HI, cls.max(0)))
spread = dr.ACT_PARAM_HI / dr.ACT_PARAM_LO
check("frictionloss is the widest range (6.7x) and every other parameter is under 1.5x",
      abs(spread[1] - 6.667) < 0.05 and np.all(np.delete(spread, 1) < 1.5), str(spread.round(2)))

cfg = dr.DRConfig()
rng = np.random.default_rng(0)
ok = True
for _ in range(300):
    f = cfg.sample(rng)
    a = np.array([f["actuator"][k] for k in dr.ACT_PARAM_NAMES])
    ok &= bool(np.all(a >= dr.ACT_PARAM_LO) and np.all(a <= dr.ACT_PARAM_HI))
    ok &= cfg.mass_scale[0] <= f["mass_scale"] <= cfg.mass_scale[1]
    ok &= cfg.latency[0] <= f["latency"] <= cfg.latency[1] and isinstance(f["latency"], int)
    ok &= cfg.init_noise[0] <= f["init_noise"] <= cfg.init_noise[1]
check("300 samples all inside the declared ranges", ok)
f0 = dr.DRConfig.none().sample(rng)
check("DRConfig.none() samples the nominal class, latency 0, init_noise 0.02, no sensor noise",
      np.allclose([f0["actuator"][k] for k in dr.ACT_PARAM_NAMES], dr.NOMINAL_CLASS)
      and f0["latency"] == 0 and f0["init_noise"] == 0.02 and not f0["sensor_noise"])

print("\n--- apply is exact, restore is exact, nothing compounds ---")
e = env.MicroduckEnv(seed=0)
m = e.model
snap = {k: getattr(m, k).copy() for k in ("dof_damping", "dof_frictionloss", "dof_armature",
                                          "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                                          "body_mass", "body_inertia", "geom_friction")}
rz = dr.DomainRandomizer(cfg)
f = rz.apply(e, np.random.default_rng(1))
act_i = common.actuated_qvel_index(m)
check("all 14 actuated dofs carry the sampled damping/frictionloss/armature",
      np.allclose(m.dof_damping[act_i], f["actuator"]["damping"])
      and np.allclose(m.dof_frictionloss[act_i], f["actuator"]["frictionloss"])
      and np.allclose(m.dof_armature[act_i], f["actuator"]["armature"]))
check("passive dofs (free joint) are untouched", np.allclose(m.dof_damping[:6], snap["dof_damping"][:6]))
check("kp goes into gainprm[0] and biasprm[1] together, keeping bias = -gain",
      np.allclose(m.actuator_gainprm[:, 0], f["actuator"]["kp"])
      and np.allclose(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0]))
check("forcerange is symmetric at the sampled limit",
      np.allclose(m.actuator_forcerange[:, 1], f["actuator"]["forcerange"])
      and np.allclose(m.actuator_forcerange[:, 0], -f["actuator"]["forcerange"]))
check("every body mass and inertia scaled by the same factor",
      np.allclose(m.body_mass, snap["body_mass"] * f["mass_scale"])
      and np.allclose(m.body_inertia, snap["body_inertia"] * f["mass_scale"]))
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
check("floor sliding friction scaled, other geoms untouched",
      np.isclose(m.geom_friction[floor, 0], snap["geom_friction"][floor, 0] * f["floor_friction"])
      and np.allclose(np.delete(m.geom_friction, floor, 0), np.delete(snap["geom_friction"], floor, 0)))
check("env latency and init_noise took the sampled values",
      e.latency == f["latency"] and e.init_noise == f["init_noise"])
check("sensor noise is on", e.sensor_noise is True)
for _ in range(5):
    rz.apply_factors(e, f)
check("applying the same factors five times gives the same model (no compounding)",
      np.allclose(m.body_mass, snap["body_mass"] * f["mass_scale"]))
rz.restore(e)
check("restore returns every array bit-for-bit",
      all(np.array_equal(getattr(m, k), v) for k, v in snap.items())
      and e.latency == 0 and e.init_noise == 0.02 and e.sensor_noise is False)

print("\n--- the env applies it at reset, and it changes the dynamics ---")
e = env.MicroduckEnv(seed=0, dr=dr.DomainRandomizer(cfg))
kps = set()
for _ in range(5):
    e.reset()
    kps.add(round(float(e.model.actuator_gainprm[0, 0]), 6))
check("five resets draw five different servo gains", len(kps) == 5)
e_nom = env.MicroduckEnv(seed=0)
e_nom.reset(seed=4)
e_dr = env.MicroduckEnv(seed=0, dr=dr.DomainRandomizer(cfg))
e_dr.reset(seed=4)
r_nom = sum(e_nom.step(e_nom.zero_action())[1] for _ in range(50))
r_dr = sum(e_dr.step(e_dr.zero_action())[1] for _ in range(50))
check("the same seed under DR gives a different rollout than nominal", r_nom != r_dr,
      f"{r_nom:.3f} vs {r_dr:.3f}")
o = e_dr.reset(seed=4)
check("observation is still 48-dim and finite under DR", o.shape == (48,) and np.isfinite(o).all())

print("\n--- the held-out variants rest on the floor and keep the action space ---")
for v in common.HELDOUT_VARIANTS:
    mv, _ = common.load(v)
    names = [mujoco.mj_id2name(mv, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mv.nu)]
    check(f"{v}: 14 actuators in the training order, >= 10 floor-collidable geoms",
          names == common.ACTUATOR_NAMES and len(common.floor_contact_geoms(mv)) >= 10,
          f"nq {mv.nq}, floor geoms {len(common.floor_contact_geoms(mv))}")
mw, _ = common.load("walk_backlash")
check("walk_backlash is NOT a held-out variant here: only its feet touch the floor",
      "walk_backlash" not in common.HELDOUT_VARIANTS and len(common.floor_contact_geoms(mw)) == 2)
eb = env.MicroduckEnv(variant="groundcontact_backlash", seed=0)
ob = eb.reset(seed=0)
check("groundcontact_backlash gives a 48-dim observation through the strided index",
      ob.shape == (48,) and eb.model.nq == 35)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all domain-randomisation checks passed")
