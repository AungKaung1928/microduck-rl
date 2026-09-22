"""The PD hold-pose baseline, and a look at whether the reward can be learned.

Two jobs, both of which existed only as numbers in the README before this file
did. The README reported a baseline return of 108.9 +- 2.8 over 20 seeds with
no script behind it, the same way it reported a 0% wrapper overhead that turned
out to be 13% once the machine was measured properly. A number with no
reproducer does not get to survive.

  1. the baseline itself: the shipped PD controller commanded to hold STAND,
     which is what `zero_action` emits, over N seeds of the full task
  2. three diagnostics on the reward, because a reward is a specification and
     this one has never had anything optimise against it:
       - what each term actually contributes, against its nominal weight
       - how much reward varies in the fallen region, which is the gradient a
         policy would have to climb to get back up
       - the per-group scale of the observation, which decides what the first
         layer of an unnormalised policy pays attention to

Single process, a few seconds, one core. Nothing here loads the machine.

Run:  python3 baseline.py --seeds 20               # v1, the step-2 reward
      python3 baseline.py --seeds 20 --reward v2   # v2, the step-3 reward
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse           # noqa: E402
import json               # noqa: E402

import numpy as np        # noqa: E402

import common             # noqa: E402
import env as _env        # noqa: E402


def episode(seed, policy, variant="groundcontact", reward="v1", **kw):
    """One full episode. `policy` is "pd" (hold STAND) or "random".

    Both are needed for the reward diagnostics below, and using only one gives
    a misleading answer in opposite directions. The PD baseline emits a
    constant action, so `action_rate` and the `prev_action` observation block
    are identically zero for it and look dead when they are not. Random actions
    exercise both but tell you nothing about the baseline's return.
    """
    e = _env.MicroduckEnv(variant=variant, seed=seed, reward=reward, **kw)
    obs = e.reset(seed=seed)
    zero = e.zero_action()
    rng = np.random.default_rng(10_000 + seed)
    rec = {"reward": [], "upright": [], "height": [], "fallen": [],
           "terms": [], "obs": []}
    done = False
    while not done:
        a = zero if policy == "pd" else rng.uniform(-1, 1, _env.ACT_DIM)
        rec["obs"].append(obs)
        obs, r, done, info = e.step(a)
        rec["reward"].append(r)
        rec["upright"].append(info["upright_cos"])
        rec["height"].append(info["trunk_height"])
        rec["fallen"].append(info["fallen"])
        rec["terms"].append(info["terms"])
    return {k: (np.array(v) if k != "terms" else v) for k, v in rec.items()}


def first_crossing(series, pred, dt):
    hit = np.flatnonzero(pred(series))
    return float(hit[0] * dt) if hit.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--reward", choices=sorted(_env.REWARDS), default="v1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out_path = a.out or ("runs/baseline.json" if a.reward == "v1" else f"runs/baseline_{a.reward}.json")
    dt = _env.CONTROL_DT
    weights = _env.REWARDS[a.reward]

    eps = [episode(s, "pd", reward=a.reward) for s in range(a.seeds)]
    rnd = [episode(s, "random", reward=a.reward) for s in range(a.seeds)]
    returns = np.array([e["reward"].sum() for e in eps])
    floor = np.array([e["fallen"].mean() for e in eps])
    tilt = np.array([first_crossing(e["upright"], lambda x: x < 0.9, dt) for e in eps])
    down = np.array([first_crossing(e["height"], lambda x: x < _env.FALL_HEIGHT, dt)
                     for e in eps])

    print(f"\nPD hold-pose baseline, {a.seeds} seeds, groundcontact, full task, reward {a.reward}")
    print(f"  return                      {returns.mean():7.1f} +- {returns.std(ddof=1):.1f}"
          f"   of a {2.0 * eps[0]['reward'].size:.0f} ceiling")
    print(f"  fraction of episode on floor {100*floor.mean():6.0f}%")
    print(f"  starts to tilt (cos < 0.9)  {np.nanmean(tilt):7.2f} s")
    print(f"  trunk reaches the floor     {np.nanmean(down):7.2f} s")

    # -- diagnostic 1: which reward terms are numerically alive ------------
    keys = list(weights)
    def term_sums(group):
        return {k: float(np.mean([sum(t[k] for t in e["terms"]) for e in group]))
                for k in keys}
    sums, sums_r = term_sums(eps), term_sums(rnd)
    rnd_return = float(np.mean([e["reward"].sum() for e in rnd]))
    print(f"\n  per-episode contribution of each reward term, {a.seeds} seeds each")
    print(f"    {'term':<12} {'weight':>9} {'PD sum':>9} {'% ret':>7}"
          f" | {'random sum':>10} {'% ret':>7}")
    for k in keys:
        print(f"    {k:<12} {weights[k]:9.1e} {sums[k]:9.3f} "
              f"{100*abs(sums[k])/abs(returns.mean()):6.2f}% | "
              f"{sums_r[k]:10.3f} {100*abs(sums_r[k])/abs(rnd_return):6.2f}%")
    dead = [k for k in keys if weights[k] < 0
            and abs(sums[k]) / abs(returns.mean()) < 0.01
            and abs(sums_r[k]) / abs(rnd_return) < 0.01]
    print(f"    Under 1% of the return under BOTH policies: "
          f"{', '.join(dead) if dead else 'none'}.")
    print("    action_rate is exactly 0 for the PD baseline because a constant\n"
          "    action has no rate; it is live under any policy that moves.")
    if a.reward == "v1":
        print("    The others are dead as written, so the README's claim that step 3\n"
              "    can show which penalty is doing the work would show three zeros.\n"
              "    joint_vel's -2e-4 was set for the ~20 rad/s of a fall; measured\n"
              "    joint speed is under 1 rad/s, which is 400x smaller once squared.")
    else:
        share = 100 * abs(sums_r["joint_vel"]) / abs(rnd_return)
        target = 5.0
        print(f"    v2: joint_vel is {share:.2f}% of the random-action return. The target\n"
              f"    is ~{target:.0f}%; scale the weight by {target / max(share, 1e-9):.1f}x to get there\n"
              f"    (-> {weights['joint_vel'] * target / max(share, 1e-9):.2e}). Changing it\n"
              f"    re-baselines: rerun this script and update the README together.")

    # -- diagnostic 2: is there a gradient out of the fallen region? -------
    allr = eps + rnd
    stand_r = np.concatenate([e["reward"][e["upright"] > 0.9] for e in allr
                              if (e["upright"] > 0.9).any()])
    fall_r = np.concatenate([e["reward"][e["upright"] < 0.3] for e in allr
                             if (e["upright"] < 0.3).any()])
    print(f"\n  reward while upright (cos > 0.9): {stand_r.mean():.3f} +- "
          f"{stand_r.std():.3f}   over {stand_r.size} steps")
    print(f"  reward while down    (cos < 0.3): {fall_r.mean():.3f} +- "
          f"{fall_r.std():.3f}   over {fall_r.size} steps")
    print(f"  level ratio {stand_r.mean()/max(fall_r.mean(), 1e-9):.0f}x, "
          f"spread ratio {stand_r.std()/max(fall_r.std(), 1e-9):.1f}x")
    print("  The level gap is large, so the RETURN clearly prefers standing and\n"
          "  the task is not degenerate. The risk is local: `upright` is floored\n"
          "  at 0 past 90 degrees and `height` is a 3 cm Gaussian that is ~1e-3\n"
          "  at floor level, so within the fallen region the per-step reward is\n"
          f"  {fall_r.mean():.2f} +- {fall_r.std():.2f}, which carries little\n"
          "  information about which way is up. Expect exploration, not the\n"
          "  reward shape, to be what decides whether recovery is learned, and\n"
          "  expect 'fall slowly and stay tilted' as the competing local optimum.")

    # -- diagnostic 3: relative scale of the observation groups ------------
    obs_pd = np.concatenate([e["obs"] for e in eps])
    obs_rd = np.concatenate([e["obs"] for e in rnd])
    print(f"\n  observation group rms after OBS_SCALE, {obs_pd.shape[0]} steps each")
    print(f"    {'group':<12} {'PD':>8} {'random':>8}")
    rms = {}
    for name, sl in _env.OBS_SLICES.items():
        a_, b_ = (float(np.sqrt((o[:, sl] ** 2).mean())) for o in (obs_pd, obs_rd))
        rms[name] = {"pd": a_, "random": b_}
        print(f"    {name:<12} {a_:8.3f} {b_:8.3f}")
    spread = max(v["random"] for v in rms.values()) / max(
        min(v["random"] for v in rms.values()), 1e-9)
    print(f"  Widest-to-narrowest under random actions: {spread:.0f}x. The 28\n"
          "  joint dims that describe the body's configuration carry far less\n"
          "  input variance than prev_action and projected gravity, so an\n"
          "  unnormalised first layer attends mostly to the policy's own last\n"
          "  output. PPO needs a running observation normaliser; OBS_SCALE is a\n"
          "  fixed guess and the measured spread says it guessed wrong.")

    os.makedirs("runs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": a.seeds, "variant": "groundcontact", "reward": a.reward,
            "weights": weights,
            "episode_steps": int(eps[0]["reward"].size),
            "ceiling": 2.0 * int(eps[0]["reward"].size),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std(ddof=1)),
            "floor_fraction": float(floor.mean()),
            "tilt_s": float(np.nanmean(tilt)), "down_s": float(np.nanmean(down)),
            "term_sums": sums, "term_sums_random": sums_r,
            "random_return": rnd_return,
            "reward_upright_mean": float(stand_r.mean()),
            "reward_upright_std": float(stand_r.std()),
            "reward_fallen_mean": float(fall_r.mean()),
            "reward_fallen_std": float(fall_r.std()),
            "obs_group_rms": rms,
        }, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
