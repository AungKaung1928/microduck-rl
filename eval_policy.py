"""Evaluate a trained policy the way the README reports it: 100 episodes x 5
seeds, deterministic (mean) actions, against the PD baseline on the same
seeds, on any model variant.

    python eval_policy.py runs/ppo_v2.pt --variant groundcontact
    python eval_policy.py runs/ppo_v2.pt --variant groundcontact_backlash

Episode j of seed s is reset with seed s*100_000 + j, so any row can be
replayed by hand. Metric definitions are metrics.py's; the same module feeds
the in-training evaluation, so the curve and this table cannot disagree.
Writes runs/eval_<tag>_<variant>.json and prints a markdown row pair.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse            # noqa: E402
import datetime as dt      # noqa: E402
import json                # noqa: E402
import time                # noqa: E402

import numpy as np         # noqa: E402

import env as _env         # noqa: E402
import metrics             # noqa: E402
from ppo import Policy     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def run(policy_fn_factory, variant, reward, episodes, seeds, action_scale):
    per_seed, eps_all = [], []
    for s in seeds:
        e = _env.MicroduckEnv(variant=variant, seed=s, reward=reward, action_scale=action_scale)
        pol = policy_fn_factory(e)
        eps = [metrics.run_episode(e, pol, seed=s * 100_000 + j) for j in range(episodes)]
        per_seed.append(metrics.aggregate(eps))
        eps_all.extend(eps)
    agg = metrics.aggregate(eps_all)
    agg["per_seed_return"] = [p["return_mean"] for p in per_seed]
    agg["per_seed_recovery"] = [p["recovery_rate"] for p in per_seed]
    agg["per_seed_upright"] = [p["upright_mean"] for p in per_seed]
    agg["seed_std_upright"] = float(np.std(agg["per_seed_upright"], ddof=1)) if len(seeds) > 1 else 0.0
    agg["seed_std_return"] = float(np.std(agg["per_seed_return"], ddof=1)) if len(seeds) > 1 else 0.0
    agg["seed_std_recovery"] = float(np.nanstd(agg["per_seed_recovery"], ddof=1)) if len(seeds) > 1 else 0.0
    return agg


def row(name, a):
    p50 = "--" if a["recover_p50_s"] is None else f"{a['recover_p50_s']:.2f}"
    p90 = "--" if a["recover_p90_s"] is None else f"{a['recover_p90_s']:.2f}"
    return (f"| {name} | {a['return_mean']:.1f} +- {a['seed_std_return']:.1f} | "
            f"{a['survival_mean']:.2f} | {a['recovery_rate']:.2f} +- {a['seed_std_recovery']:.2f} | "
            f"{p50} / {p90} | {a['pushes_unrecoverable']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="runs/ppo_<tag>.pt")
    ap.add_argument("--variant", default="groundcontact")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pol = Policy.load(a.policy)
    cfg = pol.cfg
    tag = a.tag or cfg.tag
    t0 = time.perf_counter()
    learned = run(lambda e: pol, a.variant, cfg.reward, a.episodes, a.seeds, cfg.action_scale)
    pd = run(metrics.pd_policy, a.variant, cfg.reward, a.episodes, a.seeds, cfg.action_scale)

    print(f"\n{a.episodes} episodes x {len(a.seeds)} seeds, variant {a.variant}, reward {cfg.reward}")
    print("| policy | return (+- over seeds) | survival | recovery (+- over seeds) | "
          "recover p50 / p90 s | pushes on a fallen robot |")
    print("|---|---|---|---|---|---|")
    print(row(f"PPO {tag}", learned))
    print(row("PD hold-pose", pd))

    out = a.out or f"runs/eval_{tag}_{a.variant}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"policy": a.policy, "variant": a.variant, "reward": cfg.reward,
                   "trained_on": cfg.variant, "dr": cfg.dr, "episodes": a.episodes,
                   "seeds": a.seeds, "learned": learned, "pd": pd,
                   "wall_s": time.perf_counter() - t0,
                   "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
