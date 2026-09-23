"""Step 4's table: nominal-trained versus DR-trained, on the model they were
trained on and on the two held-out variants neither has ever seen.

    python eval_gap.py --nominal runs/ppo_v2.pt --dr runs/ppo_v2dr.pt

Runs eval_policy.run for each (policy, variant) cell -- 100 episodes x 5
seeds each, the same seeds in every cell -- and prints the gap: return and
upright fraction on the held-out variant minus the same on the training
variant, per policy. The difference between the two gaps is the number domain
randomisation is worth here. Recovery is printed too but carries almost no
information (step 3). Survival is height-only, so a policy braced on its side
scores well on it; upright fraction and median trunk height are what say
whether the robot is standing. Writes runs/gap.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse            # noqa: E402
import datetime as dt      # noqa: E402
import json                # noqa: E402

import common              # noqa: E402
import eval_policy as EP   # noqa: E402
from ppo import Policy     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True, help="policy trained without DR")
    ap.add_argument("--dr", required=True, help="policy trained with DR")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(EP.SEEDS))
    ap.add_argument("--out", default="runs/gap.json")
    a = ap.parse_args()

    variants = ("groundcontact",) + common.HELDOUT_VARIANTS
    cells = {}
    for name, path in (("nominal", a.nominal), ("dr", a.dr)):
        pol = Policy.load(path)
        for v in variants:
            cells[f"{name}/{v}"] = EP.run(lambda e, p=pol: p, v, pol.cfg.reward, a.episodes,
                                          a.seeds, pol.cfg.action_scale)
            c = cells[f"{name}/{v}"]
            print(f"  {name:8s} {v:24s} return {c['return_mean']:.1f} +- {c['seed_std_return']:.1f}  "
                  f"upright {c['upright_mean']:.2f}  survival {c['survival_mean']:.2f}  "
                  f"height p50 {c['height_p50']:.3f}  recovery {c['recovery_rate']:.2f}", flush=True)

    print("\n| trained | evaluated on | return +- seeds | upright +- seeds | survival | trunk height p50 | "
          "recovery | return gap | upright gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ("nominal", "dr"):
        b = cells[f"{name}/groundcontact"]
        for v in variants:
            c = cells[f"{name}/{v}"]
            print(f"| {name} | {v} | {c['return_mean']:.1f} +- {c['seed_std_return']:.1f} | "
                  f"{c['upright_mean']:.2f} +- {c['seed_std_upright']:.2f} | {c['survival_mean']:.2f} | "
                  f"{c['height_p50']:.3f} m | {c['recovery_rate']:.2f} | "
                  f"{c['return_mean'] - b['return_mean']:+.1f} | {c['upright_mean'] - b['upright_mean']:+.2f} |")
    print("\nmean return per term, per episode:")
    for k, c in cells.items():
        print(f"  {k:32s} " + "  ".join(f"{t} {x:7.1f}" for t, x in c["terms_mean"].items()))
    with open(a.out, "w") as f:
        json.dump({"nominal_policy": a.nominal, "dr_policy": a.dr, "episodes": a.episodes,
                   "seeds": a.seeds, "cells": cells,
                   "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
