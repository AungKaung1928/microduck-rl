"""Step 4's table: nominal-trained versus DR-trained, on the model they were
trained on and on the two held-out variants neither has ever seen.

    python eval_gap.py --nominal runs/ppo_v2.pt --dr runs/ppo_v2dr.pt

Runs eval_policy.run for each (policy, variant) cell -- 100 episodes x 5
seeds each, the same seeds in every cell -- and prints the gap: recovery on
the held-out variant minus recovery on the training variant, per policy. The
difference between the two gaps is the number domain randomisation is worth
here. Writes runs/gap.json.
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
            print(f"  {name:8s} {v:24s} recovery {c['recovery_rate']:.2f} +- "
                  f"{c['seed_std_recovery']:.2f}  return {c['return_mean']:.1f}", flush=True)

    print(f"\n| trained | evaluated on | recovery | +- seeds | return | gap vs training model |")
    print("|---|---|---|---|---|---|")
    for name in ("nominal", "dr"):
        base = cells[f"{name}/groundcontact"]["recovery_rate"]
        for v in variants:
            c = cells[f"{name}/{v}"]
            gap = c["recovery_rate"] - base
            print(f"| {name} | {v} | {c['recovery_rate']:.2f} | {c['seed_std_recovery']:.2f} | "
                  f"{c['return_mean']:.1f} | {gap:+.2f} |")
    with open(a.out, "w") as f:
        json.dump({"nominal_policy": a.nominal, "dr_policy": a.dr, "episodes": a.episodes,
                   "seeds": a.seeds, "cells": cells,
                   "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
