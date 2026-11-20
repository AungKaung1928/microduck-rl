"""Push-recovery bookkeeping in metrics.summarise / aggregate, on hand-built
step arrays (no simulation).

Run:  python test_metrics.py
"""
import numpy as np

import vec_env            # noqa: F401  sets OMP_NUM_THREADS before mujoco
import env
import metrics as M

FAILED = []
W = int(M.RECOVER_WINDOW * 50)      # 2 s at 50 Hz = 100 steps


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def episode(n, down, pushes, fallen_at=()):
    up = np.ones(n, bool)
    for lo, hi in down:
        up[lo:hi] = False
    p = np.zeros(n, bool)
    p[list(pushes)] = True
    f = np.zeros(n, bool)
    f[list(fallen_at)] = True
    return M.summarise(np.ones(n), f, up, p)


print("\n--- summarise ---")
check("window is 100 steps at the 50 Hz control rate", W == 100 and env.CONTROL_DT == 0.02)
# push 10: down 10..14, up at 15 -> recovered after 5 steps (0.10 s)
# push 50: already down at step 49 -> unrecoverable, out of the denominator
# push 150: down for W + 1 steps (150..250) -> not recovered in the window
s = episode(400, [(10, 15), (40, 51), (150, 150 + W + 1)], [10, 50, 150], fallen_at=range(40, 60))
check("counts", (s["pushes"], s["pushes_unrecoverable"], s["pushes_valid"], s["recovered"]) == (3, 1, 2, 1), s)
check("recovery = 1 / 2", s["recovery"] == 0.5)
check("time to recover 0.10 s", np.allclose(s["time_to_recover"], [0.10]), s["time_to_recover"])
check("survival 1 - 20/400, return = steps", s["survival"] == 0.95 and s["return"] == 400.0)
# boundary: upright exactly W steps after the push still counts (window is inclusive)
s = episode(300, [(20, 20 + W)], [20])
check("upright at push + W counts", s["recovered"] == 1 and np.allclose(s["time_to_recover"], [W * 0.02]))
s = episode(300, [(20, 20 + W + 1)], [20])
check("upright at push + W + 1 does not", s["recovered"] == 0 and s["recovery"] == 0.0)
s = episode(100, [], [])
check("no pushes -> recovery NaN, not 0 or 1", np.isnan(s["recovery"]))

print("\n--- aggregate ---")
a = episode(400, [(10, 15)], [10])                 # 1 of 1, 0.10 s
b = episode(400, [(10, 10 + W + 1), (140, 160), (200, 200 + W + 1)], [10, 150, 200])  # 0 of 2 + 1 while down
g = M.aggregate([a, b])
check("recovery pooled over pushes, not averaged over episodes",
      (g["pushes_valid"], g["pushes_unrecoverable"]) == (3, 1) and np.isclose(g["recovery_rate"], 1 / 3),
      f'{g["recovery_rate"]} (per-episode mean would be 0.5)')
check("p50 recover time from recovered pushes only", g["recover_p50_s"] == 0.10)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all metrics checks passed")
