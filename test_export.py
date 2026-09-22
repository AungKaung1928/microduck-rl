"""ONNX export of a randomly initialised actor with a non-trivial normaliser.
Checks that the folded normaliser is really in the graph. No assets needed.

Run:  python test_export.py
"""
import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np       # noqa: E402
import torch             # noqa: E402

import ppo               # noqa: E402
import export_onnx       # noqa: E402
import env as _env       # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


torch.manual_seed(0)
ag = ppo.GaussianActorCritic(_env.OBS_DIM, _env.ACT_DIM, 64, -0.5)
with torch.no_grad():
    ag.mu[-1].weight.mul_(100.0)       # make the output depend visibly on the input
norm = ppo.RunningNorm(_env.OBS_DIM)
rng = np.random.default_rng(0)
norm.update(rng.normal(2.0, 3.0, (500, _env.OBS_DIM)))
pol = ppo.Policy(ag, norm, ppo.Config(hidden=64, tag="test"))

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "p.onnx")
m, nbytes = export_onnx.export(pol, path)
check("export writes a file", nbytes > 1000, f"{nbytes} bytes")

obs = rng.normal(2.0, 3.0, (200, _env.OBS_DIM)).astype(np.float32)
s = export_onnx.sess(path, 1)
got = np.concatenate([s.run(None, {"obs": obs[i:i + 1]})[0] for i in range(len(obs))])
ref = np.stack([pol(o) for o in obs])
check("onnx output matches the training path (numpy normaliser + torch actor) to 1e-4",
      float(np.abs(got - ref).max()) < 1e-4, f"max diff {np.abs(got - ref).max():.2e}")

# Prove the normaliser is inside the graph: an actor fed the RAW observation
# must disagree with the graph.
with torch.no_grad():
    raw = ag.mu(torch.from_numpy(obs)).numpy()
check("the graph does not equal the un-normalised actor, so normalisation is folded in",
      float(np.abs(raw - got).max()) > 1e-2, f"max diff vs raw actor {np.abs(raw - got).max():.2e}")

big = obs[:1] + 1e4
g_big = s.run(None, {"obs": big})[0]
with torch.no_grad():
    t_big = m(torch.from_numpy(big)).numpy()
check("clipping survives export (a huge observation gives the same finite answer both ways)",
      np.isfinite(g_big).all() and float(np.abs(g_big - t_big).max()) < 1e-4)

os.remove(path)
os.rmdir(tmp)
print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all export checks passed")
