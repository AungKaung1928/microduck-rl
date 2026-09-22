"""Step 5: one ONNX graph from raw 48-dim observation to 14-dim mean action.

The observation normaliser is folded into the graph -- (obs - mean) / std,
clipped -- so the runtime consumer never has to know it existed. A policy
that ships without its normaliser is a different policy.

Verification comes before timing: 1000 observations drawn from real episodes
(not white noise, whose range the normaliser has never seen) through both
paths, max |torch - onnx| must be under 1e-5. Then ONNX Runtime latency on
ONE intra-op thread, p50 / p99 over 2000 calls, batch 1, which is the shape
an on-robot controller runs.

    python export_onnx.py runs/ppo_v2.pt
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse            # noqa: E402
import datetime as dt      # noqa: E402
import json                # noqa: E402
import time                # noqa: E402

import numpy as np         # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch               # noqa: E402
import torch.nn as nn      # noqa: E402

import env as _env         # noqa: E402
from ppo import Policy     # noqa: E402


class Deployed(nn.Module):
    """normaliser + actor mean, as one module."""

    def __init__(self, policy):
        super().__init__()
        self.mu = policy.agent.mu
        self.register_buffer("mean", torch.as_tensor(policy.norm.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(policy.norm.std, dtype=torch.float32))
        self.clip = float(policy.norm.clip)

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        return self.mu(x)


def export(policy, path):
    m = Deployed(policy).eval()
    x = torch.zeros(1, _env.OBS_DIM)
    torch.onnx.export(m, (x,), path, opset_version=17, input_names=["obs"],
                      output_names=["action"], dynamo=False)
    return m, os.path.getsize(path)


def sample_observations(policy, n, seed=123):
    """Observations from the policy's own rollouts, so the normaliser sees its
    own distribution rather than white noise."""
    e = _env.MicroduckEnv(seed=seed, variant=policy.cfg.variant, reward=policy.cfg.reward,
                          action_scale=policy.cfg.action_scale)
    out, obs = [], e.reset(seed=seed)
    while len(out) < n:
        out.append(obs)
        obs, _, done, _ = e.step(policy(obs))
        if done:
            obs = e.reset()
    return np.stack(out[:n]).astype(np.float32)


def sess(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="runs/ppo_<tag>.pt")
    ap.add_argument("--path", default="")
    ap.add_argument("--n-check", type=int, default=1000)
    ap.add_argument("--n-time", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pol = Policy.load(a.policy)
    tag = pol.cfg.tag
    path = a.path or f"runs/policy_{tag}.onnx"
    m, nbytes = export(pol, path)

    obs = sample_observations(pol, a.n_check)
    with torch.no_grad():
        ref = m(torch.from_numpy(obs)).numpy()
    ref2 = np.stack([pol(o) for o in obs])            # the training-time path, numpy normaliser
    s = sess(path, 1)
    got = np.concatenate([s.run(None, {"obs": obs[i:i + 1]})[0] for i in range(len(obs))])
    diff = float(np.abs(ref - got).max())
    diff_train = float(np.abs(ref2 - got).max())
    ok = diff < 1e-5 and diff_train < 1e-4

    xs = [obs[i % len(obs)][None] for i in range(a.n_time)]
    for x in xs[:50]:
        s.run(None, {"obs": x})
    lat = np.empty(a.n_time)
    for i, x in enumerate(xs):
        t0 = time.perf_counter()
        s.run(None, {"obs": x})
        lat[i] = (time.perf_counter() - t0) * 1e3
    tl = np.empty(a.n_time)
    with torch.no_grad():
        for i, x in enumerate(xs):
            t0 = time.perf_counter()
            m(torch.from_numpy(x))
            tl[i] = (time.perf_counter() - t0) * 1e3

    print(f"\n=== {path}  {nbytes / 1e3:.1f} kB, {sum(p.numel() for p in m.parameters()):,} params ===")
    print(f"  max |torch - onnx| over {a.n_check} rollout observations: {diff:.2e}  "
          f"{'OK' if ok else 'MISMATCH -- export is wrong'}")
    print(f"  max |training path - onnx| (numpy normaliser vs folded):    {diff_train:.2e}")
    print(f"  ONNX Runtime, 1 thread, batch 1:  p50 {np.percentile(lat, 50):.3f} ms   "
          f"p99 {np.percentile(lat, 99):.3f} ms")
    print(f"  torch eager, 1 thread, batch 1:   p50 {np.percentile(tl, 50):.3f} ms   "
          f"p99 {np.percentile(tl, 99):.3f} ms")
    print(f"  control period is {1e3 / 50:.0f} ms; the policy uses "
          f"{100 * np.percentile(lat, 99) / (1e3 / 50):.1f}% of it at p99")
    res = {"policy": a.policy, "onnx": path, "bytes": nbytes, "n_check": a.n_check,
           "max_diff_torch_onnx": diff, "max_diff_train_onnx": diff_train, "verified": ok,
           "onnx_ms_p50": float(np.percentile(lat, 50)), "onnx_ms_p99": float(np.percentile(lat, 99)),
           "torch_ms_p50": float(np.percentile(tl, 50)), "torch_ms_p99": float(np.percentile(tl, 99)),
           "threads": 1, "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    out = a.out or f"runs/onnx_{tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
