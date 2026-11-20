"""Write rollout observations and Python ONNX Runtime outputs as raw float32 for bench_ort,
and time Python ORT on them with the same protocol (1 thread, batch 1, 50 warm-up, 2000 timed).

    python cpp/dump_obs.py runs/ppo_v2.pt runs/policy_v2.onnx  ->  cpp/obs.bin, cpp/ref.bin
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from export_onnx import Policy, sample_observations, sess  # noqa: E402

if __name__ == "__main__":
    pt, onnx = sys.argv[1], sys.argv[2]
    obs = sample_observations(Policy.load(pt), 1000)
    s = sess(onnx, 1)
    ref = np.concatenate([s.run(None, {"obs": obs[i:i + 1]})[0] for i in range(len(obs))])
    here = os.path.dirname(os.path.abspath(__file__))
    obs.astype(np.float32).tofile(os.path.join(here, "obs.bin"))
    ref.astype(np.float32).tofile(os.path.join(here, "ref.bin"))
    print(f"wrote {obs.shape} obs, {ref.shape} ref")
    xs = [obs[i % len(obs)][None] for i in range(2000)]
    for x in xs[:50]:
        s.run(None, {"obs": x})
    lat = np.empty(len(xs))
    for i, x in enumerate(xs):
        t0 = time.perf_counter()
        s.run(None, {"obs": x})
        lat[i] = (time.perf_counter() - t0) * 1e3
    print(f"Python ONNX Runtime {ort.__version__}, 1 thread, batch 1: n {len(xs)} "
          f"p50 {np.percentile(lat, 50):.4f} ms p99 {np.percentile(lat, 99):.4f} ms")
