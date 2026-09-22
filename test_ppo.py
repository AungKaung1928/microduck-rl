"""PPO machinery, checked against hand-derived values. No assets needed
except for the resume check, which is skipped when assets/ is absent.

Run:  python test_ppo.py
"""
import os
import shutil
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np       # noqa: E402
import torch             # noqa: E402

import ppo               # noqa: E402
import env as _env       # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("\n--- GAE, 3 steps, one env, by hand ---")
# rewards 1, 2, 3; values 0.5, 0.5, 0.5; gamma 0.9; lambda 0.8; no dones;
# last_value 1.0.
#   delta_2 = 3 + 0.9*1.0 - 0.5 = 3.4          A_2 = 3.4
#   delta_1 = 2 + 0.9*0.5 - 0.5 = 1.95         A_1 = 1.95 + 0.72*3.4 = 4.398
#   delta_0 = 1 + 0.9*0.5 - 0.5 = 0.95         A_0 = 0.95 + 0.72*4.398 = 4.11656
r = torch.tensor([[1.0], [2.0], [3.0]])
v = torch.tensor([[0.5], [0.5], [0.5]])
d = torch.zeros(3, 1)
adv, ret = ppo.compute_gae(r, v, d, torch.tensor([1.0]), 0.9, 0.8)
check("advantages match the hand recursion", torch.allclose(adv[:, 0], torch.tensor([4.11656, 4.398, 3.4]), atol=1e-6),
      str(adv[:, 0].numpy().round(5)))
check("returns are advantages plus values", torch.allclose(ret, adv + v))

print("\n--- a done at step 1 cuts the recursion; the truncation value rides in the reward ---")
# Same numbers, but step 1 ends an episode. With the caller having folded
# gamma*V(terminal) = 0.9*0.5 = 0.45 into r_1:
#   delta_2 = 3.4 (unchanged, new episode)      A_2 = 3.4
#   delta_1 = (2 + 0.45) + 0 - 0.5 = 1.95       A_1 = 1.95   (no carry from step 2)
#   delta_0 = 0.95                              A_0 = 0.95 + 0.72*1.95 = 2.354
d = torch.tensor([[0.0], [1.0], [0.0]])
r2 = r.clone()
r2[1] += 0.9 * 0.5
adv, _ = ppo.compute_gae(r2, v, d, torch.tensor([1.0]), 0.9, 0.8)
check("done cuts the carry and the bootstrap arrives through the reward",
      torch.allclose(adv[:, 0], torch.tensor([2.354, 1.95, 3.4]), atol=1e-6), str(adv[:, 0].numpy().round(4)))
adv0, _ = ppo.compute_gae(r, v, d, torch.tensor([1.0]), 0.9, 0.8)
check("without the folded bootstrap the truncated step's advantage is lower by exactly gamma*V",
      abs(float(adv[1, 0] - adv0[1, 0]) - 0.45) < 1e-6,
      "this is the ~100-step corruption the README warns about, in miniature")

print("\n--- running normaliser ---")
rng = np.random.default_rng(0)
X = rng.normal(3.0, 2.0, (1000, _env.OBS_DIM)) * np.linspace(0.1, 5, _env.OBS_DIM)
n = ppo.RunningNorm(_env.OBS_DIM)
for chunk in np.array_split(X, 13):
    n.update(chunk)
check("Welford mean matches numpy", np.allclose(n.mean, X.mean(0), atol=1e-9))
check("Welford variance matches numpy (population)", np.allclose(n.var, X.var(0), atol=1e-7))
z = n(X)
check("normalised output has ~zero mean and ~unit std per dim",
      np.abs(z.mean(0)).max() < 1e-3 and np.abs(z.std(0) - 1).max() < 1e-3)
big = n(X[0] + 1e6)
check("output is clipped at +-clip", np.all(big <= n.clip))
n2 = ppo.RunningNorm(_env.OBS_DIM)
n2.load_state_dict(n.state_dict())
check("state_dict round trip is exact", np.array_equal(n2.mean, n.mean) and n2.count == n.count)

print("\n--- actor-critic shapes and init ---")
ag = ppo.GaussianActorCritic(_env.OBS_DIM, _env.ACT_DIM, 64, -0.5)
x = torch.zeros(5, _env.OBS_DIM)
a, lp, ent, val = ag.act(x)
check("act returns (5,14) actions, (5,) logp, entropy, value",
      a.shape == (5, 14) and lp.shape == (5,) and ent.shape == (5,) and val.shape == (5,))
with torch.no_grad():
    check("mean head starts near zero (std 0.01 init), so the first actions hold STAND",
          float(ag.mu(x).abs().max()) < 1e-6)
check("initial sigma is exp(init_log_std)", abs(float(ag.log_std.exp()[0]) - np.exp(-0.5)) < 1e-6)

print("\n--- checkpoint resume is deterministic and lands on the right step ---")
# What resume can and cannot promise. It restores model, optimiser,
# normaliser, RNG state and step counter. It does NOT restore the workers'
# physics state -- the resumed chunk starts fresh episodes -- so a split run
# is not bit-identical to an uninterrupted one and the test does not pretend
# it is. What it checks: two resumes from the same checkpoint are identical,
# and the split run ends where the uninterrupted one does.
if not os.path.exists("assets/scene.xml"):
    print("  skip  assets/ missing; run ./fetch_assets.sh for this check")
else:
    tmp = tempfile.mkdtemp()
    try:
        base = dict(seed=5, num_envs=2, num_steps=32, num_minibatches=2, update_epochs=2,
                    total_steps=4 * 64, eval_every=0, ckpt_every=2, out=tmp, reward="v2")
        full = ppo.train(ppo.Config(tag="full", **base), verbose=False)
        ppo.train(ppo.Config(tag="split", chunk_steps=2 * 64, **base), verbose=False)
        ck = os.path.join(tmp, "ppo_split.ckpt.pt")
        mid = torch.load(ck, weights_only=False)
        check("the chunk stopped at 2 updates / 128 steps and saved a resumable checkpoint",
              mid["state"]["global_step"] == 128 and "optimizer" in mid and "rng" in mid)
        import shutil as _sh
        _sh.copy(ck, ck + ".bak")
        r1 = ppo.train(ppo.Config(tag="split", **base), resume=ck, verbose=False)
        w1 = torch.load(os.path.join(tmp, "ppo_split.pt"), weights_only=False)
        r2 = ppo.train(ppo.Config(tag="split", **base), resume=ck + ".bak", verbose=False)
        w2 = torch.load(os.path.join(tmp, "ppo_split.pt"), weights_only=False)
        same = all(torch.equal(w1["model"][k], w2["model"][k]) for k in w1["model"])
        check("two resumes from the same checkpoint give bit-identical weights", same)
        check("and identical normaliser state",
              w1["norm"]["count"] == w2["norm"]["count"] and np.array_equal(w1["norm"]["mean"], w2["norm"]["mean"]))
        check("split run ends at the same step count as the uninterrupted run",
              full["global_step"] == r1["global_step"] == r2["global_step"] == 256)
        check("resumed run reports finished", r1["finished"] and r1["updates"] == 4)
        pol = ppo.Policy.load(os.path.join(tmp, "ppo_full.pt"))
        act = pol(np.zeros(_env.OBS_DIM, np.float32))
        check("Policy.load gives a callable obs -> 14 actions", act.shape == (14,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    raise SystemExit(1)
print("all PPO checks passed")
