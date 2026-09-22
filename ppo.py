"""Step 3: PPO against the vector env, chunked so no run needs the box for
more than two hours at a time.

The loop is the one from ppo-from-scratch -- GAE, orthogonal init, clipped
surrogate, multi-epoch minibatches, learning-rate anneal -- with the four
things this environment demands on top:

  vectorised rollouts    N forked envs, N <= 8, one observation batch per step
  running normaliser     Welford mean/var over every observation seen. Step 2
                         measured a 13x rms spread across observation groups
                         with the 28 body-configuration dims carrying the
                         least variance, so a fixed OBS_SCALE was never going
                         to be right. The normaliser is part of the policy: it
                         is saved in the checkpoint and folded into the ONNX
                         graph at export.
  truncation bootstrap   every episode here ends on the step limit and none on
                         a terminal state. The bootstrap on `done` is
                         V(terminal_obs), which the vector env carries across
                         in `info`. Zeroing it corrupts the value target ~100
                         steps back at gamma 0.99 -- see the README.
  chunking               --chunk-steps env steps per invocation, a checkpoint
                         every --ckpt-every updates, --resume picks up the
                         model, optimiser, normaliser, RNG state and step
                         counter. A killed run loses at most one checkpoint
                         interval.

Throughput is logged every update. If the rolling rate drops more than 20%
below the mean of the first five updates the log says so: on this machine
that is the only visible sign of thermal or power throttling, and it is also
what contention looks like, so the warning says to check both.

Recommended invocation (the caller owns nice; the box check is inside):

    OMP_NUM_THREADS=1 nice -n 10 python ppo.py --total-steps 50000000 \\
        --chunk-steps 25000000 --tag v2 --reward v2

Evaluation during training uses metrics.py, the same definitions
eval_policy.py reports, so the curve and the final table agree by
construction.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse            # noqa: E402
import json                # noqa: E402
import math                # noqa: E402
import time                # noqa: E402
from dataclasses import asdict, dataclass, field   # noqa: E402

import numpy as np         # noqa: E402
import torch               # noqa: E402
import torch.nn as nn      # noqa: E402

import vec_env             # noqa: E402  (sets OMP before mujoco loads)
import env as _env         # noqa: E402
import metrics             # noqa: E402
from boxcheck import require_quiet_box   # noqa: E402

torch.set_num_threads(1)


@dataclass
class Config:
    seed: int = 0
    total_steps: int = 50_000_000
    chunk_steps: int = 0            # 0 = run to total_steps in this invocation
    num_envs: int = 8
    num_steps: int = 256            # per env per update
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 5
    num_minibatches: int = 8
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -0.5
    reward: str = "v2"
    variant: str = "groundcontact"
    action_scale: float = _env.ACTION_SCALE
    dr: bool = False
    eval_every: int = 20            # updates
    eval_episodes: int = 20
    ckpt_every: int = 10            # updates
    tag: str = "run"
    out: str = "runs"

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


# ----------------------------------------------------------------- normaliser

class RunningNorm:
    """Welford running mean/variance. Numpy, because it also has to live in
    the ONNX graph and be checked against numpy in a test."""

    def __init__(self, dim, clip=10.0, eps=1e-8):
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.count = 0
        self.clip, self.eps = float(clip), float(eps)

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.size)
        n = x.shape[0]
        if n == 0:
            return
        bmean, bvar = x.mean(0), x.var(0)
        tot = self.count + n
        delta = bmean - self.mean
        self.mean = self.mean + delta * n / tot
        self.m2 = self.m2 + bvar * n + delta ** 2 * self.count * n / tot
        self.count = tot

    @property
    def var(self):
        return self.m2 / max(self.count, 1)

    @property
    def std(self):
        return np.sqrt(self.var + self.eps)

    def __call__(self, x):
        return np.clip((np.asarray(x, np.float32) - self.mean.astype(np.float32))
                       / self.std.astype(np.float32), -self.clip, self.clip)

    def state_dict(self):
        return {"mean": self.mean.copy(), "m2": self.m2.copy(), "count": self.count,
                "clip": self.clip, "eps": self.eps}

    def load_state_dict(self, d):
        self.mean, self.m2, self.count = d["mean"].copy(), d["m2"].copy(), int(d["count"])
        self.clip, self.eps = float(d["clip"]), float(d["eps"])


# ---------------------------------------------------------------------- model

def layer_init(layer, std=math.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(inp, hidden, out, out_std):
    return nn.Sequential(
        layer_init(nn.Linear(inp, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, out), std=out_std))


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, init_log_std):
        super().__init__()
        self.critic = mlp(obs_dim, hidden, 1, 1.0)
        self.mu = mlp(obs_dim, hidden, act_dim, 0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        mean = self.mu(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(x)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Backward recursion. `dones[t]` marks that step t ended an episode.

    Truncation bootstraps are folded into `rewards` by the caller (the value
    of terminal_obs, discounted, added to the last reward), so here a done
    simply cuts the recursion. `last_value` is V of the observation after the
    final step, used only if the final step was not a done."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(N)
    next_value = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values


# ----------------------------------------------------------------- policy I/O

class Policy:
    """A trained actor plus its normaliser: obs -> mean action. What
    eval_policy.py and export_onnx.py load."""

    def __init__(self, agent, norm, cfg):
        self.agent, self.norm, self.cfg = agent, norm, cfg
        self.agent.eval()

    def __call__(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(self.norm(obs), dtype=torch.float32)
            return self.agent.mu(x).numpy()

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        cfg = Config(**ck["config"])
        agent = GaussianActorCritic(_env.OBS_DIM, _env.ACT_DIM, cfg.hidden, cfg.init_log_std)
        agent.load_state_dict(ck["model"])
        norm = RunningNorm(_env.OBS_DIM)
        norm.load_state_dict(ck["norm"])
        return cls(agent, norm, cfg)


def save_checkpoint(path, agent, opt, norm, cfg, state):
    torch.save({
        "model": agent.state_dict(), "optimizer": opt.state_dict(),
        "norm": norm.state_dict(), "config": asdict(cfg), "state": state,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
    }, path)


def make_env_kwargs(cfg):
    kw = dict(variant=cfg.variant, reward=cfg.reward, action_scale=cfg.action_scale)
    if cfg.dr:
        import dr
        kw["dr"] = dr.DomainRandomizer(dr.DRConfig())
    return kw


def evaluate(policy, cfg, episodes, seed=777_000):
    e = _env.MicroduckEnv(variant=cfg.variant, seed=seed, reward=cfg.reward,
                          action_scale=cfg.action_scale)
    eps = [metrics.run_episode(e, policy, seed=seed + i) for i in range(episodes)]
    return metrics.aggregate(eps)


# ---------------------------------------------------------------------- train

def train(cfg, resume=None, verbose=True):
    os.makedirs(cfg.out, exist_ok=True)
    stem = os.path.join(cfg.out, f"ppo_{cfg.tag}")
    agent = GaussianActorCritic(_env.OBS_DIM, _env.ACT_DIM, cfg.hidden, cfg.init_log_std)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    norm = RunningNorm(_env.OBS_DIM)
    state = {"global_step": 0, "update": 0, "trace": [], "evals": []}

    if resume:
        ck = torch.load(resume, weights_only=False)
        agent.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        norm.load_state_dict(ck["norm"])
        state = ck["state"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        if verbose:
            print(f"  resumed from {resume} at step {state['global_step']:,}")
    else:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    n_updates_total = cfg.total_steps // cfg.batch_size
    chunk_target = cfg.total_steps if cfg.chunk_steps <= 0 else min(
        cfg.total_steps, state["global_step"] + cfg.chunk_steps)
    O, A = _env.OBS_DIM, _env.ACT_DIM
    envs = vec_env.VecEnv(n=cfg.num_envs, seed=cfg.seed * 1000 + state["update"] * 17,
                          **make_env_kwargs(cfg))
    obs_np = envs.reset()
    norm.update(obs_np)

    b_obs = torch.zeros(cfg.num_steps, cfg.num_envs, O)
    b_act = torch.zeros(cfg.num_steps, cfg.num_envs, A)
    b_logp = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_rew = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_done = torch.zeros(cfg.num_steps, cfg.num_envs)
    b_val = torch.zeros(cfg.num_steps, cfg.num_envs)

    rates, ep_returns, t_chunk = [], [], time.perf_counter()
    ep_ret = np.zeros(cfg.num_envs)
    try:
        while state["global_step"] < chunk_target:
            state["update"] += 1
            upd = state["update"]
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * max(0.0, 1.0 - (upd - 1.0) / max(n_updates_total, 1))
            t0 = time.perf_counter()

            for t in range(cfg.num_steps):
                x = torch.as_tensor(norm(obs_np), dtype=torch.float32)
                with torch.no_grad():
                    action, logp, _, value = agent.act(x)
                b_obs[t], b_act[t], b_logp[t], b_val[t] = x, action, logp, value
                obs_np, rew, done, infos = envs.step(action.numpy())
                ep_ret += rew
                rew = torch.as_tensor(rew, dtype=torch.float32)
                for i, info in enumerate(infos):
                    if done[i]:
                        # Every done here is a truncation: bootstrap V(terminal_obs).
                        with torch.no_grad():
                            tv = agent.value(torch.as_tensor(norm(info["terminal_obs"]),
                                                             dtype=torch.float32))
                        if info.get("truncated", True) and not info.get("terminated", False):
                            rew[i] += cfg.gamma * tv
                        ep_returns.append(float(ep_ret[i]))
                        ep_ret[i] = 0.0
                b_rew[t] = rew
                b_done[t] = torch.as_tensor(done, dtype=torch.float32)
                norm.update(obs_np)
            state["global_step"] += cfg.batch_size

            with torch.no_grad():
                last_value = agent.value(torch.as_tensor(norm(obs_np), dtype=torch.float32))
            adv, ret = compute_gae(b_rew, b_val, b_done, last_value, cfg.gamma, cfg.gae_lambda)

            f_obs, f_act = b_obs.reshape(-1, O), b_act.reshape(-1, A)
            f_logp, f_adv, f_ret = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            idx = np.arange(cfg.batch_size)
            clipfracs, approx_kls = [], []
            for _ in range(cfg.update_epochs):
                np.random.shuffle(idx)
                for s in range(0, cfg.batch_size, cfg.minibatch_size):
                    mb = idx[s:s + cfg.minibatch_size]
                    _, newlogp, entropy, newval = agent.act(f_obs[mb], f_act[mb])
                    logratio = newlogp - f_logp[mb]
                    ratio = logratio.exp()
                    mb_adv = f_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg = -torch.min(ratio * mb_adv,
                                    torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv).mean()
                    v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                    loss = pg - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
                        approx_kls.append(((ratio - 1) - logratio).mean().item())

            dt = time.perf_counter() - t0
            rate = cfg.batch_size / dt
            rates.append(rate)
            recent = float(np.mean(ep_returns[-50:])) if ep_returns else float("nan")
            row = {"update": upd, "step": state["global_step"], "rate": rate,
                   "return50": recent, "clipfrac": float(np.mean(clipfracs)),
                   "approx_kl": float(np.mean(approx_kls)),
                   "sigma": float(agent.log_std.detach().exp().mean()),
                   "lr": opt.param_groups[0]["lr"]}
            state["trace"].append(row)
            warn = ""
            if len(rates) > 5:
                base = float(np.mean(rates[:5]))
                roll = float(np.mean(rates[-3:]))
                if roll < 0.8 * base:
                    warn = (f"  THROUGHPUT -{100 * (1 - roll / base):.0f}% vs first 5 updates: "
                            f"thermal/power limit or another job. Check load and the host.")
            if verbose:
                print(f"  upd {upd:>5}  step {state['global_step']:>11,}  {rate:7,.0f} env-steps/s"
                      f"  ret50 {recent:7.1f}  clip {row['clipfrac']:.3f}  kl {row['approx_kl']:.4f}"
                      f"  sigma {row['sigma']:.3f}{warn}", flush=True)

            if cfg.eval_every and upd % cfg.eval_every == 0:
                ev = evaluate(Policy(agent, norm, cfg), cfg, cfg.eval_episodes)
                agent.train()
                ev.update({"update": upd, "step": state["global_step"]})
                state["evals"].append(ev)
                if verbose:
                    print(f"    eval {cfg.eval_episodes} eps: return {ev['return_mean']:.1f} +- "
                          f"{ev['return_std']:.1f}  survival {ev['survival_mean']:.2f}  "
                          f"recovery {ev['recovery_rate']:.2f}  "
                          f"({ev['pushes_unrecoverable']} pushes landed on a fallen robot)", flush=True)
            if cfg.ckpt_every and upd % cfg.ckpt_every == 0:
                save_checkpoint(stem + ".ckpt.pt", agent, opt, norm, cfg, state)
    finally:
        envs.close()

    save_checkpoint(stem + ".ckpt.pt", agent, opt, norm, cfg, state)
    torch.save({"model": agent.state_dict(), "norm": norm.state_dict(), "config": asdict(cfg),
                "global_step": state["global_step"]}, stem + ".pt")
    summary = {
        "config": asdict(cfg), "global_step": state["global_step"], "updates": state["update"],
        "chunk_wall_s": time.perf_counter() - t_chunk,
        "rate_mean": float(np.mean(rates)) if rates else None,
        "rate_first5": float(np.mean(rates[:5])) if rates else None,
        "rate_last5": float(np.mean(rates[-5:])) if rates else None,
        "final_return50": state["trace"][-1]["return50"] if state["trace"] else None,
        "evals": state["evals"], "trace": state["trace"],
        "finished": state["global_step"] >= cfg.total_steps,
    }
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    if verbose:
        print(f"\n  wrote {stem}.pt (policy), {stem}.ckpt.pt (resume), {stem}.json")
        if not summary["finished"]:
            print(f"  chunk done at {state['global_step']:,} of {cfg.total_steps:,}. Continue with:\n"
                  f"    OMP_NUM_THREADS=1 nice -n 10 python ppo.py --resume {stem}.ckpt.pt --tag {cfg.tag}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    d = Config()
    for k, v in asdict(d).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=None)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force", action="store_true", help="start on a busy box. Do not.")
    a = p.parse_args()
    if a.resume:
        cfg = Config(**torch.load(a.resume, weights_only=False)["config"])
        for k in ("total_steps", "chunk_steps", "tag", "out", "eval_every", "ckpt_every"):
            if getattr(a, k) is not None:
                setattr(cfg, k, getattr(a, k))
    else:
        over = {k: v for k, v in vars(a).items() if k in asdict(d) and v is not None}
        cfg = Config(**over)
        if a.no_anneal_lr:
            cfg.anneal_lr = False
    require_quiet_box(a.force, quiet=a.quiet)
    print(f"=== ppo  tag={cfg.tag}  reward={cfg.reward}  dr={cfg.dr}  envs={cfg.num_envs}  "
          f"batch={cfg.batch_size}  total={cfg.total_steps:,} ===")
    print("  run under: OMP_NUM_THREADS=1 nice -n 10 python ppo.py ...")
    r = train(cfg, resume=a.resume or None, verbose=not a.quiet)
    print(f"  steps {r['global_step']:,}  rate {r['rate_mean']:,.0f} env-steps/s  "
          f"return50 {r['final_return50']}")
