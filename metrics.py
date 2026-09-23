"""Episode metrics, defined once and used by training and evaluation alike.

Three numbers describe a stand-and-recover episode:

    return        the sum of per-step rewards (ceiling 500 under both reward
                  versions; the two versions are not comparable to each other)
    survival      fraction of steps with the trunk above FALL_HEIGHT. Height
                  only: a robot braced on its side at 5 cm counts as surviving,
                  which is why upright_frac and trunk height are reported too
    upright_frac  fraction of steps with cos > UPRIGHT_COS (about 26 degrees)
    recovery      per push: was the robot upright (cos > UPRIGHT_COS) within
                  RECOVER_WINDOW seconds after the push landed. A push that
                  arrives while the robot is already down cannot be recovered
                  from by definition and is excluded from the denominator;
                  the count of such pushes is reported separately so it
                  cannot hide a policy that spends the episode on the floor.

Time to recover is measured from the push step to the first upright step,
and reported as p50 / p90 over all recovered pushes.
"""
import numpy as np

import common
import env as _env

UPRIGHT_COS = 0.9
RECOVER_WINDOW = 2.0            # s


def run_episode(e, policy, seed=None):
    """Roll one episode. `policy(obs) -> action`. Returns a metrics dict."""
    obs = e.reset(seed=seed)
    rewards, fallen, upright, pushed, heights, terms = [], [], [], [], [], {}
    while True:
        obs, r, done, info = e.step(policy(obs))
        rewards.append(r)
        heights.append(info["trunk_height"])
        for k, v in info["terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        fallen.append(info["fallen"])
        upright.append(info["upright_cos"] > UPRIGHT_COS)
        pushed.append(info["pushed"])
        if done:
            break
    out = summarise(np.array(rewards), np.array(fallen), np.array(upright), np.array(pushed))
    out["height_p50"] = float(np.median(heights))
    out["terms"] = {k: float(v) for k, v in terms.items()}
    return out


def summarise(rewards, fallen, upright, pushed):
    window = int(RECOVER_WINDOW * common.CONTROL_HZ)
    push_steps = np.flatnonzero(pushed)
    recovered, unrecoverable, t_rec = 0, 0, []
    for p in push_steps:
        if not upright[max(p - 1, 0)]:
            unrecoverable += 1
            continue
        seg = upright[p:p + window + 1]
        hit = np.flatnonzero(seg)
        if hit.size:
            recovered += 1
            t_rec.append(hit[0] * _env.CONTROL_DT)
    n_valid = int(push_steps.size - unrecoverable)
    return {
        "return": float(rewards.sum()),
        "survival": float(1.0 - fallen.mean()),
        "upright_frac": float(upright.mean()),
        "pushes": int(push_steps.size),
        "pushes_valid": n_valid,
        "pushes_unrecoverable": int(unrecoverable),
        "recovered": int(recovered),
        "recovery": float(recovered / n_valid) if n_valid else float("nan"),
        "time_to_recover": t_rec,
    }


def aggregate(episodes):
    """Mean +- sd over episodes, recovery pooled over pushes, p50/p90 recover time."""
    ret = np.array([e["return"] for e in episodes])
    surv = np.array([e["survival"] for e in episodes])
    valid = sum(e["pushes_valid"] for e in episodes)
    rec = sum(e["recovered"] for e in episodes)
    times = np.concatenate([e["time_to_recover"] for e in episodes]) if episodes else np.zeros(0)
    return {
        "n_episodes": len(episodes),
        "return_mean": float(ret.mean()), "return_std": float(ret.std(ddof=1)) if ret.size > 1 else 0.0,
        "survival_mean": float(surv.mean()), "survival_std": float(surv.std(ddof=1)) if surv.size > 1 else 0.0,
        "upright_mean": float(np.mean([e["upright_frac"] for e in episodes])),
        "height_p50": float(np.median([e["height_p50"] for e in episodes if "height_p50" in e]))
        if any("height_p50" in e for e in episodes) else None,
        "terms_mean": {k: float(np.mean([e["terms"][k] for e in episodes]))
                       for k in (episodes[0].get("terms") or {})} if episodes else {},
        "pushes_valid": int(valid),
        "pushes_unrecoverable": int(sum(e["pushes_unrecoverable"] for e in episodes)),
        "recovery_rate": float(rec / valid) if valid else float("nan"),
        "recover_p50_s": float(np.percentile(times, 50)) if times.size else None,
        "recover_p90_s": float(np.percentile(times, 90)) if times.size else None,
    }


def pd_policy(e):
    zero = e.zero_action()
    return lambda obs: zero
