#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tiered. Tiers 1-7 are cheap, so a reader can check every structural claim --
# action space, dof arithmetic, contact masks, the observation contract, both
# reward versions, domain randomisation, the PPO maths, the ONNX export -- in
# a few minutes without loading the box. Tier 2 briefly forks three worker
# processes and tier 6 briefly forks two; those are the heaviest of them.
# Tier 8 is the CPU benchmark and tier 9 the training runs; those need the
# machine to itself.
set -u
cd "$(dirname "$0")"

# Python comes from whatever environment is active. A venv is expected but not
# required; requirements.txt lists everything this repo imports.
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import mujoco, numpy' 2>/dev/null; then
  echo "mujoco/numpy are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "    pip install -r requirements.txt" >&2
  exit 1
fi
HAVE_TORCH=1
"$PY" -c 'import torch, onnxruntime' 2>/dev/null || HAVE_TORCH=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hr() { printf '\n=== %s ===\n' "$1"; }

if [ ! -f assets/scene_walk.xml ]; then
  cat <<'MSG'
assets/ is missing. It is gitignored on purpose: the meshes are CC BY-SA-NC.

    ./fetch_assets.sh

pulls them from a pinned upstream commit (~24 MB, one shallow clone).
MSG
  exit 1
fi

hr "1/9  model and repo assumptions -- hand-derived checks"
# Three silent failure modes live here: a reordered action space, a passive
# joint shifting qpos, and a contact bitmask that drops the robot through the
# floor. None of them raise.
"$PY" test_model.py || exit 1

hr "2/9  environment contract -- 48-dim observation, actions, pushes, vector env"
# The quiet failures here are a leaked sim-only observation, an action that
# saturates against a joint stop, and an index that reads a passive backlash
# joint on the evaluation model. None of them raise. Forks 3 workers briefly.
"$PY" test_env.py || exit 1

hr "3/9  the two rewards -- v1 unchanged to the last bit, v2 does what the decision says"
"$PY" test_reward.py || exit 1

hr "4/9  domain randomisation -- measured ranges, exact apply and restore, held-out variants"
"$PY" test_dr.py || exit 1

hr "5/9  what the MJCF contains"
"$PY" inspect_model.py --all || exit 1

hr "6/9  drop tests -- does the physics behave, does the shipped PD hold"
"$PY" drop_test.py --mode limp --z0 0.25 --seconds 5 || exit 1
"$PY" drop_test.py --mode hold --seconds 5 || exit 1
"$PY" drop_test.py --mode hold --variant walk --seconds 5 --no-render || exit 1

hr "7/9  the PD baseline under both rewards, and the three risks step 3 inherits"
# v1: 108.7 +- 2.9 of a 500 ceiling, the step-2 number. v2: the number a
# step-3 policy has to beat; TODO(measure) in the README until this has run
# with 20 seeds on an idle box.
"$PY" baseline.py --seeds 20 || exit 1
"$PY" baseline.py --seeds 20 --reward v2 || exit 1

if [ "$HAVE_TORCH" = 1 ]; then
  hr "8/9  PPO maths, checkpoint resume, ONNX export -- no training"
  # GAE against a hand-computed 3-step case including the truncation
  # bootstrap; Welford normaliser against numpy; two resumes from one
  # checkpoint bit-identical; a folded normaliser inside the ONNX graph.
  "$PY" test_ppo.py || exit 1
  "$PY" test_export.py || exit 1
else
  hr "8/9  skipped -- torch / onnxruntime not importable with $PY"
fi

hr "9/9  the runs that load the machine"
cat <<MSG
Everything above is single-core or a few seconds of forked workers. The rest
holds 8 processes for an hour or more and is chunked so no invocation runs
longer than about two hours. Close other work first; every script below
refuses to start if the 1-minute load average is above 4.

CPU throughput (the step-1 gate, ~4 min):
    nice -n 10 $PY bench.py --seconds 20 --ref-seconds 10 --tag main
    nice -n 10 $PY bench.py --sustained 8 --variant groundcontact --windows 18 --tag gc_sustained

Step 3, 50M env steps in two chunks of ~1 h at ~13,300 env-steps/s:
    OMP_NUM_THREADS=1 nice -n 10 $PY ppo.py --total-steps 50000000 --chunk-steps 25000000 --tag v2 --reward v2
    OMP_NUM_THREADS=1 nice -n 10 $PY ppo.py --resume runs/ppo_v2.ckpt.pt --tag v2
    $PY eval_policy.py runs/ppo_v2.pt --variant groundcontact

Step 4, the same with domain randomisation, then the gap table:
    OMP_NUM_THREADS=1 nice -n 10 $PY ppo.py --total-steps 50000000 --chunk-steps 25000000 --tag v2dr --reward v2 --dr
    OMP_NUM_THREADS=1 nice -n 10 $PY ppo.py --resume runs/ppo_v2dr.ckpt.pt --tag v2dr
    $PY eval_gap.py --nominal runs/ppo_v2.pt --dr runs/ppo_v2dr.pt

Step 5, export and single-thread latency (seconds, one core):
    $PY export_onnx.py runs/ppo_v2.pt

A 90-second smoke of the training loop, no numbers worth keeping:
    OMP_NUM_THREADS=1 nice -n 10 $PY ppo.py --total-steps 4096 --num-envs 2 --num-steps 128 --tag smoke
MSG
if [ -f runs/bench_main.json ]; then
  "$PY" - <<'PY'
import json
b = json.load(open("runs/bench_main.json"))
print(f"\nlast recorded sweep: gate {'PASS' if b['pass'] else 'FAIL'}, "
      f"stable {b['stable']}, worst reference drift {100*b['worst_ref_drift']:+.1f}%")
for r in b["rows"]:
    print(f"  {r['procs']} proc  {r['env_steps_per_s']:>10,.0f} env-steps/s  "
          f"{100*r['efficiency']:3.0f}% efficient")
PY
fi
for f in runs/ppo_v2.json runs/ppo_v2dr.json; do
  [ -f "$f" ] && "$PY" - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"\n{sys.argv[1]}: {r['global_step']:,} steps, {r['updates']} updates, "
      f"rate {r['rate_mean']:,.0f} env-steps/s, finished={r['finished']}")
if r["evals"]:
    e = r["evals"][-1]
    print(f"  last eval: return {e['return_mean']:.1f}, survival {e['survival_mean']:.2f}, "
          f"recovery {e['recovery_rate']:.2f}")
PY
done
exit 0
