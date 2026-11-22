# microduck-rl

PPO balance-and-push-recovery policy for the [Microduck](https://github.com/pollen-robotics/microduck_rl) biped (25 cm, 14 servos), trained in CPU MuJoCo on a laptop with no GPU: return 344.8 ± 2.9 against the PD baseline's 175.8 ± 0.2 of 500.

<img src="out/drop_groundcontact_hold.png" width="720" alt="Filmstrip: the shipped PD controller holds the stand pose for 0.79 s, then topples face-down. This is the baseline the policy has to beat.">

The baseline: the shipped PD controller holding the stand pose topples after 0.79 s.

| method | metric | value | condition |
|---|---|---|---|
| PD hold-pose, reward v2 | 175.8 ± 0.2 | 0.47 | — | 0.88 ± 0.20 | 0.00 / 0.00 | 1443 |
| PPO v2, 50M steps (`--target-kl 0.02` in chunk 2) | return | 344.8 ± 2.9 | same seeds; survival 0.73, upright 0.50 ± 0.01, median trunk 5.9 cm |
| PPO v2 seeds 1, 2 (via mujoco-vecenv-cpp) | return | 357.3 ± 7.6 / 342.2 ± 5.1 | same protocol; 3-seed mean 348.1 ± 8.1, survival 0.87 ± 0.14 |
| DR 25M → `rollers` | return / upright gap | −164.9 / −0.15 | held-out physics, 147.3 ± 2.3 |
| ONNX Runtime | p50 / p99 latency | 0.010 / 0.033 ms | 1 intra-op thread, batch 1; torch eager 0.026 / 0.128 ms |

Domain randomisation made the policy worse (147.3 against 373.6 on `rollers`); getting up after a fall is not demonstrated. Details, every table and what failed: [docs/DETAILS.md](docs/DETAILS.md). Walkthrough: https://aungkaung1928.github.io/projects/microduck-rl.html

## Quick start

Linux, Python 3.10+ (Ubuntu: the `python3-venv` package), CPU only. `test_env.py` takes ~4 s.

```bash
git clone <this repo URL>
cd microduck-rl
python3 -m venv .venv && . .venv/bin/activate                          # first time only
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu  # first time only
pip install -r requirements.txt                                        # first time only
./fetch_assets.sh          # pinned upstream model, ~24 MB, first time only
python3 test_env.py        # 27 environment contract checks
python3 test_readme.py     # every README number vs runs/*.json, prints 'all passed'
```

The trained policies (`runs/ppo_v2.pt`, `runs/policy_v2.onnx`) are tracked, so evaluation and export run without training. Training, benchmark and Docker commands: [docs/DETAILS.md#reproducing](docs/DETAILS.md#reproducing).

## How it works

1. `env.py`: 48-dim observation (sensors the robot has), 14-dim position action, 50 Hz, 250-step episodes with random pushes.
2. `ppo.py`: PPO from scratch, 8 CPU worker processes; 50M steps took 4.4 h.
3. `dr.py` + `eval_gap.py`: domain randomisation, scored on physics variants never trained on.
4. `export_onnx.py` + `cpp/`: ONNX with the normaliser inside the graph, run from Python and C++.

## Licence

Upstream code is Apache-2.0; the 3D model files are **Creative Commons
BY-SA-NC**. `assets/` is therefore gitignored and reproduced by
`fetch_assets.sh` from pinned commit `53b8971`. Code in this repository is my
own.
