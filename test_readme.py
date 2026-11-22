"""README numbers against the committed result files (no simulation, stdlib only).

Checks the step-3 chunk-2 results table, the "per-seed gains" sentence, and the
v1 / v2 PD baselines and joint-velocity term shares quoted in the reward decision,
and the step-1 scaling and sustained-throughput tables.
A README edit or a rerun that changes a JSON without the other fails here.

Run:  python test_readme.py
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load(name):
    with open(os.path.join(HERE, "runs", name)) as f:
        return json.load(f)


# docs/DETAILS.md holds every table; the short README.md repeats a few of its rows
with open(os.path.join(HERE, "docs", "DETAILS.md"), encoding="utf-8") as f:
    README = f.read()
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    SHORT = f.read()


def row(prefix):
    """Cells of the first README table row starting with `| prefix |`, label dropped."""
    for line in README.splitlines():
        if line.startswith(f"| {prefix} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")][1:]
    raise AssertionError(f"README row not found: {prefix}")


def pm(m, s, nd=1):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


ev = load("eval_v2_groundcontact.json")
ev1 = load("eval_v2_chunk1_groundcontact.json")
L, L1, PD = ev["learned"], ev1["learned"], ev["pd"]

print("short README rows are copies of DETAILS rows")
details_rows = set(README.splitlines())
for line in SHORT.splitlines():
    if line.startswith("| ") and not line.startswith("| method |"):
        check(f"short row {line.split('|')[1].strip()}", line in details_rows, line)
for n in ("344.8 ± 2.9", "175.8 ± 0.2", "147.3 against 373.6", "0.79 s"):
    check(f"short README number {n}", n in SHORT and n in README, n)

print("step 3, chunk 2 table")
r = row("PPO v2, 50M steps (`runs/ppo_v2.pt`)")
check("PPO 50M return", r[0] == pm(L["return_mean"], L["seed_std_return"]), r[0])
check("PPO 50M survival", r[1] == f"{L['survival_mean']:.2f}", r[1])
check("PPO 50M upright", r[2] == pm(L["upright_mean"], L["seed_std_upright"], 2), r[2])
check("PPO 50M recovery", r[3] == pm(L["recovery_rate"], L["seed_std_recovery"], 2), r[3])
check("PPO 50M pushes on a fallen robot", r[5] == str(L["pushes_unrecoverable"]), r[5])
r = row("PPO v2, chunk 1 = 25M steps")
check("PPO 25M return", r[0] == pm(L1["return_mean"], L1["seed_std_return"]), r[0])
check("PPO 25M survival", r[1] == f"{L1['survival_mean']:.2f}", r[1])
check("PPO 25M pushes on a fallen robot", r[5] == str(L1["pushes_unrecoverable"]), r[5])
# the PD label also starts a row of the summary table at the top: take the 7-column one
r = [l for l in README.splitlines() if l.startswith("| PD hold-pose, reward v2 |") and l.count("|") == 8][0]
cells = [c.strip() for c in r.strip().strip("|").split("|")][1:]
check("PD return", cells[0] == pm(PD["return_mean"], PD["seed_std_return"]), cells[0])
check("PD survival", cells[1] == f"{PD['survival_mean']:.2f}", cells[1])
check("PD recovery", cells[3] == pm(PD["recovery_rate"], PD["seed_std_recovery"], 2), cells[3])
check("PD pushes on a fallen robot", cells[5] == str(PD["pushes_unrecoverable"]), cells[5])

print("per-seed gains sentence")
gains = [a - b for a, b in zip(L["per_seed_return"], L1["per_seed_return"])]
text = ", ".join(f"{g:.1f}" for g in gains[:-1]) + f" and {gains[-1]:.1f}"
check("per-seed gains", f"per-seed gains {text}" in README, text)
check("mean gain +12.2", f"+{L['return_mean'] - L1['return_mean']:.1f} return on the same seeds" in README)
check("every seed went up", all(g > 0 for g in gains))

print("seeds 1, 2 (trained via mujoco-vecenv-cpp) and the 3-seed mean")
S = [L] + [load(f"eval_v2_s{i}_groundcontact.json")["learned"] for i in (1, 2)]
rets, surv = [x["return_mean"] for x in S], [x["survival_mean"] for x in S]
three = f"3-seed mean {pm(statistics.mean(rets), statistics.stdev(rets))}, survival {pm(statistics.mean(surv), statistics.stdev(surv), 2)}"
want = [f"{pm(S[1]['return_mean'], S[1]['seed_std_return'])} / {pm(S[2]['return_mean'], S[2]['seed_std_return'])}",
        f"same protocol; {three}"]
check("seeds 1, 2 summary row", row("PPO v2 seeds 1, 2 (via mujoco-vecenv-cpp)")[1:] == want, " | ".join(want))
text = " ".join(README.split())
want = (f"They score {pm(S[1]['return_mean'], S[1]['seed_std_return'])} and {pm(S[2]['return_mean'], S[2]['seed_std_return'])}"
        f" with survival {surv[1]:.3f} and {surv[2]:.3f}, so the {surv[0]:.2f} survival above is the low draw of three"
        f" (mean {pm(statistics.mean(rets), statistics.stdev(rets))}, survival {pm(statistics.mean(surv), statistics.stdev(surv), 2)})")
check("seeds 1, 2 sentence", want in text and surv[0] == min(surv), want)

print("reward decision: baselines and term shares")
b1, b2 = load("baseline.json"), load("baseline_v2.json")
check("v1 PD baseline 108.7 ± 2.9", f"**{pm(b1['return_mean'], b1['return_std'])}**" in README)
check("v2 PD baseline 175.2 ± 3.1", f"{pm(b2['return_mean'], b2['return_std'])} of 500" in README)
jv_rand = -b2["term_sums_random"]["joint_vel"] / b2["random_return"] * 100
jv_pd = -b2["term_sums"]["joint_vel"] / b2["return_mean"] * 100
check("v2 joint_vel 5.24% random / 0.17% PD", f"{jv_rand:.2f}% under random actions and {jv_pd:.2f}% under the PD" in README,
      f"{jv_rand:.2f} / {jv_pd:.2f}")
for k in ("posture", "effort", "joint_vel"):
    for pol, sums, ret in (("PD", b1["term_sums"], b1["return_mean"]), ("random", b1["term_sums_random"], b1["random_return"])):
        check(f"v1 {k} under 0.25% ({pol})", abs(sums[k] / ret) < 0.0025, f"{abs(sums[k] / ret) * 100:.3f}%")

print("step 4, DR gap table")
gap = load("gap.json")["cells"]
for line in README.splitlines():
    if not (line.startswith("| nominal | `") or line.startswith("| DR | `")):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    trained, variant = ("nominal" if cells[0] == "nominal" else "dr"), cells[1].strip("`")
    c = gap[f"{trained}/{variant}"]
    want = [pm(c["return_mean"], c["seed_std_return"]), pm(c["upright_mean"], c["seed_std_upright"], 2),
            f"{c['survival_mean']:.2f}", f"{c['height_p50']:.3f} m"]
    ref = gap[f"{trained}/groundcontact"]
    if variant != "groundcontact":
        want += [f"{c['return_mean'] - ref['return_mean']:+.1f}".replace("-", "−"),
                 f"{c['upright_mean'] - ref['upright_mean']:+.2f}".replace("-", "−")]
    check(f"gap {trained}/{variant}", cells[2:2 + len(want)] == want, f"README {cells[2:2 + len(want)]} vs JSON {want}")

print("step 5, ONNX latency row")
ox = load("onnx_v2.json")
l = next(x for x in README.splitlines() if x.startswith("| ONNX Runtime | p50 / p99 latency |"))
want = (f"{ox['onnx_ms_p50']:.3f} / {ox['onnx_ms_p99']:.3f} ms",
        f"torch eager {ox['torch_ms_p50']:.3f} / {ox['torch_ms_p99']:.3f} ms")
check("ONNX p50/p99 and torch eager", all(w in l for w in want), " | ".join(want))
l = next(x for x in README.splitlines() if x.startswith("| ONNX vs torch | max abs difference |"))
want = (f"| {ox['max_diff_torch_onnx']:.1e} |", f"{ox['n_check']:,} rollout observations")
check("ONNX vs torch max diff", all(w in l for w in want) and ox["verified"], " | ".join(want))

print("step 1, scaling and sustained-throughput tables")
bm = load("bench_main.json")
for r in bm["rows"]:
    want = [f"{r['env_steps_per_s']:,.0f}", f"{r['per_proc']:,.0f}", f"{r['speedup']:.2f}x", f"{r['efficiency']:.0%}"]
    check(f"scaling row, {r['procs']} processes", row(str(r["procs"])) == want, " | ".join(want))
tables = README.split("| elapsed | env-steps/s | vs peak |")[1:]
check("two sustained tables", len(tables) == 2)
for name, table in zip(("bench_sustained.json", "bench_gc_sustained.json"), tables):
    su = load(name)
    for line in table.splitlines()[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().split(" ←")[0] for c in line.strip().strip("|").split("|")]
        i = int(cells[0].removesuffix(" s")) // int(su["window_s"]) - 1      # window that ends at this time
        rate = su["rates"][i]
        vs = "—" if i == 0 else f"{rate / su['peak'] - 1:+.0%}".replace("-", "−")
        check(f"{name} row, {cells[0]}", cells[1:] == [f"{rate:,.0f}", vs], f"{rate:,.0f} | {vs}")

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
