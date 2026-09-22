# Issues to open on GitHub

One block per issue: title, then body. These are the known open ends at the
point where steps 3-5 are written but not yet run, so the tracker says what
is and is not done rather than the README having to.

---

**Set the v2 `joint_vel` weight from the measured share**

`REWARD_V2["joint_vel"] = -2e-3` is a placeholder (10x v1). `baseline.py
--reward v2` prints the term's share of the random-action return and the
factor that would put it at ~5%. Set it, rerun the baseline, update the
README table in the same commit. Changing it re-baselines everything trained
so far.

---

**Measure the v2 PD baseline**

The v1 baseline is 108.7 ± 2.9 over 20 seeds. The v2 number is
`TODO(measure)`: `python baseline.py --seeds 20 --reward v2` writes
`runs/baseline_v2.json`. Until then nothing trained under v2 has a stated
number to beat.

---

**Step 3 training run, 50M steps, reward v2**

`OMP_NUM_THREADS=1 nice -n 10 python ppo.py --total-steps 50000000
--chunk-steps 25000000 --tag v2 --reward v2`, two chunks of about 1 h at the
measured ~13,300 env-steps/s. Then `eval_policy.py runs/ppo_v2.pt`. Fill the
step-3 table. Report the action-scale sweep (0.2 / 0.35 / 0.5) the README
promised, or drop the promise.

---

**Step 4: DR run and the gap table**

Same budget with `--dr`, tag `v2dr`. Then `eval_gap.py --nominal
runs/ppo_v2.pt --dr runs/ppo_v2dr.pt` on `groundcontact`,
`groundcontact_backlash` and `rollers`. The README's step-2 plan named
`walk_backlash` as the held-out model; it cannot be, because only its feet
collide with the floor and a fallen robot sinks through the world. The
switch to `groundcontact_backlash` is in the README, and this issue tracks
the measurement.

---

**Sensor noise is applied by the environment, not by MuJoCo**

MuJoCo 3.12 has no `mjENBL_SENSORNOISE` flag. `env.py` now reads
`model.sensor_noise` and adds Gaussian noise to the gyro and orientation
reads itself when `sensor_noise=True`. Verify the magnitudes against a real
IMU trace once the physical robot exists (see the step-6 issue), because
0.005 rad/s is upstream's declared value, not a measurement of a specific
unit.

---

**Walking gait is out of scope**

400M steps is 8.4 h in five chunks at the measured rate. Stand-and-recover is
the deliverable; the gait stays out unless the recover result lands with
budget to spare. This issue exists so nobody reads "steps 3-5 done" as
"walks".

---

**Step 6: the physical robot**

The endpoint of the project is running the exported policy on a real
Microduck. Not ordered, no date. When it opens: ONNX Runtime on the onboard
computer, first rollouts held above the bench, torque and episode length
capped, and a re-measurement of `sensor_noise` and the servo class from the
actual unit.
