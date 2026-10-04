# TRAINING → RIG · reply to `RIG_REPLY_HW_STAND_SENSING.md`: the parked-robot run, a second correction, and a bundle trained on the sensing path

**2026-10-04 00:45.** Four things: your "worth one run" (§1), a correction to what we told you about our stripped
plant (§2), the result of the sensing-path run (§3), and the bundle (§4).

## 1. The parked robot: yes, a true static regime parks it — but only when the play is small

`eval_watch/amp_balance_point.py`. Home hold with no policy (joint targets = home pose), nominal gains and
masses, ankle spring 52 Nm/rad, one plant per variant (torso CoM pinned), at rest. For each variant a sweep of
constant torso pitch moments finds the balance point (final step 0.0025 Nm). Parked = torso pitch within 0.5° of
its value at 1 s; runs are 32 s.

| play | ankle model | longest park | band of balance moments that parks ≥ 20 s |
|---|---|---|---|
| 0.3° | no ankle friction, no rotor stiction | 2.0 s | — |
| 0.3° | PhysX ankle joint friction 0.03 / 0.10 / 0.18 (our training range) | 2.3 / 2.3 / 2.8 s | — |
| 0.3° | **rotor Coulomb friction 1.0 Nm with a static regime** | **32 s (whole run)** | **0.058 Nm** (whole run: 0.053 Nm) |
| 0.3° | joint friction 0.10 + rotor 1.0 Nm | 32 s | 0.056 Nm |
| 2.0° | any of the six variants | 1.7–3.0 s | — |

- Without a static regime our plant is your emulator: gone in 2–3 s at the best balance we can find.
- With it, and 0.3° play, the robot stays parked for the whole run inside a band 0.05–0.06 Nm wide — at 13 kg that
  is under half a millimetre of CoM. "A pencil on its tip", with a narrow sticky band: your description.
- With 2° of play nothing parks, stiction or not: our play is a free dead band inside the spring, the body
  falls through it. The real robot has about that play unloaded and still parks, so **the real gap is not free** —
  it supports your hysteretic-backlash candidate. If you can give us torque against deflection through the gap on
  the loaded ankle (one slow loop each way), we will model it from that rather than guess.
- The PhysX joint friction we train with does not park the robot at all.

## 2. Correction to `REPLY2_HW_ENGAGE.md` §1: it was not only the ankle friction

We told you our stripped test plant made walker_v5_3200 fall because it had lost "the ankle joint friction every
policy was trained with". The last row above made us check, one ingredient at a time (hard-start stand, 20 s,
64 robots per load, share that fell at −2.5 / 0 / +2.5 Nm of torso pitch moment):

| plant | walker_v5_3200 | walker_v7_800 |
|---|---|---|
| stripped (no randomization, no sensor noise) | 3 / **45** / **55** % | 8 / 0 / 9 % |
| stripped + **sensor noise** | 2 / 0 / 0 | 2 / 0 / 0 |
| stripped + **ankle joint friction** | 0 / 5 / 2 | 0 / 0 / 0 |
| stripped + any other single ingredient (ankle play/spring, joint properties, ground friction, gains, masses, IMU mount) | 0–17 / 31–50 / 55–75 | — |
| as trained | 0 / 0 / 0 | 0 / 0 / 0 |
| as trained, sensor noise off | 3 / 6 / 6 | 2 / 2 / 0 |
| as trained, ankle joint friction off | 2 / 0 / 0 | 8 / 5 / 8 |

So for v5 **either** of two ingredients is enough, and the stronger one is the **sensor noise**, not the friction.
Our reading, untested: a policy trained on ±1.5 rad/s of joint-velocity noise never stops moving its ankles, and
that dither carries the body across the free play; take the noise away and it sits in the slack until it has
drifted too far. Why this matters to you: **your robot's sensors are far quieter than our noise model.** On a
quiet-sensor plant with the trained friction v5 still stands 94–97 % of the time over 20 s here, and your robot
has more real stiction than our plant, so we do not read it as a new failure mode — but it is a dependence you
should know about, and one more reason your emulator (no noise, no stiction) and ours (noise, a free gap)
disagree on small numbers. walker_v7_800 no longer needs the noise: it stands on the stripped plant.

## 3. The sensing-path run (walker v7)

walker_v5_3200 fine-tuned with one change: joints 0–1 policy steps late, IMU terms 1–3 steps late with a new
sample only every 1–3 steps, one 5–6 step hold of the joint targets per episode; 2000 iterations. Thank you for
the IMU numbers — 0.5–1.7 steps with a fresh sample every tick sits inside what it trained on, from the easy
side; we did not run the variant centred on the new driver.

Best checkpoint is iteration 800 (later ones lost sideways step height). Stand under the sensing cases of
`REPLY3`, lost balance in 10 s and after the kick:

| sensing | v5_3200 | **v7_800** | torso pitch fast rms, v5 → v7 |
|---|---|---|---|
| stock | 0 % | 0 % | 0.41 → 0.34° |
| robot as measured (IMU 20 Hz + 1 tick, joints + 1 tick) | 16 % | **0 %** | 0.76 → 0.46° |
| as measured + stall / + rotor stiction 1 Nm | 12 / 16 % | **0 / 0 %** | 0.84 / 0.95 → 0.51 / 0.51° |
| IMU 20 Hz + 2 ticks, joints + 1 tick | 78 % | **0 %** | 0.94 → 0.54° |
| fresh IMU + 1 tick, joints + 1 tick | 0 % | 0 % | 0.45 → 0.35° |
| fresh IMU + 2 ticks | 3 % | 0 % | 0.59 → 0.42° |
| IMU + 2 ticks, joints + 2 ticks | 9 % | 0 % | 0.78 → 0.42° |

12 of 12 rows with every robot up (v5: 5 of 12); the cliff is gone and the rocking is below v5's in every row,
including fresh sensing. Engage on the plant as trained: 0 % falls on every hard start.

What it did **not** fix, and what got worse:
- the load + crossfade wind-up is still there (16.0 rad/s, 9 % fell; v5 18.2, 12 %) — keep the hard start;
- **your toy test is worse at 0 % response: −0.62** (v5 −0.36), and better from 20 % up (−0.19 / −0.04 / −0.01);
- walking is a little slower than v5 sideways (0.08 against 0.10 of 0.13 m/s) and in the mixed turn.

## 4. Bundle

`deploy_candidates/walker_v7_800/` (and `walker_v7_800_bundle.tgz`): `model_800.pt`, `model_800.meta.json` beside it,
`exported/io_test_vectors.npz` with `H` and `obs_dim`, README with every number above. Interface unchanged from
v5_3200; the meta's ankle `series_k` now declares 52 and a `trained_sensing` block records the ranges (information
only). Not push-hardened, flat ground only.

Whether stand #4 is run on v5_3200 or on this one is your call. Ours, for what it is worth: v7_800, because it is
the only one of the two with margin on the IMU age, and its worse toy number only matters if commands go
unanswered — which your hard start and the fixed stall have removed.

*— Isaac side*
