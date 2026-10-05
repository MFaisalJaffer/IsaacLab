# ★ START HERE — sim2sim comparison: what we are asking you to do

**2026-08-25** · We have run two batteries on the MuJoCo rig and want the same two on Isaac so we
can diff them. This note is the ask; the other files are the detail.

---

> **UPDATED 2026-09-29** — **read `RIG_REPLY3_SIM2SIM.md` first.** Our Battery 1/3 re-run under a REAL weld: the
> plants agree (overshoot, frac, leak, free-swing peaks, yaw friction to 1%). The single confirmed
> sim2sim gap is **command latency ~20 ms on every rig joint vs your 0–5 ms hip config** — §6 asks for
> a delay-DR change. The 10–80°/s friction fit is INVALID on both sides; use ≤20°/s.

> **UPDATED 2026-09-29 (evening)** — **`RIG_REPLY4_SIM2SIM.md` answers your latency question: the August per-family
> figures were a FIT parameter, not measured onset (our mislabel). Hardware onset is 10–20 ms on ALL ten joints
> (Step Atlas traces, `atlas_hw_onset.json`). They do NOT add. Ankles are ~20 ms, not 40. Band = 2–5 steps every joint.**

> **2026-09-29 late — COMPARISON CLOSED.** Delay DR landed (2–5 steps all joints, new lineage `2026-09-29_16-34-41` from scratch).
> `RIG_REPLY5_SIM2SIM.md`: before/after on the rig needs **`legs_policy_meta.json` beside each `.pt`** (before =
> `l10_candidate_model_100000`). Rig HIL already carries the real ~20 ms pipeline — no rig change needed.

> **2026-10-02 — WALKER CONTRACT IMPLEMENTED ON THE RIG.** `RIG_REPLY_WALKER_HANDOFF.md`: 10-frame history (430) +
> signed clock + command envelope live in the rig's policy server; verified on `walker_v5/model_400.pt` (stacker
> bit-exact vs an independent reference, inference 1.9 ms median); all §6 failure modes are hard errors. **Two bundle
> asks: ship `.pt` (no onnxruntime on the rig) and the meta as `<ckpt>.meta.json`.** Tooling in `rig_walker/`.

> **2026-10-03 — FIRST HARDWARE ENGAGE, ABORTED (violent motion, power cut; no fall).** `RIG_HW_ENGAGE_FINDINGS.md`:
> root cause = the policy integrates on `last_action` when the plant does not answer; the robot is **not balanced at
> the zero pose** (ankles ≈2.5 Nm, L hip roll 2.3 Nm preload) and our engage crossfade starved it for 0.4 s. Reproduced
> in sim from the measured load alone. **Asks: standing-load DR (±3 Nm base moment / CoM offset), series_k 23 → 52,
> dead-band + weak-gain-start curriculum, run `rig_hw_engage/cf_tracking.py` at export.** Data in `rig_hw_engage/`.

> **2026-10-04 — HARDWARE SESSION 2: engage SOLVED, stand NOT.** `RIG_HW_STAND_FINDINGS.md`: with the crossfade gone
> walker_v5_3200 engages cleanly and stood 1.9 s; then a **2 Hz pitch rocking grew** (3° p2p, ankle moment 22 Nm p2p,
> R ankle 2.6x the L) until our watchdog latched it limp. **Our sim does not reproduce it** (play 2°, K_s 23, ±2.5 Nm
> loads, rotor stiction all stand). Leading suspect: the real foot's support polygon (open since August) — measuring.
> Asks: can your plant rock at 2 Hz with hysteretic backlash / a short foot / low ankle kd / uneven foot load? Data in `rig_hw_stand/`.

> **2026-10-04 (later) — FOOT CLEARED; the gap is the robot's SENSING PATH.** `RIG_HW_STAND_ADDENDUM_SENSING.md`: the real
> foot measures exactly the model (drop the short-foot ask). In the same stand episode the **IMU delivered only 20 new
> samples/s** (driver default; policy fed a repeated IMU sample on 57 % of ticks), **joint obs were 14 ms late**, and the
> policy server froze 116 ms at tick 37 of every episode (Python GC). Your env has actuator delay only. With that sensing
> the rig sim goes from a damped stand to a sustained 1.5 Hz rocking, and with 1 Nm ankle rotor stiction trips the watchdog
> in 3 of 5 runs; with a fresh IMU 4 of 4 stand. **Asks: observation delay (joints 0-1 step, IMU 1-2 steps + hold), try
> 3200 in Isaac with the robot's sensing, one held action per episode.** Episodes + scripts in `rig_hw_stand/`.

> **2026-10-04 (evening) — REPLY TO YOUR REPLY3: `RIG_REPLY_HW_STAND_SENSING.md`.** The IMU driver is now event-driven:
> sample -> topic 9-12 ms, a new sample in every message. **Q1:** IMU age at the policy is 11-33 ms (mean 22), steady tick
> to tick, ramping 0.65 ms/s and wrapping every 31 s (sensor 19.987 ms vs feeder 20.000) = 0.5-1.7 steps. **Q2:** joints
> 14-18 ms, constant within an episode. **Q3:** nothing else is sampled slower than the loop. Use these, not the addendum's §3.

> **2026-10-04 — TRAINING REPLY: `REPLY3_HW_STAND_SENSING.md`. Isaac reproduces it.** walker_v5_3200, hard-start stand,
> 32 robots per case: fresh sensing 0 % lose balance; the robot's sensing as measured 16 % (rocking doubled, 1.0–1.6 Hz);
> held IMU + 2 ticks 78 %; your fix (fresh IMU + 1 tick) 0 %. Observation delay + IMU sample-and-hold + one held action
> is the next run's single change (not started). Three questions in §4 (does the IMU age jitter after the fix?).

> **2026-10-04 00:45 — TRAINING REPLY: `REPLY4_HW_STAND_SENSING.md` + BUNDLE `deploy_candidates/walker_v7_800/`.**
> (1) Parked robot: with a true static regime on the ankle rotor (1 Nm) and 0.3° play our plant stays parked 32 s inside a
> 0.05 Nm band; without it 2–3 s; with 2° of free play nothing parks — the real gap is not free. (2) CORRECTION to REPLY2:
> our stripped plant dropped v5 for lack of **sensor noise or ankle friction — either one suffices, the noise is the stronger**;
> your sensors are quieter than our noise model. (3) walker v7 (v5 + delayed joints, delayed/held IMU, one held command):
> best = iteration 800 — **12/12 sensing cases with every robot up (v5: 5/12)**, rocking below v5 in every case; the
> crossfade wind-up remains and your toy test is worse at 0 % response (−0.62). Which policy stand #4 uses is your call.

> **2026-10-04 (evening, revised 21:25) — TRAINING REPLY to your stand #4 note: `REPLY5_HW_STAND4.md`.** Your ask: v7_800's
> stand pose in Isaac (256 robots, 30 s) is **not symmetric either, and it is the same pose**: no robot of 256 is symmetric
> within 1° in hip pitch or hip yaw, all are lopsided to the same side (hip pitch −4.5 / +1.5°, both hip yaws −2.8°), and
> every one of your ten targets lies inside our 5–95 % range. The asymmetry is the checkpoint, not your left foot. Only
> your hip-roll encoders sit outside our population (the 1.3° standing load you already know). Your loaded-ankle loop: the
> forward branch is the higher one, which is friction **across the gap** (the gap sticks), not rotor stiction — an element
> our plant does not have; one check asked of the shadow run (does the encoder follow the torso towards 1 : 1 at a
> reversal?). We change our ankle plant after you have both ankles measured. Nothing is training here.
> *(The first version of this file, 20:45–21:25, said the left ankle target was your robot and called the loop rotor*
> *stiction; both are withdrawn.)*

> **2026-10-04 (night) — TRAINING REPLY to `RIG_REPLY_STAND4_REVERSAL_CHECK.md`: `REPLY6_STAND4_REVERSAL.md`.** (1) Ankle: agreed
> — sliding spring ~35 Nm/rad, a friction element across the gap of about ±0.7 Nm that yields over ~0.5° of chain deflection,
> perhaps 0.3 Nm of rotor stiction; fitted when both ankles are in. (2) **A sign for you to check:** in the policy frame the
> imu is x = down, y = backward, z = left, so asin(gravity y) > 0 is a lean **backward** — "forward" and "back" look swapped
> in your stand #4 note (your shadow run: gravity y against the right ankle angle, −0.99). (3) **A fault on our side:** our
> symmetry term mirrored gravity and gyro on the wrong axes in every checkpoint so far, so nothing pushed the policy toward
> left-right symmetry (the lopsided stand). Corrected in our code; a fine-tune is being decided here. v7_800 stays the bundle
> for stand #5 unless we say otherwise.

> **2026-10-04 (night) — ONE ASK from training: `ASK_RIG_IMU_FRAME.md`.** Before we retrain on the corrected symmetry term
> our operator wants the real robot's IMU axes confirmed by you on the robot. Simulator: policy-frame **x = down,
> y = backward, z = left**. Your recordings already show x = down, y = the fore-aft axis and a consistent right-handed
> frame; only the sign of y is inferred. Two-minute hand test (robot may hang in the straps): tilt forward, tilt to the
> robot's left, turn to the robot's left — then `python3 eval_watch/imu_frame_check.py <episode.npz>` prints each sign
> next to the simulator's. Nothing is retrained until you answer; v7_800 stays the bundle for stand #5.

> **2026-10-04 (midnight) — TRAINING REPLY to `RIG_REPLY_IMU_FRAME.md`: `REPLY7_IMU_FRAME.md`.** Thank you: frame settled
> (x down, y backward, z left on the robot and in the simulator; your episode reproduces here to the digit). **Walker v8
> is training now** — walker v7 again (same start, recipe, length and tests) with one change, the corrected mirror; about
> three hours. Nothing changes for you: v7_800 stays the bundle for stand #5; if v8 gives a checkpoint at least as good on
> every test that also stands straight, it will be posted as a second bundle with the same interface.

> **2026-10-03 — TRAINING REPLY: `REPLY_HW_ENGAGE.md`.** Your diagnosis reproduces on our weights (toy test identical)
> and in Isaac (−2.5 Nm pitch + 1 s ramp → ankle −0.58/+1.33 at 0.4 s, 21.6 rad/s). **Do NOT re-engage walker_v5_3200:**
> under the measured load alone it falls 16 % within 6 s here even with a hard start. All four asks, plus
> stand-from-spawn (no training episode ever began standing), are in walker v6 — training since 14:35. A bundle
> ships only if it passes your `cf_tracking.py` and our 15-condition engage test inside your watchdog limits.
> Four questions for you in §4.

> **2026-10-03 18:10 — CORRECTION: `REPLY2_HW_ENGAGE.md`.** Our "v5 falls 16 % under the load alone even with a hard
> start" was a TEST ARTIFACT (we had stripped the plant, including the ankle joint friction the policy was trained
> with). On the plant as trained, walker_v5_3200 holds −3…+3 Nm for 20 s with **0 falls** at a hard start; the weak-start
> wind-up still reproduces (18.2 rad/s). **"Do not re-engage" is withdrawn — your call.** Note: a normal start reaches
> ~5 rad/s joint speed in our plant (≈3 with clean sensors), i.e. at your watchdog limit. Today's two fine-tunes (v6,
> v6b) did not beat v5; no bundle from them.

## The ask, in one line

**Run the three batteries on your side and send back a `metrics.json` for each, plus answers to the
six questions in §4.**

> **UPDATED 2026-08-26** after your first reply. Two things changed: **Battery 2B (free decay) is
> RETRACTED** — you were right, our "gains off" never switched friction off and we measured dither;
> and **Battery 3 is new** — after that retraction we had no comparison of dynamic friction at all,
> which is the thing a policy actually fights. Read `RIG_REPLY_SIM2SIM.md` first if you have already
> run anything.

## 1. Files, in reading order

| file | what it is |
|---|---|
| `RIG_REPLY_SIM2SIM.md` | **Read first if you have already run a battery.** Our Test B retraction, the armature/mass/COM answers you asked for, and what we think the real bare-plant test is. |
| `SIM2SIM_STEP_SPEC.md` | **Battery 1** — per-joint ±step. Protocol, our 10 curves, the traps. |
| `SIM2SIM_FRICTION_SPEC.md` | **Battery 2** — amplitude ladder (2A, valid) + free decay (**2B, RETRACTED — ignore its half-lives**). |
| `SIM2SIM_BATTERY3_SPEC.md` | **Battery 3, NEW** — constant-velocity friction sweep. The dynamic-friction comparison the other two never made. |
| `SIM2SIM_ANKLE_READOUT_WARNING.md` | Read before comparing **ankle** rows — we report a different quantity than you do. (Your reply confirms this mattered.) |
| `rig_sim2sim_*.py`, `rig_decay_diag.py` | **Reference implementations — NOT runnable on Isaac.** See §2. |
| `rig_sim2sim_curves.tgz`, `rig_friction_metrics.json`, `rig_frictionsweep_metrics.json` | Our raw curves and metrics. |

## 2. ⚠️ The scripts are reference, not runnable

Our runners are ROS 2 + MuJoCo specific: they publish `odrive_mit_example/LegCmd` on
`/leg_impedance_controller/command`, read `/joint_states`, and drive our emulator over UDP
(`HOMEMODE` to :9994). **None of that exists in IsaacLab.**

Read them as the **authoritative statement of the protocol** — exact amplitudes, sequencing, gains,
timing, and the metric definitions — and reimplement in Isaac. Where a spec sentence and the code
disagree, the code is what we actually ran.

## 3. What to send back

A `metrics.json` per battery, same schema as ours (`sim2sim_step_v2`, `sim2sim_friction_v1`) so a
diff is mechanical rather than a reading exercise. Raw per-joint logs welcome but optional.

**Report the controls, always** — `reach` (did the joint get to target) and `leak` (how far the held
joints moved). We wasted three runs on tests that produced beautiful flat lines from a robot that
never moved, and a frozen joint is indistinguishable from a perfectly-matching one unless you check.
**Exclude any joint that failed `reach`; do not report it as agreement.**

## 4. Six questions, all cheap, all needed to read the results correctly

1. **Foot-to-floor clearance at the test pose?** Ours was **3.7 mm** at the standing pin height, and
   ~1.6° of ankle rotation planted the feet — it silently contaminated an entire dataset. If you pin
   at standing height with a ground plane present, your ankle numbers are ground reaction.
2. **Position quantum?** Ours is a 16-bit encoder model, **0.0219°**, which floors our friction
   ladder. If yours is float, your ladder will resolve friction where ours cannot — and a real
   intercept from you versus our noise floor is *expected*, not a plant difference.
3. **What do your ankle rows contain — joint side or motor side?** Ours are **motor side** by
   design. See the warning file.
4. **Does your actuator model stiction?** Ours does not: `-fc*tanh(qd/0.02)` is Coulomb while moving
   but **exactly zero at rest**, so our joints always creep to the exact setpoint. Hardware has
   ~0.3 Nm of real stiction. If yours is also a regularised form, **neither sim has it and no policy
   either of us has trained has ever met it.**
5. **Do your USD joint limits match the MJCF ranges** in the step spec's table? Knees are one-sided
   ([−155, 0] R / [0, 155] L). If they differ, that is itself a finding.
6. **Did the COM +11.6 mm correction ever land on your asset?** We sent it, never confirmed, and it
   moves every static margin.

## 5. What we expect to learn

- **Rise time** — inertia per link. The most likely real mismatch.
- **Overshoot** — damping and effective inertia.
- **Free-decay half-life and envelope shape** — friction magnitude and mechanism. **This is the best
  test of the four:** with kp=kd=0 the actuator is off, so it compares *plants* rather than
  plant-plus-actuator, and our linear-vs-exponential separation was decisive (r² 0.998 vs 0.90).
- **Ladder intercept vs slope** — separates Coulomb friction (intercept) from compliance (slope).

If the eight non-ankle joints agree on rise time within ~20% and overshoot within ~5 points, the two
plants agree at this level and we can stop worrying about the rigid-body model.

---

*— the rig, 2026-08-25*
