# RIG → TRAINING · Sim2sim step-response battery — spec, our ten curves, and the traps

**2026-08-25** · A per-joint comparison of Isaac vs our MuJoCo rig. One joint stepped at a time,
others held, base pinned, no ground contact. **Both sides run their FULL stack including the
actuator model** — the TV actuator on your side, the MIT emulator with series compliance on ours.
The question is whether the two sims give a policy the same experience, not whether the bare
rigid-body plants agree.

**→ Please run this protocol on Isaac and send back a `metrics.json` in our schema
(`sim2sim_step_v2`).** The numbers in §3 are ours to diff against.

`rig_sim2sim_step.py` is attached as the authoritative statement of the protocol — exact
amplitudes, sequencing, gains, timing, metric definitions. **It is ROS 2 + MuJoCo specific and will
NOT run on Isaac** (it publishes `odrive_mit_example/LegCmd` and drives our emulator over UDP);
reimplement it. Where this prose and the code disagree, the code is what we actually ran.
See `00_README_SIM2SIM.md` for the full ask.

---

## 1. Protocol — do not derive these locally

**Amplitudes are HARDCODED and must be identical on both sides.** Joint ranges are strongly
asymmetric and two are one-sided, so a uniform ±20° is invalid — but deriving from each sim's own
limits would let a USD/MJCF limit difference silently make us run *different tests*.

| joint | +step | −step | MJCF range |
|---|---|---|---|
| right_hip_pitch_04 | +20 | −20 | [−127, 60] |
| right_hip_roll_04 | +7 | −20 | [−130, 12] |
| right_hip_yaw_03 | +20 | −20 | [−90, 90] |
| right_knee_04 | — | −20 | [−155, 0] one-sided |
| right_ankle_02 | +20 | −7 | [−13, 72] |
| left_hip_pitch_04 | +20 | −20 | [−60, 127] |
| left_hip_roll_04 | +20 | −7 | [−12, 130] |
| left_hip_yaw_03 | +20 | −20 | [−90, 90] |
| left_knee_04 | +20 | — | [0, 155] one-sided |
| left_ankle_02 | +7 | −20 | [−72, 13] |

**If your USD limits differ from the MJCF ranges above, tell us — that is itself a finding.**

- **Gains** (per joint, both sides): kp `[150,150,60,150,60, 150,150,60,150,60]`,
  kd `[2.5,1.5,1.0,1.0,0.5, 2.5,1.5,1.0,1.0,0.5]`, order right→left ×
  (hip_pitch, hip_roll, hip_yaw, knee, ankle).
- **Base pinned, and LIFTED so the feet are genuinely clear of the floor.** ⚠️ See §4 — this bit
  us hard. Pinning at the standing height is NOT contact-free: our feet clear the floor plane by
  **3.7 mm** there, and the foot reaches 13.5 cm to the toe, so **~1.6° of ankle rotation plants
  it**. We now lift the base ~0.6 m. **Please verify and report your own foot-to-floor clearance
  at the test pose** — if you pin at standing height with a ground plane present, your ankle
  results are ground reaction, not actuator response.
- **Gravity on**, −9.81 z. It loads the chain and is part of what we compare.
- **Sequence per joint:** 1.0 s at 0 → step, hold 2.0 s → 0, 1.5 s → other direction, 2.0 s → 0, 1.5 s.
- **Log ≥100 Hz**: time, target, all ten positions, all ten velocities. Report your rate.

## 2. Controls — please report these, they are not optional

We wasted three runs on tests that produced beautiful flat lines from a robot that never moved.
**A frozen joint and a perfectly-matching joint look identical unless you check.**

- **`reach`** — did the test joint get to target? **This gates the result.** Exclude any joint that
  did not move; do not report it as agreement.
- **`leak`** — how far the *held* joints moved. **This is data, not a failure.** With kp 150 the
  hold is not rigid, so leak is real coupled dynamics and a quantity both sims should reproduce.
  We see 5.7° on the knee when the hip steps. If you see 2°, that is a genuine inertia difference
  and exactly what this battery is for.

Two specific traps from our side, in case they map onto yours:
- Our emulator has a HELD state where **the joints are frozen and nothing can move them**. Sending
  `RESET` alone leaves it there. Every step logged a flat line.
- Our controller has a preflight that drops to **LIMP** without a live command stream, and silently
  ignores commands until it re-engages. We now poll for `ENGAGED` and abort if it never arrives.

Your TV actuator's position PD overriding `set_joint_effort_target` is the same family of problem.
If it does anything similar here, the battery will read as agreement when nothing happened.

## 3. Our results (MuJoCo rig, full stack, K_s 52)

| joint | tgt | reached | frac | rise 10-90 | overshoot % | leak |
|---|---|---|---|---|---|---|
| right_hip_pitch | +20 / −20 | 19.97 / −20.03 | 0.998 / 1.002 | 70 / 70 ms | 13.7 / 14.8 | 2.33 |
| right_hip_roll | +7 / −20 | 6.96 / −20.03 | 0.994 / 1.002 | 80 / 80 ms | 10.6 / 17.8 | 3.90 |
| right_hip_yaw | +20 / −20 | 19.97 / −20.03 | 0.998 / 1.002 | 40 / 30 ms | 1.5 / 1.9 | 0.10 |
| right_knee | −20 | −20.03 | 1.002 | 30 ms | 20.4 | 2.77 |
| right_ankle | +20 / −7 | 19.97 / −7.03 | 0.998 / 1.004 | 29 / 20 ms | 26.3 / 27.4 | 0.84 |
| left_hip_pitch | +20 / −20 | 19.97 / −20.03 | 0.998 / 1.002 | 79 / 71 ms | 14.5 / 14.1 | 2.28 |
| left_hip_roll | +20 / −7 | 19.97 / −7.03 | 0.998 / 1.004 | 80 / 70 ms | 17.5 / 11.4 | 3.97 |
| left_hip_yaw | +20 / −20 | 19.97 / −20.03 | 0.998 / 1.002 | 40 / 40 ms | 1.5 / 1.7 | 0.14 |
| left_knee | +20 | 19.97 | 0.998 | 20 ms | 20.1 | 2.70 |
| left_ankle | +7 / −20 | 6.96 / −20.03 | 0.994 / 1.002 | 20 / 30 ms | 26.2 / 27.2 | 0.78 |

**Every joint reaches target (frac 0.994–1.004) and left/right pairs mirror to ~0.5%** — a useful
self-check that the battery measures what it claims.

Rough signature by joint type: hips ~70–80 ms rise with 11–18% overshoot; yaw ~30–40 ms and almost
no overshoot (1.5%); knees and ankles ~20–30 ms with 20–27% overshoot. **If your rise times are
within ~20% and overshoot within ~5 points, the two plants agree at this level.**

## 4. ⚠️ The trap that nearly cost us this test — check your foot clearance

Our first run showed both ankles settling at **63–67% of commanded target**, stable, zero velocity.
It looked exactly like a broken series-compliance model, and it was the only anomaly in an
otherwise clean sweep. We were one step from sending it to you as a suspected bug in our ankle.

**It was ground contact.** The scene has a floor plane at z=0, and at the standing pin height the
feet clear it by **3.7 mm**. The foot reaches 13.5 cm to the toe, so **~1.6° of ankle rotation
plants it** — after which the motor is pushing against ground reaction, which absorbs 7.7 Nm
happily. *"Base pinned" is not the same as "no contact".*

Lifting the base 0.6 m fixed every joint: ankles went 0.633 → 0.998, **and the leak numbers moved
across the board too** (knee 6.76 → 2.77, hip_pitch 5.74 → 2.33), because planted feet had been
coupling both legs through the floor. **The entire first dataset was contaminated, not just the
ankles.**

Two things follow:

1. **Verify and report your own foot-to-floor clearance at the test pose.** If Isaac pins at
   standing height with a ground plane present, you have the same problem and your ankle numbers
   are ground reaction. It is easy to miss precisely because the other nine joints still look fine.
2. **Our ankles are healthy** — any earlier suggestion otherwise is withdrawn. §3 holds the
   corrected numbers. Independent cross-check: on hardware the same day, `HOME` drove the real left
   ankle from −17.93° to −0.01°, dead on target, consistent with the corrected sim.

## 5. What to compare, and roughly what matters

1. **`frac` (reached/target)** — should be ~1.0 for the eight non-ankle joints. A joint below ~0.9
   means its actuator is not achieving setpoint.
2. **rise time** — scales with inertia/kp. A joint 20%+ off its counterpart on our side points at a
   mass or inertia difference in that link. This is the most likely real mismatch.
3. **overshoot** — damping and effective inertia.
4. **leak** — coupled inertia; compare per stepped joint.
5. **left/right symmetry within your own run** — free self-check needing no cross-comparison.

Our steady-state errors are ≤0.5° on the eight good joints, so differences above ~1° are real.

## 6. Also worth a cheap look while you are in there

A static query, no simulation: **total mass, per-link mass, COM at zero pose, foot collision
polygon extents.** If those differ, the step responses will differ and we would be chasing a
symptom. Ours: total **13.06 kg**, COM x **+0.0131 m** at zero pose (i.e. on the ankle axis after
the +11.6 mm correction), foot polygon x **−0.064 .. +0.148 m**, ankle axis 7.7 cm from heel /
13.5 cm from toe.

**Did the COM +11.6 mm fix ever land on your asset?** We sent it, never confirmed, and it changes
every static margin.

---

*— the rig, 2026-08-25*
