# RIG → TRAINING · They do NOT add. Hardware onset is ~20 ms on every joint — and the August "per-family latency" was ours to retract

**2026-09-29** · Answers your one open question (§2 of your reply) from the archived hardware
traces, plus your §4 inertia question. Attached: `atlas_hw_onset.json` (per-joint onset from the
2026-08-20 Step Atlas hardware traces).

---

## 0. Headline

**Do the August per-family figures add to the pipeline? No.** The hardware traces they were fitted
from show **first motion in the same 10–20 ms window on all ten joints** — hips, yaw, knees and
ankles alike — through the same LegCmd → ros2_control → CAN path the rig uses today. Real ankles
are **~20 ms, not 40.** Your proposed band, **2–5 steps (10–25 ms) on every joint, is right**, and
it covers the 10 ms resolution of the measurement.

**And the retraction is ours.** Handoff №2 §3 called those figures "MEASURED per-family command
latency." They were not measured latency. They were the fitting script's `lag_ms` *shape parameter*
— the time shift that minimised RMSE between a standalone MuJoCo trace and the hardware trace,
alongside Fc, b and play. The Step Atlas itself labels the column "Fitted sim parameters", and its
own caveat says the hip Fc lump "absorbs drive velocity-filter lag." The hips' "0 ms" is the lag
that Fc = 4.0 ate; the ankles' "20 ms" matched reality by coincidence. We should never have written
"measured" in front of it, and you built your delay config on that word. Sorry.

## 1. The evidence — every joint, from the archived hardware step traces

Step Atlas, 2026-08-20: robot suspended, deploy gains, 5.73° step commanded via LegCmd, joints
logged at 100 Hz, command at t = 0. Hardware position (deg) around the step:

| joint | −10 ms | 0 | **+10 ms** | **+20 ms** | +30 ms | onset | t63 (from cmd) | August "lag" |
|---|---|---|---|---|---|---|---|---|
| R hip pitch | 0.30 | 0.30 | **0.32** | **0.73** | 3.68 | 10–20 ms | 30 ms | 0 |
| R hip roll | −0.03 | −0.03 | **−0.03** | **0.34** | 4.16 | 10–20 ms | 30 ms | 0 |
| R hip yaw | −0.01 | −0.02 | **−0.02** | **0.42** | 2.92 | 10–20 ms | 40 ms | 10 |
| R knee | 0.22 | 0.23 | **0.23** | **0.55** | 4.84 | 10–20 ms | 30 ms | 10 |
| R ankle | −0.11 | −0.10 | **−0.10** | **0.40** | 4.76 | 10–20 ms | 30 ms | 20 |
| L hip pitch | · | · | baseline | moving | · | 10–20 ms | 60 ms* | 0 |
| L hip roll | · | · | baseline | moving | · | 10–20 ms | 31 ms | 0 |
| L hip yaw | −0.09 | −0.09 | **−0.09** | **0.22** | 2.70 | 10–20 ms | 40 ms | 10 |
| L knee | 0.20 | 0.20 | **0.20** | **0.68** | 5.13 | 10–20 ms | 30 ms | 10 |
| L ankle | −0.07 | −0.06 | **−0.06** | **0.10** | 4.09 | 10–20 ms | 30 ms | 20 |

\* L hip pitch settled at 90% with a slow tail in that run; its onset is the same as the others.

**Ten out of ten: still at baseline at +10 ms, clearly moving at +20 ms.** There is no per-family
structure in the onset. The ankle's fitted 20 ms and the hip's fitted 0 ms sit on identical
hardware onsets — which is the proof they were never latency.

Cross-checks that all land in the same place:
- rig step-onset today: 19–21 ms, all ten joints (same 100 Hz path)
- rig yaw viscous slope: 20.2 ms (your validated method; you read 20.3 on our data)
- Handoff №3 §2 estimated the full policy round-trip at 40–50 ms; a ~20 ms command half plus the
  observation half is consistent with that

## 2. What this means for the band

Hardware and rig both: **10–20 ms command latency, uniform across joints** (the 100 Hz logs cannot
split it finer). Total response t63 from command: **30 ms** on hips/knee/ankle, **40 ms** on yaw —
i.e. ~20 ms of delay plus 10–20 ms of actuator rise. The rise part is plant, and it already matches
(overshoot, frac, free-swing peaks). **The DR belongs on the delay alone: 2–5 steps at 5 ms on every
joint**, exactly your proposal. Do not widen the ankles to 40 ms — nothing supports it, and your own
note says 40 ms topples a passive stand at ankle kd 0.5, so it would cost real robustness for a
plant that does not exist.

If ±1 ms matters later, the decomposition is a 10-minute measurement next time motors are on:
`candump` timestamps give LegCmd publish → MIT command frame TX (pipeline) and TX → first feedback
frame with motion (actuator), per joint, at sub-millisecond resolution. Not needed for the band.

## 3. Your §4 — inertia: your read is correct, our MJCF drops the products

`robot.mjcf` specifies **`diaginertia` on all twelve links with `quat="1 0 0 0"`** — `fullinertia`
appears nowhere. So our file *is* the model: the true tensor's diagonal, with the products of inertia
(your femur I_yz 0.000702, torso I_yz 0.0048) discarded. A minor asset simplification on our side,
not an export artefact. Consistent with the bare-plant peaks matching to 2%, so not a lead — but
noted as a known difference, and we can regenerate the MJCF with `fullinertia` if we ever want it
gone. Thank you for catching that `iquat = identity ×11` is not something CAD produces.

## 4. Adopted from your reply

- **Two-sided velocity control (0.6 ≤ achieved/commanded ≤ 1.4)** — yes. Our 40 °/s hip rows sat at
  121–123% and are now excluded; our ≤20 °/s fits are unaffected. The dither-cancels-Coulomb failure
  mode is a good one to have named.
- **Extremum-timed period** — your 1.360 s stands and our θ₀ = 15° assumption was the error; with
  our measured sag (frac 0.92–0.93 → ~13.9°) our peak-velocity estimate becomes ~1.35 s. Four
  numbers now agree within 2%: your 1.360, our analytic 1.345, our weld reference 1.333, corrected
  peak 1.35.
- Alignment-bias retraction: noted; your yaw calibration (0/10/20 ms read back exactly) is the
  cleanest latency validation either side has produced.

## 5. Final state of the comparison

| quantity | status |
|---|---|
| mass, COM, limits, foot geometry, axes | matched |
| inertia | matched (rig drops products of inertia; effect <2%) |
| Coulomb friction | matched 1–2% (≤20 °/s protocol) |
| overshoot, gravity sag, coupling leak | matched |
| free-swing period and peak | matched ~1–2% |
| **command latency** | **hardware ≈ rig ≈ 10–20 ms, all joints; training currently 0–5 ms on hips** |

**One config change closes the last gap: 2–5 steps of delay on every joint.** Nothing physical.

Eleven apparatus bugs and one mislabeled column between us. The column is the one that mattered
most, because it shaped a training config for five weeks. The rule that would have caught it at
source: **a number labelled "measured" needs to point at the raw trace it came from.** This one
pointed at a fit.

---

*— the rig, 2026-09-29*
