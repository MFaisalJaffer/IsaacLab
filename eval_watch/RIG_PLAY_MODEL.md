# Ankle free-play (backlash) model — implementation spec for the rig simulator

**Context:** hardware measurement = ankles have a **15° total free band (±7.5°)** from linkage
looseness; all other joints ≈ tight. This is the model we added to the training simulator,
validated against the hardware symptoms, so you can port it to yours.

## 1. The mechanism being modeled

```
 MOTOR ──────► LOOSE LINKAGE ──────► FOOT
 (encoder      (±7.5° free            (wanders freely inside the band;
  reads HERE;   travel)                motor cannot feel it,
  = policy's                           encoder cannot see it)
  jpos obs)
```

Torque transmits only when the linkage engages at a band edge (±7.5° from the motor angle).
The IMU is the only sensor that sees the consequence (body tilt).

## 2. The model — a deadband on the PD position error

```
 PD torque
    ▲                     rigid joint (old sim):        /
    │                     torque = kp·err everywhere   /
    │                                     ____________/
    │                                    /
 ───┼──────────══════════════════──────► position error (target − measured)
    │         /   free band =
    │        /    "play" (±play/2):
    │       /     ZERO stiffness torque
```

Inside ±play/2 of the target: no corrective stiffness (link wanders under load).
Outside: ordinary PD, shifted — the linkage has "caught."

## 3. Exact change (port this)

```python
# Actuator config: one new field. (0,0) = rigid = legacy-exact.
play_range: tuple = (0.0, 0.0)   # free-play band, rad, drawn U(range) per env

# In the actuator's compute(), BEFORE the (delayed) PD stage:
err      = q_target - q_measured
half     = play * 0.5                       # play: per-env, per-joint tensor
err_eff  = sign(err) * max(abs(err) - half, 0)  # deadband the error
q_target = q_measured + err_eff             # then run your normal PD / delay
```

Per-episode randomization (our training values, post-measurement):

| joints | band draw |
|---|---|
| ankles | U(10°, 16°) total band — brackets your measured 15° |
| all others | U(0°, 0.5°) |

Each joint draws independently (asymmetric wear left/right).
**For a faithful reproduction of YOUR robot in YOUR sim: ankles = 15°, others = 0.**

**Known approximations** (fine at ≤2°, acceptable at 15°, listed for honesty):

1. kd damping stays active inside the band — real play also frees damping; hardware
   grease/friction partially damps anyway.
2. With an actuation delay, the shrink is computed against current position but applied
   up to 40 ms later.
3. **Observation caveat — matters at 15°:** in this version the sim policy still *observes
   the true foot angle*; your hardware encoder is motor-side and cannot. Our next extension
   reports the motor-tracked angle in the ankle jpos obs so sim is exactly as blind as the
   robot. If your simulator can model the linkage as a real passive joint (e.g., a limited
   free joint in MuJoCo), that is the higher-fidelity equivalent and gets the encoder story
   right for free.

## 4. Validation — what the model reproduces (use to verify your port)

| policy | ankle band | settled pitch (12 s stand) | net drift / 12 s |
|---|---|---|---|
| 33800 (the one you rig-tested) | 0° (rigid, old sim) | +3.6° | 2.5 cm |
| 33800 | 2° | +5.9° | 3.8 cm |
| 33800 | 7.5° | +4.0° (p95 11.5°) | **22 cm** (max 57) |
| 33800 | 15° (your measurement) | oscillates, p95 6.7° | **58 cm** (max 114) |
| servo build (43200) | 15° | roll 3× better than 33800 | ~52 cm |

**Two testable predictions for your hardware:**

1. An unsupported quiet stand should **wander** tens of cm over ~10 s rather than fall —
   station-keeping, not survival, is the gap at your band size.
2. Roll-axis attitude correction should already work on the servo build even with loose
   ankles (it routes through the hips, which are tight) — pitch is the axis that suffers,
   because its correction path runs through the loose joint itself.

## 5. What training does with it next

Band-width curriculum from small toward the 10–16° range (walk-quality gated), the
motor-side-encoder obs extension, and tilt-hold practice rebalanced toward pitch — teaching
the policy to punch through the band or route pitch correction via hips and weight shift.
Every degree the linkage gets tightened mechanically is performance the policy doesn't
have to buy back — both tracks are worth running.
