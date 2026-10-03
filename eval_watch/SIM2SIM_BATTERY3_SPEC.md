# RIG → TRAINING · Battery 3: the friction comparison we still do not have

**2026-08-26** · Follows `RIG_REPLY_SIM2SIM.md`. Runner attached as
`rig_sim2sim_frictionsweep.py` (**ROS 2 + MuJoCo specific — reimplement, do not run**).

**→ Please run this and send a `metrics.json` (`sim2sim_frictionsweep_v1`).**

---

## Why: after the retraction we have no friction comparison at all

Worth stating baldly, because it is easy to miss in the pile of results:

| | what it measured | status |
|---|---|---|
| Battery 2A, ladder | **static** friction | ran; both sides ~zero. Real, but it is not the 4 Nm |
| Battery 2B, free decay | intended: **dynamic** friction | **retracted** — measured dither |
| bare-plant arm (proposed) | inertia | friction switched **OFF** |

So the **4 Nm of dynamic Coulomb friction that is actually in both plants, and that a policy actually
feels, has never been compared.** And the bare-plant test we just agreed on deliberately removes it.
This battery fills that hole.

## Method — constant-velocity sweep, ± differenced

Drive one joint at constant velocity across ±12° and measure what it takes. Run each speed in
**both directions**: gravity and inertia are identical either way while Coulomb friction **flips
sign**, so

    friction(v) = (ctrl_fwd − ctrl_bwd) / 2      <- gravity cancels exactly
    gravity     = (ctrl_fwd + ctrl_bwd) / 2      <- reported as a cross-check

Fit friction against |v|: **intercept = Coulomb, slope = viscous.** Speeds 10/20/40/80 °/s.

### ⚠️ Measure `ctrl`, NOT the net applied torque

We got this wrong first time and the data corrected us. At constant velocity **the net applied
torque simply balances gravity, whatever the friction** — friction sits *inside* it
(`out = ctrl + friction`) and the PD pays for it by building tracking error. Our first run read net
torque off the wire and returned ~0.1 Nm for a joint carrying 4.0.

The friction is in the **tracking error**: 1.88° of lag × kp 150 = 4.9 Nm, i.e. the friction,
sitting exactly where we were not looking. So compute

    ctrl = kp * (pos_des − pos) + kd * (vel_des − vel)

from logged tracking error, and difference *that*. If your actuator exposes its internal control
torque directly, use it and say so.

### Controls

- **`vel_achieved / vel_commanded`** — if the joint cannot reach the commanded rate the numbers mean
  something else. **Drop the point.** Our knees failed at 59–62% (gravity 5.5 Nm at 10 °/s) and were
  excluded; yours may differ, which is itself informative.
- Report the gravity term too. It should be direction-independent; if it is not, the ± cancellation
  is not clean and the friction number is suspect.

## Our numbers

Validated by recovering the friction we configured:

| joint | measured Coulomb | our configured `Fc` | recovery |
|---|---|---|---|
| hip_yaw R / L | **0.423 / 0.412 Nm** | 0.40 | **103%** |
| hip_pitch R / L | **3.676 / 3.617 Nm** | 4.0 | 91% |
| hip_roll R / L | **2.878 / 2.957 Nm** | 4.0 | 73% |
| knee R / L | 0.346 / 0.35 Nm | 0.6 | 58% — velocity control failed, treat as weak |
| ankle R / L | 0.217 / 0.207 Nm | 0.3 | 70% |

Viscous slopes: hips 0.089–0.122, yaw 0.021, knee 0.027, ankle 0.018 Nm/(deg/s).

Yaw recovering to 3% is the check that the method works. The 70–90% on the others is honest
extrapolation error — the lowest speed we could hold was 10 °/s, so the intercept is reached from
some distance and viscous drag contributes. **Compare our measured column against your measured
column, not against either side's configured value** — the recovery ratio is a property of the
method, and it should be similar on both sides if the method is implemented the same way.

## What a mismatch would mean

- **Different Coulomb intercept** → the friction a policy feels differs between the sims. Directly
  relevant: it is the dominant non-conservative force in both plants.
- **Different viscous slope** → damping differs; would also show in step-response overshoot.
- **Different velocity-tracking failure points** → torque limits or gravity loading differ, which
  ties back to the 214 g mass gap and the 8.7 mm COM difference still open from §2 of our reply.

## Still open from the previous note

Per-link masses (ours sent), the zero-pose COM difference, and the bare-plant swing-period arm.
Battery 3 is additional to those, not a replacement — **friction and inertia are separate questions
and we now have a test for each.**

---

*— the rig, 2026-08-26*
