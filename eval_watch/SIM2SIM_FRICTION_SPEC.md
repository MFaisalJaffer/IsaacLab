# RIG → TRAINING · Friction battery — the two probes the step test is blind to

**2026-08-25** · Companion to `SIM2SIM_STEP_SPEC.md`. Same conditions: full stack both sides, base
pinned **and lifted clear of the floor**, no contact.

**→ Please run both tests below on Isaac and send back a `metrics.json` in our schema
(`sim2sim_friction_v1`).** `rig_sim2sim_friction.py` is attached as the authoritative statement of
the protocol, but it is **ROS 2 + MuJoCo specific and will not run on Isaac** — reimplement it.
See `00_README_SIM2SIM.md` for the full ask and the questions we need answered alongside.

**Why this exists:** the ±20° step battery cannot see friction. At kp 150 a 20° step commands 52 Nm
of restoring torque, so even 4 Nm of Coulomb friction is a 7% effect. We proved the blindness on
our own model — our hips carry a nominal `Fc = 4.0 Nm` and the step battery reported
`frac = 0.998`, i.e. indistinguishable from frictionless.

---

## Test A — amplitude ladder

Same step, at **0.5, 1, 2, 5, 10, 20°**. Fit steady-state error against amplitude. The two
mechanisms separate by *shape*, so one fit yields both:

    constant error (intercept)     -> Coulomb / stiction;  tau_c = radians(intercept) * kp
    proportional error (slope)     -> compliance / series spring

⚠️ **Units:** the intercept is in degrees and kp is Nm/rad. We shipped this wrong first time and got
answers 57× too large. Convert.

## Test B — free decay ★ the more useful of the two

Set **kp = kd = 0** on one joint, drive to 15°, release, let gravity swing it. Fit the envelope of
successive extrema:

    linear decay       -> Coulomb friction
    exponential decay  -> viscous damping

Two reasons this is the better probe. It is **decisive** — our linear/exponential r² separation is
0.06–0.16, not marginal. And **the actuator is switched off entirely**, so it isolates the plant:
no TV actuator, no MIT emulation, no gain assumptions. It is the one test in this whole exchange
that compares physics rather than physics-plus-actuator.

---

## Our results

### A. Amplitude ladder — quantisation-limited ON OUR SIDE ONLY

| joint | intercept | in LSB | implied Coulomb | nominal `Fc` |
|---|---|---|---|---|
| hip_pitch (R/L) | 0.0255 / 0.0256° | 1.16 / 1.17 | **<0.067 Nm** | 4.0 |
| hip_roll (R/L) | −0.0401 / 0.0255° | 1.84 / 1.16 | **<0.105 Nm** | 4.0 |
| hip_yaw (R/L) | 0.0255° | 1.16 | **<0.027 Nm** | 0.4 |
| knee (R/L) | −0.0401 / 0.0255° | 1.84 / 1.16 | **<0.105 Nm** | 0.6 |
| ankle (R/L) | 0.0255 / −0.0401° | 1.16 / 1.84 | **<0.042 Nm** | 0.3 |

Every intercept lands within **1.2–1.8 LSB** of our MIT position quantiser (16-bit over ±12.5 rad =
**0.0219°**). So on our side these are **upper bounds, not measurements**.

**This asymmetry matters for the comparison:** our joint state passes through a 16-bit encoder
model by design; yours almost certainly does not. **Your ladder should resolve real friction where
ours cannot.** If you measure a genuine intercept and we report the quantisation floor, that is
*expected*, not a plant difference. Please report your own position quantum (or "float, none").

### B. Free decay — decisive

| joint | peaks | linear r² | exp r² | half-life | verdict |
|---|---|---|---|---|---|
| right_hip_pitch | 39 | **0.998** | 0.934 | 2.01 s | **Coulomb** |
| right_hip_roll | 21 | 0.950 | 0.939 | 2.49 s | ambiguous |
| right_hip_yaw | 36 | **0.999** | 0.910 | 1.60 s | **Coulomb** |
| right_knee | 30 | **1.000** | 0.905 | 1.55 s | **Coulomb** |
| right_ankle | 0 | — | — | — | **did not swing** |
| left_hip_pitch | 51 | **0.999** | 0.881 | 1.51 s | **Coulomb** |
| left_hip_roll | 25 | 0.956 | 0.920 | 2.21 s | ambiguous |
| left_hip_yaw | 37 | **0.999** | 0.839 | 1.58 s | **Coulomb** |
| left_knee | 13 | **1.000** | 0.905 | 1.79 s | **Coulomb** |
| left_ankle | 0 | — | — | — | **did not swing** |

**Half-life is the number to compare** — ours is 1.5–2.5 s. It is a direct proxy for friction
magnitude and needs no actuator model on either side.

**Our ankles do not free-swing.** With kp=kd=0 the ankle is still tied to a floating rotor through
the series spring, so it is not truly free. If yours oscillates freely, expect no comparison there —
that is the series model, not a plant difference.

## ★ What this revealed about our own model — and probably a gap in both

Putting A and B together characterises our friction precisely:

- **Dynamically: Coulomb.** Free decay gives a linear envelope on 6/10 joints, r² ≥ 0.998.
- **Statically: absent.** The ladder shows <0.1 Nm against a nominal 4.0 Nm.

Both follow from the model being `-fc * tanh(qd / 0.02)`: it approximates `-fc * sign(qd)` while
moving, and goes to **exactly zero at qd = 0**. There is no stiction. A joint always creeps to its
exact setpoint.

**The real robot does not do this.** We measured **~0.3 Nm of genuine stiction** on the ankle
today, three independent ways (dead-band torque 0.36, release residual 0.30° × kp 60, rest noise
±0.26). At kp 60 that is a ±0.3° dead zone the policy has to live with — a joint that stops short
and *stays* short.

**So: does your actuator model have stiction?** If it uses the same tanh/regularised form, neither
sim has it, and no policy either of us trains has ever encountered a joint that refuses to close
the last fraction of a degree. That seems worth fixing on both sides — and worth randomising, given
stiction varies part to part and with wear.

---

*— the rig, 2026-08-25*
