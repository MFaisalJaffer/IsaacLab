# ⚠️ READ BEFORE COMPARING ANKLE CURVES — our two sims report DIFFERENT quantities

**2026-08-25, urgent addendum to `SIM2SIM_STEP_SPEC.md`.** If you have already run the battery,
this affects how you read the ankle rows. Nothing else in the spec changes.

---

## The problem

**For the two ankle joints, our emulator reports the MOTOR-side position. Isaac reports the true
joint position. These are not the same signal.**

Our ankles carry a series spring (`K_s` 52 Nm/rad) between a virtual motor state and the joint.
`_encoder()` deliberately returns the motor state, because that is what the real hardware encoder
can see — the spring, the linkage and the foot all sit *downstream* of it, and ~54% of true motion
is invisible to it. We model that on purpose so the policy is blind in sim exactly as it is on the
robot.

Isaac has no such indirection: your ankle readout is the actual joint.

## What that does to a naive comparison

- **Steady state agrees.** With the spring unloaded at rest, motor ≈ joint, so `frac` is comparable.
  Our corrected ankle numbers (0.994–1.004) can be compared to yours directly.
- **Transients do NOT agree, and are not supposed to.** Our ankle overshoot of **26–27%** is partly
  motor-side two-mass dynamics through the spring — a mode your joint-side readout will not contain.
  Rise time is affected likewise.

**So a mismatch in ankle rise time or overshoot is expected and is not evidence of a plant
difference.** Comparing those two curves naively manufactures a discrepancy that does not exist.
The other eight joints have no series element and are directly comparable throughout.

## What to do

1. **Compare ankle `frac` / steady state — yes.**
2. **Compare ankle rise time and overshoot — no**, unless you can also log motor-side state through
   your actuator model, in which case send both and we will compare like with like.
3. **Tell us which quantity your ankle rows actually contain**, so we label them correctly rather
   than inferring.

If you would rather have a directly comparable ankle number, we can re-run ours with
`ANKLE_SERIES_K=0` (rigid, motor == joint) as a second arm. Say the word — it is one restart, and
it would give a clean apples-to-apples ankle comparison at the cost of not representing the real
robot.

## Unrelated but same file

The contact warning in §4 of the spec matters more than this one: **"base pinned" is not "no
contact"**, our feet cleared the floor by 3.7 mm at the standing pin height, and ~1.6° of ankle
rotation planted them. Please verify and report your own foot-to-floor clearance at the test pose.

---

*— the rig, 2026-08-25*
