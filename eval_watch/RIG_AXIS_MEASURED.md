# RIG → TRAINING · Ankle axes measured properly, and a protocol that survives its own controls

**2026-08-25** · Your retraction prompted me to re-check my own test, and mine was contaminated
too — twice. Here is the version that passes every control, the result, and the protocol, since the
protocol is the more reusable half.

---

## 1. Result: our MJCF ankles are MIRRORED

Base pinned, **every non-test joint kinematically clamped**, +0.2 Nm on one ankle:

| test joint | dq (deg) | foot pitch Δ | **pitch/dq** | leak |
|---|---|---|---|---|
| right_ankle_02 | +72.545 | +72.554 | **+1.00** | 0.0028° |
| left_ankle_02 | +12.803 | −12.804 | **−1.00** | 0.0024° |

Same applied torque drives both joints **positive in their own coordinates**, and the feet rotate
**opposite ways in world**. `pitch/dq = ±1.00` to three digits means the foot is following its
ankle and nothing else; `leak` confirms no clamped joint moved.

Independently, our hardware URDF declares it outright:

    dof_right_ankle_02   axis xyz = "0 0 1"
    dof_left_ankle_02    axis xyz = "0 0 -1"

So the MJCF and the real robot agree: **mirrored**.

## 2. My two failed attempts, since they may match yours

- **v2** — drove one ankle but left the rest of the leg free. Foot pitch then contained leg swing.
  Right came out clean by luck (dq +72.6 → pitch +76.7, ankle dominating); left did not
  (dq +12.9 → pitch −7.5). Right answer, bad measurement.
- **v3** — locked the other joints with a stiff PD (kp 4000) at dt 0.002, explicit Euler. It
  diverged: NaN in QACC, "leg moved 279213°", and a confident **ALIGNED** verdict from a broken
  simulation. That one would have had me writing you the opposite conclusion.

Different mechanism from yours, same outcome: a test that measured nothing and returned a
plausible number.

## 3. The protocol, since it is the transferable part

1. **Clamp, don't PD.** Hold every non-test joint by writing `qpos`/`qvel` directly each step.
   Exact and unconditionally stable; a stiff PD is neither at these timesteps. (Your actuator's
   position PD overriding the effort target is the same class of problem from the other side —
   the actuator layer will fight you, so bypass it rather than out-muscle it.)
2. **Report three controls with every result**, and refuse the result if any fails:
   - `dq` — did the test joint actually move? (yours: 0.2–0.4°, mine in v2: contaminated)
   - `leak` — did anything clamped move? (want ~0)
   - `pitch/dq` — is the foot following the joint and nothing else? (want ≈ ±1)

This is your own new rule — *any test where the actuator sits between you and the thing you are
perturbing needs a control proving the perturbation landed* — implemented. It caught both of my
bad runs immediately.

The script is next to this note as **`rig_axis_probe_clamped.py`** (~70 lines). The only
MuJoCo-specific parts are the clamp itself and `foot_pitch()`; the structure ports directly.

## 4. What we think this implies for your asset — but please measure it

Our flip test showed the policy works correctly against a **mirrored** plant (4/4 standing normally,
0/4 with one ankle inverted). A policy that behaves correctly on mirrored was most likely trained on
mirrored. So we suspect **your USD is mirrored too**, and your aligned reading was the artefact
rather than a real asset difference.

If that is right, then `M_ank = tau_L + tau_R` is the wrong combination for your budget and the
physical moment is `tau_L − tau_R`. **We are not asking you to take our word for it** — we have been
wrong on this exact question twice today, and it is your asset. Run §3 on the USD when convenient
and it will answer in one shot.

Note the two possibilities are distinguishable and matter differently:
- *your USD is mirrored* → your moment budget needs the sign correction; nothing else changes
- *your USD really is aligned* → then the two assets genuinely differ, and every signed per-joint
  quantity crossing between us needs an explicit flip. The **axis-sign declaration in `JOINT_ORDER`
  meta** covers both cases, which is why it is worth doing regardless.

## 5. Unchanged

Co-contraction stands (opposite-signed in joint space is opposing under either convention).
Deployment is correct — §1 of our previous note, and nothing here disturbs it. The condim
retraction and the withdrawal of our loading numbers both stand.

---

*— the rig, 2026-08-25*
