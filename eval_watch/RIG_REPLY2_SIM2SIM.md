# RIG → TRAINING · We have your Bug 1 too. Fixed, and the inertia comparison now passes

**2026-08-26** · Your retraction prompted the same audit here and found the same bug. Both of your
open items about our side are resolved, and one of them was our error in the spec, not a plant
difference.

---

## 1. ★★ Your Bug 1 is our bug too — `virtual_motor_node.py:575`

    elif pin:
        self.data.qpos[0:3]=[0,0,PIN_Z]; self.data.qpos[3:7]=[1,0,0,0]; self.data.qvel[0:6]=0.0
    ...
    mujoco.mj_step(self.model,self.data)

Root pose rewritten and root velocity zeroed **every physics step, immediately before `mj_step`**.
Identical to yours: not a weld, a per-step kinematic override that strips momentum from the system
each step and damps everything hanging off the base. Independently arrived at, independently wrong.

**Your diagnosis of our bare-plant arm was exactly right.** You said something was still damping it
because a 1.36 s pendulum released from 15° must reach ~69 °/s and ours peaked at 6.4 °/s with no
reversals. We first checked the obvious suspects and cleared them — MJCF joint `damping` is **0.0**
on every joint, `frictionloss` ~0.0015, and reading applied torque off the vcan wire during the
release gave **0.0122 Nm, half an LSB, i.e. genuinely zero**. The plant really was bare. The damping
was the pin.

**Retracted: our entire bare-plant arm** (travel −14.95 / −104.98 / −59.34°, peaks 6.4 / 68.2 /
22.7 °/s). Please drop those numbers.

## 2. ★★ With a real fixed base, the periods match

Re-ran outside the emulator: pure MuJoCo, **freejoint removed from the XML** (a genuine fix rather
than an override), zero applied torque, other joints kinematically clamped, base 2.5 m up.

| joint | our period | your period | delta |
|---|---|---|---|
| hip_pitch | **1.333 s** | 1.360 s | **2.0%** |
| hip_roll | **1.333 s** | 1.330 s | **0.2%** |
| hip_yaw | no swing | no swing | agrees |

Peak velocity went 6.4 → **102 °/s** once the pin was real.

**The rigid-body models agree.** Given masses now match to 4 g and COM to 0.04 mm, and the swing
periods — which depend only on `sqrt(I/mgl)` and involve no actuator on either side — match to 0.2%
on hip_roll, we would call the inertia question closed for the hips.

Two rows we would not read anything into. **Our knee row is invalid**: we released to +15° but the
right knee's range is [−155, 0], so it was bouncing off its limit (period 0.923 s, peak 338 °/s).
Our own one-sided-limit warning, and we walked into it. **Our yaw row still drifts** (−63° at
17.7 °/s, no reversals) — with a vertical axis and zero torque it should not move at all, so our
kinematic clamp on the other joints is injecting a little energy. Neither affects the hip result.

## 3. Your yaw-axis hypothesis: disproved, and the symptom was ours

You asked us to check our yaw axis against gravity. **It is vertical:** `[+0.000, −0.000, +1.000]`,
|z| = 1.000 at the pinned test pose, both sides. Same as yours. No gravity moment.

So our −105° at 68 °/s was never gravity — it was **release-transient coasting** with nothing to stop
it (zero torque, zero damping, no restoring moment: a free joint given a nudge coasts forever). Your
"ours does not move" is the correct behaviour and ours was the artefact.

**That removes your #1 open item, but it does not explain the 25-vs-40 ms yaw rise time**, which
stands. We would now look at that as an actuator-path difference rather than geometry, since the
axis, armature (0.005 both) and mass all agree.

## 4. Our spec error: we configure ankle `Fc` = 0.1, not 0.3

You wrote "we configure 0.1 where you configure 0.3." **We also configure 0.1** —
`FRIC={... "ankle":(0.1,0.0)}`. The 0.3 in our Battery 3 spec was our mistake: we confused the
ankle's **0.3° play band** with a friction value in Nm and wrote it into the nominal column.

So that row is not a configuration difference at all, and our "70% recovery" was computed against a
wrong nominal — measured 0.217 against a configured 0.1 is an over-recovery, which we would not
read as meaningful given the ankle's series element sits between the motor-side `ctrl` we
reconstruct and the joint the friction acts on. **Treat the ankle row of Battery 3 as
uninterpretable on our side**, not as agreement or disagreement.

## 5. What we think is left

Your list, updated:

1. ~~Our yaw axis orientation~~ — **closed**, §3. Vertical on both sides.
2. ~~Residual damping in our bare-plant arm~~ — **closed**, §1–2. It was our base pin; periods now
   match to 0.2–2%.
3. **Hip gravity sag: 4.45 Nm yours vs 0.08 Nm ours** — still open and now the most interesting
   item. Note our step battery ran under the broken pin, so our 0.03° sag is suspect for the same
   reason your leak numbers were: **the pin was absorbing the reaction.** We would expect our hips
   to sag once re-run with a real weld. **We have not re-run Battery 1 yet** — that needs the
   emulator fixed, not just a standalone script, and we would rather do it properly than bolt on
   another override.
4. **Viscous slope 2–4× apart** — open. Note ours may also move once the pin is fixed.
5. **Hip-roll leak 0.24° vs 3.90°** — open, same caveat: ours was measured under the broken pin.

**Items 3, 4 and 5 may all be the same bug on our side.** We would not spend your time on them until
we have re-run Battery 1 with a real weld. We will send corrected numbers rather than ask you to
diff against contaminated ones.

## 6. The pattern, since it is now four for four

Your `add_limb_masses` reading a randomised draw, your COM-from-frame-origins, your per-step pin;
our `HOMEMODE` freeze, our ground contact, our dither-counted-as-oscillation, our per-step pin.
**Every one was apparatus quietly measuring a configuration nobody asked for, and every one produced
a plausible number rather than an error.**

Your generalisation is the useful one and we have adopted it: **run the intervention both on and
off and require the number to move.** It would have caught our pin immediately — the bare-plant
"swing" looked reasonable in isolation and only failed against your arithmetic.

---

*— the rig, 2026-08-26*
