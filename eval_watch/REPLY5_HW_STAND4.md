# TRAINING → RIG · reply to the stand #4 note (walker_v7_800, 2026-10-04): the lopsided stand pose is the checkpoint

> **Revised 2026-10-04 21:25, about 40 minutes after the first version of this file.** If you read the first one,
> two things changed. (a) §1 no longer says your left ankle target is "your robot": it is inside our own
> robot-to-robot spread. The only place your robot sits outside our population is the hip roll. (b) §2 no longer
> equates your ankle loop with rotor stiction: the loop in your table runs the opposite way to what rotor stiction
> gives. The Isaac numbers are now from a second run that recorded the spread; they moved by at most 0.2°.

Thank you — 30 s without a snap or a growing rock, with the caveats you list, is the first hardware result of this
line that looks like our simulator. Understood on the left foot: we will not use any left-against-right difference
from 3–4 October, nor stand #3's rocking amplitude, as a target.

## 1. Your ask: the stand pose of v7_800 in Isaac

`eval_watch/amp_stand_pose.py`, numbers in `eval_watch/stand_pose_v7_800.json`. Flat stand at zero command, hard
start, plant as trained (randomization and sensor noise on, fresh sensing), 256 robots, 30 s, all 256 up. Each
robot's own mean over 2–30 s; then the mean, the spread (standard deviation) and the 5–95 % range across the 256
robots. Degrees.

| | hip pitch L / R | hip roll L / R | hip yaw L / R | knee L / R | ankle L / R |
|---|---|---|---|---|---|
| **policy target, your robot** (2–6 s) | −4.7 / +0.9 | +0.4 / −0.4 | −2.8 / −3.4 | at the stop | −2.5 / −1.5 |
| policy target, Isaac mean | −4.5 / +1.5 | +0.7 / −0.4 | −2.7 / −2.9 | at the stop (raw action −4.7 / +2.7) | −1.1 / −1.4 |
| policy target, Isaac 5–95 % | −5.3…−3.7 / +0.5…+2.5 | +0.4…+1.2 / −0.6…−0.2 | −4.0…−2.0 / −3.9…−2.0 | — | −2.8…+0.8 / −2.8…+0.1 |
| **measured, your robot** | −4.2 / +1.5 | −0.9 / −0.5 | −2.7 / −3.3 | 0.0 / −0.2 | −3.1 / −0.2 |
| encoder, Isaac mean ± spread | −3.9 ± 0.6 / +1.9 ± 0.8 | +0.2 ± 0.3 / +0.3 ± 0.3 | −2.3 ± 0.6 / −2.4 ± 0.8 | 0.0 / 0.0 | −2.2 ± 0.7 / +0.1 ± 0.6 |
| link, Isaac mean | −3.8 / +2.0 | +0.1 / +0.4 | −2.2 / −2.4 | 0.0 / −0.1 | −3.3 / +1.5 |

Mirroring swaps left and right and negates every joint, so a mirror-symmetric pose has L + R = 0:

| L + R of the policy target | hip pitch | hip roll | hip yaw | ankle |
|---|---|---|---|---|
| your robot | −3.8 | 0.0 | −6.2 | −4.0 |
| Isaac mean ± spread | −3.1 ± 0.7 | +0.3 ± 0.4 | −5.7 ± 0.6 | −2.6 ± 1.2 |
| Isaac 5–95 % | −4.2…−1.8 | −0.2…+1.0 | −6.7…−4.6 | −4.6…−0.6 |
| Isaac robots symmetric within 1°, of 256 | **0** | 246 | **0** | 23 |

**It is not symmetric in Isaac either, and it is the same pose.** Not one of our 256 robots stands symmetric
within 1° in hip pitch or in hip yaw, and all 256 are lopsided to the same side: the left hip pitch about 4.5° one
way and the right 1.5° the other, both hip yaws about −2.8° (the same sign, i.e. the pelvis turned a few degrees
over the feet). Every one of your ten targets, and all four of your L + R sums, lie inside our 5–95 % range.
**The asymmetry in your §1 table is the checkpoint, not your left foot.** Torso lean in Isaac: pitch −0.5°,
roll +0.1°.

Where your robot is and is not inside our population:

- **Ankles: inside.** Your left ankle target (−2.5°) is 1.3° past our mean, but our own robots spread by ±1.2° on
  that joint, so it is an ordinary plant draw here. Drive torque, kp × (target − encoder) at kp 60: yours
  +0.6 / −1.4 Nm, ours +1.1 ± 0.7 / −1.6 ± 0.8 Nm — the same sign pattern, the right ankle carrying more in both.
  So this table cannot show the damaged foot one way or the other.
- **Hip roll: outside, and you already know it.** Your hip-roll encoders sit at −0.9 / −0.5°, ours at
  +0.2 ± 0.3 / +0.3 ± 0.3°, with nearly the same targets. That is 1.3° of offset on your left hip roll — the same 1.3° you
  report under a plain zero-pose hold in your §3, so it is the robot and not the policy. As drive torque (kp 150)
  it is about 2 Nm more than ours on both hip rolls in the same joint direction (yours +3.4 / +0.3 Nm, ours
  +1.3 ± 0.6 / −1.8 ± 0.5 Nm): a sideways load our robots do not have, or stiction holding the offset.

Why the checkpoint is lopsided: the mirror-symmetry loss was on in these runs (coefficient 1.0), but it is a soft
penalty. On the bundle's own golden standing inputs, mirrored, the network's left–right error averages 0.7–1.1° on
hip pitch and hip yaw, and all 256 robots settle on the same side. walker_v5_3200 is lopsided in a different
joint (hip roll target −0.6 / −3.3°, L + R −3.9 ± 0.5, none of 256 within 1°; its hip pitch and ankle sums are
+0.6° and −0.4°). It stands 30 s like this in all 256 robots and on yours. If the lopsided stance gets in the
way of your left-against-right diagnostics, say so and a symmetric stand becomes a requirement of the next
fine-tune; otherwise we leave it.

## 2. Your loaded-ankle loop: we read it as friction across the gap, not at the rotor

This is the measurement we asked for in `REPLY4` §1; thank you. We only have the binned table (`rig_hw_stand4/` is
not on this machine), so please check the reading below against the raw traces.

How each kind of friction shows in your plot (drive torque kp × encoder, against pitch − encoder, slow rocking):

- **at the rotor** (the encoder's side of the gap): on a reversal the encoder stays put, so the torque stays
  constant while the deflection changes; and the forward branch is the *lower* one (the friction helps the drive
  hold the load);
- **across the gap** (between the encoder's side and the foot's side): on a reversal the deflection stays put and
  the torque changes — foot, gap and rotor move as one piece against the drive's kp; the forward branch is the
  *higher* one;
- **at the shank–foot hinge itself** (beside the whole drive): no loop in this plot.

Your forward branch is the higher one in every bin from −2° to +3° (by 0.5–1.45 Nm), and you write that the torque
drops "before the deflection changes much". Both are the second kind. So we read: **a spring of about 43 Nm/rad
with a friction of roughly 0.3–0.7 Nm acting across the gap, which sticks** — your "hysteretic backlash"
candidate. Our plant has no such element: its play is a free dead band inside the spring, and its rotor friction
is on the wrong side of the spring (it would make the loop run the other way).

**One check on the shadow run:** at a reversal, for the first degree of torso motion, does the encoder follow the
torso more closely than while sliding (towards 1 : 1 — gap locked, our reading), or less (towards 0 — rotor
stuck)? While sliding you measured 0.42 : 1.

Two consequences if the reading holds:

1. For torque changes smaller than the loop (about ±0.7 Nm per ankle, the first ~0.7° of motion at kp 60) the gap
   and the spring are out of the picture and the motor holds the foot directly. If the lock is rigid, a home hold
   then has 2 × kp = 120 Nm/rad against your mgh of 86 Nm/rad, which is stable; once the gap slides it has
   2 × 25 = 50 Nm/rad, which is not. That would explain your parked robot without any rotor stiction (our plant
   parked only with rotor stiction and 0.3° of play, `REPLY4` §1). How rigid the lock is, the binned table cannot
   say: its steepest step after a reversal is about twice the spring's slope, and 1° bins over several loops
   would show that even for a rigid lock. The 1 : 1 check above answers it.
2. The loaded real ankle is easier to stand on than our plant. Our stand tests in `REPLY3` / `REPLY4` used
   52 Nm/rad with 2° of *free* play, where the motor has no authority inside the gap. Their absolute numbers are
   pessimistic for the robot; the comparison of v5 against v7 on them stands.

What we will change in our plant once you have both ankles after the repair — not before, one hand-rocked loop on
one ankle is too little to train on: stiffness from your two ankles, no free dead band under load, and a friction
element across the spring with a static regime, sized from your loops. For that fit, a slow single loop at two or
three amplitudes would help most: loop height against amplitude separates a friction element from a gap.

## 3. Nothing else from us

Noted that the timing fixes hold on the robot (repeats 5 % / 3 %, longest gap 26 ms, IMU 11–33 ms old, joints
14 ms): inside what v7 trained on, from the easy side. Nothing is training here, and we are not planning a run
before your stand #5 and the two-ankle measurement.

*— Isaac side, 2026-10-04*
