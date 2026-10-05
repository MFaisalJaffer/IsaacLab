# TRAINING → RIG · reply to the stand #4 note (walker_v7_800, 2026-10-04): the stand pose is the checkpoint, not your foot

Thank you — 30 s without a snap or a growing rock, with the caveats you list, is the first hardware result of this
line that looks like our simulator. Understood on the left foot: we will not use any left-against-right difference
from 3–4 October, nor stand #3's rocking amplitude, as a target.

## 1. Your ask: the stand pose of v7_800 in Isaac

`eval_watch/amp_stand_pose.py`. Flat stand at zero command, hard start, plant as trained (randomization and sensor
noise on), 256 robots, 30 s, all 256 up. Means over 2–30 s and over the robots, degrees. Mirroring swaps left and
right and negates every joint, so a mirror-symmetric pose has L + R = 0.

| | hip pitch L / R | hip roll L / R | hip yaw L / R | knee L / R | ankle L / R |
|---|---|---|---|---|---|
| **your robot, measured** (2–6 s) | −4.2 / +1.5 | −0.9 / −0.5 | −2.7 / −3.3 | 0.0 / −0.2 | −3.1 / −0.2 |
| Isaac, encoder (motor side) | −3.7 / +1.9 | +0.2 / +0.3 | −2.3 / −2.3 | 0.0 / 0.0 | −2.2 / +0.2 |
| Isaac, link | −3.7 / +2.0 | +0.1 / +0.4 | −2.3 / −2.3 | 0.0 / −0.1 | −3.1 / +1.5 |
| **your robot, policy target** | −4.7 / +0.9 | +0.4 / −0.4 | −2.8 / −3.4 | at the stop (clipped) | −2.5 / −1.5 |
| Isaac, policy target | −4.4 / +1.5 | +0.7 / −0.5 | −2.8 / −2.9 | at the stop (raw −4.7 / +2.7) | −1.2 / −1.3 |
| Isaac, L + R of the target | −3.0 | +0.3 | −5.6 | 0 | −2.5 |

Spread across the 256 robots of each robot's own mean target: 0.5–0.6° on the hips, 0.8–1.1° on the ankles. Torso
lean in Isaac: pitch −0.5°, roll +0.1°.

**So it is not symmetric in Isaac either, and it is the same pose.** Your policy targets agree with ours to within
0.6° on hip pitch, hip roll and hip yaw, and on the right ankle (0.2°): left hip pitch about 4.5° one way and the
right 1–1.5° the other, both hip yaws about −2.8° (the same direction, i.e. the pelvis turned about 2–3° over the
feet), ankle targets of the same sign on both sides. The spread across robots is a fifth of the offsets, so this is
the checkpoint's stand, not a plant draw. **The asymmetry in your §1 table is the checkpoint.**

What is *not* in Isaac, and so is your robot:
- the left ankle target: −2.5° on the robot against −1.2° here (the right one matches) — the policy asks the left
  ankle for about 1.3° more, which is what a softer left chain would produce;
- the left hip roll sitting at −0.9° against +0.2° here — your standing load on that hip.

For reference, walker_v5_3200 was asymmetric in a different way (hip roll −0.6 / −3.3° target, hips and ankles
mirror-symmetric to 0.7°). The mirror-symmetry loss was on in every one of these runs; it does not force a
symmetric stand because our plant is not symmetric (the torso CoM), and we have not tried to make it one. It
stands 30 s in all 256 robots like this, so we are not planning to change it unless you see a reason on the robot
— the twist puts a little more of the standing torque on one ankle, which is the only thing we can think of that
would matter to a repaired foot.

## 2. What we take from your §3 (loaded right ankle)

No torque-free zone, about 43 Nm/rad, and a loop of about 1.4 Nm on reversal: that is a spring in series with a
Coulomb friction of about ±0.7 Nm that sticks — which is the rotor friction with a static regime that parked our
robot in `REPLY4` §1 (it took 1.0 Nm there; your breakaway numbers were 0.6–1.2 Nm). It also says our "ankle rig"
setting — 2° of *free* play at 52 Nm/rad — is wrong for a standing robot: free play is the unloaded behaviour.
So the next change to our plant, once you have repeated the measurement on both ankles, is:

- ankle chain under load: stiffness around 43 Nm/rad (band to be set from your two ankles), **no free dead band**;
- friction in the chain with a static regime, around 0.7 Nm (band from your loop);
- free play only as an unloaded effect, if at all.

We will not retrain on the single hand-rocked loop; please send both ankles after the repair, and if a slow
controlled loop is possible, the loop width against amplitude.

One consequence to keep in mind on your side: our stand test in `REPLY3`/`REPLY4` used the 2° free play, so its
absolute numbers are pessimistic for the real ankle; the comparison between v5 and v7 on it stands.

## 3. Nothing else from us

Noted that the timing fixes hold on the robot (repeats 5 % / 3 %, longest gap 26 ms, joints 14 ms): that is inside
what v7 trained on. No new run is planned here until your stand #5.

*— Isaac side, 2026-10-04*
