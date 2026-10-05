# RIG → TRAINING · reply to your stand #4 reply (2026-10-04): the reversal check. The gap does lock after a reversal, but not rigidly

We read the revised version (21:25).

## 1. Stand pose: agreed, it is the checkpoint

We withdraw the suggestion that the lopsided pose might be our left foot. Thank you for recording the spread; we
will read future stands against `stand_pose_v7_800.json`.

A symmetric stand is not needed for our diagnostics. Our left-against-right checks are made with the drives holding
the zero pose, where the policy plays no part. Please leave it.

Hip roll: agreed, that one is the robot. The left hip roll sits about 1.2° off under a plain zero-pose hold in
every session (about 3 Nm). Hanging in the straps at the zero pose today it read −0.6° (about 1.6 Nm), and standing
−1.0° to −1.2°, so part of it is there without floor contact. We do not know the cause yet.

## 2. Your check on the shadow run

`rig_hw_stand4/reversal_check.py`. For each stroke between two torso reversals, the slope of encoder angle against
torso pitch, in windows of torso travel since the reversal. Five strokes of 6–13°, drives holding zero at kp 60.

**Right ankle (intact):**

| torso travel since the reversal | encoder ° per torso ° (median; each stroke) | chain stiffness that implies |
|---|---|---|
| 0–0.3° | 0.03 (0.23, 0.03, 0.02, 0.02, 0.09) | — |
| 0.3–0.6° | 0.65 (0.61, 0.65, 0.66, 0.06, 0.75) | 110 Nm/rad |
| 0.6–1.0° | 0.80 (0.80, 0.82, 0.71, 0.78, 0.86) | 240 Nm/rad |
| 1–2° | 0.73 (0.64, 0.71, 0.77, 0.81, 0.73) | 160 Nm/rad |
| 2–4° | 0.47 (0.41, 0.55, 0.47, 0.57, 0.32) | 54 Nm/rad |
| 4–8° | 0.35 (0.43, 0.30, 0.40, 0.31) | 33 Nm/rad |
| 8–14° | 0.38 (0.39, 0.37, 0.38, 0.34) | 36 Nm/rad |

So the answer to your question is **towards 1 : 1, but only part of the way**, and in three stages:

1. **First 0.3°: the encoder does not move.** That is the rotor-stuck signature, or flex somewhere else (hips,
   frame); this test cannot tell which. If it is the rotor, it is about 0.3 Nm of stiction, the size we measured on
   the drive in August.
2. **From 0.3° to about 2°: the encoder follows the torso at 0.65–0.80 : 1.** This is your reading, friction
   across the gap that sticks. It is not a rigid lock: the chain is 110–240 Nm/rad there, three to seven times the
   sliding value. The torque changes by about 1.3 Nm over this stretch.
3. **Beyond about 4°: sliding at 0.35–0.38 : 1**, a spring of about 35 Nm/rad. The 42.6 Nm/rad in our note was one
   straight line through locked and sliding stretches together; 35 is the sliding spring.

**No creep.** While the torso was held still for 1.3–2.4 s the right ankle's torque changed by 0.1 Nm or less, so
the loop is friction and not a slow relaxation.

**On your consequence 1 (the parked robot).** With the lock as measured, each ankle gives 60 × 0.65…0.80 = 39–48 Nm
per rad of torso lean, so two intact ankles give 78–96 Nm/rad against our mgh of 86. That is marginal, not the 120 of
a rigid lock. Once sliding it is 2 × 22 = 44, which is unstable. This matches what the operator sees: the robot can
be parked by hand at one precise point and drifts off it within a minute or two.

**Left ankle, for reference only (damaged):** it locks the same way from 0.3° to 1° (0.70–0.75 : 1) but is back to
0.37 by 1–2°, so its lock lets go about 1° earlier.

**Limits.** Five strokes, rocked by hand, all but one of similar size. The first two windows rest on 0.3° of travel
each.

## 3. Next from us

After the foot repair: both ankles, slow single loops at three amplitudes with the drives holding zero, as you ask
(we plan ±1°, ±3° and ±6° of torso lean), and the hip roll read hanging and standing. Then stand #5 on v7_800.

`rig_hw_stand4/` was not copied when the note went out. It is in the drop folder now, with `reversal_check.py`
added.

*— the rig, 2026-10-04*
