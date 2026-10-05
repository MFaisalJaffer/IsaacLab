# TRAINING → RIG · one ask: which way do the real robot's IMU axes point? (2026-10-04)

This sharpens §2 of `REPLY6_STAND4_REVERSAL.md` and replaces its "check against the video". Before we retrain with
the corrected symmetry term (`REPLY6` §3), our operator wants the real robot's IMU axes **confirmed by you on the
robot**, not inferred by us. It is a two-minute hand test; the robot can hang in the straps, the foot does not
matter.

## What we need to know

In our simulator the policy's gravity and gyro inputs are in this frame (the URDF's `imu` link, measured in the
running simulator):

| policy-frame axis | points |
|---|---|
| x | **down** |
| y | **backward** |
| z | **to the robot's left** |

Is that also what the real robot feeds the policy?

## What your two recordings already show

`eval_watch/imu_frame_check.py` on `shadow_ep_…4526t.npz` and `stand4_ep_…1501t.npz`:

- The policy's gravity is the gravity vector in the IMU's own frame, straight from its quaternion (w, x, y, z), and
  the policy's gyro is the IMU gyro unchanged (differences 2e-6 and 1e-8). So the policy frame on the robot **is**
  whatever frame your IMU source reports in.
- Upright, gravity reads (1.00, 0.00, 0.01): **x is down**, as in the simulator.
- Gravity and gyro belong to one right-handed frame (d g/dt against −ω × g: slope 0.97 for y, 0.96 for z).
- Your fore-aft hand rocking shows in gravity **y** (−5° … +8°) and hardly in z (0.7°): **y is the fore-aft axis.**

What the recordings cannot tell us is the **sign**: whether +y is backward (then +z is left) or forward (then +z
is right). We inferred backward from the ankle encoders and the URDF's joint signs (`REPLY6` §2); that is an
inference, and it is the one thing we ask you to settle by hand.

## The test

Record one episode (drives off or holding, policy in preview). From upright, make three moves **in this order**,
each 2–3 s out, 2 s held, back to upright:

- **A** tilt the torso **forward** (chest toward the toes), 8° or more;
- **B** tilt the torso to the **robot's left**, 8° or more;
- **C** turn the torso to the **robot's left** about the vertical (counter-clockwise seen from above), 20° or more.

Then `python3 eval_watch/imu_frame_check.py <episode.npz>` prints the sign of each excursion next to what our
simulator gives:

| move | simulator: gravity | simulator: gyro while moving out |
|---|---|---|
| A forward tilt | gravity **y goes negative** (−0.14 at 8°) | gyro **z positive** |
| B tilt to the left | gravity **z goes positive** (+0.14 at 8°) | gyro **y positive** |
| C turn to the left | — | gyro **x negative** |

Please send the script's output and the episode, and say which way the operator actually moved if a move went the
other way.

Two questions about the sensor itself, if you know them:

1. Is the frame your IMU source reports the chip's own frame as it is mounted, or does the driver remap axes? If
   it remaps, how?
2. Was the mapping ever checked by hand against the simulator's frame, or only through the policy standing?

## What we do with the answer

- **All five signs as in the table:** the robot and the simulator agree; we retrain with the corrected symmetry
  term, and "forward" and "back" are swapped in your stand #4 note as described in `REPLY6` §2.
- **Any sign different:** then the robot's frame and the simulator's differ, which matters more than our symmetry
  term. We stop and sort that out with you first; nothing is retrained on it.

Nothing is training here meanwhile, and v7_800 stays the bundle for stand #5.

*— Isaac side, 2026-10-04*
