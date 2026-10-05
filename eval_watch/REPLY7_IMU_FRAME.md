# TRAINING → RIG · reply to `RIG_REPLY_IMU_FRAME.md` (2026-10-04): thank you — the retrain is launched

Thank you for doing the hand test the same night, and for the plain answers on the driver.

- **Frame: settled.** We ran `imu_frame_check.py` on your `imu_frame_test_ep_…4513t.npz` and get your output to
  the digit: three clean, separate moves (forward tilt 18°, tilt to the left 17°, turn to the left 132°), all five
  signs as in the simulator, gravity and gyro one right-handed frame (slopes 0.98 and 0.99 — the roll pair now has
  a real signal). The robot and the simulator feed the policy the same frame: **x down, y backward, z left.**
- **Your calibration** (down from stillness, the sideways axis from rocking, the sign of forward from a held lean)
  fixes exactly the one sign our recordings could not; nothing to change on your side.
- **Forward and back:** agreed, and noted that your scripts and the plot are replaced.
- **Your two observations on "the rate, hardly the lean":** both point the same way as ours. The integrated
  gradients for v5_3200 (gyro −0.37 of −0.50, gravity −0.02) are an independent measurement of what we saw in the
  exported network (ankle target 0.3° per degree of lean, 38° per rad/s of pitch rate). And a robot that leans
  1.3–2.3° and stays there is what a controller with rate feedback and almost no lean feedback does, strap or not.
  Whether the corrected symmetry term changes that is one of the things the new run is read for.

**What is running here now: walker v8.** It is walker v7 again — same start (v5_3200's checkpoint), same recipe,
same length, same tests at the same iterations — with one change, the corrected mirror. So it reads against v7
line by line. About three hours. Beyond v7's tests each checkpoint also gets the stand pose (is it symmetric now?).

**Nothing changes for you.** v7_800 stays the bundle for stand #5. If v8 produces a checkpoint that is at least as
good as v7_800 on every test and stands straight, we will post it as a second bundle with the same interface, and
which one you put on the robot stays your call.

*— Isaac side, 2026-10-04*
