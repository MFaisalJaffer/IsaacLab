# TRAINING → RIG · reply to `RIG_REPLY_STAND4_REVERSAL_CHECK.md` (2026-10-04): agreed on the ankle; one sign for you to check; one fault found on our side

Thank you for the check and for the recordings. Three things: your ankle result as we will model it (§1), a
forward/back label in your note that we think is swapped (§2), and a fault in our training that your recordings
led us to (§3). Nothing here changes the interface or the bundle.

## 1. The ankle: agreed

We re-ran `reversal_check.py` on your shadow recording and get your table to the digit. Your three stages, as the
model we will fit once both ankles are in:

- a **sliding spring of about 35 Nm/rad** (your stage 3);
- in parallel, a **friction element across the gap** of about ±0.7 Nm that yields gradually: after a reversal its
  torque swings by about 1.0 Nm over the first 0.45° of chain deflection (about 130 Nm/rad on top of the spring;
  your 0.65–0.80 : 1 stretch) and by about 1.4 Nm in all by 4° of torso travel;
- possibly **0.3 Nm of rotor stiction** on top (your dead first 0.3°), to be decided by the slow loops.

Your three amplitudes (±1°, ±3°, ±6°) are the right test: ±1° stays inside the lock, ±6° is mostly sliding, and
the loop height against amplitude gives the yield distance. Agreed on the parked robot: with the lock as measured
it is marginal, not stable. And noted: no symmetric stand needed, hip roll is the robot.

## 2. A sign to check: we think "forward" and "back" are swapped in your stand #4 note

Your scripts take torso pitch as asin of gravity y in the policy frame and call positive "leaning forward". By our
model positive is **backward**:

- The policy's gravity and gyro are in the URDF's `imu` link frame. Measured in our simulator
  (`eval_watch/mirror_check.py`): **imu x = down, y = backward, z = left.** A forward lean gives gravity y < 0.
- Your own recording agrees, given the URDF's joint signs: in the shadow run gravity y and the right ankle angle
  correlate at −0.99, and by the URDF a forward lean over a flat foot *raises* the right ankle angle (positive is
  toe-up; its range is −13° … +72°). So gravity y rises as the robot leans back.
- Gravity y and gyro z are consistent with each other in your data (d g_y/dt = −w_z, slope 0.97), so this is only
  about which way is called forward, not about the signals the policy gets.

If you confirm it against the video or the operator, then in your stand #4 note: the lean from 6 s was 1.3–2.3°
**forward**, the ankles were pushing the body **backward**, the hand rocking went 8° back and 5° forward, and the
left ankle's dead stretch is over the first 3° of **backward** lean. The stiffness, the loop and the strap estimate
do not change in size.

One number for the strap estimate: in Isaac a free stand of v7_800 already holds about 2.7 Nm at the ankles
(+1.1 / −1.6 Nm, the same signs as your +0.6 / −1.3 Nm in 2–6 s), because the stance keeps the CoM about 2 cm ahead
of the ankle axis. So about 2 Nm of ankle push is the stance in both worlds. Measured from your 2–6 s window, the
push then grows to 4.4–6.2 Nm at 1.0–1.9° of extra lean, which is 0.4–1.3 Nm more than gravity needs for that
lean: that part asks for a strap, and it is smaller than the 1–3 Nm in your note.

## 3. A fault on our side: our symmetry term mirrored the IMU on the wrong axes

In `REPLY5` §1 we said the stand is lopsided because the mirror-symmetry loss is "a soft penalty". That was not the
reason. Checking the frame for §2 we found that the term mirrors the gravity and gyro inputs as if they were in a
standard body frame (x forward, y left, z up). In the imu frame above, that flips the fore-aft component of gravity
and the pitch rate, and leaves the sideways component and the roll rate alone. So the term told the policy that a
forward lean is the mirror image of a backward lean, and never mirrored a sideways lean.

- Test (`mirror_check.py`): 16 pairs of robots on one plant, the second of each pair driven with the mirrored
  actions. The second robot's gravity y / z follow the first's at +0.98 / −0.98 and its gyro y / z at −0.97 / +0.95:
  the opposite of what our term assumed on all four. The joint mirror (swap sides, negate) is right.
- It has been so in every checkpoint you have had from us with that term on, v5_3200 and v7_800 included.
- What it means for those checkpoints: nothing ever pushed them toward left-right symmetry, which fits the
  lopsided stand and v5 and v7 being lopsided in different joints; and correct balance reactions to the IMU were
  being penalized. They learned them anyway — v7_800 stood 30 s on your robot — but at the stand its ankle targets
  move only 0.3° per degree of lean against 38° per rad/s of pitch rate: of the IMU it uses the rate, hardly the
  lean. Whether that is the term's doing we do not know yet.
- The term is corrected in our code. Whether we fine-tune with it before your stand #5 is being decided here.
  **v7_800 stays the bundle for stand #5 unless we tell you otherwise.**

For your own tools, the mirror in the policy frame is: joints swap sides and negate; command flips vy and wz;
gravity flips **z**; gyro flips **x and y**; the gait phase swaps its two pairs.

*— Isaac side, 2026-10-04*
