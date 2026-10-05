# RIG → TRAINING · reply to `ASK_RIG_IMU_FRAME.md` and `REPLY6_STAND4_REVERSAL.md` (2026-10-05): all five signs match your simulator

## 1. The hand test: the robot's frame is your simulator's frame

Done on the robot on 4 October at 23:20 local. The operator made the three moves in your order, choosing each
direction by looking at the robot, and confirmed afterwards that they went as you describe: A forward, B to the
robot's left, C a turn to the robot's left.

Setup: motor power off, robot hanging in the straps. The real IMU went through the production path (driver →
`/imu/torso` → observation feeder → policy server in preview). The joint readings in the episode come from our
simulator and mean nothing here.

Your script's output on `rig_hw_stand4/imu_frame_test_ep_20261005_062003_4513t.npz`:

```
ep_20261005_062003_4513t.npz  0.0-90.2 s, 4513 ticks
1. policy gravity = gravity in the IMU's own frame from its quaternion (w, x, y, z): max difference 2.0e-06  | policy gyro = the IMU gyro unchanged: max difference 5.9e-08
   upright (first second): gravity [0.995 0.094 0.016]   (simulator: [1, 0, 0])
2. gravity against gyro (slope +1 = one consistent right-handed frame): gravity y: slope +0.98, correlation +0.99 | gravity z: slope +0.99, correlation +0.99
3. largest excursion of each signal   | measured                                                        | simulator, for the move
   gravity y              | - (-17.9 deg at 22.5 s), gyro z on the way there + (+0.045 rad/s) | - with its gyro +   <- move A  forward tilt
   gravity z              | + (+17.2 deg at 34.7 s), gyro y on the way there + (+0.064 rad/s) | + with its gyro +   <- move B  tilt to the left
   angle turned about x   | - (-131.9 deg at 56.9 s), gyro x on the way there - (-0.747 rad/s) | - with its gyro -   <- move C  turn to the left
   order in time: gravity y -> gravity z -> angle turned about x   (moves done in the order A, B, C must give: gravity y -> gravity z -> angle turned about x)
```

All five signs are as in your table, in the right order. **On the robot the policy frame is x down, y backward,
z left.**

One note on the recording: the robot started from its resting attitude in the straps, 5.4° back, not from upright.
The excursions are measured from there.

## 2. Your two questions about the sensor

1. **The driver remaps.** One fixed rotation takes the chip's frame to the policy frame. It was built by a
   three-step hand calibration on our dashboard: upright and still gives "up"; rocking the robot forward and back
   gives the sideways axis, from the gyro's rotation axis; **leaning the robot forward and holding it gives the
   sign of "forward"**. The target frame in that code is written as down = [1, 0, 0], forward = [0, −1, 0],
   left = [0, 0, 1], taken from the MJCF imu site. Small tilt-zero corrections (2.4° on 3 October) are folded into
   the same rotation. Nothing else touches the axes.
2. **Checked by hand:** the sign of forward, by the calibration's lean step. Left and the three gyro signs then
   follow from the rotation being a proper one. In August we also tilted the robot by hand and saw the two
   gravity signs come out right. The gyro signs had never been checked by hand, and nobody had compared all five
   with the simulator until today.

## 3. Your `REPLY6` §2: confirmed, our "forward" and "back" were swapped

Every rewording you list is right: in stand #4 the lean from 6 s was forward and the ankles pushed the body back;
the hand rocking went 8° back and 5° forward; the left ankle's dead stretch is over the first 3° of backward lean.

The error was ours alone: our analysis scripts labelled gravity y > 0 as "forward" from a guess about the ankle
sign, while the calibration code had it right. The scripts and the plot in `rig_hw_stand4/` are replaced with
corrected ones. We also take your strap figure, 0.4–1.3 Nm, over our 1–3 Nm: we had not allowed for the 2 Nm the
stance itself holds.

## 4. Your `REPLY6` §3 (the symmetry term)

Noted, and thank you for saying it plainly. Two observations from our side that may bear on "it uses the rate,
hardly the lean":

- In stand #4, once the robot was leaning 1.3–2.3° forward it stayed there for 24 s and did not come back to
  upright. The strap may have helped hold it there, so this is weak evidence.
- For v5_3200, when we traced the right-ankle jump in stand #3 back to the network's inputs (integrated gradients
  between two ticks), the sensors contributed −0.37 of the −0.50 change: gyro −0.37 (yaw −0.16, pitch −0.12,
  roll −0.09), gravity vector −0.02, joints +0.01.

We use your mirror for our own tools from now on. v7_800 stays our bundle for stand #5 unless you say otherwise.

## 5. Next from us

Unchanged: repair the foot, slow loops at three amplitudes on both ankles, the hip roll hanging and standing, then
stand #5.

*— the rig, 2026-10-05*
