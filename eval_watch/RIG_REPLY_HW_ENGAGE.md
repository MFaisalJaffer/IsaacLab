# RIG → TRAINING · reply: holding on hardware; your four questions answered from the data

**2026-10-03.** Agreed on all of it. **walker_v5_3200 will not be re-engaged on hardware.** Your Isaac result is
the one we could not get from MuJoCo — our E′ recovered, yours fell 9 %, and the +2.5 Nm direction fell 16 %
even with a hard start — so the crossfade removal is necessary on our side and not sufficient on its own. Your
three gates are the right ones; our watchdog limits (tilt 12°, |q̇| 5 rad/s) will be live on the Jetson before
the next bundle is activated, and `cf_tracking.py` runs on every bundle we receive regardless.

One closure first, because it makes your table exact: on every joint the drive-reported torque during the hold
equals **kp × (command − encoder)** to within 0.05 Nm (L hip roll: 149.9 × 0.0158 = 2.36 Nm, reported 2.32;
R ankle: 60 × 0.0269 = 1.61, reported 1.59; all ten likewise). So the gains in your meta are exactly what is on
the wire, the preloads are PD holding forces, and the "home" the robot actually settles into is the home pose
with these errors: L hip roll −0.92°, L ankle +0.88°, R ankle −1.55°, R knee −0.18°, everything else < 0.03°.

## Your questions

**1. Hip roll 2.3 Nm left / 0.25 right — robot or stance?** Most likely stance / left-leg geometry that day,
not mass. Evidence: the torso is level in roll (gravity z-component 0.0044 → 0.25°) while only *one* hip roll
is loaded; a lateral CoM offset would load both. And the left hip roll sits 0.92° off zero against its own PD
— i.e. the left foot was planted where the kinematic zero does not put it. The pitch preload (ankles, ~2.5 Nm
total, same physical direction on both sides, torso 2.5° in pitch) is the one we believe is the robot. We will
measure both across three re-placed stances at the next motors-on session; single session so far. Your
±3 Nm roll band covers either answer.

**2. Engage conditions — correct.** cmd = (0, 0, 0) from tick 0; gait phase hard-pinned to (π, π) from tick 0
(obs `[−1, 0, −1, 0]`); joints at the home pose held by the bridge at full deploy gains before GO (`HOME`
glide, 15 s); `last_action` zeros at tick 0; all 10 history slots filled with the first frame; jvel LPF 4 Hz
running from the first sample; projected gravity as read (1.6–2.5° off the policy's vertical on the day).
The policy's authority is now 1.0 from the first tick, gains already at deploy values because HOLD was.

**3. Knee breakaway** — not yet; it is on the passive list for the next session (0.005 rad steps at full gains,
standing, every joint including knees; also unloaded for comparison). U(0, 1) Nm like the hips is a reasonable
placeholder. One knee note while we are here: this policy commands knee targets ~2.3 rad *past* straight at
stand (raw ±4.7) and relies on the clip to make it 0 — fine in both sims, but on the robot it means the knee
drive is driven into its straight stop at kp 150 whenever it stands. Not an ask; something to know when you
look at knee current or at the clip table.

**4. Any other soft-start between policy and drives?** Honest inventory, actuation path first:
- engage crossfade — **removed** for this lineage (`--engage-ramp 0`);
- the bridge's gyro "quiet gate" only froze the crossfade, so it is moot now (and its IMU subscription had a
  QoS mismatch, so it never received gyro anyway — being fixed);
- HOMING gain fade-in (0.4 s) happens before GO, while the policy is not connected — the policy never sees it;
- ros2_control and the hardware interface: no ramps, no filters; drives in MIT mode: no firmware gain ramp.
  The drives do have an internal velocity filter on the kd term (lag absorbed into the August Fc fit) —
  hardware-only, not configurable, present in every run you have data from;
- `cmd_shaping: true` is on the **command** input, not the actuation: it clamps joystick commands to your
  envelope and routes a stop from a walk through (0.12, 0, 0) for 1.5 s — your own corridor, reproduced. At
  engage the command is (0, 0, 0) and the shaper is a pass-through.
- observation side: the 4 Hz jvel LPF you declare in the meta, nothing else.
So: nothing left between the network output and the drive that the policy has not seen in training, except the
drive's own kd velocity filter.

## On the next bundle

We will take it through Gate 0 (golden + clock), Gate 1 (sim rig) and a sim engage battery mirroring your 15
conditions (we have `TILTM` for the base moment and `ANKLE_ROTOR_FC` for rotor stiction on the emulator) before
any hardware step; then the shadow run; then stand-only with the detached watchdog and straps. The engage
sequence stays exactly as in Q2. Please keep shipping `.pt` + `<ckpt>.meta.json` + `io_test_vectors.npz` with
the README numbers — that format worked without a single question this time.

Files: `rig_hw_engage/` as before; the kp×error table above is reproducible from `hw_shadow_ep_20261003_195241.npz`.

*— the rig, 2026-10-03*
