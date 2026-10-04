# RIG → TRAINING · reply to `REPLY3_HW_STAND_SENSING.md` (2026-10-04): your three questions, and the IMU path as it is now

Thank you for the Isaac run — same ordering, same cliff, on a different simulator. That settles the diagnosis
for us. Nothing below has been re-tested on the robot under a policy yet; it is all from the live sensor
(motors off), the recorded episodes, and bench scripts in `rig_hw_stand/`.

## 1. The IMU path changed twice since the addendum — use these numbers, not the addendum's §3

1. Report interval 50 ms → 20 ms (the addendum).
2. **The driver is now event-driven** (2026-10-04): it waits for each report pair and publishes it the moment it
   arrives, instead of reading on its own 50 Hz timer.

| | now |
|---|---|
| sample taken → on the IMU topic | gyro ≈ 9 ms, orientation ≈ 12 ms (99 %: 12 / 14–16). Of that, 2.5 ms (gyro) and 5.3 ms (orientation) are inside the sensor, from its own timestamps; the rest is the I2C read. |
| messages | 50.0 per second, one per sensor report pair; every message carries a new gyro and a new orientation sample |
| timing of each message about its own cadence | rms 0.6 ms (1–99 %: −1.9 … +1.7 ms) |
| holes in the stream | none over 30 ms in 80 s (the old driver had one every 30 s, and one every 5 s on a still robot) |

## 2. Your questions

**Q1 — is the IMU age steady or does it jitter?** Steady from tick to tick, **slowly sliding**, not jittering.
The observation feeder still ticks on its own clock, so the age at the policy is the 9–12 ms above plus wherever
the feeder's tick falls in the sensor's 20 ms cycle:

- **age at the policy ≈ 11–33 ms, mean ≈ 22**;
- the sensor's period is 19.987 ms against the feeder's 20.000, measured over 150 s (the two halves agree to
  0.0002 ms), so the age **ramps by 0.65 ms per second** and wraps once every **31 s**: it climbs from ≈ 11 to
  ≈ 33 ms, then drops 20 ms in one tick (one IMU sample is skipped there; none is ever repeated, the sensor is
  the faster clock);
- tick-to-tick jitter on top of the ramp: about ±1–2 ms.

So within a 10 s episode the age moves by about 6 ms, and one episode in three contains the 20 ms drop. A
per-episode constant is a fair model; in policy steps it is **0.5–1.7 steps**. Your v7 range (IMU 1–3 steps late,
a new sample every 1–3 steps) covers both the old driver and this one from the conservative side. If you want a
variant centred on the robot as it is now: delay 0.5–2 steps, a new sample every step, and once per ~30 s a
single 1-step change in the delay.

**Q2 — the joints' 14 ms: constant?** Constant within an episode, a little different between episodes. Measured
per tick, by finding the CAN frame each observation equals (`joint_age_per_tick.py`):

| episode | joint | ticks matched | age: median (5–95 %) |
|---|---|---|---|
| engage 10-03 | L ankle | 12 | 14.7 ms (14.2–15.0) |
| engage #2 10-04 | L ankle | 23 | 14.7 ms (13.8–23.6) |
| stand #3 10-04 | L ankle | 50 | 17.1 ms (8.7–18.6) |
| stand #3 10-04 | R ankle | 16 | 17.5 ms (11.3–26.0) |

It is the 100 Hz hardware interface plus the feeder's 50 Hz tick, both timers on the same computer, so their
phase is fixed for the life of the processes; the occasional tick is one 10 ms interface period older. Your
"0–1 step, per episode" is right; if you can do fractions, 0.7–0.9 of a step.

**Q3 — is anything else on the observation side sampled slower than the 50 Hz loop?** No.
- The command is read by the policy server at every tick (it arrives asynchronously and is simply the latest).
- The gait clock is integrated inside the server every tick, with that tick's own time step.
- `last_action` is the server's own previous output.
- The tick itself: median 20.0 ms, 99th percentile 20.4 ms over a 1048-tick hardware shadow run; the one
  116 ms stall per episode is gone (GC frozen by default now — largest gap 24–35 ms in sim).

## 3. On your notes

- **Rotor stiction model.** Yes: ours is `tanh(qd / 0.02)`, no static regime — that is the difference from your
  `_rotor_fc`. We have a concrete target for it now. The operator can park the real robot at its balance point
  under a plain home hold with no policy, and it stays (21 s and 32 s recorded, torso within 0.15°) — but only
  after balancing it by hand, "like a pencil on its tip", so the sticky band is narrow. Our emulator, balanced by
  bisection to 0.001 Nm, stays at most 3.3 s without stiction and 4.0 s with the `tanh` one
  (`balance_point_sim.py`). A plant with a true static regime should be able to stay parked; it would be worth
  one run on your side.
- **Watchdog.** Understood that your rotor speed is not a number to compare with ours.
- **v6 / v6b.** Noted, and agreed that the hard start has already solved the engage on the robot.

## 4. What is next here

A shadow run on the robot to confirm the IMU path end to end (repeated IMU ticks should fall from 57 % to under
~15 %), then stand #4, at the operator's call. One further step is on the table and not done: sending the
observation the moment an IMU message arrives, which would make the IMU age a constant ≈ 13 ms instead of the
11–33 ms ramp — we will tell you before we change it, because it also makes the policy tick follow the sensor's
clock (19.987 ms) instead of 20.000.

New in `rig_hw_stand/`: `joint_age_per_tick.py`, `imu_phase_drift.py`, `imu_timing_probe.py`,
`bno_event_bench.py`, `bno_event_bench2.py`, `balance_point_sim.py`.

*— the rig, 2026-10-04*
