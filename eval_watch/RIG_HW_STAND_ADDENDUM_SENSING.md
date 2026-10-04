# RIG → TRAINING · Addendum to `RIG_HW_STAND_FINDINGS.md` (2026-10-04): the foot is cleared — the gap is in the robot's sensing path and loop timing

## 0. Correction first

Suspect #1 in the last note (short foot) is **withdrawn**. The operator measured the real foot: 21.2 cm long,
6.4 cm behind the ankle axis, 14.8 cm in front — identical to the model. Please drop the "short foot" variant
from ask 1 of that note.

## 1. What we then measured in the same hardware episode

All from `rig_hw_stand/hw_stand3_ep_20261004_023018_92t.npz` (policy rows + CAN wire tap, one clock) and a
passive probe of the live IMU. Scripts are in `rig_hw_stand/`.

| path | robot | rig sim | your env (walker_v5 `env.yaml`) |
|---|---|---|---|
| IMU: new samples per second | **20** — our driver enabled the BNO08x at its library default (50 ms report interval) and republished at 50 Hz. The policy got a **bit-identical IMU sample on 57 % of ticks** (holds of 2–3 ticks). `imu_staleness.py` | 50 (fresh every tick) | fresh every step |
| IMU age at the policy input | 20–70 ms, mean ≈ 45 (by construction: 50 ms report + 20 ms driver poll + 20 ms feeder tick; see §4) | 0 | 0 |
| joint pos/vel age at the policy input | **14 ms** behind the CAN bus (median of 8 joints, fit residual ≤ 2.7 %). `obs_latency.py` | 0 | 0 |
| observation received → command on the CAN bus | 18 ms. `act_latency.py` | 18 ms | actuator delay 2–5 × 5 ms |
| policy loop stall | **one 116 ms gap at tick 37 (0.74 s) of every episode** — Python's garbage collector doing a full pass in the policy server (measured 102–106 ms). The last command is held for six ticks. | same server, same stall | none |

We found only the actuator `min_delay 2 / max_delay 5` in your `env.yaml` and no observation delay — correct us
if it lives elsewhere. If that is right, the loop the policy closes through the IMU on the robot is several
times slower than anything it trained on, and stepped at 20 Hz.

In stand #3 the stall landed on a swing extreme: right-ankle action −0.81 held for 116 ms; the first torso
peak (1.8°) is at 0.86 s — the rocking grows out of the stall. Figure: `rig_hw_stand/hw_stand3_imu_hold.png`.

## 2. Sim reproduction (rig MuJoCo, walker_v5_3200, hard start, play 2°, K_s 52)

Sim-only knobs in the policy server: `IMU_HOLD_MS` (the sensor yields a new sample every N ms),
`IMU_DELAY_TICKS`, `JNT_DELAY_TICKS`. "Measured sensing" = IMU held at 20 Hz + 20 ms, joints + 20 ms.
"Fresh IMU" = the same transport delays with a new IMU sample every tick (what our driver fix gives).
Numbers are the fast (> 1 Hz) part only, 15–20 s stands.

| run | result | torso pitch rms | ankle encoder rms L / R | ankle torque p2p L / R |
|---|---|---|---|---|
| **robot, stand #3** | watchdog at 1.9 s | 0.77° | 3.2° / 5.5° | 11.1 / 17.7 Nm |
| stock sensing | stands | 0.07° | 0.5° / 0.7° | 4.2 / 4.6 |
| IMU 20 Hz only | stands | 0.14° | 0.8° / 1.1° | 4.5 / 5.9 |
| measured sensing | stands, rocks at 1.5 Hz | **0.32°** | 2.7° / 3.4° | 11.1 / 12.1 |
| measured sensing + 2 Nm left-heavy | stands, rocks | 0.24° | 2.4° / 3.1° | 10.1 / 11.6 |
| measured sensing + 1 Nm ankle rotor stiction | **watchdog trip in 3 of 5 runs** (5.3, 9.2, 15.8 s; ankle 8.1–9.0 rad/s), 2 stood 20 s | 0.22–0.35° | 2.5–4.0° / 3.1–4.6° | 9–13 / 11–13 |
| measured sensing, IMU pipeline one tick slower (+40 ms) | watchdog at 0.57 s (ankle 12.9 rad/s) | — | — | — |
| fresh IMU | stands | 0.09° | 0.8° / 1.0° | 4.5 / 6.3 |
| fresh IMU + 1 Nm ankle rotor stiction | **stood 20 s in 4 of 4 runs** | 0.19° | 2.1° / 2.7° | 8.5 / 9.7 |
| IMU 40 ms + joints 40 ms (constant) | watchdog at 4.8 s, 1.8 Hz | 0.39° | 4.1° / 4.8° | 11.6 / 12.4 |

Kick test (5 Nm pitch moment for 0.25 s at t = 6 s; `ringdown.py`, successive half-swings in degrees):

| sensing | half-swings after the kick | verdict |
|---|---|---|
| stock | 0.25, 0.14, 0.05, then noise | dies in one swing |
| measured | 0.25, 0.46, 0.36, 0.31, 0.25, 0.28, 0.32, 0.29 … (13 swings above 0.2°) | **sustained at 1.5 Hz** |
| fresh IMU | 0.36, 0.29, 0.23, 0.18, 0.14, 0.09 | decays to the background wobble |

What this does and does not show. The 20 Hz IMU alone moves the sim from a well-damped stand to a sustained
rocking, and with the robot's measured ankle rotor stiction (0.6–1.2 Nm) it reaches the same ending as the
robot — an ankle whip past 8 rad/s. One more 20 ms on the IMU path fails at engage, so the robot was run at
the edge of a cliff. But the sim's rocking is about 40 % of the robot's amplitude and 1.5 Hz against ~2 Hz, so
something else on the robot still adds to it (true hysteretic backlash is the candidate we have not modelled).
We are not claiming the robot will stand once the sensing is fixed; we are claiming it could not have been a
fair test of the policy.

## 3. What the rig changed (applied 2026-10-04, operator-approved; not yet re-tested on the robot)

1. **IMU driver: gyro and game-rotation-vector reports every 20 ms, published at 50 Hz** — a new sample for
   every policy tick instead of 20 per second. Not the 10 ms / 100 Hz we first intended: the sensor's I2C bus runs
   at 100 kHz and the stock driver needs ~5.8 ms per packet, so 10 ms reports collapse its loop (bench in
   `rig_hw_stand/bno_bench.py`). Expected IMU age at the policy ≈ 20–40 ms (was 20–70).
2. **Policy server: `gc.freeze()` after start-up, no automatic collection while a client is connected** —
   largest tick gap 24–35 ms in sim instead of 114.
3. The joint path (14 ms) stays: it is the 100 Hz hardware interface plus the 50 Hz feeder tick.

**Where the cliff is.** With a finer sim knob (IMU age in 2 ms steps, fresh every tick, joints 14 ms old, 1 Nm
ankle rotor stiction, play 2°, 20 s stands):

| IMU age | result |
|---|---|
| 16 ms | 1 stood; 1 tripped at 0.73 s on a right-knee bend at engage (not understood) |
| 20 ms | 4 of 4 stood |
| 26 ms | 2 of 2 stood |
| 36 ms | 2 of 2 tripped (5.4, 11.9 s) |
| 40 ms | 4 of 4 tripped (5–17 s) |

So for this checkpoint the stand survives an IMU that is ≤ ~26 ms old and not one that is ≥ ~36 ms old, and
the robot after our fix sits at about 30. We are looking at an event-driven read in the driver to bring it to
≈ 15 ms. **For training this is the number that matters: a policy that only stands with an IMU fresher than
~30 ms has no margin on this robot.**

## 4. One estimator that failed — do not repeat it

We tried to measure the IMU's lag against the CAN bus from the leg kinematics plus the ankle spring
(`imu_latency.py`). On a sim episode with a true +40 ms IMU delay it returned −18 ms, so the hardware number it
gave is discarded. The IMU age above is structural (report interval and poll periods), not cross-correlated.

## 5. Asks

1. **Observation delay in training**, separate for the two groups, on top of the actuator delay:
   joints 0–1 policy step (0–20 ms); IMU terms (projected gravity, base angular velocity) 1–3 steps (20–60 ms),
   so the robot's ~30 ms sits inside the range rather than at its edge. A sample-and-hold on the IMU terms (new value every 1–3 steps) is the cheap way to cover a slow sensor.
2. **Does 3200 rock in Isaac with the robot's sensing as it was?** IMU terms held at 20 Hz and one step late,
   joints one step late, hard-start stand, with and without ~1 Nm ankle rotor stiction. If Isaac shows the
   sustained rocking too, the diagnosis has a second, independent confirmation — and that configuration becomes
   the stand-stability gate from the last note (1–4 Hz torso oscillation under 1° p2p, not growing), with the
   kick above added.
3. **One held action per episode**: repeat the previous action for 5–6 policy steps at a random time. We are
   removing the known cause, but a policy that survives one missed 100 ms is worth having.
4. Still wanted from the last note: hysteretic ankle backlash 1–2°; ankle rotor friction with a true static
   regime (the table shows it is what tips the 20 Hz case over).

Files added to `rig_hw_stand/`: `imu_staleness.py`, `imu_rate_probe.py`, `obs_latency.py`, `act_latency.py`,
`rock_strength.py`, `ringdown.py`, `bno_bench.py`, `imu_latency.py` (failed, kept for the record),
`hw_stand3_imu_hold.png`,
and the sim episodes behind the table in `sim_latency_eps/`.

*— the rig, 2026-10-04*
