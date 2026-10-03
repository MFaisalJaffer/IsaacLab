# Gate 4 — first live engage of walker_v5_3200 on hardware (2026-10-03) — ABORTED BY OPERATOR, cause found and reproduced in sim

**Outcome, stated precisely:** on GO the robot moved violently — left knee to 1.6 rad at 20 Nm, joint
velocities 15–22 rad/s, torso to 18° — and the operator cut motor power at 0.88 s. **The robot did not
fall.** Whether it would have recovered or fallen if left running is unknown. Straps were slack (robot on
its feet); no damage reported.

Evidence: `~/hwlog/20261003_125723/` on the Jetson — raw CAN (`candump`, through the power cut), ROS bag
(joints, raw IMU, LegCmd, controller status), policy episodes `ep_20261003_195937_3742t.npz` (engage, wire
tap 0 drops) and `shadow_ep_20261003_195241_1594t.npz`, `events.jsonl`, bridge/feeder/controller/server logs.

## Timeline (wire tap, 100 Hz)

| t | what the drives were told / did |
|---|---|
| 0–0.3 s | engage crossfade: gains 77→91 (knee), 31→36 (ankle); commanded positions ≈ held home pose. Policy's ankle actions ramp −0.04 → −0.23 → −0.51 (ticks 5/10/15) while the ankle encoder stays at 0.00 (1.2 Nm applied at 0.3 s, no motion). |
| 0.4 s | ankle cmd −0.16 rad, encoder −0.05, 4.1 Nm; hip pitch cmd 0.07, encoder 0.03. Policy ankle action −1.42. Torso still 1.6°. |
| 0.5–0.6 s | ankle encoder −0.19 → −0.35 (body lets go), gyro 2–3 rad/s, policy commands knee bends: L knee cmd +0.37 → measured 1.61 rad at 20.6 Nm, R knee −0.69. |
| 0.88 s | last feedback frame on every node — motor power cut. |

Not the policy's fault in the sense of a bad network, not a sensor fault, not a wiring/order fault
(tick-0 actions on real sensors matched the sim stand joint-for-joint in the shadow run).

## Root cause

The policy integrates on its own `last_action` history whenever the plant does not answer its commands
(offline counterfactual on the logged episode: zeroing last_action in the 10-frame history removes the
wind-up entirely; zeroing velocity noise, joint offsets or levelling gravity changes nothing). Two things
together gave it ~0 % plant response for 0.4 s:

1. **The bridge's 1.0 s engage crossfade** (α-scaled commands at 50–70 % gains) — built in August for
   policies whose first targets jumped; with the hard phase pin this policy's first targets *are* the home
   pose, so the ramp protected nothing and opened a dead window.
2. **The real standing load on the ankles.** From the 30 s shadow hold: ankles carry 0.9 / 1.6 Nm, left hip
   roll 2.3 Nm (right 0.25) at the zero pose — the CoM is a few cm off the ankle axis and more weight sits
   left. Against that load, with the 52 Nm/rad series spring, a small ankle command first winds the spring
   (0.4 s: encoder 3° ≈ 2.7 Nm of spring + preload ≈ the 4 Nm reported) before the body moves at all. The
   sim robot at home carries ≈ 0 Nm, so its body answers immediately.

The IMU-reported lean (1.6–2.5°) is **not** the cause: reproducing it as an IMU offset with a vertical sim
robot (Exp. A) does not wind up. The IMU sits on a tilted bracket and its software zero dates from
2026-08-15, before the ankle refix, so that reading is some mix of real posture and stale reference.

## Sim study (walker_v5_3200, stock rig, HARD START)

| exp | engage ramp | perceived lean | ankle model | standing load | ankle action @ tick 20 | result |
|---|---|---|---|---|---|---|
| HW | 1.0 s | 1.6° (IMU) | real | real (0.9/1.6 Nm ankles) | **−1.42 / +1.25** | violent, power cut 0.88 s |
| A | 1.0 s | 2.5° IMU offset | stock | none | −0.05 / −0.07 | calm |
| B | 1.0 s | IMU offset | + 2° play | none | −0.20 / +0.23 | calm |
| C | 1.0 s | IMU offset | + 3 Nm rotor stiction (`ANKLE_ROTOR_FC`) | none | −0.24 / +0.28 | onset reproduced, ⅕ magnitude, recovers |
| D | **0** | IMU offset | + 3 Nm rotor stiction | none | −0.02 / −0.01 | quiet |
| E | 1.0 s | none (real lean from load) | stock | **+2.5 Nm pitch moment** | **+1.37 / −0.58** | HW wind-up reproduced (mirror direction) |
| E′ | 1.0 s | none | stock | **−2.5 Nm** (HW direction) | **−0.64 / +2.98**, peak −3.2 / +3.0 | full HW signature: hips 1.4, jvel 11.7 rad/s, torso 8.3°; recovered in sim |
| F | **0** | none | stock | −2.5 Nm | −0.11 / +0.03 (peak ±0.37 @ 0.2 s) | quiet; ankle answers 70 % within 0.1 s |

Toy-plant rollout of the policy's own loop: wind-up rate is monotonic in plant response — 0 % runs away,
20 % mild, 50 % settles, 100 % stable. Tooling: `tools/walker/engage_trial*.py`, `engage_summary.py`,
`cf_tracking.py`, `breakaway*.py`, `noisefloor.py`; emulator knob `ANKLE_ROTOR_FC` (default 0).

## Breakaway / stiction from the same data (standing load, ramp-level gains, 0.1° motion threshold)

| joint | preload at HOME | moves at (reported) | error at motion | note |
|---|---|---|---|---|
| L / R hip pitch | 0.18 / 0.33 Nm | 0.8 / 0.55 Nm | 0.2° | little stiction; lag is bandwidth |
| R hip roll | 0.25 Nm | 0.7 Nm | 0.35° | |
| L hip roll | **2.3 Nm** | pushed *away* from command until 2.4–3.4 Nm | 0.6° | robot leans onto left hip at "home" |
| L / R hip yaw | 0.17 / 0.14 Nm | 0.35 / 0.2 Nm | ≤0.4° | |
| L / R knee | 0.21 / 0.47 Nm | no probe (command stayed 0); held ≥0.7 / 0.55 Nm | | |
| L / R ankle | 0.94 / 1.59 Nm | motor 0.6 / 1.2 Nm; **body only at ~4 Nm** | 1–2° | motor motion to 0.4 s = spring wind-up (K_s 52) |

Caveats: one event; gains ramping (35–105), not deploy values; drive-reported (current-based) torque;
upper bounds (joints moved at these torques, may move at less).

## Actions

1. **`--engage-ramp 0` for this lineage** (sim-verified, F). Recommended as the default in `loop_simimu.sh`
   and `loop_realhw.sh`.
2. **Detached watchdog on the Jetson** (nohup): E-STOP at tilt > 12°, any |jvel| > 5 rad/s, or action
   saturation. The session's watchdog died with the interrupted ssh.
3. **Stale-joint gap**: after the power cut ros2_control kept republishing the last values with fresh
   timestamps; the feeder's 0.3 s gate never tripped. Liveness must come from CAN feedback.
4. Bridge `/imu` subscription QoS incompatible with `imu_body_pub` → the gyro quiet-gate has been inert.
5. **IMU zero**: establish torso vertical physically (level on the frame at HOME), then re-zero; current zero
   predates the ankle refix.
6. **To training**: (a) the standing load distribution at the zero pose (ankles ≈ 2.5 Nm total, hip roll
   2.3 L / 0.25 R) — their robot stands balanced at zero, ours does not; (b) actuator dead-band / stiction
   randomization so the policy does not integrate when a request goes unanswered.
7. Passive breakaway ramp per joint at full gains next motors-on session (0.005 rad steps) — gives knees
   and proper numbers.
