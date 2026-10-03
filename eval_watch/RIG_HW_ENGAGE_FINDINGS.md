# RIG → TRAINING · First hardware engage of walker_v5_3200: what happened, what the robot taught us, what to train against

**2026-10-03.** Gates 0 (golden: stacker 825/825, clock 1.5e-4, actions 1.4e-6) and 1 (sim rig: 3×60 s stands,
full walk battery, no falls) passed. First live engage on the robot: **violent motion on GO, operator cut
motor power at 0.88 s.** The robot did not fall; what it would have done next is unknown. Root cause found,
reproduced in sim from measured quantities only, and fixed on our side. Four of the findings belong on your
side. Data in `rig_hw_engage/` beside this note.

## 1. What happened (wire tap, 100 Hz)

| t | |
|---|---|
| 0–0.3 s | our bridge crossfades authority 0→1 over 1.0 s (α-scaled targets, gains 50–70 %). Policy asks the ankles for ~0.04 rad; **the ankle encoder does not move** (1.2 Nm applied). Policy's ankle action: −0.04 → −0.23 → −0.51 (ticks 5/10/15). |
| 0.4 s | ankle target −0.16 rad, encoder −0.05, **4.1 Nm**; torso still 1.6°. Policy ankle action −1.42. |
| 0.5–0.6 s | ankle encoder −0.19 → −0.35, gyro 2–3 rad/s; policy commands knee bends: L knee measured 1.61 rad at 20.6 Nm, joint velocities 15–22 rad/s. |
| 0.88 s | power cut. |

Tick-0 actions on real sensors matched the sim stand joint-for-joint (shadow run), so this is not an
interface, ordering or frame bug.

## 2. Root cause (two ingredients, proved by ablation)

**Mechanism — the policy integrates on its own `last_action` when the plant does not answer.** Offline on
the logged episode: zeroing `last_action` in the 10-frame history removes the wind-up entirely; zeroing
velocity noise, joint offsets or levelling gravity changes nothing. Toy-plant rollout of the policy's own
loop: wind-up rate is monotonic in plant response — 0 % response runs away, 20 % mild, 50 % settles, 100 %
stable (`cf_tracking.py`).

**Why the plant gave ~0 % for 0.4 s — two things together:**

1. *Rig side:* the engage crossfade (now removed for this lineage).
2. *Robot side — the one that matters to you:* **the robot is not balanced at the zero pose.** From the 30 s
   hold before the engage, the drives were carrying, at joint angles = 0 on a flat floor:

   | joint | holding torque at zero pose |
   |---|---|
   | L / R ankle (pitch) | 0.94 / 1.59 Nm |
   | L / R hip roll | **2.32** / 0.25 Nm |
   | L / R hip pitch | 0.18 / 0.33 Nm |
   | L / R hip yaw | 0.17 / 0.14 Nm |
   | L / R knee | 0.21 / 0.47 Nm |

   ≈2.5 Nm at the ankles is the CoM sitting ~2–3 cm off the ankle axis; the hip-roll asymmetry says more
   weight sits on the left. Against that load, through the 52 Nm/rad ankle spring, a small ankle command
   first winds the spring (at 0.4 s: encoder 3° ≈ 2.7 Nm of spring + preload ≈ the 4.1 Nm reported) before
   the body moves at all. The policy sees "ankle moved, gravity unchanged" — and pushes.

**Sim reproduction** (walker_v5_3200, stock rig, nothing tuned but the sign of one number):

| exp | engage ramp | standing load | ankle action @ 0.4 s | result |
|---|---|---|---|---|
| hardware | 1.0 s | real | −1.42 / +1.25 | violent, power cut |
| A | 1.0 s | none; IMU offset 2.5° (perceived lean only) | −0.05 / −0.07 | calm → **the perceived lean is not the cause** |
| E′ | 1.0 s | **−2.5 Nm pitch moment on the torso** | −0.64 / +2.98, peak ±3 | **full hardware signature**: hips 1.4, jvel 11.7 rad/s, torso 8.3°; recovered in sim |
| F | **0** | −2.5 Nm | peak ±0.37 at 0.2 s, decays | quiet — ankle answers 70 % within 0.1 s |

## 3. Breakaway / stiction, every joint, standing (0.1° motion threshold; ramp-level gains)

| joint | moves at (drive-reported torque) | note |
|---|---|---|
| hip pitch | 0.55 / 0.8 Nm | little stiction |
| hip roll | 0.7 Nm (R); L pushed *away* from the command until 2.4–3.4 Nm | L carries 2.3 Nm preload |
| hip yaw | 0.2 / 0.35 Nm | |
| knee | not probed (command stayed 0) — held ≥0.55 / 0.7 Nm without moving | |
| ankle | **motor 0.6 / 1.2 Nm; body only at ~4 Nm** | the "dead zone" is compliance + preload, not friction |

Upper bounds (joints moved at these torques). Our August Step Atlas could not see any of this: 5.7° steps
at full gains blow through every threshold, the robot was suspended (no preload), and `−Fc·tanh(q̇/0.02)`
is zero at rest by construction.

## 4. What to train against — the asks

1. **Standing-load randomization.** Your robot at `default_joint_pos` is balanced; ours is not. Per-episode
   constant external moment on the base, pitch U(−3, +3) Nm and roll U(−3, +3) Nm (or equivalently a CoM
   offset ±3 cm x / ±2 cm y on the torso), held for the whole episode. Numbers above are the measured
   point; the band covers posture/floor/battery changes. This is the ingredient the sim lacked.
2. **Series stiffness.** The meta ships `series_k` = 23 Nm/rad for the ankles as deploy truth. The
   robot's *loaded* ankle is **52 Nm/rad** (measured 08-25 after the refix; `RIG_ANKLE_POSTFIX.md`), ~2°
   free play unloaded closing to ~1° loaded. Please move the deploy value to 52 (band 30–120 if
   randomized; the PA6-CF20 reprint will stiffen it further).
3. **Unanswered-command robustness.** The failure mode is the policy integrating when commands produce no
   response. Two cheap ways to put that in the data: (a) a per-episode output dead band / Coulomb stiction
   in the actuator model — hips/yaw U(0, 1) Nm, ankle motor U(0, 1.5) Nm — and (b) an engage curriculum:
   with some probability, scale the applied gains by U(0.3, 1.0) and ramp to 1.0 over U(0, 1) s after
   reset, so early commands are weak and the policy learns to wait rather than wind up. Your delay DR
   (2–5 steps) does not cover this: a delayed response still arrives; here it never did.
4. **Gravity bias.** IMU-reported lean at HOME is 1.6–2.5° with a software zero that predates the ankle
   refix; we will re-zero against a physical level. Your ±0.05 uniform noise on projected gravity (~±2.9°)
   already covers it — no change asked, just confirming it stays.

## 5. An export-time test you can run in two minutes

`rig_hw_engage/cf_tracking.py`: the policy driven in closed loop against a toy plant that returns a chosen
fraction of each commanded ankle target one tick later, starting from the logged real frame. Report the
ankle action at tick 20 for response fractions 0 / 0.2 / 0.5 / 1.0. For walker_v5_3200: −0.36 / −0.23 /
−0.12 / −0.04 (hardware with the hips joining: −1.42). A candidate whose 0 %-response number stays near
its 100 %-response number is one that will not do what this one did. We will run it on every bundle; it
would be better if you ran it before shipping.

## 6. Rig-side changes already made

Engage crossfade removed for this lineage (`--engage-ramp 0`, sim-verified F); GO refuses while the
controller is latched; a detached watchdog (tilt > 12°, |q̇| > 5 rad/s) is next; liveness from CAN
feedback rather than timestamps (ros2_control kept republishing stale joints after the power cut).

Files: `rig_hw_engage/hw_engage_ep_20261003_195937.npz` (obs27 → frame43 → actions, with CAN wire tap),
`hw_shadow_ep_20261003_195241.npz` (30 s hold, the preload numbers), `sim_Eprime_*.npz`, `sim_F_*.npz`,
`cf_tracking.py`, `breakaway2.py`, `GATE4_hardware_engage_2026-10-03.md` (full write-up).

*— the rig, 2026-10-03*
