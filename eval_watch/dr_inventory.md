# Domain-randomization inventory — kbot_legs_rough training

Everything the policy is randomized over, verified against the live config
(2026-07-12; sources: `velocity_env_cfg.py` → `config/kbot/rough_env_cfg.py` →
`config/kbot_legs/rough_env_cfg.py` → `kscale_legs.py` / `kbot_tv_actuator.py`).
All "per reset" = redrawn per env at every episode reset; "per step" = every control step.

## 1. Body / physical properties
| What | Range | When |
|---|---|---|
| Foot–ground static friction | uniform **[0.4, 1.4]** (64 buckets, feet bodies) | per reset |
| Foot–ground dynamic friction | uniform **[0.3, 1.1]** (kept ≤ static via make_consistent) | per reset |
| Ground restitution | uniform **[0.0, 0.2]** (PhysX analog of MJX contact-compliance DR) | per reset |
| Per-link masses (torso + all leg links, legs-retargeted list) | scale **×[0.8, 1.2]**, inertia recomputed | per reset |
| Base add-mass / base CoM shift | **DISABLED** (root "base" isn't a rigid body in this asset) | — |

## 2. Actuators (per-joint TVCurveActuator, 10 joints)
| What | Range | When |
|---|---|---|
| Command **delay** (positions/velocities/efforts through a DelayBuffer) | uniform integer **0–8 physics steps = 0–40 ms**, drawn per env *per joint-actuator* | per reset (constant within episode) |
| kd (damping) | scale **×[0.75, 1.0]** — DOWN-only, kd ∈ [3.75, 5.0], never above base 5.0 | per reset |
| kp (stiffness) | **NOT randomized** — fixed 150 (deliberate: kp±25% + low-kd draws caused the value-loss-inf crashes) | — |
| Torque–velocity motoring limit | scale **×[0.85, 1.0]** (tv_randomization=0.15; only weaker — hot/worn motor). T-V curves themselves pre-derated ×0.85 for sim2real margin | **per step** |
| Joint armature | scale **×[0.8, 1.2]** | per reset |
| Joint friction | **SILENT NO-OP — verified by probe 2026-07-12**: all 10 joints have base friction 0.0 and the event scales it (0 × anything = 0). The sim is friction/stiction-free at the joints. **QUEUED as next DR rung — values in hand** (rig MJCF: hips/knees 0.0015, yaw 0.001, **ankles 0.1 Nm**): two abs-mode events after the rung-2 milestone — ankles uniform [0, 0.3] Nm, all other joints [0, 0.05] Nm. The rig's 67×-outlier ankle stiction is a prime suspect for the stand limit-cycle (gyro ~1 rad/s, micro-replants). Armature already mirrors the rig exactly and is randomized — no action. | per reset |

## 3. Initial state (every episode)
| What | Range |
|---|---|
| Base XY position / yaw | ±0.5 m / ±π |
| Base linear velocity | x,y ±0.3 m/s; z ±0.1 m/s |
| Base angular velocity | roll/pitch/yaw ±0.2 rad/s |
| Joint positions | default pose **± 0.2 rad** offset (reset_joints_by_offset) |
| Joint velocities | ±1.0 rad/s |

## 4. Sensing
| What | Range | When |
|---|---|---|
| IMU mount position | ±5 cm per axis | per reset |
| IMU mount rotation | ±0.1 rad roll/pitch/yaw | per reset |
| Obs noise (policy group only; uniform, additive): projected gravity ±0.05, joint_pos ±0.05 rad, joint_vel ±0.5 rad/s, imu_ang_vel ±0.1 rad/s | | per step |
| No noise on: velocity commands, last_action, gait_phase. Critic (privileged) group: noise-free | | — |

## 5. External disturbances
| What | Range | When |
|---|---|---|
| Random base pushes (velocity injection, XY) | curriculum-ramped **0.01 → 0.7 m/s** (legs cap; kbot base was 2.0) over iters ~500→5500; **fast-forwarded on resume** via KBOT_RESUME_STEP_OFFSET, so mature runs train at 0.7 constant | every 5–15 s per env |
| base_external_force_torque | present but zero ranges (placeholder, no-op) | — |

## 6. Task / command distribution
| What | Range |
|---|---|
| vx command | ±1.0 m/s |
| vy command | ±0.5 m/s |
| wz command (direct, heading mode OFF) | ±1.0 rad/s |
| Standing envs (cmd = 0) | **20%** of resamples |
| Command resample interval | every 10 s |

## 7. Terrain
Flat plane only (`KBOT_FLAT=1`): terrain-type/roughness randomization and terrain
curriculum are **disabled** in this experiment.

## Deliberately NOT randomized (with reasons)
- **kp** — fixed 150 (crash history; HIL port assumes uniform kp150/kd5).
- **Within-episode delay jitter** — delay is constant per episode; the real path
  varies per packet. Known gap; candidate lever on the stillness ladder.
- **MIT 16-bit quantization** — resolution analysis says negligible; unmodeled.
- **Gait clock frequency** (1.4 Hz), **action scale** (0.5), obs/action layout — part
  of the deploy interface contract with the HIL port; must never drift silently.
