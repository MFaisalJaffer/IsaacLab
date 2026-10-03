# K-Bot Isaac Lab port plan: T-V actuators + legs-only model

Two goals, both about closing the sim-to-real gap that lets the current
Isaac kbot learn "too easily":

1. **Port the velocity-dependent (T-V curve) actuator model** from the MJX
   setup (`ksim_kbot/common.py:TVCurveMITActuators` + `TV_CURVES`) into Isaac.
2. **Add the legs-only kbot** (10 DOF, no arms) as its own Isaac task,
   matching the MJX `kbot-v2-legs` morphology.

---

## Background: what's there now

- Robot cfg: `source/isaaclab_assets/isaaclab_assets/robots/kscale.py`
  - `_JOINT_META` (20 joints): per-joint `kp, kd, torque, vmax, arm`
  - `_ACTUATORS`: one `DelayedPDActuatorCfg` per joint — **flat `effort_limit`**
    (no T-V rolloff), `velocity_limit`, kp/kd, armature, `min_delay/max_delay`.
- Task: `manager_based/locomotion/velocity/config/kbot/` (rough/flat/RNN/LSTM),
  registered as `Isaac-Velocity-Rough-Kbot-v0` etc.
- USD: `robots/temp_kbot_usd/robot.usd` — now regenerated with
  `collision_from_visuals=True` (full-body collision). Termination fixed to
  `bad_orientation` + torso contact (see rough_env_cfg `__post_init__`).

The MJX T-V model (source of truth):
- `ksim_kbot/common.py`:
  - `TV_CURVES`: per-motor-type `(omega[], tau[])` measured curves, motor types
    `"04"` (GIM_8108, peak 18.7 Nm @0.85), `"03"` (GIM_6010, 9.35), `"02"` (same as 03).
    All tau already ×0.85 for sim-to-real margin.
  - `TVCurveMITActuators`: per-step, per-joint
    `max_tau_motoring = interp(|qvel|, omega, tau)`,
    `max_tau_braking = ctrl_clip` (constant),
    `effective = where(sign(ctrl)==sign(qvel), motoring, braking)`,
    then `ctrl = clip(ctrl, -eff, +eff)`. Plus `tv_curve_randomization=0.15`
    (per-step scale in [0.85, 1.0], only down).

Joint→motor-type mapping is the numeric suffix in the joint name:
`*_04` (hip_pitch, knee), `*_03` (hip_roll/yaw, shoulder_pitch/roll), `*_02`
(ankle, shoulder_yaw, elbow), `*_00` (wrist).

> ⚠️ Naming mismatch to resolve: the **full kbot** URDF uses `hip_roll_03`,
> but the **legs-only** MJX robot uses `hip_roll_04`. They are different motor
> assignments — the legs model needs its own joint metadata, not a copy of the
> full-kbot meta.

---

## Part A — Port the T-V curve actuator into Isaac

Isaac actuator options, easiest → most faithful:

1. **`DCMotorActuatorCfg` (built-in, linear approximation).** Isaac's
   `DCMotorActuator` already clamps torque with a *linear* speed-torque
   relationship (`saturation_effort`, `velocity_limit`). Quick swap from
   `DelayedPDActuatorCfg`; gets the qualitative "less torque at speed"
   behavior but not our exact piecewise curve. Good 80% first step.

2. **Custom `TVCurveActuator` (faithful).** Subclass Isaac's
   `DelayedPDActuator` (to keep the action-delay behavior) and override
   `compute()` to apply the piecewise T-V clamp + braking exception +
   per-step randomization, exactly mirroring `TVCurveMITActuators`.

### Recommended: custom `TVCurveActuator`

Steps:
1. New file `source/isaaclab_assets/isaaclab_assets/robots/kbot_tv_actuator.py`:
   - Copy `TV_CURVES` (the omega/tau arrays) from `ksim_kbot/common.py`.
   - `class TVCurveActuatorCfg(DelayedPDActuatorCfg)` with an extra field:
     `motor_type: str` (or per-joint curve arrays), `tv_randomization: float = 0.15`.
   - `class TVCurveActuator(DelayedPDActuator)`: in `compute(control_action,
     joint_pos, joint_vel)`:
       a. call `super().compute(...)` to get the PD torque + apply delay,
       b. build `max_tau_motoring = torch.interp-equivalent(|joint_vel|, omega, tau)`
          (torch has no interp; use `torch.searchsorted` + manual lerp, or
          precompute on a fixed grid and index — vectorized over joints/envs),
       c. `max_tau_braking = effort_limit`,
       d. `eff = where(sign(applied)==sign(joint_vel), motoring, braking)`,
       e. multiply motoring limit by `uniform(1-tv_rand, 1.0)` per step,
       f. `applied = clamp(applied, -eff, eff)`; return.
2. In `kscale.py`, replace the `_ACTUATORS` construction with grouped
   `TVCurveActuatorCfg` — one per motor type (`"04"`, `"03"`, `"02"`, `"00"`),
   `joint_names_expr` regex grouping joints by suffix, each pointing at its curve.
3. Keep `kp/kd` from `_JOINT_META`; the T-V curve replaces the flat
   `effort_limit` as the torque ceiling. Reconcile the torque numbers: the MJX
   curves are the trusted spec — use those tau peaks, not the current 42/11.9.

### Validation
- Add a logging hook (or a one-off eval) that records applied torque vs
  `|joint_vel|` and confirm it tracks the curve (high speed → capped torque).
- Expect training to slow / episode-length to climb more gradually than the
  flat-actuator run — that's the realism cost, as intended.

---

## Part B — Add the legs-only kbot (10 DOF)

Mirror what we did for the full kbot, but for `kbot-v2-legs`.

1. **Convert legs URDF → USD** (full-body collision):
   ```bash
   ./isaaclab.sh -p scripts/tools/convert_urdf.py \
     ksim_kbot/.../kbot-v2-legs/robot.urdf \
     source/isaaclab_assets/.../temp_kbot_legs_usd/robot.usd --headless
   ```
   (convert_urdf.py already patched with `collision_from_visuals=True`.)
   - Verify the legs URDF's collision coverage like we did for the full kbot
     (it likely also only has foot collision → collision_from_visuals handles it).
   - Confirm the root/torso body names for the termination + height-scanner
     prim paths (legs MJCF root was `base`/`Torso...`; check the URDF).

2. **Legs robot cfg** — `kscale_legs.py` (or `KBOT_LEGS_CFG` in kscale.py):
   - 10-joint `_JOINT_META_LEGS`: hip_pitch_04, **hip_roll_04**, hip_yaw_03,
     knee_04, ankle_02 (× L/R). Use the legs-robot motor assignment (note
     hip_roll is `_04` here, not `_03`).
   - `TVCurveActuator` grouped by motor type (reuses Part A).
   - `KBOT_LEGS_USD` path, `_INIT_JOINT_POS` for 10 joints.

3. **Legs env cfg + registration**:
   - `config/kbot_legs/rough_env_cfg.py` (+ flat, +rnn): subclass the velocity
     rough env, set `self.scene.robot = KBOT_LEGS_CFG`, fix:
       - `height_scanner.prim_path` to the legs root body,
       - `bad_orientation` termination (works as-is),
       - contact termination body name = legs torso body,
       - any reward/obs `body_names`/`joint_names` referencing arms (remove),
       - foot body names for air-time / feet rewards (`.*FOOT`).
   - `config/kbot_legs/__init__.py`: register `Isaac-Velocity-Rough-KbotLegs-v0`
     (+ -Play), pointing at the legs env cfg + a legs rsl_rl agent cfg.
   - Action term `joint_names=[".*"]` will resolve to the 10 legs joints
     automatically (verify action dim == 10 in the log).

4. **Launch** (fits alongside full kbot given GPU headroom):
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Velocity-Rough-KbotLegs-v0 --headless --num_envs 4096
   ```

---

## Part C — Domain-randomization alignment (supporting, do with A)

Current Isaac DR has gaps vs MJX; fix in the kbot env `EventCfg`:
- **Friction**: ranges are degenerate `(0.8,0.8)/(0.6,0.6)` → make real, e.g.
  static `(0.4, 1.25)`, dynamic `(0.3, 1.0)` (MJX used 0.1–2.0; pick a sane band).
- **T-V randomization**: handled inside `TVCurveActuator` (±15%, down only).
- Keep existing mass/COM/external-force/push-curriculum/action-delay (already good).
- Optional: joint zero-offset reset noise (MJX `JointZeroPositionRandomizer` ±0.05).

---

## Suggested sequence

1. **Part A first** (T-V actuator on the full kbot) — biggest realism lever,
   reusable by the legs model. Validate torque-vs-speed, re-train, compare how
   much slower/harder it learns.
2. **Part C** alongside A (friction + zero-offset) — cheap, same file.
3. **Part B** (legs-only) — reuses the `TVCurveActuator` from A. Stand up the
   USD + cfg + registration, launch in parallel.

Open question to decide before B: is the **legs-only** robot the deployment
target, or the **full kbot**? That determines which gets the realism investment
first. (MJX work was legs-only; the Isaac task as-shipped is full kbot.)
