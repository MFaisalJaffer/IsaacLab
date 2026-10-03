"""Gait-phase observation + foot-phase reward for the legs walker (Option 2).

Ports the MJX `FeetPhaseReward` + `TimestepPhaseObservation` (ksim_kbot) into
Isaac Lab. A fixed-frequency gait clock gives each foot an anti-phase target
height (a cubic-bezier bump 0 -> max_foot_height -> 0). The reward matches the
actual foot height to that clock, which DIRECTLY drives an alternating stepping
gait — breaking the stand/sway local optimum that velocity-tracking alone falls
into (it only rewards matching velocity, and swaying is cheaper than stepping).

The phase is also exposed as an observation ([cos, sin] per foot) so the policy
can time its steps. This grows the policy obs by 4 (39 -> 43), so a model trained
with this CANNOT resume a 39-d checkpoint — it needs a fresh run.

Reference (same Bezier profile as MJX / mujoco_playground gait.py):
    x in [0,1] from phase; height = smoothstep bump peaking at swing height.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _phase(env: "ManagerBasedRLEnv", gait_freq: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env left/right foot phase in [-pi, pi], anti-phase (trot)."""
    steps = env.episode_length_buf.float()                 # (N,) resets to 0 on episode reset
    phase_dt = 2.0 * math.pi * gait_freq * env.step_dt
    base = steps * phase_dt
    phi_l = torch.remainder(base + math.pi, 2.0 * math.pi) - math.pi          # start phase 0
    phi_r = torch.remainder(base + 2.0 * math.pi, 2.0 * math.pi) - math.pi    # start phase pi
    return phi_l, phi_r


def _freq_of_cmd(cmd_vx_abs: torch.Tensor, freq_map: tuple) -> torch.Tensor:
    """SPEED-ADAPTIVE gait frequency (2026-08-04, user design): humans co-modulate
    cadence AND stride with speed; a fixed clock forces stride = v/f to absorb all
    speed change (the coupling behind the 1.15 Hz hip_pitch blowup, the isometric
    0.30-stride pull, and the 36-deg lean crutch at 1.0 Hz slow-support). Linear
    map between measured-viable anchors: f = f_min at v <= v_lo, f_max at
    v >= v_hi. Deterministic in the COMMAND (policy observes cmd -> no new obs,
    no Goodhart surface; MJX heritage COMMANDED f over 1.25-1.5, so this is the
    source design plus a speed law)."""
    f_min, f_max, v_lo, v_hi = freq_map
    t = ((cmd_vx_abs - v_lo) / max(v_hi - v_lo, 1e-6)).clamp(0.0, 1.0)
    return f_min + (f_max - f_min) * t


def _phase_adaptive(
    env: "ManagerBasedRLEnv", freq_map: tuple, command_name: str = "base_velocity"
) -> tuple[torch.Tensor, torch.Tensor]:
    """INTEGRATED per-env anti-phase clock whose rate follows the commanded speed.

    The stateless _phase (phase = steps * 2pi*f*dt) cannot express a time-varying
    f — changing f would rewrite the whole history and JUMP the phase. This
    integrates instead: phi += 2pi * f(|cmd_vx|) * dt once per env step, with a
    per-env buffer stored on the env. Matches _phase's conventions exactly at
    constant f: same start phases (L=0 mid-cycle ref, R=pi), same free-running
    behavior at stand (consumers pin their own outputs, as today).
    """
    st = getattr(env, "_kbot_adaptive_phase", None)
    if st is None:
        st = {"phi": torch.zeros(env.num_envs, device=env.device), "step": -1}
        env._kbot_adaptive_phase = st
    step = int(env.common_step_counter)
    if step != st["step"]:                       # integrate ONCE per env step
        st["step"] = step
        cmd = env.command_manager.get_command(command_name)
        f = _freq_of_cmd(cmd[:, 0].abs(), freq_map)
        st["phi"] = st["phi"] + 2.0 * math.pi * f * env.step_dt
        # fresh episodes restart the clock (mirrors the stateless steps-based start)
        st["phi"] = torch.where(
            env.episode_length_buf <= 1, torch.zeros_like(st["phi"]), st["phi"]
        )
    base = st["phi"]
    phi_l = torch.remainder(base + math.pi, 2.0 * math.pi) - math.pi
    phi_r = torch.remainder(base + 2.0 * math.pi, 2.0 * math.pi) - math.pi
    return phi_l, phi_r


def _phase_signed(
    env: "ManagerBasedRLEnv", gait_freq: float, command_name: str = "base_velocity", back_threshold: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """INTEGRATED anti-phase clock that runs BACKWARD while the commanded forward speed is negative.

    Walker v4 (2026-10-02): the multi-cycle tracker learned backward walking as the forward stride with
    the clock reversed (a forward-running clock demanded opposite leg motions for the same clock value and
    was never learned). The walker and the robot must feed the policy the same clock: phi += dir * 2 pi f dt
    with dir = -1 when cmd_vx < -back_threshold, else +1. Same start phases and wrap as _phase.
    """
    st = getattr(env, "_kbot_signed_phase", None)
    if st is None:
        st = {"phi": torch.zeros(env.num_envs, device=env.device), "step": -1}
        env._kbot_signed_phase = st
    step = int(env.common_step_counter)
    if step != st["step"]:                       # integrate ONCE per env step
        st["step"] = step
        cmd = env.command_manager.get_command(command_name)
        direction = torch.where(cmd[:, 0] < -back_threshold, -1.0, 1.0)
        st["phi"] = st["phi"] + direction * 2.0 * math.pi * gait_freq * env.step_dt
        st["phi"] = torch.where(env.episode_length_buf <= 1, torch.zeros_like(st["phi"]), st["phi"])
    base = st["phi"]
    phi_l = torch.remainder(base + math.pi, 2.0 * math.pi) - math.pi
    phi_r = torch.remainder(base + 2.0 * math.pi, 2.0 * math.pi) - math.pi
    return phi_l, phi_r


def _swing_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Smooth (cubic-bezier / smoothstep) foot-height bump.

    0 at phi = +-pi (foot planted), peaks at `swing_height` at phi = 0 (mid-swing).
    """
    x = ((phi + math.pi) / (2.0 * math.pi)).clamp(0.0, 1.0)

    def _bez(p0: float, p1: torch.Tensor | float, t: torch.Tensor) -> torch.Tensor:
        b = 3.0 * t**2 - 2.0 * t**3                        # smoothstep
        return p0 + (p1 - p0) * b

    up = _bez(0.0, swing_height, 2.0 * x)                  # first half: rise 0 -> swing
    down = _bez(swing_height, 0.0, 2.0 * x - 1.0)          # second half: fall swing -> 0
    return torch.where(x <= 0.5, up, down)


def gait_phase_obs(
    env: "ManagerBasedRLEnv",
    gait_freq: float = 1.4,
    freq_map: tuple | None = None,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    signed: bool = False,
) -> torch.Tensor:
    """Observation: [cos phi_l, sin phi_l, cos phi_r, sin phi_r], (N, 4).

    signed=True (walker v4): the clock runs backward while cmd_vx < 0 (see _phase_signed).

    When the commanded velocity is ~zero, the phase is pinned to a fixed
    standing pose so the policy gets a consistent "stand" signal.

    Pin = (pi, pi): BOTH feet at the "planted" phase. The old MJX-derived pin
    (pi/2, pi) was measured to drive stand fidget: pi/2 is the mid-swing-descent
    phase, and via the anti-phase walking prior it reads as "right foot rising"
    -> the 69k build tapped its RIGHT foot 385/min at stand (L only 82/min).
    (pi, pi) is also mirror-INVARIANT (swap L<->R gives the same obs), so the
    rsl_rl mirror-symmetry loss forces a symmetric stand action: lifting a
    single foot at stand becomes structurally impossible from a symmetric state,
    and both-lift = flight is penalized -> both-planted is the stable optimum.
    """
    phi_l, phi_r = (_phase_adaptive(env, freq_map, command_name) if freq_map
                    else (_phase_signed(env, gait_freq, command_name) if signed else _phase(env, gait_freq)))
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    standing = cmd_norm < stand_still_threshold
    # ANNEALED PIN ENTRY (2026-08-13, user-approved): the hard snap to (pi,pi)
    # was the third cliff of this campaign — an OBS discontinuity at the stand
    # switch (the gait drumbeat stopping dead) that the from-scratch policy
    # only ever experienced while dying; probes: upright 1.02 m through the
    # corridor, collapse to 0.17 m within 1 s of the snap. Lineage 1 stood
    # under the SAME pin, but learned it when height-collapse was survivable
    # (0.55 floor added later) — the pin is learnable, the SNAP into it under
    # lethal rules is not. So: glide each foot's phase to pi along the
    # shortest arc over anneal_s after the (corridor-published) stand onset;
    # s=1 -> exactly the proven (pi,pi) contract. Envs standing WITHOUT a
    # corridor onset (probes pinning flags directly; pre-corridor flicker)
    # get s=1 = the old hard pin, so eval semantics are backward-compatible.
    # HIL amendment (lineage-2 builds): blend to pi over ~1 s after stand
    # onset instead of snapping — end state unchanged.
    anneal_s = 1.0
    onset = getattr(env, "_stand_onset_time", None)
    if onset is not None:
        now = env.episode_length_buf.float() * env.step_dt
        s = ((now - onset) / anneal_s).clamp(0.0, 1.0)
        s = torch.where(onset > -1.0e5, s, torch.ones_like(s))
    else:
        s = torch.ones_like(cmd_norm)
    two_pi = 2.0 * math.pi
    d_l = torch.remainder(math.pi - phi_l + math.pi, two_pi) - math.pi
    d_r = torch.remainder(math.pi - phi_r + math.pi, two_pi) - math.pi
    pin_l = torch.remainder(phi_l + s * d_l + math.pi, two_pi) - math.pi
    pin_r = torch.remainder(phi_r + s * d_r + math.pi, two_pi) - math.pi
    phi_l = torch.where(standing, pin_l, phi_l)
    phi_r = torch.where(standing, pin_r, phi_r)
    return torch.stack([torch.cos(phi_l), torch.sin(phi_l), torch.cos(phi_r), torch.sin(phi_r)], dim=1)


def _cmd_track_gate(
    env: "ManagerBasedRLEnv",
    asset,
    command_name: str,
    lin_sensitivity: float,
    ang_sensitivity: float = 0.5,
) -> torch.Tensor:
    """Multiplicative gate in (0,1]: ~1 when the robot tracks the COMMANDED velocity
    (linear XY in the heading frame + angular yaw), ->0 when it ignores a commanded
    component. Uncommanded components (=0) don't gate. Stops the policy from FARMING
    the gait rewards by stepping/marching in place when told to move OR turn (e.g.
    commanded to turn left but it just runs feet_phase for reward without turning)."""
    cmd = env.command_manager.get_command(command_name)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_err = torch.sum((cmd[:, :2] - vel_yaw[:, :2]) ** 2, dim=1)
    yaw_err = (cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]) ** 2
    gate_lin = torch.exp(-lin_err / lin_sensitivity)
    gate_ang = torch.exp(-yaw_err / ang_sensitivity)
    gate_lin = torch.where(torch.norm(cmd[:, :2], dim=1) > 1e-3, gate_lin, torch.ones_like(gate_lin))
    gate_ang = torch.where(torch.abs(cmd[:, 2]) > 1e-3, gate_ang, torch.ones_like(gate_ang))
    return gate_lin * gate_ang


def feet_phase_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    gait_freq: float = 1.4,
    freq_map: tuple | None = None,
    max_foot_height: float = 0.12,
    foot_offset: float = 0.05,
    sensitivity: float = 0.01,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    translation_gated: bool = True,
    translation_gate_sensitivity: float = 0.25,
) -> torch.Tensor:
    """exp(-||foot_z - ideal_z||^2 / sensitivity), gated by a non-zero command.

    foot_z is the foot body height above ground (world z minus the planted
    foot-body offset). ideal_z is the gait-clock target per foot. asset_cfg
    must resolve to the two foot bodies (L/R order is irrelevant — anti-phase
    is symmetric).

    If ``translation_gated`` (MJX `translation_gated`), the reward is multiplied
    by a Gaussian on the body's actual XY velocity tracking error (yaw frame).
    The policy then only earns the gait-clock reward when it is ALSO translating
    in the commanded direction — killing the "march/circle in place" optimum.
    """
    asset = env.scene[asset_cfg.name]
    phi_l, phi_r = (_phase_adaptive(env, freq_map, command_name) if freq_map
                    else _phase(env, gait_freq))
    ideal_l = _swing_height(phi_l, max_foot_height)
    ideal_r = _swing_height(phi_r, max_foot_height)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_offset      # (N, 2)
    err = (foot_z[:, 0] - ideal_l) ** 2 + (foot_z[:, 1] - ideal_r) ** 2
    reward = torch.exp(-err / sensitivity)

    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    reward = reward * (cmd_norm > stand_still_threshold).float()

    if translation_gated:
        # gate by tracking of the FULL command (linear XY + angular yaw) so the
        # policy can't earn feet_phase by stepping in place when told to move OR turn.
        reward = reward * _cmd_track_gate(env, asset, command_name, translation_gate_sensitivity)

    return reward


def flight_phase_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize a FLIGHT PHASE: timesteps where NO foot is in contact (both feet
    airborne) — the defining feature of a hop/run. A walk always keeps >=1 foot on
    the ground, so this returns 0 for a clean walking gait and 1 whenever the robot
    leaves the ground entirely. Apply with a NEGATIVE weight.

    Together with feet_phase_reward (which rewards each foot LIFTING in anti-phase),
    this pushes the policy toward the walking ideal: one foot swings up while the
    other stays planted — instead of bouncing both feet off the ground.

    Contact uses the max net-force over the sensor's short history (same idiom as
    feet_slide) so a single-substep gap doesn't false-trigger. Gated by a non-zero
    velocity command so a settling/standing robot is never penalized.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > force_threshold
    )  # (N, n_feet) bool
    airborne = (in_contact.sum(dim=1) == 0).float()  # (N,) 1 when no foot is down
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    return airborne * (cmd_norm > stand_still_threshold).float()


def feet_alternation_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    gait_freq: float = 1.4,
    freq_map: tuple | None = None,
    step_separation: float = 0.30,
    sensitivity: float = 0.03,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    speed_scale_sep: bool = False,
    sep_min: float = 0.10,
    sep_max: float = 0.30,
) -> torch.Tensor:
    """Reward the FORE-AFT foot separation tracking an anti-phase step clock, so
    the legs must ALTERNATE which one leads.

    feet_phase rewards alternating foot HEIGHT only (it is blind to fore-aft
    position), which a 'one-leg-always-front' vertical-bob shuffle can fully
    satisfy. This adds the missing dimension: the foot that is lifting on the
    feet_phase clock must ALSO be swinging forward, while the planted one sweeps
    back -> a real alternating step.

    sep = foot0_x - foot1_x along the forward axis (heading frame); the ideal
    swings +/-step_separation in anti-phase, using the SAME body order + phi_l as
    feet_phase so height and fore-aft stay aligned (lift-at-back -> swing-forward
    -> plant-at-front). Centered at 0 by L/R symmetry, so it doesn't fight the
    robot's neutral stance offset. Gated by a non-zero command.
    """
    asset = env.scene[asset_cfg.name]
    phi_l, _ = (_phase_adaptive(env, freq_map, command_name) if freq_map
                else _phase(env, gait_freq))
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]            # (N, 2, 3)
    rel = foot_pos_w - asset.data.root_pos_w[:, None, :]                    # (N, 2, 3)
    yq = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, rel.shape[1], -1)
    foot_rel = quat_apply_inverse(yq.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)
    sep = foot_rel[:, 0, 0] - foot_rel[:, 1, 0]                             # forward sep, foot0 - foot1
    cmd = env.command_manager.get_command(command_name)
    if speed_scale_sep:
        # STRIDE-SHORTENING lever (2026-07-29): the fixed 0.30 m target demands
        # the SAME leg splay at every speed — ~2.8x the natural stride v/f at
        # cmd 0.15, which is exactly where hip_pitch runs hottest (10.5 rms vs
        # 8.6 @0.50: quasi-static splay torque, the inverse of the coupling
        # that killed the cadence lever). Scale the demanded separation with
        # the commanded speed: target = clamp(|cmd_vx|/f, sep_min, sep_max) —
        # legs stay near-vertical at low speed, unchanged at high speed.
        _f = _freq_of_cmd(cmd[:, 0].abs(), freq_map) if freq_map else gait_freq
        target = (cmd[:, 0].abs() / _f).clamp(min=sep_min, max=sep_max)
    else:
        target = step_separation
    ideal = target * torch.sin(phi_l)                                       # +sep: foot0 leads
    reward = torch.exp(-((sep - ideal) ** 2) / sensitivity)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    reward = reward * (cmd_norm > stand_still_threshold).float()
    # gate by command tracking (lin+ang) — no fore-aft-step reward for marching in place
    return reward * _cmd_track_gate(env, asset, command_name, 0.6)


def knee_swing_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    gait_freq: float = 1.4,
    freq_map: tuple | None = None,
    flex: float = 0.55,
    sensitivity: float = 0.1,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward the KNEE flexing during swing and extending at stance, on the gait
    clock. The walker otherwise locks its knees (~0 deg of motion) and clears the
    foot by hip-yaw circumduction; this gives a positive pull to BEND the knee
    instead. Target = +/-flex * swing-bump(phi) (the same 0->1->0 Bezier as the
    height clock), so the knee is most bent at mid-swing and straight at stance.

    Left/right knees flex with OPPOSITE sign (the URDF joint axes are mirrored:
    the bent-knee pose is L=+50, R=-50 deg), so the target is +flex for the left
    knee and -flex for the right. Gated by a non-zero command.
    """
    asset = env.scene[asset_cfg.name]
    names = asset.data.joint_names
    li = next(i for i, n in enumerate(names) if "knee" in n and "left" in n)
    ri = next(i for i, n in enumerate(names) if "knee" in n and "right" in n)
    phi_l, phi_r = (_phase_adaptive(env, freq_map, command_name) if freq_map
                    else _phase(env, gait_freq))
    bump_l = _swing_height(phi_l, 1.0)                  # 0->1->0 over the swing
    bump_r = _swing_height(phi_r, 1.0)
    knee_l = asset.data.joint_pos[:, li]
    knee_r = asset.data.joint_pos[:, ri]
    err = (knee_l - flex * bump_l) ** 2 + (knee_r + flex * bump_r) ** 2   # L +, R -
    reward = torch.exp(-err / sensitivity)
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :3], dim=1)
    # NOT command-track gated (unlike feet_phase/alternation): knee flexion is a gait
    # MECHANIC that should happen whenever stepping. The ang-tracking gate crushed
    # this to ~0 on turn commands (poor yaw tracking) so the knees never developed
    # and locked straight. Only gate on "is commanded to move at all".
    return reward * (cmd_norm > stand_still_threshold).float()


def _push_release(env: "ManagerBasedRLEnv", gate: torch.Tensor) -> torch.Tensor:
    """RELIEF COMPLETION (2026-08-09, reward-conflict audit): zero a statue-
    shaping gate while a sustained burst is ON that env. The audit found SEVEN
    stand penalties (stand_upright -15, flat_orientation -3, stand_gyro -2.5,
    brace_ema -2.5, stand_action_rate/-magnitude -1, joint_deviation_hip -0.5)
    all live during pushes — collectively out-bidding the +3.0 push_brace
    bonus and taxing every exploratory twitch toward the correct response the
    moment it starts. Same exogenous event mask as the stand_pose relief, so
    the can't-Goodhart property carries: quiet stand pays full price.
    `gate` may be bool or float; returns float."""
    pushed = getattr(env, "_sustained_push_active", None)
    g = gate.float() if gate.dtype == torch.bool else gate
    if pushed is None:
        return g
    return g * (~pushed).float()


def _push_or_hold_release(env: "ManagerBasedRLEnv", gate: torch.Tensor) -> torch.Tensor:
    """MOTION-tax release for pushes AND tilt-servo holds (2026-08-16, rig
    report): during a hold, the CORRECTION movement must be untaxed (gyro,
    action-rate, stillness, pose terms) — but the TILT terms (stand_upright,
    flat_orientation, stand_tilt_wall) deliberately do NOT use this gate: they
    stay live during holds, which is the entire point of the hold."""
    g = _push_release(env, gate)
    held = getattr(env, "_tilt_hold_active", None)
    if held is None:
        return g
    return g * (~held).float()


def _push_or_hold_soften(env: "ManagerBasedRLEnv", gate: torch.Tensor, floor: float = 0.25) -> torch.Tensor:
    """PARTIAL relief (lineage 5, 2026-08-18 — hardware jerk report): scale a
    gate DOWN to `floor` during bursts/holds instead of zeroing it. The full
    release taught corrections that cost nothing to be abrupt — the rig
    measured 'no stabilization below ~3 deg, then jerky commands all at once
    past 4 deg'. With a floor, corrections stay affordable (75% discount) but
    a smooth ramp now beats a slam. Used by the ACTION-smoothness taxes only;
    outcome taxes (gyro, joint motion) keep the full release — they would
    fight the correction itself, not its style."""
    g = gate.float() if gate.dtype == torch.bool else gate
    pushed = getattr(env, "_sustained_push_active", None)
    held = getattr(env, "_tilt_hold_active", None)
    relieved = torch.zeros_like(g, dtype=torch.bool)
    if pushed is not None:
        relieved = relieved | pushed
    if held is not None:
        relieved = relieved | held
    return g * torch.where(relieved, torch.full_like(g, floor), torch.ones_like(g))


def _rehome_scale(env: "ManagerBasedRLEnv", factor: float = 0.25) -> torch.Tensor | float:
    """RE-HOME GRACE (2026-08-16, lineage 4 — pose_rehome_probe verdict): for a
    short window after a burst/hold ENDS, soften the standing motion taxes so
    the corrective step back to the default stance is cheap. The probe showed
    the policy survives pushes but PARKS in the deviated stance (width ratchets
    36 -> 46 -> 51 cm across pushes, flat for 10 s): the way home is a step,
    and the step was fully taxed the moment the burst relief switched off.
    stand_stance_geometry stays LIVE during the grace — it is the gradient
    home; this scale only discounts the taxes on traveling. Published by
    sustained_push_bursts as env._rehome_until (per-env episode seconds)."""
    ru = getattr(env, "_rehome_until", None)
    off = getattr(env, "_stance_off_nominal", None)
    if ru is None and off is None:
        return 1.0
    if ru is None:
        return torch.where(off, torch.full_like(off, factor, dtype=torch.float),
                           torch.ones(off.shape, device=off.device))
    now = env.episode_length_buf.float() * env.step_dt
    discount = now < ru
    if off is not None:
        # QUIET RE-HOME (2026-08-22, hardware observation): the feet also need
        # an affordable way home when the stance drifted WITHOUT a push. The
        # post-burst window alone left slow scuffing as the only cheap path
        # (measured: ~10 cm of planted-foot travel per 5 s quiet hold).
        discount = discount | off
    return torch.where(discount, torch.full_like(ru, factor), torch.ones_like(ru))


def stand_still_joint_motion(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint motion while commanded to STAND (cmd ~ 0) so the robot holds
    still instead of fidgeting (measured ~15 deg ankle jitter at ~380 deg/s while
    standing). Returns sum(joint_vel^2), active ONLY when not commanded to move.

    Apply with a NEGATIVE weight. This is a PENALTY, not a stand bonus, so it does
    NOT recreate the stand-still attractor: it costs only when moving-while-standing
    (zero when actually still) and switches off the instant a move is commanded, so
    it never discourages starting to walk.
    """
    asset = env.scene[asset_cfg.name]
    motion = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return motion * _push_or_hold_release(env, standing) * _rehome_scale(env)


def stand_body_gyro(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    lin_coeff: float = 0.5,
) -> torch.Tensor:
    """Penalize BODY motion (base angular velocity + a smaller linear-sway term)
    while commanded to STAND. Targets the HIL-measured stillness gap: on the real
    control path the stand ran |gyro| mean 1.07 rad/s (vs ~calm in Isaac).

    CRITICAL DISTINCTION vs the failed 'least-motion' experiments: this penalizes
    the torso-motion OUTCOME, NOT joint/ankle velocity. Ankle corrections are the
    balance mechanism (penalizing them made the robot fall in ~1 s — measured);
    this term leaves the ankles entirely free and only asks that their work keep
    the BODY quiet. Pairs with 0-40 ms delay DR: the delay makes sim standing as
    'busy' as the rig, and this term demands calmness under those conditions,
    which is what transfers. Penalty gated to cmd~0 -> no stand-still attractor.
    """
    asset = env.scene[asset_cfg.name]
    ang = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    lin = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return (ang + lin_coeff * lin) * _push_or_hold_release(env, standing)


def push_by_setting_velocity_cmd_scaled(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    velocity_range: dict,
    standing_scale: float = 0.35,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push event with command-aware magnitude: STANDING envs get pushes scaled by
    ``standing_scale`` (realistic nudges), MOVING envs get the full curriculum push.

    WHY (measured 2026-07-13): after the curriculum fast-forward fix, standing envs
    were shoved at the full 0.7 m/s every 5-15 s and stand stillness REGRESSED 3x
    (lifts 43 -> 110/min, both-down 97 -> 91%) — the stand penalties punish exactly
    the recovery motion the shoves force, so the policy adopted a permanently busy
    stance. Every historically 'still' checkpoint was secretly trained under the
    re-ramp bug's GENTLE pushes. This keeps full-strength pushes for walking
    (robustness) while standing practice gets bump-scale disturbances.
    The push curriculum keeps writing ``velocity_range`` on this event's params —
    the split is applied on top of whatever the curriculum sets.
    """
    from isaaclab.envs.mdp.events import push_by_setting_velocity

    cmd = env.command_manager.get_command(command_name)[env_ids]
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    moving_ids = env_ids[~standing]
    standing_ids = env_ids[standing]
    _q = getattr(env, "_quiet_stand", None)      # LINEAGE 10: see draw_quiet_stand
    if _q is not None and len(standing_ids) > 0:
        standing_ids = standing_ids[~_q[standing_ids]]
    if len(moving_ids) > 0:
        push_by_setting_velocity(env, moving_ids, velocity_range, asset_cfg)
    if len(standing_ids) > 0:
        scaled = {k: (v[0] * standing_scale, v[1] * standing_scale) for k, v in velocity_range.items()}
        push_by_setting_velocity(env, standing_ids, scaled, asset_cfg)


def stand_action_rate(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize CHANGES in the commanded action while standing — the 'statue-maker'.

    The rig's own hold phase proves this robot stands at ~zero gyro when its PD
    targets are FROZEN; the policy fidgets because it re-decides targets @50 Hz
    (partly chasing joint-vel obs noise). This term teaches: when undisturbed,
    converge to constant actions and let the PD's passive stiffness do the holding.
    CRUCIALLY different from the fatal joint-velocity penalties: commands freeze,
    JOINTS stay free — the PD still balances around the fixed target. Gated to
    cmd~0 and a penalty, so push recovery still pays (falling costs -150; moving
    the commands to recover costs pennies). Measured motivation @152k: rig gyro
    trend 0.8 -> 1.05 -> 1.2 as friction-DR vigor grew — vigor calibration isn't
    the binding lever, command jitter is."""
    da = env.action_manager.action - env.action_manager.prev_action
    motion = torch.sum(torch.square(da), dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    # L5: partial relief — jerk always costs something (see _push_or_hold_soften)
    return motion * _push_or_hold_soften(env, standing, 0.25) * _rehome_scale(env)


def stand_foot_slide(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet SLIDING while in contact at STAND — closes the skating exploit.

    Measured on the statue build (171500): body statue-still (gyro 0.1) but the
    planted feet skate ~15 cm/s (4+ m cumulative per 30 s stand; worst envs slide
    outward 9 cm / 10.7 m skate) — user-visible as feet sliding outwards to brace.
    Enabled by: stand_action_rate makes CONSTANT outward pressure free, and the
    global feet_slide (-0.1) prices slow slip at ~nothing. This term makes planted
    mean PLANTED at stand: xy speed of in-contact feet, gated cmd~0. Distinct from
    the fatal joint-vel penalties (slip isn't the balance mechanism; a gripping
    foot balances better) and from foot LIFTS (stand_feet_planted)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = (body_vel.norm(dim=-1) * contacts.float()).sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return slide * standing


def stand_upright(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize TILT (projected-gravity xy, squared) while commanded to STAND.

    Companion to stand_body_gyro: measured @105k that penalizing only body
    ROTATION RATE lets the optimizer adopt a rigid static LEAN (gyro-free but
    tilted 9-10 deg in sim AND rig). This term prices the lean itself. Gated to
    cmd~0 so walking dynamics (which legitimately lean) are untouched; the global
    flat_orientation_l2 (-3.0) stays as the mild always-on term."""
    asset = env.scene[asset_cfg.name]
    tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return tilt_sq * _push_release(env, standing)


def stand_still_feet_lift(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    force_threshold: float = 1.0,
    vel_release_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize feet LEAVING the ground while commanded to STAND (cmd ~ 0) and CALM.
    Returns the number of feet not in contact (0 when both planted), active only when
    commanded to stand. Directly stops the 'step in place' — the walking gait leaked
    into standing (feet planted only ~30% of the time) because in-place stepping
    doesn't translate, so track_lin_vel didn't penalize it. Apply with NEGATIVE weight.

    CALM-GATE (2026-07-22, lineage-2 @40k): released when the base is MOVING
    (>vel_release_threshold) — a genuine push. Pairs with the stand_pose calm-gate:
    together they free the lateral CATCH-STEP during a real perturbation (the only
    lateral recovery this ankle-roll-less robot has besides hip-roll bracing), so the
    policy is no longer cornered into a rigid brace. Quiet-stand fidget (base calm)
    still pays full price.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > force_threshold
    )  # (N, n_feet)
    lifted = (~in_contact).sum(dim=1).float()  # feet off the ground
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    calm = asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_release_threshold
    # EVENT-GATE (2026-08-07): also release while a sustained burst is ON this
    # env. The calm-gate only opens ~100 ms in (once v>0.15); the probe showed
    # the 0-100 ms PREVENTION window stayed taxed, so early protective steps
    # were priced and the policy stepped only when already doomed.
    pushed = getattr(env, "_sustained_push_active", None)
    if pushed is not None:
        calm = calm & ~pushed
    held = getattr(env, "_tilt_hold_active", None)
    if held is not None:
        calm = calm & ~held
    return lifted * (standing & calm).float() * _rehome_scale(env)


def stand_pose_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    sensitivity: float = 0.5,
    vel_coeff: float = 0.01,
    vel_release_threshold: float = 0.15,
    torque_deadband: float = 12.0,
    torque_gate_tau: float = 4.0,
) -> torch.Tensor:
    """POSITIVE reward for holding the default (straight-leg) pose AND being STILL when
    commanded to STAND (cmd ~ 0) AND calm: exp(-(||q - q_default||^2 +
    vel_coeff*||qdot||^2) / sensitivity). Rewards the full stand TARGET directly
    (planted default pose + least motion) so the policy learns a quiet stand natively.

    The position-only version got the feet PLANTED (84-91% both-down) but left the
    ankles vibrating ~360 deg/s (it doesn't see velocity). Adding the velocity term to
    the SAME positive target — learned from a fresh run — makes 'still' the target
    rather than slamming a velocity penalty onto a jittery policy (that broke the plant).
    Gated to cmd~0, so no stand-still attractor / never discourages walking.

    CALM-GATE (2026-07-22, lineage-2 @40k): also released when the base is actually
    MOVING (>vel_release_threshold) — a genuine lateral push/excursion. Rationale:
    the robot has NO ankle-roll DOF, so its only lateral balance authority is hip
    roll or STEPPING; but this reward (demanding q=default) plus stand_feet_planted
    penalized the catch-STEP during a push, cornering the policy into a rigid
    hip-roll BRACE (the only push-resistance left). Releasing the stand-hold reward
    during a real perturbation makes catch-stepping FREE exactly when needed, while
    quiet standing still gets the full gradient. Releasing a REWARD (not a penalty)
    during motion can't be Goodharted — the policy still WANTS the bonus so it
    returns to a quiet stand to collect it. POSITIVE weight.
    """
    asset = env.scene[asset_cfg.name]
    pos_err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    vel_err = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    reward = torch.exp(-(pos_err + vel_coeff * vel_err) / sensitivity)
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    calm = asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_release_threshold
    # EVENT-GATE (2026-08-07): release while a sustained burst is ON this env —
    # same rationale as stand_feet_planted (see there); frees the 0-100 ms
    # prevention window the velocity calm-gate cannot see.
    pushed = getattr(env, "_sustained_push_active", None)
    if pushed is not None:
        calm = calm & ~pushed
    # TILT-HOLD release (2026-08-16, rig H4): a tilted statue at DEFAULT pose
    # collects this bonus in full — correcting the tilt deviates joints and
    # would be TAXED by it. Withhold during holds so correction is free.
    held = getattr(env, "_tilt_hold_active", None)
    if held is not None:
        calm = calm & ~held
    # H4 TORQUE-GATE (2026-07-23, user): withhold the stand reward when the hip
    # rolls are over-torquing (the A-frame brace, ~20 Nm) — the brace is
    # INVISIBLE to the pose term (it holds q=default via isometric stall), so
    # gate the reward on the torque directly. Graded exp falloff above
    # torque_deadband (12 Nm exempts the ~9 Nm geometric gravity load): a
    # braced robot (20 Nm) keeps only exp(-8/4)=13% of stand_pose; a legs-
    # vertical <=12 Nm stand keeps 100%. Removes the brace's REWARD basis
    # (vs adding a payable penalty) — the "don't give reward" idea. Graded (not
    # a hard cliff) so the policy gets a smooth 'lower torque -> more reward'
    # gradient. Hip roll = joint indices 2,3.
    hr_torque = asset.data.applied_torque[:, 2:4].abs().max(dim=1)[0]
    torque_gate = torch.exp(-(hr_torque - torque_deadband).clamp(min=0.0) / torque_gate_tau)
    return reward * (standing & calm).float() * torque_gate


def randomize_joint_play(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    ankle_play_range: tuple = (0.005, 0.035),
    other_play_range: tuple = (0.0, 0.009),
    ankle_floor_deg: float = 1.0,
    series_k_range: tuple | None = None,
):
    """FREE-PLAY DR (2026-08-16, user sim2real hypothesis; TRAINING-WIRED for
    lineage 4): per-reset redraw of each joint's backlash band (see
    TVCurveActuator). Ankles draw the widest (hardware measurement: 15-deg
    total band from linkage looseness); everything else small-to-zero.
    CURRICULUM: if ankle_play_curriculum is active it publishes
    env._ankle_play_cap (radians); ankles then draw U(0.6*cap, cap) so the
    band widens 2 -> 16 deg only as the fleet copes (thermostat pattern).
    Without the curriculum attr, the static ankle_play_range applies.
    NOTE actuators lazily create _play on first compute; on the very first
    reset (before any physics step) _play may not exist yet — those envs
    keep play=0 until their next reset, which is harmless at birth.

    ALSO DRAWS ANKLE K_s (2026-08-28, rig RIG_ANKLE_POSTFIX §4) when
    `series_k_range` is set: log-uniform per env over the requested band.
    It lives in THIS event rather than its own term on purpose. Every probe in
    eval_watch/ already nulls `randomize_joint_play` by name, so folding the
    stiffness draw in here means no probe can silently evaluate on a randomised
    plant — a separate term would have needed all ~15 null-lists updated, and
    the last time we relied on that (the plant-curriculum string mismatch,
    2026-08-21) a whole battery ran at the wrong K_s without anyone noticing.
    Nulling this event leaves every env at the actuator's scalar nominal."""
    robot = env.scene["robot"]
    cap = getattr(env, "_ankle_play_cap", None)
    for name, act in robot.actuators.items():
        play = getattr(act, "_play", None)
        if play is None:
            continue
        if series_k_range is not None and "ankle" in name:
            k_env = getattr(act, "_series_k_env", None)
            if k_env is not None:
                # LINEAGE 10 from-scratch: series_k_band_curriculum publishes a
                # moving LOWER bound (60 -> 20) so a newborn is not handed a
                # passively unstable ankle before it can stand at all.
                k_lo = getattr(env, "_series_k_lo", None)
                lo = math.log(k_lo if k_lo is not None else series_k_range[0])
                hi = math.log(series_k_range[1])
                u = torch.empty(len(env_ids), k_env.shape[1], device=k_env.device)
                k_env[env_ids] = u.uniform_(lo, hi).exp()
        if "ankle" in name and cap is not None:
            # FULL-SPECTRUM draw (2026-08-16, user call after the baseline
            # battery): U(0.6cap, cap) trained only 9.6-16 deg once the cap
            # ratcheted up — near-rigid ankles went OUT of distribution and
            # every rigid probe arm collapsed ~6x vs play arms. Fixed 1-deg
            # floor keeps the whole tight-to-slack spectrum trained, so a
            # linkage repair on hardware can't strand the policy.
            rng = (min(math.radians(ankle_floor_deg), cap), cap)
        else:
            rng = ankle_play_range if "ankle" in name else other_play_range
        n = len(env_ids)
        play[env_ids] = torch.empty(n, play.shape[1], device=play.device).uniform_(*rng)


def walk_at_spawn(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    command_name: str = "base_velocity",
    spawn_window_steps: int = 10,
):
    """STAND IS NEVER ASSIGNED AT SPAWN — only via mid-episode resample.

    DEATH-FORENSICS PART 2 (2026-08-12): even with the base_height grace, the
    stand-commanded spawn is a 100% deterministic collapse-to-floor (0.27-0.33
    m by step ~24-31) — the stand branch of a from-scratch policy has NEVER
    learned anything because stand-spawns died at step ~15 since iteration
    zero (chicken-and-egg; lineage 1 never hit it because it warm-started
    already standing, and the 0.55 floor was added late as an anti-exploit).
    The learnable path into standing is FROM MOTION: walk, then decelerate
    when a mid-episode resample (every 10 s) draws stand. This 10 Hz interval
    event catches just-reset envs (episode age <= spawn_window_steps) that
    were drawn STANDING and converts them to walkers with a fresh velocity
    draw (the command term zeroes vel_command for standers each update, so
    the flag flip alone is not enough — must redraw). Runs as interval mode
    because commands are resampled AFTER reset events in _reset_idx, which
    would overwrite a reset-mode flip."""
    term = env.command_manager.get_term(command_name)
    young = env.episode_length_buf[env_ids] <= spawn_window_steps
    ids = env_ids[young]
    if len(ids) == 0:
        return
    stand = term.is_standing_env[ids]
    s = ids[stand]
    if len(s) == 0:
        return
    r = term.cfg.ranges
    n = len(s)
    dev = env.device
    term.vel_command_b[s, 0] = torch.empty(n, device=dev).uniform_(*r.lin_vel_x)
    term.vel_command_b[s, 1] = torch.empty(n, device=dev).uniform_(*r.lin_vel_y)
    term.vel_command_b[s, 2] = torch.empty(n, device=dev).uniform_(*r.ang_vel_z)
    term.is_standing_env[s] = False


def stand_tilt_wall(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    deadband_deg: float = 3.0,
) -> torch.Tensor:
    """ALWAYS-ON stand attitude wall (2026-08-16, rig report fix): pay
    (tilt - deadband)+ per step while stand-commanded — and deliberately NOT
    push-released and NOT hold-released. This is the one tilt term that must
    survive disturbances: the rig proved (and sim reproduced: correlation
    -0.37, ~0.003 rad servo response) that the policy has NO static-attitude
    servo, because tilted-quasi-static states occurred only during bursts
    where every tilt penalty was released. The deadband keeps the push
    counter-lean (0.5-2 deg) free; the settled 3.6 deg lean and any held
    5-15 deg tilt sit on a firm linear slope (the quadratic terms are mush
    there — the same lesson as walk_upright_wall). Apply NEGATIVE weight."""
    asset = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
    tilt = torch.asin(pg)
    excess = (tilt - math.radians(deadband_deg)).clamp(min=0.0)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return excess * standing


def stand_height_slope(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    target_height: float = 0.85,
) -> torch.Tensor:
    """STAND EXEMPTION companion (2026-08-13): with the height KILL removed at
    stand, this prices sinking as a SLOPE instead — pay (target_height - h)+
    per step while stand-commanded. Closes the sit-exploit niche the original
    0.55 floor was built against (a sitter is still, upright-torsoed, and
    otherwise penalty-free) AND fixes the learning geometry: stand_pose's
    exp-kernel is flat at deep postures (all bad crouches pay ~= 0), so a
    downed stander has no local gradient — this term slopes every centimeter
    of rising. Push-released like all stand shaping (a burst crouch-catch is
    legitimate). Apply with NEGATIVE weight (~-3: sitting at 0.15 m costs
    ~2.1/step, upright >= 0.85 m costs exactly 0)."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    sink = (target_height - h).clamp(min=0.0)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return sink * _push_release(env, standing)


def stand_tilt_excursion(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    tilt_ref_deg: float = 5.0,
) -> torch.Tensor:
    """EXCURSION pricing (lineage 8, 2026-08-24) — the counter to a SLOW SWAY.

    Measured problem: on the compliant plant the quiet stand oscillates, worst
    |tilt| median 8.6 deg (p90 10.7, max 15.4), and 7.4k iterations of training
    shrank the TAILS (p90 13.2 -> 10.7) while leaving the MEDIAN flat. The rig
    sees the same distribution on their plant, so it is the policy, not physics.

    Why the existing quadratic can't fix it: stand_upright pays the MEAN of
    tilt^2, and for an oscillation of amplitude A the mean of A^2 sin^2 is
    A^2/2 — a 9 deg sway costs exactly what sitting still at 6.4 deg costs.
    Nothing in that term prefers "stop swinging" over "sway about a smaller
    centre". stand_gyro prices the RATE, but a slow excursion has a low rate.

    This prices the PEAK: (tilt/tilt_ref)^4, so the cost is dominated by the
    excursions and nearly vanishes when calm. At tilt_ref 5 deg: 2 deg costs
    0.026 units, 5 deg costs 1.0, 9 deg costs 10.5 — a 400x ratio across the
    band where the quadratic only spans 20x. Zero-deadband and push/hold-
    released, exactly like stand_upright (same gating, different exponent).
    Introduced FROM SCRATCH: bolted onto a policy that already sways, it would
    be fighting an entrenched strategy. NEGATIVE weight."""
    asset = env.scene[asset_cfg.name]
    tilt = torch.asin(asset.data.projected_gravity_b[:, :2].norm(dim=1).clamp(-1.0, 1.0))
    ref = math.radians(tilt_ref_deg)
    # BOUNDED (2026-08-24, same day): unbounded, this term is a value-function
    # bomb — a newborn lives at large tilt and (30deg/5deg)^4 = 1296 pays ~260
    # /step against rewards of order 1 (L8 first attempt: VF 39-113, reward
    # -4.6, ep_len falling 125->94, noise 0.85). Saturate at 20x the reference
    # cost (reached ~10.6 deg): excursions are priced across the band we care
    # about, and beyond it the robot is falling, which termination and
    # stand_upright already handle.
    excursion = ((tilt / ref) ** 4).clamp(max=20.0)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return excursion * _push_release(env, standing)


def stand_stance_geometry(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    nominal_width: float = 0.33,
    width_deadband: float = 0.05,
    stagger_deadband: float = 0.03,
    stagger_coeff: float = 1.0,
) -> torch.Tensor:
    """POSE RE-HOMING part 1 (2026-08-16, lineage 4 — user/rig observation,
    confirmed by pose_rehome_probe on model_55200): after surviving a push the
    JOINTS partially re-home (stand_pose pulls them) but the FOOT GEOMETRY
    ratchets and parks — width 36 -> 46 -> 51 cm across three pushes, stagger
    3-4x baseline, flat for 10 s. Mechanism: stand_pose is a joint kernel that
    is nearly blind to width (10 cm wider ~ 1 deg mean joint error), and the
    corrective step home is taxed (feet_lift/action taxes), so the deviated
    stance is a local optimum — and in training another burst is always
    coming, which makes braced-wide genuinely optimal HERE but wrong on
    hardware where no push follows.

    This term gives the missing gradient: deadbanded penalty on |stance width
    - nominal| and on fore-aft stagger, body-frame, standing envs only.
    Released during bursts AND holds (protective stepping stays free); LIVE
    during the post-push re-home grace (see _rehome_scale — it IS the
    gradient home while the travel taxes are softened). NEGATIVE weight."""
    asset = env.scene[asset_cfg.name]
    fp_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    n = fp_w.shape[0]
    q = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    fp_b = quat_apply_inverse(q.reshape(-1, 4), fp_w.reshape(-1, 3)).reshape(n, 2, 3)
    width = (fp_b[:, 0, 1] - fp_b[:, 1, 1]).abs()
    stagger = (fp_b[:, 0, 0] - fp_b[:, 1, 0]).abs()
    pen = ((width - nominal_width).abs() - width_deadband).clamp(min=0.0) \
        + stagger_coeff * (stagger - stagger_deadband).clamp(min=0.0)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    # publish "my feet are not where they should be" so the lift tax can be
    # discounted exactly while a correction is warranted (see _rehome_scale)
    env._stance_off_nominal = (pen > 0.0) & (standing > 0.0)
    return pen * _push_or_hold_release(env, standing)


def stand_foot_scuff(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    speed_deadband: float = 0.01,
) -> torch.Tensor:
    """SCUFF PENALTY (2026-08-22, hardware observation: the feet SLIDE back to
    nominal stance instead of stepping). Economics that produced it: lifting a
    foot at stand costs stand_still_feet_lift (-3.0 => ~0.9 per step taken),
    while sliding was priced only by the global feet_slide (-0.1) => ~0.05 for
    the same 5 cm correction. Sliding won by ~20x. This prices the scuff:
    in-contact foot speed ABOVE a deadband, at stand.

    The deadband is the graveyard lesson (stand_foot_slide v1, 2026-07:
    penalizing ALL slip suppressed the ankle micro-corrections that ARE the
    balance mechanism). Quiet-balance foot motion sits under ~1 cm/s; the
    measured scuff runs 2-4 cm/s per foot, so only deliberate dragging pays.
    Push/hold-released: protective stepping and its contact scrub stay free.
    NEGATIVE weight."""
    cs = env.scene.sensors[sensor_cfg.name]
    contact = (cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
               .norm(dim=-1).max(dim=1)[0] > 1.0)
    asset = env.scene[asset_cfg.name]
    speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    scuff = ((speed - speed_deadband).clamp(min=0.0) * contact.float()).sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return scuff * _push_or_hold_release(env, standing)


def joint_vel_rel_motorside_filtered(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cutoff_hz: float = 4.0,
) -> torch.Tensor:
    """MOTOR-SIDE jvel with the rig's 4 Hz one-pole low-pass (handoff #3 §2,
    user-approved 2026-08-23). The rig runs JVEL_LPF_HZ=4 live and DEPENDS on
    it: the unfiltered engage self-destructed at 1.2 s (|jvel| 15.8 rad/s) via
    the ~15 Hz series-compliance mode; filtered, the same checkpoint ran 23 s
    calm (0.09 rad/s). Training and rig MUST match, so the filter lives in the
    observation here too and is part of the deploy contract.
    Cost: ~14 deg phase at the ~1 Hz body dynamics — accepted by both sides."""
    raw = joint_vel_rel_motorside(env, asset_cfg)
    a = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * env.step_dt)
    prev = getattr(env, "_jvel_lpf_state", None)
    if prev is None or prev.shape != raw.shape:
        prev = raw.clone()
    state = prev + a * (raw - prev)
    env._jvel_lpf_state = state.detach()
    return state


def joint_pos_rel_motorside(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """MOTOR-SIDE ENCODER obs (2026-08-16, lineage 4, free-play package): the
    hardware ankle encoder is on the MOTOR side of a linkage with a measured
    15-deg free band — it cannot see where the foot actually is inside the
    band. The rigid-sim obs (true link angle) hands the policy information the
    robot will never have, so at 15-deg play the sim policy balances on data
    that vanishes on hardware. This reports each joint's virtual motor angle
    (tracked by TVCurveActuator: commanded target clamped to +-play/2 of the
    true link angle) relative to default. For play=0 joints the motor angle
    IS the link angle — exact-legacy obs, so old checkpoints eval unchanged.
    Wired into the POLICY group only; the critic keeps privileged truth."""
    asset = env.scene[asset_cfg.name]
    out = (asset.data.joint_pos - asset.data.default_joint_pos).clone()
    for act in asset.actuators.values():
        mp = getattr(act, "motor_pos", None)
        if mp is not None:
            out[:, act.joint_indices] = mp - asset.data.default_joint_pos[:, act.joint_indices]
    return out


def joint_vel_rel_motorside(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Companion to joint_pos_rel_motorside: motor-side joint velocity. When
    the linkage is slack the encoder sees ~zero motion regardless of how the
    link wanders — that blindness is the point. Exact-legacy at play=0
    (motor_vel is reported as the true joint_vel for engaged/rigid joints)."""
    asset = env.scene[asset_cfg.name]
    out = (asset.data.joint_vel - asset.data.default_joint_vel).clone()
    for act in asset.actuators.values():
        mv = getattr(act, "motor_vel", None)
        if mv is not None:
            out[:, act.joint_indices] = mv - asset.data.default_joint_vel[:, act.joint_indices]
    return out


class stand_transition_corridor(ManagerTermBase):
    """STAND LEARNABILITY PACKAGE part 1 (2026-08-13, stand_quality probes):
    the stand command was a binary cliff — walking at 1.01 m, then collapse to
    0.39 m within 0.5 s of the switch, every robot, every time (and in
    training, stand tenures averaged 12 steps before termination: standing has
    NEVER been successfully practiced — stand_pose has paid 0.000 for the
    lineage's whole life; the spawn fix moved the absorbing failure to the
    resample, it didn't remove it).

    This 10 Hz interval manager converts every freshly-drawn stand into a
    DECELERATION CORRIDOR: ~decel_s seconds of slow walking (decel_speed,
    above the 0.1 stand threshold so the gait clock keeps running), THEN the
    true stand begins. It also publishes env._stand_onset_time (episode-time
    seconds of the true stand start; -1e6 when not standing) which the
    stand-onset termination graces (part 2) and the burst shield (part 3)
    read. Spawn-drawn stands (age <= spawn_window_steps) are left to
    walk_at_spawn, which owns that case."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n = env.num_envs
        self._decel_until = torch.full((n,), -1.0, device=env.device)
        env._stand_onset_time = torch.full((n,), -1.0e6, device=env.device)
        # shared with the burst shield: corridor-active = _stand_corridor_until >= 0
        env._stand_corridor_until = self._decel_until

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._decel_until[env_ids] = -1.0
        self._env._stand_onset_time[env_ids] = -1.0e6

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor,
        command_name: str = "base_velocity",
        decel_speed: float = 0.12,
        decel_s: float = 1.5,
        spawn_window_steps: int = 12,
    ):
        term = env.command_manager.get_term(command_name)
        now = env.episode_length_buf.float() * env.step_dt
        ids = env_ids
        t = now[ids]
        onset = env._stand_onset_time
        flag = term.is_standing_env[ids]
        corridor = self._decel_until[ids] >= 0.0
        # 1) corridor expiry -> the true stand begins; stamp the onset
        expire = corridor & (t >= self._decel_until[ids])
        if expire.any():
            e = ids[expire]
            term.is_standing_env[e] = True
            onset[e] = now[e]
            self._decel_until[e] = -1.0
        # 2) corridor maintenance: keep the slow-walk command pinned
        active = corridor & ~expire
        if active.any():
            a = ids[active]
            term.is_standing_env[a] = False
            term.vel_command_b[a, 0] = decel_speed
            term.vel_command_b[a, 1:] = 0.0
        # 3) freshly-drawn stand (mid-episode resample) -> open a corridor
        fresh = flag & ~corridor & (onset[ids] < -1.0e5) & \
                (env.episode_length_buf[ids] > spawn_window_steps)
        if fresh.any():
            s = ids[fresh]
            term.is_standing_env[s] = False
            term.vel_command_b[s, 0] = decel_speed
            term.vel_command_b[s, 1:] = 0.0
            self._decel_until[s] = now[s] + decel_s
        # 4) stand ended (resampled back to walking) -> clear the onset
        ended = ~flag & ~corridor & (onset[ids] > -1.0e5)
        if ended.any():
            onset[ids[ended]] = -1.0e6


def _outside_stand_grace(env: "ManagerBasedRLEnv", stand_grace_s: float) -> torch.Tensor:
    """True where the env is NOT inside the stand-onset grace window."""
    onset = getattr(env, "_stand_onset_time", None)
    if onset is None or stand_grace_s <= 0.0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    now = env.episode_length_buf.float() * env.step_dt
    return (now - onset) >= stand_grace_s


def bad_orientation_stand_grace(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    stand_grace_s: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """STAND PACKAGE part 2b: standard bad_orientation, released for
    stand_grace_s after a stand transition so a wobbly first stand is a
    lesson, not an execution. Mid-episode/walking standard unchanged."""
    from isaaclab.envs.mdp.terminations import bad_orientation
    return bad_orientation(env, limit_angle, asset_cfg) & _outside_stand_grace(env, stand_grace_s)


def base_height_after_grace(
    env: "ManagerBasedRLEnv",
    minimum_height: float,
    grace_steps: int = 30,
    stand_grace_s: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """root_height_below_minimum with a spawn-settle GRACE window.

    DEATH-FORENSICS VERDICT (2026-08-12, death_forensics.txt, user-driven):
    ~30% of episodes draw a STAND command, and the stand-commanded spawn
    settle is a deep crouch that crosses the 0.55 m line at step ~15 —
    EXECUTED before the skill can ever be practiced. Proof: with all DR
    nulled and rel_standing=1.0, 31238/31253 episodes died at step 16 (a
    100% deterministic death loop); every spawn-draw isolation (z, joints,
    velocity, gains, tilt, friction, drop height, bursts — single AND
    combined) left the ~44% early-death mass untouched. Near-absorbing
    failure: no surviving stand-settle exists to reinforce, so -150 death
    teaches nothing (same shape as the instant-on push cliff). This grace
    (~0.6 s) lets the settle complete so the skill is learnable; the
    mid-episode standard is UNCHANGED. Also the likely cause of lineage-1's
    ~620-660 ep_len ceiling (same standers, same termination, all along).
    Wean by shrinking grace_steps once stand-settle survival is established."""
    from isaaclab.envs.mdp.terminations import root_height_below_minimum
    below = root_height_below_minimum(env, minimum_height, asset_cfg)
    alive = below & (env.episode_length_buf > grace_steps) & _outside_stand_grace(env, stand_grace_s)
    # STAND EXEMPTION (2026-08-13, user proposal): while commanded to STAND,
    # sinking never terminates — the room lineage 1 actually learned standing
    # in (the 0.55 floor postdates its stand skill). A 1.5 s grace only
    # DELAYS the execution (collapse completes in ~1 s, the robot is down,
    # grace expires). Tip-overs still terminate via bad_orientation; walkers
    # keep the full floor; the sit-exploit niche this reopens is priced by
    # the stand_height_slope penalty (a continuous climb gradient the
    # exp-kernel stand_pose cannot provide from deep postures). WEAN: restore
    # the stander floor once stand_pose sustains ~0.3+.
    cmd = env.command_manager.get_command("base_velocity")
    standing_cmd = torch.norm(cmd[:, :3], dim=1) < 0.1
    return alive & ~standing_cmd


def walk_upright_wall(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    pitch_deadband_deg: float = 2.0,
    roll_deadband_deg: float = 3.0,
) -> torch.Tensor:
    """Deadbanded upright SPEC for WALKING (2026-08-10, user call): zero cost
    inside the band, LINEAR cost beyond it — a firm wall where the quadratic
    flat_orientation_l2 is mush (5 deg costs ~0.4% of step income at w=-3).

    Measured need (freq_map posture cells, deploy build): pitch is a CONSTANT
    ~7 deg forward-lean HABIT (p95 ~10; first-qtr == last-qtr so it is posture,
    not stride oscillation) — priced by a 2 deg pitch line. Roll p95 2.7-4.3
    deg is the no-ankle-roll weight-transfer cycle — near the morphology's
    physical floor, so its line sits at 3 deg to discipline without fighting
    the mechanism (a 2 deg roll wall would Goodhart into wide-stance/slow-
    cadence shuffles). Axes priced separately for exactly that reason.
    Walking-gated (stand has its own -15 quadratic term); push-released so
    the counter-lean under a genuine shove stays free. Lineage-2 timing: the
    newborn has NOT yet formed the lean habit — this wall means it never does.
    Apply with NEGATIVE weight (~-4: 7 deg pitch costs ~6% of step income,
    3 deg costs ~1%, in-band exactly 0)."""
    asset = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b
    pitch = torch.asin(pg[:, 0].clamp(-1.0, 1.0)).abs()
    roll = torch.asin(pg[:, 1].clamp(-1.0, 1.0)).abs()
    excess = (pitch - math.radians(pitch_deadband_deg)).clamp(min=0.0) + \
             (roll - math.radians(roll_deadband_deg)).clamp(min=0.0)
    cmd = env.command_manager.get_command(command_name)
    walking = (torch.norm(cmd[:, :3], dim=1) >= stand_still_threshold).float()
    return excess * _push_release(env, walking)


def flat_orientation_l2_push_released(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Standard flat_orientation_l2, released while a burst is on the env —
    the counter-lean IS tilt; taxing it during a real push fights the skill
    (reward-conflict audit 2026-08-09). Quiet envs pay the normal price."""
    from isaaclab.envs.mdp.rewards import flat_orientation_l2
    base = flat_orientation_l2(env, asset_cfg)
    return _push_release(env, torch.ones_like(base)) * base


def joint_deviation_hip_push_released(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Standard joint_deviation_l1 for the hip group, released while a burst
    is on the env — the lean/step posture is hip-roll deviation by definition
    (reward-conflict audit 2026-08-09)."""
    from isaaclab.envs.mdp.rewards import joint_deviation_l1
    base = joint_deviation_l1(env, asset_cfg)
    return _push_or_hold_release(env, torch.ones_like(base)) * base


def push_brace_shaping(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    tilt_cap_deg: float = 18.0,
    lateral_min: float = 0.85,
    lean_hi: float = 0.08,
) -> torch.Tensor:
    """v2: pay the UPWIND COUNTER-LEAN under a sustained lateral push.

    v1 (load asymmetry toward the downwind foot) was INVERTED by the probe
    (2026-08-08, brace_20N/25N_BRACERUN): downwind load shift happens
    PASSIVELY on any pushed structure — it launched at +0.63 within 100 ms
    with zero learning, and saturated asymmetry (+0.9) is the signature of
    TOPPLING over the downwind foot's edge. Fallers averaged +0.80@200 ms;
    survivors +0.00. The survivor strategy is the human one: lean the CoM
    INTO the push (upwind), which keeps the net load CENTERED and preserves
    support margin on both sides (required lean ~= F*h/W ~= 11-14 cm at
    20-25 N; hip roll can supply it; the 2-4/64 survivors prove it).

    Paid quantity: CoM displacement from the FEET MIDPOINT projected on the
    UPWIND direction (-push_dir), clamped [0, lean_hi] and normalized. The
    passive topple moves the CoM DOWNWIND -> clamps to ZERO, so the failure
    mode can never collect (v1's fatal flaw). Quiet stand: CoM ~ centered
    -> ~0; the event mask keeps it 0 anyway. Same gates as push_step.
    sensor_cfg kept for cfg-signature stability (unused in v2).
    """
    pushed = getattr(env, "_sustained_push_active", None)
    pdir = getattr(env, "_sustained_push_dir", None)
    if pushed is None or pdir is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    lateral = pdir[:, 1].abs() > lateral_min
    pg = asset.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
    early = torch.asin(pg) < math.radians(tilt_cap_deg)
    # CoM proxy (root) displacement from feet midpoint, yaw frame
    mid_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)   # (N, 3)
    diff_w = asset.data.root_pos_w - mid_w
    yq = yaw_quat(asset.data.root_quat_w)
    diff_b = quat_apply_inverse(yq, diff_w)
    upwind = -(diff_b[:, :2] * pdir).sum(dim=-1)          # + = CoM into the push
    pay = (upwind / lean_hi).clamp(0.0, 1.0)
    return pay * (pushed & standing & lateral & early).float()


def push_step_shaping(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    tilt_cap_deg: float = 15.0,
    spread_lo: float = 0.45,
    spread_hi: float = 0.65,
    lateral_min: float = 0.85,
) -> torch.Tensor:
    """Pay for the EARLY protective side-step under a sustained lateral push.

    WHY (2026-08-08, stepout_probe.txt): relief-alone is falsified — after a
    full 15.7k-iter run with the statue rewards event-released, survivors at
    20 N lateral stand were 1-3/64 and the step still came at ~600 ms / ~26 deg
    tilt (a doomed lunge, just a bigger one). The intermediate behavior sits in
    a fitness valley (a half-committed early step dies MORE than the practiced
    brace-lunge), which local PPO exploration does not cross. This term pays
    the valley directly. Goodhart routes closed by construction:
      - pays only while a burst is ON this env (event mask -> quiet stand: 0)
      - pays only for NEAR-LATERAL pushes (|dir_y| > lateral_min) — the actual
        killer cell; fore-aft has an ankle-pitch strategy and survives
      - pays only at tilt < tilt_cap — the late lunge (~26 deg) earns ZERO, so
        only stepping BEFORE the lean establishes collects anything
      - the paid quantity is foot-to-foot spread along the push axis — the
        counter-lean brace moves the BASE, not the feet, so it cannot inflate
        it; dead-zone at normal stance width (spread_lo) so standing pays 0
    POSITIVE weight, keep SMALL (a nudge across the valley, not an attractor —
    death already costs ~40+ future reward; this only makes the first early
    steps sampled by exploration net-positive instead of taxed).
    """
    pushed = getattr(env, "_sustained_push_active", None)
    pdir = getattr(env, "_sustained_push_dir", None)
    if pushed is None or pdir is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    lateral = pdir[:, 1].abs() > lateral_min
    pg = asset.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
    early = torch.asin(pg) < math.radians(tilt_cap_deg)
    # foot-to-foot vector, world -> yaw frame (push dir is body-frame; tilt<15
    # deg gate keeps yaw the only meaningful rotation between the frames)
    fp = asset.data.body_pos_w[:, asset_cfg.body_ids, :]           # (N, 2, 3)
    diff_w = fp[:, 0, :] - fp[:, 1, :]                             # (N, 3)
    yq = yaw_quat(asset.data.root_quat_w)
    diff_b = quat_apply_inverse(yq, diff_w)                        # (N, 3)
    spread = (diff_b[:, :2] * pdir).sum(dim=-1).abs()              # along push
    pay = ((spread - spread_lo) / (spread_hi - spread_lo)).clamp(0.0, 1.0)
    return pay * (pushed & standing & lateral & early).float()


class stand_foot_anchor(ManagerTermBase):
    """Penalize a planted foot's NET displacement from where it touched down.

    Replaces stand_foot_slide, which was rig-disproven (2026-07-15): penalizing
    instantaneous in-contact slip SPEED is a Goodhart trap — model_204800 halved
    the priced quantity while net world-frame drift got 2.5x WORSE on the rig
    (17.2 vs 6.8 cm/30s; the policy re-routed sliding into unpriced forms:
    unloaded micro-motion / slow coherent ratcheting). This term prices the
    DESTINATION instead: each foot anchors its world-xy at (sustained) touchdown,
    and while in contact at stand it pays |pos - anchor| EVERY step the
    displacement persists. Properties that matter:
    - drift vs sway: balance sway is bounded oscillation around the anchor
      (small constant cost); ratcheting drift is growing distance (growing
      cost) — the discrimination velocity kernels can't make, so it never
      fights the ankle-balance mechanism (penalizing that is proven fatal).
    - push recovery: a real recovery step breaks contact and plants a NEW
      anchor — only slide-without-stepping is priced, no push gating needed.
    - micro-lift ratcheting (lift, move, re-plant to re-anchor) is priced by
      stand_feet_planted; the two terms close both displacement routes.
    - FROZEN anchors at a calm stand (v2, gate-218k autopsy): v1 released the
      anchor after a sustained lift (release_steps=5), and the policy found it
      in 10k iters — micro-lift ratcheting: lifts/min rose 35/40 -> 50/55 with
      SHORTER taps (median 9.0 -> 6.9 mm) as it re-planted feet to reset the
      meter; drift regressed 10.2 -> 12.3 cm/30s and tilt/gyro paid for the
      tapping. v2 rule: at a calm stand there is NO legitimate reason for a
      foot to end up somewhere else — anchors NEVER release while (standing &
      base calm). A tap lands against its OLD anchor and keeps paying;
      tap-in-place stays ~free. Outside the frozen state (walking, or |base
      lin vel| > vel_release_threshold = push recovery) anchors track every
      touchdown continuously, so they are fresh when the robot stops and
      recovery steps are never priced. (Don't fix this with a longer timer:
      that prices recovery steps against stale anchors — an unlearnable
      penalty, the policy cannot observe anchors.)
    NB the policy cannot OBSERVE drift (proprioception is translation-blind) —
    this shapes an open-loop disposition (motor patterns with zero net thrust),
    not a station-keeping controller. Gate builds on net world-frame drift
    (skate_diag.py), never on slip speed.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        n, k = env.num_envs, len(asset_cfg.body_ids)
        self._anchor = torch.zeros(n, k, 2, device=env.device)
        self._valid = torch.zeros(n, k, dtype=torch.bool, device=env.device)
        self._prev_frozen = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._valid[env_ids] = False
        self._prev_frozen[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        stand_still_threshold: float = 0.1,
        contact_force_threshold: float = 1.0,
        vel_release_threshold: float = 0.15,
        deadband: float = 0.0,
        max_dist: float = 0.3,
    ) -> torch.Tensor:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
            > contact_force_threshold
        )
        asset = env.scene[asset_cfg.name]
        pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
        cmd = env.command_manager.get_command(command_name)
        standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
        calm = asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_release_threshold
        frozen = standing & calm  # (n,)
        fz = frozen.unsqueeze(-1)  # (n, 1) -> broadcast over feet
        # entering the frozen state mid-swing: that foot's anchor is a stale walk
        # touchdown — invalidate so its landing re-anchors instead of being priced
        entering = (frozen & ~self._prev_frozen).unsqueeze(-1)
        self._valid &= ~(entering & ~contact)
        self._prev_frozen = frozen.clone()
        # anchor writes: continuous touchdown-tracking outside frozen; inside
        # frozen only anchorless feet (post-reset / post-invalidation) anchor
        setmask = contact & (~fz | ~self._valid)
        self._anchor = torch.where(setmask.unsqueeze(-1), pos, self._anchor)
        self._valid |= contact
        # pay for persistent displacement of planted feet at a calm stand,
        # BEYOND the deadband (v3): a ~5 cm chalk circle around each anchor is
        # free, so balance-preserving micro-corrections (the v2 killer: taxing
        # them suppressed corrective steps -> falls 61->52/64) cost nothing;
        # only sustained migration past the circle — the actual skating —
        # accumulates cost. Anchors stay FROZEN at calm stand (the v1 killer
        # was tap-to-re-anchor ratcheting), so tapping outward still walks the
        # foot out of its circle and pays.
        dist = ((pos - self._anchor).norm(dim=-1) - deadband).clamp(min=0.0, max=max_dist)
        active = (contact & self._valid & fz).float()
        return (dist * active).sum(dim=1)


def ang_vel_error_l1(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """L1 yaw-rate tracking error — the anti-VEER term (2026-07-18).

    The rig caught a SYSTEMATIC left veer (+0.031 rad/s, ~7 m circles) on the
    terrain-era build; sim confirmed it's policy-learned (+0.08, 7/8 envs).
    It was FREE under track_ang_vel_z_exp: the exp kernel (std 0.5) pays 97.5%
    of full reward at 0.08 rad/s error — no gradient at small errors, and the
    mirror loss only constrains L/R gait symmetry, not slow heading bias.
    L1 has CONSTANT gradient down to zero error, so a persistent 0.08 bias
    accumulates real cost while genuine turn-tracking (large commanded wz,
    small error) pays ~nothing extra. Applies at walk AND stand (yaw rotation
    at stand is equally unwanted — the rig also saw settle pirouettes).
    """
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    return torch.abs(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])


def tv_headroom_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.9,
) -> torch.Tensor:
    """Price MOTORING torque above `threshold` of the live T-V limit (2026-07-18).

    The rail-riding autopsy (289400): knees rode the T-V clamp 86-87% of the
    walk. Saturated gaits are (a) a metal hazard — motors pinned at the
    thermal/torque limit continuously — and (b) engine-sensitive by
    construction: physics engines agree about unsaturated dynamics (210000,
    knees ~25%, sims agreed 100%/100%) and diverge at the clamp edge (289400:
    sim 106% vs rig 73%). Cost = mean over TV-actuated joints of
    max(0, |tau|/limit - threshold), MOTORING only (same-sign torque/velocity;
    braking at the hard cap is transient physics, not propulsion strategy —
    and pricing |tau| vs the motoring limit while braking was the known
    saturation-metric artifact). Below threshold: exactly free.
    """
    robot = env.scene["robot"]
    n = robot.data.root_pos_w.shape[0]
    total = torch.zeros(n, device=robot.data.root_pos_w.device)
    cnt = 0
    for act in robot.actuators.values():
        lim = getattr(act, "tv_motoring_limit", None)
        if lim is None:
            continue
        eff = act.applied_effort
        qd = robot.data.joint_vel[:, act.joint_indices]
        motoring = (torch.sign(eff) == torch.sign(qd)).float()
        # frac CLAMPED to 1.0: the actuator classifies motoring/braking at its
        # own substep, but this term reads joint_vel at the END of the
        # decimation window — a sign flip inside the window (dithering ankle)
        # makes a braking-clamped 22 Nm read against a near-zero motoring
        # limit -> frac in the HUNDREDS (first deploy logged -11.5/s vs the
        # 0.18/s design max and would have re-collapsed the fleet). Clamping
        # prices proximity-to-ceiling only, which is the design intent.
        frac = (eff.abs() / lim.clamp(min=1e-6)).clamp(max=1.0)
        total += ((frac - threshold).clamp(min=0.0) * motoring).sum(dim=1)
        cnt += eff.shape[1]
    return total / max(cnt, 1)


def stand_action_magnitude(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    deadband: float = 1.0,
) -> torch.Tensor:
    """Price |raw action| above `deadband` at commanded STAND (2026-07-20).

    Thermal audit: at stand the policy RAILS its actions — knees at the +/-8
    clip (targets 4 rad past the hyperextension stop, ~20 Nm continuous stall)
    and hip rolls ~+/-2 (targets ~1 rad outward = the snowplow demand, ~19 Nm
    vs the ~9 Nm geometric requirement). Sim doesn't model heat, so railing is
    free; on hardware it is 4 motors at continuous stall while "standing
    still". The v5.1 loss regularizer tolerates boundary-sitting (|mu|~8 costs
    ~0.04) and per-joint clamps can't help joints whose demand clamps AT the
    physical limit (hip roll) — so price the action magnitude itself, gated to
    stand: holding a calm stand needs |a| ~ 0.1-0.3, the deadband exempts
    everything sane, and walking/recovery are untouched (cmd-gated). The
    policy fully observes and controls its own actions — the safe lever class
    (143800 precedent), unlike the unobservable-quantity terms that failed.
    """
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    a = env.action_manager.action
    excess = (a.abs() - deadband).clamp(min=0.0).mean(dim=1)
    # L5: partial relief — railed corrections still pay 25% (anti-slam)
    return excess * _push_or_hold_soften(env, standing, 0.25)


def stand_torque_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    deadband: float = 12.0,
) -> torch.Tensor:
    """Price |applied joint torque| above `deadband` Nm at commanded STAND.

    THE thermal term (2026-07-20). Trace at stand: hip-roll targets +/-0.70 rad
    (legs commanded ~40 deg APART) while q ~ 0 — the planted feet cannot slide
    outward, so the joints never move and the motors stall at ~18.5 Nm
    CONTINUOUSLY. The policy is using motor stall x ground friction as an
    A-frame brace (same trick as the knee hyperextension brace, horizontal).
    Free in sim (no thermal model); on hardware it is 2 motors at ~2-3x
    continuous rating while "standing still".

    Why the other terms cannot touch it:
      - per-joint action clips: the knee stall was against the knee's OWN joint
        stop (clamp fixed it: 20.2 -> 0.2 Nm). This stall is against the
        GROUND — target 0.70 rad is far inside the 2.27 rad joint limit, so no
        joint-limit clamp binds.
      - tv_headroom: motoring-gated; a stalled joint is zero-velocity/braking
        classified, so it is structurally exempt.
      - stand_action_magnitude: post-clip action ~1.4 vs deadband 1.0 -> ~3% of
        the stand_pose reward. Too weak, and one step removed from the physics.
    This term prices the literal thermal quantity. Deadband 12 Nm exempts the
    ~9 Nm the stance geometry genuinely requires (m*g/2 x 0.14 m lever) plus
    margin; only the brace pays. Gated to stand, so walking/recovery are
    untouched.
    NB the brace presumably buys some lateral stability — expect a stability
    cost (cf. the -5.0 tv_headroom collapse) and watch ep_len/falls.
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    n = robot.data.root_pos_w.shape[0]
    excess = torch.zeros(n, device=robot.data.root_pos_w.device)
    cnt = 0
    for act in robot.actuators.values():
        eff = act.applied_effort
        excess += (eff.abs() - deadband).clamp(min=0.0).sum(dim=1)
        cnt += eff.shape[1]
    return (excess / max(cnt, 1)) * standing


def stand_hip_roll_offset(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    vel_release_threshold: float = 0.15,
    deadband: float = 0.3,
) -> torch.Tensor:
    """Price |hip-roll ACTION| above `deadband` at a CALM stand — the scoped
    anti-A-frame term (2026-07-21), targeting the brace's CAUSE, not outcomes.

    Reward-structure autopsy of why every prior lever failed on the ~18.5 Nm
    hip-roll stall: the brace and lateral balance SHARE every outcome channel
    (foot slide, foot displacement, torque), so outcome penalties kept hitting
    the balance mechanism — stand_torque -0.3/12Nm left the stall at 19 Nm and
    tripled stand falls (2->15/32) by suppressing transient CATCH torques (the
    avoidable thing) instead of the persistent stall (the entrenched thing).
    The brace's UNIQUE signature is upstream: a large CONSTANT hip-roll action
    offset (~+/-1.4 = commanding legs 40 deg apart into ground friction), which
    stand_action_rate actively protects (constant = zero rate) and the global
    action-magnitude deadband (1.0) barely grazes. Honest gravity support
    needs ~9 Nm = 0.06 rad PD offset = ACTION ~0.12 — a 10x separation.
    User spec 2026-07-21: 9 Nm continuous is hardware-fine; the target state
    is legs-vertical, ~9 Nm quiescent, MODULATION for lateral balance (the
    brace saturates the motor and leaves release-only authority; the vertical
    stance has +/-9 Nm of two-sided authority — strictly more controllable).
    Deadband 0.3 action = 0.15 rad target offset = 22 Nm PD demand ceiling —
    generously above anything gravity-legitimate, far below the brace.
    Scoped to HIP ROLL ONLY: sagittal balance (ankle/hip-pitch, the proven
    do-not-touch mechanisms) is untouched; walking is cmd-gated out; push
    recovery is exempted via base-velocity release (anchor-v2's sound pattern).
    Action indices 2,3 = left/right hip_roll per the fixed joint ordering.
    """
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
    asset = env.scene["robot"]
    calm = asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_release_threshold
    gate = (standing & calm).float()
    a_hr = env.action_manager.action[:, 2:4]
    excess = (a_hr.abs() - deadband).clamp(min=0.0).mean(dim=1)
    return excess * gate


class sustained_push_curriculum(ManagerTermBase):
    """ADAPTIVE curriculum for sustained_push_bursts (2026-08-05, user idea).

    Measured problem: dropping the full 10-30 N range on the policy at once put
    it permanently at its failure boundary. TWICE it climbed to ep_len ~548 then
    destabilized (noise_std running away 0.32 -> 0.45), because it never gets a
    stable regime to consolidate in. The impulse pushes already ramp
    (velocity_push_curriculum) — this does the same for sustained forces, but
    gated on PERFORMANCE rather than step count, since the measured ceiling is a
    competence limit, not a schedule.

    Difficulty level in [0,1] scales force AND duration between (start_*, end_*).
    Every `review_every` steps: raise it if the fleet's mean episode length is
    above `ep_up`, lower it if below `ep_down`, hold in between. Rate-limited so
    it can only move `rate` per review — the policy always trains near, but not
    beyond, what it can currently survive. Level is logged to TensorBoard.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._level = 0.0
        self._last_review = 0
        self._ep_hist: list[float] = []          # COMPLETED episode lengths

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids,
        start_force: tuple = (3.0, 8.0),
        end_force: tuple = (10.0, 30.0),
        start_dur: tuple = (0.3, 0.8),
        end_dur: tuple = (0.5, 2.5),
        ep_up: float = 620.0,
        ep_down: float = 480.0,
        rate: float = 0.02,
        review_every: int = 200,
        min_samples: int = 200,
    ):
        # ---- TRUE mean episode length (BUGFIX 2026-08-06) ----
        # CurriculumManager.compute() is called from _reset_idx with env_ids = the
        # envs being reset, and episode_length_buf is not zeroed until later in
        # that function — so these values ARE the completed episode lengths.
        # The previous proxy, episode_length_buf.mean()*2 over ALL envs, is
        # LENGTH-BIASED (inspection paradox: envs spend more time inside long
        # episodes, so it estimates E[L^2]/E[L], ~45% high with a fall/timeout
        # mix). It read ~870 while the fleet was actually at ~600, so the level
        # raced 0 -> 1.0 in 440 iters and pinned at max difficulty — the exact
        # thing the curriculum exists to prevent. (User caught it by asking
        # whether the ramp actually pauses below the threshold.)
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]

        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()                # fresh window each review
            if ep_now > ep_up:
                self._level = min(1.0, self._level + rate)
            elif ep_now < ep_down:
                self._level = max(0.0, self._level - rate)
        L = self._level
        f = (start_force[0] + (end_force[0] - start_force[0]) * L,
             start_force[1] + (end_force[1] - start_force[1]) * L)
        d = (start_dur[0] + (end_dur[0] - start_dur[0]) * L,
             start_dur[1] + (end_dur[1] - start_dur[1]) * L)
        # getattr-not-None, NOT hasattr: probes null the event to None, and
        # hasattr(None-attr) is True -> None.params would crash every probe.
        if getattr(env.event_manager.cfg, "sustained_push", None) is not None:
            env.event_manager.cfg.sustained_push.params["force_range"] = f
            env.event_manager.cfg.sustained_push.params["duration_range"] = d
        return {"sustained_push_level": L,
                "sustained_push_force_max": f[1],
                "sustained_push_dur_max": d[1]}


class ankle_play_curriculum(ManagerTermBase):
    """FREE-PLAY thermostat (lineage 4, 2026-08-16): widen the ankle backlash
    band cap from cap_start toward cap_end only while the fleet copes — the
    same completed-episode ep-len thermostat as sustained_push_curriculum
    (620 up / 480 down hysteresis, fresh window per review). Publishes
    env._ankle_play_cap (RADIANS) for randomize_joint_play's per-reset draws
    (U(0.6*cap, cap)) and logs the cap in degrees to TensorBoard. Both this
    and the push thermostat gate on the same signal, so they alternate
    naturally as headroom allows (documented, acceptable). Hardware truth the
    ramp targets: ankles 15-deg total band; cap_end 16 brackets it."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._cap_deg = None          # set from cap_start on first call
        self._last_review = 0
        self._ep_hist: list[float] = []

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids,
        cap_start_deg: float = 2.0,
        cap_end_deg: float = 16.0,
        step_deg: float = 1.0,
        ep_up: float = 620.0,
        ep_down: float = 480.0,
        review_every: int = 200,
        min_samples: int = 200,
    ):
        if self._cap_deg is None:
            self._cap_deg = float(cap_start_deg)
        # completed-episode capture (see sustained_push_curriculum BUGFIX
        # 2026-08-06: called from _reset_idx, env_ids are the resetting envs,
        # episode_length_buf not yet zeroed -> true completed lengths)
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]
        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()
            if ep_now > ep_up:
                self._cap_deg = min(cap_end_deg, self._cap_deg + step_deg)
            elif ep_now < ep_down:
                self._cap_deg = max(cap_start_deg, self._cap_deg - step_deg)
        env._ankle_play_cap = math.radians(self._cap_deg)
        return {"ankle_play_cap_deg": self._cap_deg}


class plant_friction_curriculum(ManagerTermBase):
    """FRICTION THERMOSTAT (lineage 7R, 2026-08-21): ramp the fitted Coulomb
    friction from scale_start (0.4 = learnable, near the old plant's hips)
    toward 1.0 (the rig-fitted truth) only while the fleet copes — the same
    completed-episode ep-len thermostat as the push/wobble ramps. WHY: L7
    trained at 100% fitted friction from birth destabilized by 5k iters
    (ep_len 550->430, push level retreated to 0, noise 0.47->0.63, walking
    never formed at air_time 0.009) — the lineage-2 law again: a policy
    permanently at its failure boundary never consolidates. Deploy truth is
    unchanged: the ramp ENDS at the fitted values; probes null this term so
    evals always measure 100%. Writes the three fitted-friction events' param
    ranges = base * scale each review."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._level = 0.0
        self._last_review = 0
        self._ep_hist: list[float] = []

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids,
        scale_start: float = 0.4,
        ep_up: float = 620.0,
        ep_down: float = 480.0,
        rate: float = 0.02,
        review_every: int = 200,
        min_samples: int = 200,
        base_hip_pitch_roll: tuple = (2.0, 5.0),
        base_knee: tuple = (0.3, 0.9),
        base_yaw: tuple = (0.2, 0.6),
    ):
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]
        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()
            if ep_now > ep_up:
                self._level = min(1.0, self._level + rate)
            elif ep_now < ep_down:
                self._level = max(0.0, self._level - rate)
        sc = scale_start + (1.0 - scale_start) * self._level
        for ev_name, base in (("randomize_joint_friction_hip_pitch_roll", base_hip_pitch_roll),
                              ("randomize_joint_friction_knees", base_knee),
                              ("randomize_joint_friction_yaws", base_yaw)):
            ev = getattr(env.event_manager.cfg, ev_name, None)
            if ev is not None:
                ev.params["friction_distribution_params"] = (base[0] * sc, base[1] * sc)
        return {"plant_friction_level": self._level, "plant_friction_scale": sc}


class series_stiffness_curriculum(ManagerTermBase):
    """PLANT-STABILITY RAMP (lineage 8R, 2026-08-24). The rig-verified ankle
    K_s = 23 Nm/rad makes the robot PASSIVELY UNSTABLE (33 Nm/rad at the body
    vs gravity's ~86): a newborn random policy topples before it can generate
    a learning signal, and lineage 8 from scratch never crossed ep_len 400 in
    3.5k iters while the value function exploded.

    This is the lineage-2 law applied to plant STABILITY for the first time:
    ramp K_s from `k_start` (stiff enough that the plant is passively stable
    and a gait can form) down to `k_end` = the measured 23, on the same
    completed-episode thermostat as every other ramp here. Deploy truth is
    unchanged — the ramp ENDS at the rig's number — and probes null this term
    so every evaluation measures the real plant.

    k_start 150 gives a body-level ratio (kp+K_s)/K_s = 1.4 (near-rigid);
    k_end 23 gives 3.6, the hardware measurement."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._level = 0.0
        self._last_review = 0
        self._ep_hist: list[float] = []

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids,
        k_start: float = 150.0,
        k_end: float = 23.0,
        ep_up: float = 620.0,
        ep_down: float = 480.0,
        rate: float = 0.02,
        review_every: int = 200,
        min_samples: int = 200,
    ):
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]
        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()
            if ep_now > ep_up:
                self._level = min(1.0, self._level + rate)
            elif ep_now < ep_down:
                self._level = max(0.0, self._level - rate)
        k = k_start + (k_end - k_start) * self._level
        for nm, act in env.scene["robot"].actuators.items():
            if "ankle" in nm and hasattr(act, "_series_k"):
                act._series_k = float(k)
        return {"series_stiffness_level": self._level, "series_k_now": k}


class sustained_push_bursts(ManagerTermBase):
    """EVENT (2026-08-04): random SUSTAINED-force bursts on the base link.

    Closes the measured robustness hole (push_response.txt): the policy
    micro-adjusts to trained IMPULSE pushes within 80 ms, but a sustained
    20 N lateral lean gets ~no response for 300-400 ms and knocks it over at
    ~1.4 s — sustained forces never occur in training (impulse-only pushes),
    are barely observable (no base-lin-vel/force obs; only joint-coupling +
    slow gravity drift), and the anti-brace stand design releases corrections
    on MOTION, which a quasi-static lean never triggers. Training WITH bursts
    teaches the policy to read the coupling cue and counter early.

    Runs as a 10 Hz per-env state machine (interval mode, 0.1 s ticks):
    rest U(rest_range) -> burst: horizontal body-frame force, direction
    uniform, |F| U(force_range), duration U(duration_range) -> rest. Wrench
    persists between ticks (external-wrench buffer semantics), zeroed at
    burst end and on episode reset. Bursts start MID-EPISODE — the onset
    transition is exactly the micro-adjustment moment being trained.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n = env.num_envs
        self._wrench = torch.zeros(n, 1, 3, device=env.device)
        self._zeros = torch.zeros(n, 1, 3, device=env.device)
        self._until = torch.zeros(n, device=env.device)         # burst end (episode s)
        self._next = torch.zeros(n, device=env.device)          # next burst start
        # RAMPED ONSET (2026-08-09, phase 2): force ramps 0 -> target over
        # ramp_range seconds instead of instant-on. The surgical program
        # (7-term relief + noise surgery + 3.0 bonus) produced a clean
        # negative: 0/64 survivors, ~0.5 cm active lean, and the former
        # survivors' early-step strategy REGRESSED — because on an instant-on
        # cliff every intermediate response still dies, so nothing partial is
        # ever reinforced. A ramp makes partial responses survivable mid-ramp:
        # the skill can be learned incrementally INSIDE each burst (answer
        # 5 N, then 10, then 20). Tighten ramp_range toward instant once the
        # skill exists. Targets are stored; wrench = target * ramp_scale.
        self._target = torch.zeros(n, 2, device=env.device)     # fx, fy target
        self._t0 = torch.zeros(n, device=env.device)            # burst start time
        self._ramp = torch.ones(n, device=env.device)           # ramp seconds
        # TILT-SERVO HOLDS (2026-08-16, rig report): quasi-static base MOMENTS
        # (roll/pitch) held 1-3 s at stand — creates the "tilted but stable,
        # feet planted" state the rig proved is missing from training (static
        # tilt regulation ~zero on hardware AND in the sim reproduction).
        # Lives INSIDE this term because set_external_force_and_torque writes
        # BOTH buffers — a second event would clobber (wrench-clobber lesson).
        self._torque = torch.zeros(n, 1, 3, device=env.device)
        self._h_target = torch.zeros(n, 2, device=env.device)   # mx, my target
        self._h_t0 = torch.zeros(n, device=env.device)
        self._h_ramp = torch.ones(n, device=env.device)
        self._h_until = torch.zeros(n, device=env.device)
        self._h_next = torch.zeros(n, device=env.device)
        env._tilt_hold_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._rng_ready = False
        # EVENT-GATED RELIEF mask (2026-08-07, stepout_probe.txt): per-env "a
        # burst is on me right now". The stand statue rewards read this to
        # release DURING the push. The existing calm-gate (vel>0.15) releases
        # only ~100 ms in, AFTER the lean is established — the probe shows the
        # whole prevention window (0-100 ms) is still taxed, and the policy
        # learned exactly those economics: hold pose until doomed, then lunge
        # (100% step attempts, 95% of them too late to save a 20 N lateral).
        # The event knows the push started at t=0; the gate is exogenous to the
        # policy, so it keeps the calm-gate's can't-Goodhart property.
        env._sustained_push_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        # unit xy push direction while active (zeros when idle) — read by
        # push_step_shaping so it can pay ONLY support-widening toward the push
        env._sustained_push_dir = torch.zeros(n, 2, device=env.device)
        # RE-HOME GRACE (lineage 4): per-env episode-time until which the
        # standing travel taxes are softened (see _rehome_scale). Set at
        # burst AND hold END so the step back to nominal stance is cheap.
        env._rehome_until = torch.zeros(n, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._wrench[env_ids] = 0.0
        self._until[env_ids] = 0.0
        self._target[env_ids] = 0.0
        self._env._sustained_push_active[env_ids] = False
        self._env._sustained_push_dir[env_ids] = 0.0
        self._torque[env_ids] = 0.0
        self._h_target[env_ids] = 0.0
        self._h_until[env_ids] = 0.0
        self._env._tilt_hold_active[env_ids] = False
        self._env._rehome_until[env_ids] = 0.0
        # first burst lands 2-6 s into the episode (after settle)
        n = self._next[env_ids].shape[0] if not isinstance(env_ids, slice) else self._next.shape[0]
        self._next[env_ids] = 2.0 + 4.0 * torch.rand(n, device=self._next.device)
        # first tilt-hold waits longer (a stand must exist first)
        self._h_next[env_ids] = 4.0 + 6.0 * torch.rand(n, device=self._next.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor,
        force_range: tuple = (5.0, 25.0),
        duration_range: tuple = (0.3, 2.0),
        rest_range: tuple = (3.0, 8.0),
        lateral_bias: float = 0.0,
        lateral_spread_deg: float = 20.0,
        ramp_range: tuple = (0.0, 0.0),
        stand_shield_s: float = 0.0,
        hold_torque_range: tuple = (0.0, 0.0),
        hold_duration_range: tuple = (1.0, 3.0),
        hold_rest_range: tuple = (4.0, 10.0),
        hold_ramp_range: tuple = (0.3, 0.8),
        hold_roll_bias: float = 0.7,
        rehome_grace_s: float = 0.0,
    ):
        robot = env.scene["robot"]
        now = env.episode_length_buf.float() * env.step_dt      # per-env episode time
        ids = env_ids
        t = now[ids]
        # LINEAGE 10 (2026-09-23): QUIET standing envs get no bursts and no
        # holds. The reward audit showed ~40% of standing time was released
        # and the rest was push recovery, so a quiet stand was never a trained
        # regime -- the reward improved while the quiet-stand probe stayed
        # flat. See draw_quiet_stand. Walking envs are pushed regardless.
        _q = getattr(env, "_quiet_stand", None)
        if _q is not None:
            _cmd = env.command_manager.get_command("base_velocity")
            quiet_now = _q[ids] & (torch.norm(_cmd[ids, :3], dim=1) < 0.1)
        else:
            quiet_now = torch.zeros_like(t, dtype=torch.bool)
        # end expired bursts (state lives in _target now; wrench is derived)
        ending = (self._target[ids].abs().sum(dim=1) > 0) & (t >= self._until[ids])
        if ending.any():
            e = ids[ending]
            self._target[e] = 0.0
            self._wrench[e] = 0.0
            self._next[e] = now[e] + torch.empty(len(e), device=t.device).uniform_(*rest_range)
            if rehome_grace_s > 0.0:
                env._rehome_until[e] = now[e] + rehome_grace_s
        # start due bursts
        idle = self._target[ids].abs().sum(dim=1) == 0
        due = idle & (t >= self._next[ids]) & ~quiet_now
        # STAND PACKAGE part 3 — burst shield: never START a burst on an env
        # whose (true) stand began less than stand_shield_s ago; a 1-second-old
        # stand being shoved is another absorbing layer while the skill is
        # young. Pushes-on-standers resume past the shield (still the end goal).
        if stand_shield_s > 0.0:
            onset = getattr(env, "_stand_onset_time", None)
            if onset is not None:
                fresh_stand = (t - onset[ids]) < stand_shield_s
                corr = getattr(env, "_stand_corridor_until", None)
                if corr is not None:
                    fresh_stand = fresh_stand | (corr[ids] >= 0.0)
                due = due & ~fresh_stand
        if due.any():
            s = ids[due]
            k = len(s)
            ang = torch.rand(k, device=t.device) * 2.0 * math.pi
            if lateral_bias > 0.0:
                # DIRECTION BIAS (2026-08-05, burst_audit.txt): the event fires
                # correctly (19.6 N mean, 21% of env-steps) but uniform direction
                # x 20-30% standers left the ACTUAL failure mode — a near-lateral
                # lean while standing — at only ~6.6% of bursts, so escalating
                # force/duration never touched it. This concentrates bursts on
                # +/-y (the no-ankle-roll worst case: lateral resistance has no
                # ankle strategy available, only hip-roll or a side-step).
                is_lat = torch.rand(k, device=t.device) < lateral_bias
                side = torch.where(torch.rand(k, device=t.device) < 0.5, 1.0, -1.0)
                spread = math.radians(lateral_spread_deg)
                lat_ang = side * (math.pi / 2) + (torch.rand(k, device=t.device) - 0.5) * 2.0 * spread
                ang = torch.where(is_lat, lat_ang, ang)
            mag = torch.empty(k, device=t.device).uniform_(*force_range)
            self._target[s, 0] = mag * torch.cos(ang)
            self._target[s, 1] = mag * torch.sin(ang)
            self._t0[s] = now[s]
            self._ramp[s] = torch.empty(k, device=t.device).uniform_(*ramp_range).clamp(min=1e-3)
            self._until[s] = now[s] + torch.empty(k, device=t.device).uniform_(*duration_range)
        # ---- TILT-SERVO HOLDS (rig report 2026-08-16): ramped base MOMENTS ----
        if hold_torque_range[1] > 0.0:
            h_ending = (self._h_target[ids].abs().sum(dim=1) > 0) & (t >= self._h_until[ids])
            if h_ending.any():
                e = ids[h_ending]
                self._h_target[e] = 0.0
                self._h_next[e] = now[e] + torch.empty(len(e), device=t.device).uniform_(*hold_rest_range)
                if rehome_grace_s > 0.0:
                    env._rehome_until[e] = now[e] + rehome_grace_s
            h_idle = self._h_target[ids].abs().sum(dim=1) == 0
            h_due = h_idle & (t >= self._h_next[ids])
            # holds start ONLY on established stands (the rig scenario), past
            # the same shield window as pushes
            cmd = env.command_manager.get_command("base_velocity")
            standing_now = torch.norm(cmd[ids, :3], dim=1) < 0.1
            h_due = h_due & standing_now & ~quiet_now
            if stand_shield_s > 0.0:
                onset = getattr(env, "_stand_onset_time", None)
                if onset is not None:
                    h_due = h_due & ((t - onset[ids]) >= stand_shield_s)
            if h_due.any():
                s = ids[h_due]
                k = len(s)
                mag = torch.empty(k, device=t.device).uniform_(*hold_torque_range)
                sign = torch.where(torch.rand(k, device=t.device) < 0.5, 1.0, -1.0)
                is_roll = torch.rand(k, device=t.device) < hold_roll_bias
                self._h_target[s, 0] = torch.where(is_roll, mag * sign, torch.zeros_like(mag))
                self._h_target[s, 1] = torch.where(is_roll, torch.zeros_like(mag), mag * sign)
                self._h_t0[s] = now[s]
                self._h_ramp[s] = torch.empty(k, device=t.device).uniform_(*hold_ramp_range).clamp(min=1e-3)
                self._h_until[s] = now[s] + torch.empty(k, device=t.device).uniform_(*hold_duration_range)
            h_scale = ((now - self._h_t0) / self._h_ramp).clamp(0.0, 1.0)
            self._torque[:, 0, 0] = self._h_target[:, 0] * h_scale
            self._torque[:, 0, 1] = self._h_target[:, 1] * h_scale
            env._tilt_hold_active = self._h_target.abs().sum(dim=1) > 0

        # derive the wrench: target scaled by ramp progress (instant-on when
        # ramp_range=(0,0) — scale clamps to 1 immediately)
        scale = ((now - self._t0) / self._ramp).clamp(0.0, 1.0)
        self._wrench[:, 0, 0] = self._target[:, 0] * scale
        self._wrench[:, 0, 1] = self._target[:, 1] * scale
        # publish the mask/dir from the TARGET (not the wrench): relief and the
        # shaping bonuses must engage at ramp START (scale ~0), not once the
        # force is already large.
        env._sustained_push_active = self._target.abs().sum(dim=1) > 0
        env._sustained_push_dir = self._target / self._target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # single owner of BOTH external buffers (forces AND torques)
        robot.set_external_force_and_torque(self._wrench, self._torque, body_ids=[0])


class torque_thermal_ema(ManagerTermBase):
    """W2 (2026-07-24): GENTLE-WALKING thermal law — price sustained per-joint
    RMS torque above each motor's continuous rating. Motor winding heat is
    copper loss ~ i^2 ~ tau^2 low-passed by the thermal mass; this term IS that
    model: per-joint EMA of tau^2 (~1.5 s time constant, ~2 gait cycles),
    penalized only above rating^2. Cost = sum over joints of
    max(0, EMA(tau^2)/rating^2 - 1) — i.e. "fractional overheat" per motor.

    Why this form (the stand-campaign laws, translated):
      - EMA discrimination (H3's win): a transient catch/push-off spike barely
        moves a 1.5 s EMA -> ~free; a gait that LIVES hot pays every step. The
        instantaneous stand_torque (falls 3x) and tv_headroom -5.0 (fleet
        collapse) both lacked exactly this.
      - tau^2, not |tau|: heating is RMS-based; walk probes @71-93k show
        rms >> mean on knee (7.8 mean/11 rms) & hip pitch (9.0/10.7) — an
        |tau| deadband at rating would read those as FREE while the windings
        cook. tau^2 also naturally prices needless spikes.
      - NO motoring gate: stall and braking currents heat identically —
        tv_headroom's motoring gate is structurally blind to stalls (the
        A-frame lesson); this term sees them.
      - NO command gate: one thermal law for stand AND walk. Post-H3/H4 stand
        (hip roll 2.7 Nm) is far under rating -> already free; the measured
        walking VIGOR FLOOR (ankle sat 36-56% ~independent of speed) is the
        payer. User 2026-07-24: tracking MAY drop — the sanctioned escape is
        a slower/softer gait, the escape stand-balance never had.
    Ratings (REAL, user-confirmed 2026-07-24): _04 (hip_pitch/hip_roll/knee)
    7.5 Nm continuous / 22 Nm stall; _02 ankle + _03 yaw 5.0 Nm continuous /
    11 Nm stall. (Supersedes the earlier 9.0/4.5 guesses; the stand-era "9 Nm"
    for hip-roll was an approximate comfort bound, real continuous is 7.5.)
    Reset zeroes the EMA
    (cold motors at episode start; ~1.5 s grace is physical).
    Expected initial bill @0.3 walk: ankles ~2.0 each, knees ~0.49, hip
    pitch ~0.41 -> ~5.8 * weight.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._state: list | None = None  # lazy: [(act, rating_sq, ema), ...]

    def _build(self, env: "ManagerBasedRLEnv", ratings: dict):
        self._state = []
        for act in env.scene["robot"].actuators.values():
            r = torch.tensor(
                [next(v for sfx, v in ratings.items() if n.endswith(sfx))
                 for n in act.joint_names],
                device=env.device,
            )
            ema = torch.zeros(env.num_envs, len(act.joint_names), device=env.device)
            self._state.append([act, r.pow(2), ema])

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        if self._state is not None:
            for entry in self._state:
                entry[2][env_ids] = 0.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        ema_alpha: float = 0.013,
        ratings: dict | None = None,
    ) -> torch.Tensor:
        if self._state is None:
            self._build(env, ratings or {"_04": 7.5, "_03": 5.0, "_02": 5.0})
        total = torch.zeros(env.num_envs, device=env.device)
        for entry in self._state:
            act, rating_sq, ema = entry
            tau_sq = act.applied_effort.pow(2)
            ema += ema_alpha * (tau_sq - ema)
            total += (ema / rating_sq - 1.0).clamp(min=0.0).sum(dim=1)
        return total


class stand_hip_roll_brace_ema(ManagerTermBase):
    """H3 (2026-07-23, user idea): TIME-WINDOWED version of stand_hip_roll_offset.

    The instantaneous stand_hip_roll_offset (-1.5) was SAFE but INEFFECTIVE @48k
    (brace unchanged 20 Nm, policy paid & kept it) — and going harder risks
    clipping a transient balance CATCH (the |action| spike a real correction
    needs). Fix: penalize only the SUSTAINED commanded hip-roll offset via an EMA
    (~0.5 s). The BRACE holds the command high forever -> EMA -> high -> pays; a
    CATCH is a brief spike -> EMA barely moves -> ~free. This is the "price the
    CAUSE (sustained command), allow the transient" principle in its cleanest
    form — lets us crank the weight without correction-suppression.

    Measures the COMMANDED hip-roll action (indices 2,3), NOT the joint position:
    the braced joint sits at q~0 (feet pinned), so only the COMMAND reveals the
    brace. Hip-roll-scoped, stand-gated, push-recovery-exempt (calm-gate).
    ema_alpha 0.04 ~ 25-step / 0.5 s time constant at 50 Hz.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._ema = torch.zeros(env.num_envs, 2, device=env.device)  # L/R hip roll

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._ema[env_ids] = 0.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "base_velocity",
        stand_still_threshold: float = 0.1,
        vel_release_threshold: float = 0.15,
        deadband: float = 0.3,
        ema_alpha: float = 0.04,
    ) -> torch.Tensor:
        a_hr = env.action_manager.action[:, 2:4].abs()
        # EVENT-HOLD (2026-08-09): don't accumulate the EMA while a burst is on
        # — a legitimate 2.5 s counter-lean would otherwise load the meter and
        # bill the robot for ~1-1.5 s AFTER the push ends (tau ~0.5 s). The
        # brace this term hunts is the QUIET-STAND A-frame; pushes are exempt.
        pushed = getattr(env, "_sustained_push_active", None)
        upd = ema_alpha * (a_hr - self._ema)
        if pushed is not None:
            upd = upd * (~pushed).float().unsqueeze(1)
        self._ema = self._ema + upd
        excess = (self._ema - deadband).clamp(min=0.0).mean(dim=1)
        cmd = env.command_manager.get_command(command_name)
        asset = env.scene["robot"]
        standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
        calm = asset.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_release_threshold
        return excess * _push_or_hold_release(env, standing & calm)


# =============================================================================
# LINEAGE 10 (2026-09-23) — reward-economy fix after the L9 gate was lost.
# Audit (eval_watch/LINEAGE10_PROPOSAL.md): 177k -> 382k the optimizer sold
# ~0.10 units of standing for ~0.90 of gait, a 9:1 trade the economy priced.
# The entire standing penalty pot was 2.9% of the gait carrots, the only
# positive standing term (stand_pose) was saturated, and ~40% of standing
# time was push-released so a QUIET stand was never a trained regime.
# =============================================================================

def draw_quiet_stand(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, quiet_fraction: float = 0.5):
    """Per-episode QUIET flag (LINEAGE 10, Change 1). Standing time in a
    flagged env gets NO sustained bursts, NO tilt holds, NO velocity shoves, so
    every standing term is live at full weight for the whole stand. The other
    half keeps the entire disturbance program unchanged.

    The flag is NOT observed by the policy: with 50/50 mixing it cannot tell
    which regime it is in until a push arrives, so it must stand calmly AND
    stay recoverable. Mixing, not separating populations, is what prevents a
    brittle statue. Walking is pushed regardless of the flag.

    This is the change that makes the training reward and the quiet-stand
    probe finally measure the same robot."""
    if not hasattr(env, "_quiet_stand"):
        env._quiet_stand = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._quiet_stand[env_ids] = torch.rand(len(env_ids), device=env.device) < quiet_fraction


def stand_calm(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    tilt_sigma_deg: float = 3.0,
    gyro_sigma: float = 0.3,
) -> torch.Tensor:
    """POSITIVE calm-stand kernel (LINEAGE 10, Change 2). exp(-(tilt/s)^2) *
    exp(-(|w_xy|/s_w)^2) while standing. The penalties only cap the downside;
    this makes CALMER always worth MORE, on the same scale as the gait carrots
    (feet_phase 3.5 / knee_swing 4.0) so standing quality can compete with gait
    quality in the optimizer instead of being 2.9% of it. Bounded kernel ->
    cannot Goodhart into an explosion. POSITIVE weight.

    FULLY RELEASED in bursts/holds (2026-09-26 fix, was softened x0.25). The
    x0.25 version taxed the CORRECT push response: leaning 4-6 deg into a 20 N
    push collapses k_tilt to ~0.06, forfeiting ~1.0/step of carrot for the whole
    burst while stand_upright/excursion were fully released. Measured cost at
    64.8k: 20 N survival 44-67% vs 95-100% for L8-same-age/L9, and 100% of L10
    survivors STEPPED (L8/L9 30-88%) -- the lean-and-brace response was never
    learned. Same gating as stand_upright now: the term is about the quiet
    stand, and a push is not one."""
    asset = env.scene[asset_cfg.name]
    tilt = torch.asin(asset.data.projected_gravity_b[:, :2].norm(dim=1).clamp(-1.0, 1.0))
    k_tilt = torch.exp(-(tilt / math.radians(tilt_sigma_deg)) ** 2)
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    k_gyro = torch.exp(-(w / gyro_sigma) ** 2)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold).float()
    return k_tilt * k_gyro * _push_or_hold_release(env, standing)


def walk_hip_abduction(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stand_still_threshold: float = 0.1,
    deadband_deg: float = 3.0,
) -> torch.Tensor:
    """SPLAYED-WALK tax (LINEAGE 10, Change 5; user-spotted, measured by
    walk_width_probe: 64-71 cm foot separation at 0.3 m/s vs 28.2 cm anatomical,
    hip roll 16-20 deg, present in every checkpoint, never priced -- feet_phase
    is height, feet_alternation fore-aft, stance_geometry stand-gated).
    Cause-priced per the campaign law: sum over hip-roll joints of
    max(|q| - deadband, 0), WALKING envs only, released in bursts/holds like
    the other motion terms. Weight is RAMPED 0 -> -2 by reward_weight_thermostat
    (a narrow walk on a compliant ankle has less lateral margin and may not be
    learnable cold). asset_cfg.joint_names must select the hip-roll joints."""
    asset = env.scene[asset_cfg.name]
    roll = asset.data.joint_pos[:, asset_cfg.joint_ids].abs()
    pen = (roll - math.radians(deadband_deg)).clamp(min=0.0).sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    walking = (torch.norm(cmd[:, :3], dim=1) >= stand_still_threshold).float()
    return pen * _push_or_hold_release(env, walking)


class reward_weight_thermostat(ManagerTermBase):
    """Ramp a reward term's WEIGHT from w_start to w_end on the same
    completed-episode ep-len thermostat as every other ramp here (620 up /
    480 down). Nothing hard is learnable without a ramp; from scratch the
    fleet must be walking before a walk-shape tax means anything."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._level = 0.0
        self._last_review = 0
        self._ep_hist: list[float] = []

    def __call__(self, env: "ManagerBasedRLEnv", env_ids, term_name: str,
                 w_start: float = 0.0, w_end: float = -2.0, ep_up: float = 620.0,
                 ep_down: float = 480.0, rate: float = 0.02, review_every: int = 200,
                 min_samples: int = 200):
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]
        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()
            if ep_now > ep_up:
                self._level = min(1.0, self._level + rate)
            elif ep_now < ep_down:
                self._level = max(0.0, self._level - rate)
        w = w_start + (w_end - w_start) * self._level
        tc = env.reward_manager.get_term_cfg(term_name)
        if tc.weight != w:
            tc.weight = w
            env.reward_manager.set_term_cfg(term_name, tc)
        return {"level": self._level, "weight_now": w}


class series_k_band_curriculum(ManagerTermBase):
    """FROM-SCRATCH stability ramp for a RANDOMISED K_s (LINEAGE 10). A newborn
    topples on a soft ankle (33 Nm/rad at the body vs gravity ~86), so the
    log-uniform draw in randomize_joint_play starts on (lo_start, hi) and the
    lower bound anneals lo_start -> lo_end on the ep-len thermostat. Deploy
    truth unchanged (the band ends at the rig's 20-120); probes null the draw
    so evaluation is unaffected. Publishes env._series_k_lo."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._level = 0.0
        self._last_review = 0
        self._ep_hist: list[float] = []

    def __call__(self, env: "ManagerBasedRLEnv", env_ids, lo_start: float = 60.0,
                 lo_end: float = 20.0, ep_up: float = 620.0, ep_down: float = 480.0,
                 rate: float = 0.02, review_every: int = 200, min_samples: int = 200):
        if env_ids is not None and len(env_ids) > 0:
            self._ep_hist.extend(env.episode_length_buf[env_ids].float().tolist())
            if len(self._ep_hist) > 4000:
                self._ep_hist = self._ep_hist[-4000:]
        step = int(env.common_step_counter)
        if step - self._last_review >= review_every and len(self._ep_hist) >= min_samples:
            self._last_review = step
            ep_now = sum(self._ep_hist) / len(self._ep_hist)
            self._ep_hist.clear()
            if ep_now > ep_up:
                self._level = min(1.0, self._level + rate)
            elif ep_now < ep_down:
                self._level = max(0.0, self._level - rate)
        env._series_k_lo = lo_start + (lo_end - lo_start) * self._level
        return {"level": self._level, "k_lo_now": env._series_k_lo}


class regime_report(ManagerTermBase):
    """TensorBoard visibility per regime (LINEAGE 10). Logs mean tilt and
    body-frame stance width for QUIET-standing, DISTURBED-standing and walking
    envs separately, so the drift that lost the L9 gate (reward up, quiet
    stand worse) is visible in training rather than only in a probe."""

    def __call__(self, env: "ManagerBasedRLEnv", env_ids, asset_cfg: SceneEntityCfg,
                 command_name: str = "base_velocity", stand_still_threshold: float = 0.1):
        asset = env.scene[asset_cfg.name]
        cmd = env.command_manager.get_command(command_name)
        standing = torch.norm(cmd[:, :3], dim=1) < stand_still_threshold
        q = getattr(env, "_quiet_stand", None)
        if q is None:
            q = torch.zeros_like(standing)
        tilt = torch.asin(asset.data.projected_gravity_b[:, :2].norm(dim=1).clamp(-1, 1)) * 180.0 / math.pi
        fp = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
        n = fp.shape[0]
        qq = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        fb = quat_apply_inverse(qq.reshape(-1, 4), fp.reshape(-1, 3)).reshape(n, 2, 3)
        width = (fb[:, 0, 1] - fb[:, 1, 1]).abs() * 100.0
        out = {"quiet_stand_frac": float((standing & q).float().sum() / standing.float().sum().clamp(min=1.0))}
        for name, m in (("quiet", standing & q), ("disturbed", standing & ~q), ("walk", ~standing)):
            if bool(m.any()):
                out[f"{name}_tilt_deg"] = float(tilt[m].mean())
                out[f"{name}_width_cm"] = float(width[m].mean())
        return out
