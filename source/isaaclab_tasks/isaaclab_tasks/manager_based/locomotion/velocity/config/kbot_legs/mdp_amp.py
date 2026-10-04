"""Env-side pieces for AMP (observation groups and the walk gate)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

from .mdp_gait import _cmd_track_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def walk_gate(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", stand_still_threshold: float = 0.1,
              track_gate_lin: float = 0.3, track_gate_ang: float = 0.25, mode: str = "speed_fraction",
              yaw_cmd_max: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(N, 1) float in [0, 1]: commanded to move x body actually tracking that command.

    Everything AMP does is multiplied by this: the style reward and (above 0.5) the
    discriminator's policy samples. Standing — quiet or disturbed — is the lineage's own
    regime and must neither be paid for style nor teach the discriminator what 'walking'
    looks like. The second factor is mdp_gait._cmd_track_gate (exp(-lin_err^2/lin) *
    exp(-yaw_err^2/ang)), the same gate the lineage's gait carrots use: PILOT 0 gated on
    the command alone and the policy learned to STAND under a walk command — a still robot
    collected ~0.06/step of style, more than perfect velocity tracking pays (0.04/step).
    track_gate_lin = 0 disables the tracking factor.
    (The quiet-stand flag is NOT consulted: it is drawn for every env at reset and only has
    meaning while standing; gating on it silently dropped ~35% of the walking envs.)
    """
    cmd = env.command_manager.get_command(command_name)
    gate = (torch.norm(cmd[:, :3], dim=1) >= stand_still_threshold).float()
    if yaw_cmd_max > 0.0:
        # PILOT 1b: the dataset holds only symmetric straight strides, so a turning stride is
        # penalised by the judge whatever the weight; with this, envs commanded to turn get no
        # style (and teach the judge nothing) — turning is learned from yaw tracking alone while
        # the shared weights keep the straight-walking style.
        gate = gate * (cmd[:, 2].abs() <= yaw_cmd_max).float()
    asset = env.scene[asset_cfg.name]
    if mode == "speed_fraction":
        # PILOT 0d: fraction of the commanded planar velocity actually achieved, projected on the
        # commanded direction: 0 standing, 1/2 at half speed, 1 at full. Monotone in progress, no
        # kernel width to tune (0.3 let a still robot keep 74% of style, 0.08 shut 97% of walkers out).
        v = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)[:, :2]
        c = cmd[:, :2]
        cn = torch.norm(c, dim=1)
        frac_lin = ((v * c).sum(dim=1) / cn.clamp(min=1e-3) ** 2).clamp(0.0, 1.0)
        # dataset v3: a turn-in-place command has ~no planar part, so credit the achieved yaw rate
        # instead; blend by which component dominates the command (0.3 m lever: 0.3 rad/s ~ 0.09 m/s).
        wz_cmd = cmd[:, 2]
        frac_yaw = (asset.data.root_ang_vel_b[:, 2] * wz_cmd / wz_cmd.abs().clamp(min=1e-3) ** 2).clamp(0.0, 1.0)
        w_lin = cn / (cn + 0.3 * wz_cmd.abs()).clamp(min=1e-3)
        frac = w_lin * frac_lin + (1.0 - w_lin) * frac_yaw
        gate = gate * frac.clamp(0.0, 1.0)
    elif track_gate_lin > 0.0:
        gate = gate * _cmd_track_gate(env, asset, command_name, track_gate_lin, track_gate_ang)
    return gate.unsqueeze(1)


def axis_bias(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, p_axis: float = 0.6, command_name: str = "base_velocity",
              min_cmd: float = 0.1) -> None:
    """Interval event (every step). With probability p_axis a freshly drawn command keeps ONE component
    (forward/backward, lateral or yaw) and zeroes the others. The walker's prior (multi-cycle tracker) knows
    the pure directions; uniform sampling of all three at once almost never produces them, and the 2026-10-01
    wide-command stage taught nothing. Mixed commands still make up the remaining 1 - p_axis.

    The kept component is also moved out of the dead zone: its size is rescaled from [0, max] to
    [min_cmd, max], sign kept (min_cmd = the 0.1 stand-still threshold every gate uses). The lateral band is
    only 0.13 m/s wide, so without this 77% of the lateral-only draws are below the threshold and count as
    standing — side-stepping would get ~1/20 of the practice instead of ~1/5.

    The lineage's own command events are respected (found with eval_watch/amp_cmd_probe.py, 2026-10-02):
      * standing envs and the stand-entry deceleration corridor (mdp_gait stand corridor: 1.5 s of
        (0.12, 0, 0) before a stand) are never masked — a lateral/yaw mask zeroed the corridor command, i.e.
        40% of the stand entries were the abrupt stop the corridor exists to prevent;
      * a command re-drawn by another event without a resample (walk_at_spawn turns a stand drawn at spawn
        into a fresh walking command) is detected by comparing with the last command written here and is
        treated like any fresh draw."""
    term = env.command_manager.get_term(command_name)
    cmd = term.vel_command_b
    tl = term.time_left
    protect = term.is_standing_env.clone()
    corridor = getattr(env, "_stand_corridor_until", None)
    if corridor is not None:
        protect |= corridor >= 0.0
    mask = getattr(env, "_amp_axis_mask", None)
    if mask is None:
        mask = torch.ones(env.num_envs, 3, device=env.device)
        env._amp_axis_mask = mask
        fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        fresh = (tl > env._amp_axis_prev_tl + 1.0e-6) | (cmd != env._amp_axis_last_cmd).any(dim=1)
    fresh &= ~protect
    env._amp_axis_prev_tl = tl.clone()
    ids = fresh.nonzero(as_tuple=False).squeeze(-1)
    if len(ids) > 0:
        pure = torch.rand(len(ids), device=env.device) < p_axis
        axis = torch.randint(0, 3, (len(ids),), device=env.device)
        onehot = torch.nn.functional.one_hot(axis, 3).float()
        mask[ids] = torch.where(pure.unsqueeze(1), onehot, torch.ones_like(onehot))
        if min_cmd > 0.0:
            r = term.cfg.ranges
            for k, (lo, hi) in enumerate((r.lin_vel_x, r.lin_vel_y, r.ang_vel_z)):
                sel = ids[pure & (axis == k)]
                v = cmd[sel, k]
                top = torch.where(v >= 0, torch.full_like(v, hi), torch.full_like(v, -lo))
                size = min_cmd + v.abs() * (top - min_cmd) / top.clamp(min=1.0e-6)
                cmd[sel, k] = torch.where((v != 0) & (top > min_cmd), torch.sign(v) * size, v)
    cmd *= torch.where(protect.unsqueeze(1), torch.ones_like(mask), mask)
    env._amp_axis_last_cmd = cmd.clone()


def root_drift(env: "ManagerBasedRLEnv", max_dist: float = 1.5, max_yaw: float = 1.5, command_name: str = "base_velocity",
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: the body is more than max_dist m (or max_yaw rad) away from where its commands would have
    taken it since the reset. In the tracker this is what turned creeping into striding: without it a policy
    is paid a full episode for moving at a third of the commanded speed. Called once per env step."""
    asset = env.scene[asset_cfg.name]
    yaw_now = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    xy = getattr(env, "_amp_ref_xy", None)
    if xy is None:
        xy = asset.data.root_pos_w[:, :2].clone()
        env._amp_ref_xy = xy
        env._amp_ref_yaw = yaw_now.clone()
    yaw = env._amp_ref_yaw
    fresh = env.episode_length_buf <= 1
    xy[fresh] = asset.data.root_pos_w[fresh, :2]
    yaw[fresh] = yaw_now[fresh]
    c = env.command_manager.get_command(command_name)
    yaw += c[:, 2] * env.step_dt
    cs, sn = torch.cos(yaw), torch.sin(yaw)
    xy[:, 0] += (cs * c[:, 0] - sn * c[:, 1]) * env.step_dt
    xy[:, 1] += (sn * c[:, 0] + cs * c[:, 1]) * env.step_dt
    dist = torch.norm(asset.data.root_pos_w[:, :2] - xy, dim=1)
    yaw_err = wrap_to_pi(yaw_now - yaw).abs()
    return (dist > max_dist) | (yaw_err > max_yaw)


# ---------------------------------------------------------------- walker v5: stepping anchor from the reference cycles
_LIFT_PROFILES: dict = {}


def _lift_profile(lib_file: str, device) -> tuple[dict, torch.Tensor]:
    """The multi-cycle library and its foot heights above the ground per cycle and phase sample, (K, N, 2) [R, L]."""
    from . import mdp_trackmulti as tm

    lib = tm.load_lib(lib_file, device)
    key = (lib_file, str(device))
    if key not in _LIFT_PROFILES:
        _LIFT_PROFILES[key] = lib["base_z"].unsqueeze(-1) + lib["feet"][..., 2]
    return lib, _LIFT_PROFILES[key]


def cmd_cycle(cmd: torch.Tensor, lib: dict) -> torch.Tensor:
    """Index of the reference cycle whose direction dominates the command. Components are compared in units of
    each cycle's own speed (forward/backward ~0.4 m/s, side ~0.13 m/s, pivot ~0.56 rad/s)."""
    scale = lib["cmd"].abs().amax(dim=0).clamp(min=1.0e-6)
    return ((cmd / scale) @ (lib["cmd"] / scale).T).argmax(dim=1)


def _clock_phase_now(env: "ManagerBasedRLEnv", gait_freq: float, command_name: str, back_threshold: float = 0.05) -> torch.Tensor:
    """Left-leg phase of the signed gait clock for the state just reached — the value the next observation will
    show — WITHOUT advancing the shared clock state (mdp_gait._phase_signed). Rewards run before the observation
    in a step; a reward that integrated the clock would leave the first observation after a reset one step stale."""
    st = getattr(env, "_kbot_signed_phase", None)
    if st is None:
        return torch.zeros(env.num_envs, device=env.device)
    if st["step"] == int(env.common_step_counter):  # already integrated for this state (called after the observation)
        return st["phi"]
    cmd = env.command_manager.get_command(command_name)
    direction = torch.where(cmd[:, 0] < -back_threshold, -1.0, 1.0)
    phi = st["phi"] + direction * 2.0 * math.pi * gait_freq * env.step_dt
    return torch.where(env.episode_length_buf <= 1, torch.zeros_like(phi), phi)


def ref_foot_lift(env: "ManagerBasedRLEnv", lib_file: str, gait_freq: float, rel_sigma: float = 0.5, command_name: str = "base_velocity",
                  stand_still_threshold: float = 0.1, track_gate_lin: float = 0.0,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(-sum_feet (z - z_ref)^2 / sigma^2): each foot's height above the ground follows the reference cycle's
    lift profile at the gait clock's phase. The cycle is the one whose direction dominates the command; the
    profile depends on the phase only, so it holds at any commanded speed (cadence is fixed by the clock, stride
    length then follows from the speed). sigma = rel_sigma x that cycle's peak lift (10.7 cm forward/backward,
    ~7 cm side-steps and pivots), so a foot that never lifts scores the same low value in every direction.

    No command-tracking gate by default: the lineage's gate (exp of the INSTANTANEOUS velocity error) pays a
    smooth slide more than a stepping turn (measured: gate 0.84 sliding vs 0.76 stepping on the pivot), and
    marching in place is already ruled out by the fell-behind termination.

    Why (walker v4, 2026-10-02): with the tracker's pose/feet rewards gone, the judge alone did not hold the
    stepping. By iteration 400 side-steps and pivots were done with 0.6-1.0 cm clearance and no detectable
    strides, by 1200 forward/backward had dropped from 10-13 cm to 2-4 cm — feet sliding, while survival and
    speed (the only things the checkpoint score measured) kept rising. The lineage needed its clock-driven
    feet_phase carrot for the same reason; this is that carrot with the reference cycles' own timing and height
    (their swing peaks at phase ~0.17, not at 0 as feet_phase assumes). Flat ground only (heights are world z).
    Paid only under a moving command."""
    from . import mdp_trackmulti as tm

    asset = env.scene[asset_cfg.name]
    lib, prof = _lift_profile(lib_file, env.device)
    cmd = env.command_manager.get_command(command_name)
    ref = tm._ref(prof, cmd_cycle(cmd, lib), _clock_phase_now(env, gait_freq, command_name))  # (N, 2) [R, L]
    ids = getattr(env, "_amp_feet_ids", None)
    if ids is None:
        ids = asset.find_bodies(tm.FEET, preserve_order=True)[0]
        env._amp_feet_ids = ids
    z = asset.data.body_pos_w[:, ids, 2] - env.scene.env_origins[:, 2:3]
    k = cmd_cycle(cmd, lib)
    sigma = rel_sigma * (prof.amax(dim=1) - prof.amin(dim=1)).mean(dim=1)[k]
    r = torch.exp(-((z - ref) ** 2).sum(dim=1) / sigma ** 2)
    r = r * (torch.norm(cmd[:, :3], dim=1) >= stand_still_threshold).float()
    if track_gate_lin > 0.0:
        r = r * _cmd_track_gate(env, asset, command_name, track_gate_lin)
    return r


# ---------------------------------------------------------------- walker v6: what the first hardware engage taught (2026-10-03)
def randomize_unanswered(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, deadband_max: float = 1.0, rotor_fc_max: float = 1.5,
                         engage_p: float = 0.5, engage_gain_min: float = 0.3, engage_ramp_max: float = 1.0,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Reset event. Puts UNANSWERED COMMANDS in the data (rig RIG_HW_ENGAGE_FINDINGS.md §4.3): on the first live
    engage the ankles did not answer for 0.4 s and the policy escalated its own commands (ankle action -0.04 ->
    -1.42) into a violent motion. Per episode:
      * rigid joints (hips, yaw, knees): output dead band U(0, deadband_max) Nm — small PD torques do nothing;
      * series-elastic joints (ankles): rotor Coulomb friction with stiction U(0, rotor_fc_max) Nm;
      * with probability engage_p, a weak start: all gains x U(engage_gain_min, 1), ramping to 1 over
        U(0, engage_ramp_max) s after the reset (the ramp itself is applied by engage_gain_ramp)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device
    for act in asset.actuators.values():
        shape = (env.num_envs, len(act.joint_names))
        if getattr(act, "_series_k", 0.0) > 0.0:
            if rotor_fc_max > 0.0:
                if act._rotor_fc is None:
                    act._rotor_fc = torch.zeros(shape, device=dev)
                act._rotor_fc[env_ids] = torch.rand(n, shape[1], device=dev) * rotor_fc_max
        elif deadband_max > 0.0:
            if act._deadband is None:
                act._deadband = torch.zeros(shape, device=dev)
            act._deadband[env_ids] = torch.rand(n, shape[1], device=dev) * deadband_max
    if engage_p > 0.0:
        if getattr(env, "_engage_g0", None) is None:
            env._engage_g0 = torch.ones(env.num_envs, device=dev)
            env._engage_T = torch.zeros(env.num_envs, device=dev)
        weak = torch.rand(n, device=dev) < engage_p
        g0 = engage_gain_min + (1.0 - engage_gain_min) * torch.rand(n, device=dev)
        env._engage_g0[env_ids] = torch.where(weak, g0, torch.ones_like(g0))
        env._engage_T[env_ids] = torch.rand(n, device=dev) * engage_ramp_max


def engage_gain_ramp(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Interval event (every step): gain scale g(t) = g0 + (1 - g0) * min(t / T, 1) on every actuator, t = time
    since the reset. g0 and T come from randomize_unanswered (or from a probe that sets env._engage_g0 / _engage_T)."""
    g0 = getattr(env, "_engage_g0", None)
    if g0 is None:
        return
    t = env.episode_length_buf.float() * env.step_dt
    prog = torch.where(env._engage_T > 1.0e-6, (t / env._engage_T.clamp(min=1.0e-6)).clamp(0.0, 1.0), torch.ones_like(t))
    g = (g0 + (1.0 - g0) * prog).unsqueeze(1)
    for act in env.scene[asset_cfg.name].actuators.values():
        act._gain_scale = g


def stand_watchdog(env: "ManagerBasedRLEnv", max_joint_speed: float = 5.0, max_tilt_deg: float = 12.0, arm_s: float = 0.3,
                   settle_s: float = 1.5, push_grace_s: float = 1.0, command_name: str = "base_velocity",
                   stand_still_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: the rig's hardware watchdog, in training (user decision 2026-10-03). While the robot is told
    to STAND, any joint faster than max_joint_speed (rad/s) or a torso tilt beyond max_tilt_deg ends the episode
    like a fall — on the robot the same event cuts motor power (E-STOP at tilt > 12 deg, |qd| > 5 rad/s). Until
    now a jerk at the start of a stand cost almost nothing: the standing penalties are averaged over a 20 s
    episode, bad_orientation only fires at 57 deg and has a 1.5 s grace at stand onset.

    Armed:
      * a stand kept from spawn (the hardware engage: stand at the zero pose from tick 0): arm_s after the
        reset — the training reset leaves joint offsets and base velocity that the real engage does not have;
      * a stand entered from walking (through the stop corridor): settle_s after the stand onset;
      * not during a push burst and for push_grace_s after it (stepping out of a shove is legitimate).
    Walking is never checked."""
    asset = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)
    now = env.episode_length_buf.float() * env.step_dt
    standing = term.is_standing_env & (torch.norm(cmd[:, :3], dim=1) < stand_still_threshold)
    onset = getattr(env, "_stand_onset_time", None)
    keep = getattr(env, "_spawn_stand_keep", None)
    armed = torch.zeros_like(standing)
    if keep is not None:
        armed |= standing & keep & (now >= arm_s)
    if onset is not None:
        armed |= standing & (onset > -1.0e5) & ((now - onset) >= settle_s)
    if onset is None and keep is None:  # probes / evals without the training-side stand machinery
        armed |= standing & (now >= arm_s)
    pushed = getattr(env, "_sustained_push_active", None)
    if pushed is not None:
        until = getattr(env, "_wd_push_until", None)
        if until is None:
            until = torch.zeros(env.num_envs, device=env.device)
            env._wd_push_until = until
        until[env.episode_length_buf <= 1] = 0.0
        until[pushed] = now[pushed] + push_grace_s
        armed &= now >= until
    tilt = torch.asin(asset.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0))
    fast = asset.data.joint_vel.abs().amax(dim=1) > max_joint_speed
    return armed & (fast | (tilt > math.radians(max_tilt_deg)))


# ---------------------------------------------------------------- walker v7: the robot's sensing path (2026-10-04)
class DelayedObs(ManagerTermBase):
    """Observation term wrapper: the inner term's value as the ROBOT'S SENSING PATH delivers it.

    Rig RIG_HW_STAND_ADDENDUM_SENSING.md: on the robot the IMU gave 20 new samples a second (a repeated sample on
    57 % of ticks) 20-70 ms old and the joint readings were ~14 ms late; this env delivered every observation
    fresh at every step. walker_v5_3200 engaged cleanly, stood 1.9 s and then rocked at ~2 Hz until the watchdog
    cut it; both the rig's sim and ours (eval_watch/amp_stand_sensing_test.py) reproduce the loss of balance from
    the sensing alone, and a fresh IMU one tick late stands.

    Per episode and per robot, shared by every term of the same `sensor`:
      delay  U{delay_range} policy steps — the value the policy sees is that many steps old;
      hold   U{1..hold_max} steps — the sensor yields a NEW sample only every `hold` steps (random phase) and
             repeats the previous one in between, bit-identical, noise included.
    The term's own noise model is applied HERE, to the fresh sample (so a repeated sample repeats its noise), and
    must be removed from the term cfg. The group's history stacking then runs on what this returns, exactly like
    the rig's stacker does on what its sensors deliver."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        for v in cfg.params["inner_params"].values():
            if isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)
        self._maxd = int(cfg.params["delay_range"][1])
        self._buf = None            # (N, maxd + 1, dim): what the sensor output k steps ago, k = 0 newest
        self._held = None           # the sample the sensor is currently holding
        self._out = None
        self._fill = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._stamp = -1
        self._draw(env, torch.arange(env.num_envs, device=env.device))

    def _draw(self, env, env_ids) -> None:
        p = self.cfg.params
        st = getattr(env, "_sense", None)
        if st is None:
            st = {}
            env._sense = st
        key = p["sensor"]
        if key not in st:
            z = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            st[key] = {"delay": z.clone(), "hold": torch.ones_like(z), "phase": z.clone(), "epoch": z.clone() - 1}
        s = st[key]
        # one draw per episode for the whole sensor, whichever of its terms resets first
        epoch = (env.common_step_counter - env.episode_length_buf[env_ids]).long() if hasattr(env, "episode_length_buf") else torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
        new = s["epoch"][env_ids] != epoch
        ids = env_ids[new]
        if len(ids) > 0:
            lo, hi = p["delay_range"]
            s["delay"][ids] = torch.randint(int(lo), int(hi) + 1, (len(ids),), device=env.device)
            s["hold"][ids] = torch.randint(1, int(p["hold_max"]) + 1, (len(ids),), device=env.device)
            s["phase"][ids] = torch.randint(0, 64, (len(ids),), device=env.device)
            s["epoch"][ids] = epoch[new]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self._env.device)
        self._fill[env_ids] = True
        self._draw(self._env, env_ids)

    def __call__(self, env: "ManagerBasedRLEnv", inner, inner_params: dict, sensor: str, delay_range: tuple, hold_max: int, noise=None) -> torch.Tensor:
        step = int(env.common_step_counter)
        if self._out is not None and step == self._stamp and not bool(self._fill.any()):
            return self._out                                   # an extra compute() in the same env step: nothing new arrived
        self._stamp = step
        v = inner(env, **inner_params)
        if noise is not None and getattr(env.cfg.observations.policy, "enable_corruption", False):
            v = noise.func(v, noise)
        s = env._sense[sensor]
        if self._buf is None:
            self._buf = v.unsqueeze(1).repeat(1, self._maxd + 1, 1)
            self._held = v.clone()
        fresh = ((env.episode_length_buf + s["phase"]) % s["hold"]) == 0
        upd = fresh | self._fill
        self._held = torch.where(upd.unsqueeze(1), v, self._held)
        self._buf = torch.cat([self._held.unsqueeze(1), self._buf[:, :-1]], dim=1)
        if bool(self._fill.any()):                             # a new episode starts with every slot = its first sample
            self._buf[self._fill] = self._held[self._fill].unsqueeze(1)
            self._fill[:] = False
        self._out = self._buf[torch.arange(env.num_envs, device=env.device), s["delay"].clamp(max=self._maxd)]
        return self._out


def randomize_action_hold(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, steps: tuple = (5, 6), t_range_s: tuple = (0.3, 18.0), p: float = 1.0) -> None:
    """Reset event: with probability p, once in the episode the joint targets are frozen for U{steps} policy
    steps starting at U(t_range_s) — the rig's policy server stalled 116 ms at tick 37 of every episode (Python
    GC) and the last command was held for six ticks. The freeze itself is applied by action_hold."""
    if getattr(env, "_hold_t0", None) is None:
        env._hold_t0 = torch.full((env.num_envs,), 1.0e9, device=env.device)
        env._hold_n = torch.zeros(env.num_envs, device=env.device)
    n = len(env_ids)
    on = torch.rand(n, device=env.device) < p
    t0 = (t_range_s[0] + (t_range_s[1] - t_range_s[0]) * torch.rand(n, device=env.device)) / env.step_dt
    env._hold_t0[env_ids] = torch.where(on, torch.floor(t0), torch.full_like(t0, 1.0e9))
    env._hold_n[env_ids] = torch.randint(int(steps[0]), int(steps[1]) + 1, (n,), device=env.device).float()


def action_hold(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Interval event (every step): actuators keep their previous joint targets while the hold drawn by
    randomize_action_hold is active. The policy keeps running; its commands are ignored for those steps."""
    t0 = getattr(env, "_hold_t0", None)
    if t0 is None:
        return
    t = env.episode_length_buf.float()
    mask = ((t >= t0) & (t < t0 + env._hold_n)).unsqueeze(1)
    for act in env.scene[asset_cfg.name].actuators.values():
        act._hold_mask = mask
