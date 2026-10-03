"""Reference-cycle tracking terms (AMP_PLAN §5 stage 2b — the "tracking pilot").

A DeepMimic-style task: the policy is paid for putting the ten leg joints where a
reference stride cycle says they should be at the current gait phase, while the usual
task/safety terms keep it walking at the cycle's speed.

Phase: the same stateless clock shape as mdp_gait._phase (phi_l = 2*pi*f*t, phi_r =
phi_l + pi) PLUS a per-env offset drawn at reset by the reference-state-initialisation
event (`track_rsi`), which also puts the joints on the reference pose at that phase.
Observation, reward and the RSI event all read the phase from `_phase_l` here, so they
can never disagree about "where in the stride we are".

Kernel width: exp(-rmse^2/sigma^2) is blind when the error is far outside sigma (a 20 deg
start against sigma = 5 deg is e^-16: no gradient — measured, pilot 1). `track_report`
therefore keeps sigma = clamp(current rmse, sigma_min, sigma_max): the reward is always
about e^-1 at the mean error and tightens as tracking improves (self-paced).

The cycle file comes from eval_watch/amp_build_ref.py: cycle_q (N, 10) in the robot's
joint order, L/R-symmetric, indexed by the left-leg phase; cycle_base_z for the RSI height.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, yaw_quat

from .mdp_gait import _cmd_track_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CYCLES: dict[tuple[str, str], dict] = {}


def load_cycle(path: str, device: str) -> dict:
    key = (path, str(device))
    if key not in _CYCLES:
        d = np.load(path, allow_pickle=True)
        _CYCLES[key] = {
            "q": torch.tensor(d["cycle_q"], dtype=torch.float32, device=device),
            "qd": torch.tensor(d["cycle_qd"], dtype=torch.float32, device=device),
            "base_z": torch.tensor(d["cycle_base_z"], dtype=torch.float32, device=device),
            "period": float(d["period_s"]),
            "speed": float(d["speed_mps"]),
            "vel_b": torch.tensor([float(x) for x in d["vel_b"]] if "vel_b" in d else [float(d["speed_mps"]), 0.0], dtype=torch.float32, device=device),
            "joint_names": [str(s) for s in d["joint_names"]],
        }
    return _CYCLES[key]


def _phase_l(env: "ManagerBasedRLEnv", gait_freq: float) -> torch.Tensor:
    """Left-leg phase (rad, unwrapped): stateless clock + per-env RSI offset."""
    steps = env.episode_length_buf.float()
    base = steps * (2.0 * math.pi * gait_freq * env.step_dt)
    off = getattr(env, "_track_phase0", None)
    if off is None:
        off = torch.zeros(env.num_envs, device=env.device)
        env._track_phase0 = off
    return base + off


def _ref_at_phase(table: torch.Tensor, phi_l: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a (N, ...) cycle table at left-leg phase phi_l (any range)."""
    n = table.shape[0]
    u = torch.remainder(phi_l, 2.0 * math.pi) / (2.0 * math.pi) * n
    i0 = torch.floor(u).long() % n
    i1 = (i0 + 1) % n
    w = (u - torch.floor(u))
    if table.ndim > 1:
        w = w.unsqueeze(1)
    return (1.0 - w) * table[i0] + w * table[i1]


def _check_order(env: "ManagerBasedRLEnv", asset, cycle: dict, asset_cfg: SceneEntityCfg) -> None:
    if getattr(env, "_track_order_checked", False):
        return
    ids = asset_cfg.joint_ids
    names = list(asset.joint_names) if ids == slice(None) else [asset.joint_names[i] for i in ids]
    assert names == cycle["joint_names"], f"cycle joint order {cycle['joint_names']} != robot {names}"
    env._track_order_checked = True


# ---------------------------------------------------------------- observation
def ref_phase_obs(env: "ManagerBasedRLEnv", gait_freq: float) -> torch.Tensor:
    """[cos phi_l, sin phi_l, cos phi_r, sin phi_r] — same layout as mdp_gait.gait_phase_obs."""
    phi_l = _phase_l(env, gait_freq)
    phi_r = phi_l + math.pi
    return torch.stack([torch.cos(phi_l), torch.sin(phi_l), torch.cos(phi_r), torch.sin(phi_r)], dim=1)


# ---------------------------------------------------------------- reset event (RSI)
def track_rsi(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, cycle_file: str,
              gait_freq: float, height_margin: float = 0.01) -> None:
    """Reference-state initialisation: random phase, joints on the reference pose, base at the
    reference height moving at the reference speed along its (already randomised) heading."""
    asset = env.scene[asset_cfg.name]
    cyc = load_cycle(cycle_file, env.device)
    _check_order(env, asset, cyc, asset_cfg)
    _phase_l(env, gait_freq)  # ensure the offset buffer exists
    phi0 = torch.rand(len(env_ids), device=env.device) * 2.0 * math.pi
    env._track_phase0[env_ids] = phi0
    q = _ref_at_phase(cyc["q"], phi0)
    qd = _ref_at_phase(cyc["qd"], phi0)
    asset.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    root = asset.data.root_state_w[env_ids].clone()
    root[:, 2] = env.scene.env_origins[env_ids, 2] + _ref_at_phase(cyc["base_z"], phi0) + height_margin
    fwd = torch.zeros(len(env_ids), 3, device=env.device)
    fwd[:, :2] = cyc["vel_b"]  # the cycle's own body velocity (backward / lateral cycles included)
    root[:, 7:10] = quat_apply(yaw_quat(root[:, 3:7]), fwd)
    root[:, 10:13] = 0.0
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


# ---------------------------------------------------------------- rewards
def track_ref_pose(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, cycle_file: str, gait_freq: float,
                   sigma_deg: float = 5.0, cmd_gate_lin: float = 0.0, cmd_gate_ang: float = 0.5,
                   command_name: str = "base_velocity") -> torch.Tensor:
    """exp(-mean_j (q_j - q_ref_j)^2 / sigma^2): 1 on the reference, e^-1 at sigma rms per joint.
    sigma_deg is rewritten every iteration by track_report (self-paced).

    cmd_gate_lin > 0 multiplies by mdp_gait._cmd_track_gate: the carrot is only paid while
    the BODY tracks the commanded velocity (forward speed, zero lateral, zero yaw). Without
    it the pose carrot (~5/step) outweighs the heading terms and the policy copies the
    joints while drifting sideways and turning (pilot 1: 0.21 m/s lateral, 0.32 rad/s)."""
    asset = env.scene[asset_cfg.name]
    cyc = load_cycle(cycle_file, env.device)
    _check_order(env, asset, cyc, asset_cfg)
    q_ref = _ref_at_phase(cyc["q"], _phase_l(env, gait_freq))
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    err2 = ((q - q_ref) ** 2).mean(dim=1)
    r = torch.exp(-err2 / math.radians(sigma_deg) ** 2)
    if cmd_gate_lin > 0.0:
        r = r * _cmd_track_gate(env, asset, command_name, cmd_gate_lin, cmd_gate_ang)
    return r


def track_ref_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, cycle_file: str, gait_freq: float,
                  sigma: float = 1.5) -> torch.Tensor:
    """Same kernel on joint velocities (rad/s); sigma also self-paced by track_report."""
    asset = env.scene[asset_cfg.name]
    cyc = load_cycle(cycle_file, env.device)
    qd_ref = _ref_at_phase(cyc["qd"], _phase_l(env, gait_freq))
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    err2 = ((qd - qd_ref) ** 2).mean(dim=1)
    return torch.exp(-err2 / sigma ** 2)


# ---------------------------------------------------------------- curriculum: logging + self-paced sigma
class track_report(ManagerTermBase):
    """Logs rms tracking error (deg) per joint family and the fraction of envs within 5 deg,
    and sets the tracking kernels' sigma to the current rms error (clamped)."""

    def __call__(self, env: "ManagerBasedRLEnv", env_ids, asset_cfg: SceneEntityCfg, cycle_file: str, gait_freq: float,
                 sigma_min_deg: float = 5.0, sigma_max_deg: float = 20.0, vel_sigma_min: float = 1.5, vel_sigma_max: float = 6.0):
        asset = env.scene[asset_cfg.name]
        cyc = load_cycle(cycle_file, env.device)
        phi_l = _phase_l(env, gait_freq)
        q_ref = _ref_at_phase(cyc["q"], phi_l)
        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        err = (q - q_ref) * 180.0 / math.pi
        rms_env = err.pow(2).mean(dim=1).sqrt()
        rmse = float(rms_env.mean())
        qd_ref = _ref_at_phase(cyc["qd"], phi_l)
        vrms = float((asset.data.joint_vel[:, asset_cfg.joint_ids] - qd_ref).pow(2).mean().sqrt())
        sigma = min(max(rmse, sigma_min_deg), sigma_max_deg)
        vsig = min(max(vrms, vel_sigma_min), vel_sigma_max)
        rm = env.reward_manager
        if "track_ref_pose" in rm.active_terms:
            rm.get_term_cfg("track_ref_pose").params["sigma_deg"] = sigma
        if "track_ref_vel" in rm.active_terms:
            rm.get_term_cfg("track_ref_vel").params["sigma"] = vsig
        out = {"rmse_deg": rmse, "frac_within_5deg": float((rms_env < 5.0).float().mean()), "sigma_deg": sigma,
               "vel_rmse": vrms, "vel_sigma": vsig}
        names = cyc["joint_names"]
        for fam in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle"):
            cols = [i for i, n in enumerate(names) if fam in n]
            out[f"rmse_{fam}_deg"] = float(err[:, cols].pow(2).mean().sqrt())
        return out
