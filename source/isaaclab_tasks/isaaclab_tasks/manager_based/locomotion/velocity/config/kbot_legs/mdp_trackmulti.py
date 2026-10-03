"""Multi-cycle reference tracking (option 2, 2026-10-01): the velocity COMMAND selects which clean
one-period cycle the policy must follow.

Same machinery as mdp_track (gait clock + per-env RSI phase offset, pose/velocity kernels, self-paced
sigma), but with a small LIBRARY of cycles that share one period: forward stride, the stride reversed
(backward), side-steps and pivots. Each env draws a cycle at reset; its command is that cycle's own
kinematic body velocity (vx, vy, wz), written into the command term every step. The policy observes
exactly what the main walker observes (command + gait clock, 43-D), so its weights warm-start the AMP
stage directly — the recipe that worked for forward walking. A 94-clip tracker with a reference
observation could not do that, and it shuffled at 1/3 of the clip speed (v5-v7).

Library file: eval_watch/amp_refs/multicycle_v1.npz — names (K), cycle_q/cycle_qd (K, N, J),
cycle_base_z (K, N), cmd (K, 3), weights (K), period_s. Mirror pairs are stored half a period apart
(ref_mirror(phi) = M ref(phi - pi)), which is what symmetry.py's phase swap assumes.
KBOT_TM_ONLY="backward,side_left" restricts the draw (evaluation / rendering).
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat

from .mdp_track import _phase_l

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LIBS: dict[tuple[str, str], dict] = {}


def load_lib(path: str, device: str) -> dict:
    key = (path, str(device))
    if key not in _LIBS:
        d = np.load(path, allow_pickle=True)
        names = [str(s) for s in d["names"]]
        only = os.environ.get("KBOT_TM_ONLY", "")
        keep = [i for i, n in enumerate(names) if not only or n in [x.strip() for x in only.split(",")]]
        assert keep, f"KBOT_TM_ONLY={only} matches none of {names}"
        w = np.asarray(d["weights"], np.float32)[keep] if "weights" in d else np.ones(len(keep), np.float32)
        _LIBS[key] = {
            "names": [names[i] for i in keep],
            "q": torch.tensor(d["cycle_q"][keep], dtype=torch.float32, device=device),
            "qd": torch.tensor(d["cycle_qd"][keep], dtype=torch.float32, device=device),
            "base_z": torch.tensor(d["cycle_base_z"][keep], dtype=torch.float32, device=device),
            "cmd": torch.tensor(d["cmd"][keep], dtype=torch.float32, device=device),
            "weights": torch.tensor(w / w.sum(), dtype=torch.float32, device=device),
            # +1 = the gait clock runs forward, -1 = backward. BACKWARD WALKING IS THE FORWARD STRIDE WITH THE
            # CLOCK REVERSED: with a forward-running clock the same clock value demanded opposite leg motions for
            # forward and backward, and the policy never learned that flip (backward stuck at 17.5 deg, v1a-c).
            "clock_dir": torch.tensor(np.asarray(d["clock_dir"], np.float32)[keep] if "clock_dir" in d else np.ones(len(keep), np.float32), device=device),
            "period": float(d["period_s"]),
            "joint_names": [str(s) for s in d["joint_names"]],
            # feet [R, L] relative to the base with the base upright, per phase sample (K, N, 2, 3)
            "feet": torch.tensor(d["feet"][keep], dtype=torch.float32, device=device) if "feet" in d else None,
        }
        lib = _LIBS[key]
        print(f"[multi] library {os.path.basename(path)}: {len(keep)} cycles, period {lib['period']:.2f} s: "
              + ", ".join(f"{n} cmd {[round(float(x), 3) for x in c]}" for n, c in zip(lib["names"], lib["cmd"])))
    return _LIBS[key]


def _cyc(env: "ManagerBasedRLEnv") -> torch.Tensor:
    c = getattr(env, "_tm_cycle", None)
    if c is None:
        c = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._tm_cycle = c
    return c


def _tm_phase(env: "ManagerBasedRLEnv", lib: dict, gait_freq: float) -> torch.Tensor:
    """Left-leg phase (rad): RSI offset + clock direction x elapsed clock (stateless within an episode)."""
    off = getattr(env, "_track_phase0", None)
    if off is None:
        off = torch.zeros(env.num_envs, device=env.device)
        env._track_phase0 = off
    return off + lib["clock_dir"][_cyc(env)] * env.episode_length_buf.float() * (2.0 * math.pi * gait_freq * env.step_dt)


def tm_phase_obs(env: "ManagerBasedRLEnv", lib_file: str, gait_freq: float) -> torch.Tensor:
    """[cos phi_l, sin phi_l, cos phi_r, sin phi_r] with the signed clock (same layout as gait_phase_obs)."""
    phi_l = _tm_phase(env, load_lib(lib_file, env.device), gait_freq)
    phi_r = phi_l + math.pi
    return torch.stack([torch.cos(phi_l), torch.sin(phi_l), torch.cos(phi_r), torch.sin(phi_r)], dim=1)


def _ref(table: torch.Tensor, k: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a (K, N, ...) table at cycle k, left-leg phase phi (any range)."""
    n = table.shape[1]
    u = torch.remainder(phi, 2.0 * math.pi) / (2.0 * math.pi) * n
    i0 = torch.floor(u).long() % n
    i1 = (i0 + 1) % n
    w = u - torch.floor(u)
    if table.ndim > 2:
        w = w.unsqueeze(1)
    return (1.0 - w) * table[k, i0] + w * table[k, i1]


def _check_order(env: "ManagerBasedRLEnv", asset, lib: dict, asset_cfg: SceneEntityCfg) -> None:
    if getattr(env, "_tm_order_checked", False):
        return
    ids = asset_cfg.joint_ids
    names = list(asset.joint_names) if ids == slice(None) else [asset.joint_names[i] for i in ids]
    assert names == lib["joint_names"], f"library joint order {lib['joint_names']} != robot {names}"
    env._tm_order_checked = True


FEET = ["KB_D_501R_R_LEG_FOOT", "KB_D_501L_L_LEG_FOOT"]  # same [R, L] order as the puppet tables


def _feet_rel(env: "ManagerBasedRLEnv", asset) -> torch.Tensor:
    """Feet positions relative to the root in the gravity-aligned (yaw) frame, (N, 2, 3), order [R, L]."""
    ids = getattr(env, "_tm_feet_ids", None)
    if ids is None:
        ids = asset.find_bodies(FEET, preserve_order=True)[0]
        env._tm_feet_ids = ids
    d = asset.data.body_pos_w[:, ids] - asset.data.root_pos_w[:, None, :]
    q = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, d.shape[1], -1)
    return quat_apply_inverse(q.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)


def _root_ref(env: "ManagerBasedRLEnv", asset) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the root WOULD be if the body moved exactly at the commanded velocity since the reset."""
    xy = getattr(env, "_tm_ref_xy", None)
    if xy is None:
        xy = asset.data.root_pos_w[:, :2].clone()
        env._tm_ref_xy = xy
        env._tm_ref_yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2].clone()
    return xy, env._tm_ref_yaw


def _write_cmd(env: "ManagerBasedRLEnv", lib: dict) -> None:
    term = env.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = lib["cmd"][_cyc(env)]


# ---------------------------------------------------------------- events
def _draw_probs(env: "ManagerBasedRLEnv", lib: dict, env_ids: torch.Tensor) -> torch.Tensor:
    """Adaptive cycle draw (KBOT_TM_ADAPT=1, default). A cycle the policy is bad at has short episodes, so
    with a fixed draw it gets only a sliver of the STEPS (backward: ~20% of draws but a few % of practice,
    stuck at 17.5 deg in v1a-c while the single backward tracker learned it in 100 iterations). Here
    p_k ~ base_k / mean_episode_length_k x rmse_k, i.e. equal practice time, tilted toward what lags."""
    base = lib["weights"]
    if os.environ.get("KBOT_TM_ADAPT", "1") != "1":
        return base
    K = base.shape[0]
    ema = getattr(env, "_tm_len", None)
    if ema is None:
        ema = torch.full((K,), 200.0, device=env.device)
        env._tm_len = ema
    prev = _cyc(env)[env_ids]
    length = env.episode_length_buf[env_ids].float()
    for i in range(K):
        m = (prev == i) & (length > 1)
        cnt = int(m.sum())
        if cnt > 0:
            a = min(1.0, cnt / 400.0)
            ema[i] = (1.0 - a) * ema[i] + a * length[m].mean()
    rm = getattr(env, "_tm_rmse", None)
    tilt = (rm / rm.mean()).clamp(0.5, 2.0) if rm is not None else torch.ones(K, device=env.device)
    w = base / ema.clamp(min=20.0) * tilt
    w = w / w.sum()
    w = torch.minimum(torch.maximum(w, 0.5 * base), torch.full_like(w, 0.6))
    w = w / w.sum()
    env._tm_p = w
    return w


def tm_rsi(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, lib_file: str, gait_freq: float,
           height_margin: float = 0.01) -> None:
    """Reset: draw a cycle and a phase; joints on that cycle's pose, base at its height moving at its velocity."""
    asset = env.scene[asset_cfg.name]
    lib = load_lib(lib_file, env.device)
    _check_order(env, asset, lib, asset_cfg)
    _tm_phase(env, lib, gait_freq)  # ensure the offset buffer exists
    n = len(env_ids)
    k = torch.multinomial(_draw_probs(env, lib, env_ids), n, replacement=True)
    _cyc(env)[env_ids] = k
    phi0 = torch.rand(n, device=env.device) * 2.0 * math.pi
    env._track_phase0[env_ids] = phi0
    asset.write_joint_state_to_sim(_ref(lib["q"], k, phi0), _ref(lib["qd"], k, phi0) * lib["clock_dir"][k].unsqueeze(1), env_ids=env_ids)
    root = asset.data.root_state_w[env_ids].clone()
    root[:, 2] = env.scene.env_origins[env_ids, 2] + _ref(lib["base_z"], k, phi0) + height_margin
    c = lib["cmd"][k]
    vel = torch.zeros(n, 3, device=env.device)
    vel[:, :2] = c[:, :2]
    root[:, 7:10] = quat_apply(yaw_quat(root[:, 3:7]), vel)
    root[:, 10:12] = 0.0
    root[:, 12] = c[:, 2]
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    ref_xy, ref_yaw = _root_ref(env, asset)
    ref_xy[env_ids] = root[:, :2]
    ref_yaw[env_ids] = euler_xyz_from_quat(root[:, 3:7])[2]
    _write_cmd(env, lib)


def tm_command(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, lib_file: str) -> None:
    """Interval event (every step): the command IS the env's cycle velocity (the command term resamples at reset)."""
    _write_cmd(env, load_lib(lib_file, env.device))


# ---------------------------------------------------------------- rewards
def tm_pose(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, lib_file: str, gait_freq: float, sigma_deg: float = 5.0) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lib = load_lib(lib_file, env.device)
    k = _cyc(env)
    q_ref = _ref(lib["q"], k, _tm_phase(env, lib, gait_freq))
    err2 = ((asset.data.joint_pos[:, asset_cfg.joint_ids] - q_ref) ** 2).mean(dim=1)
    # per-cycle self-paced width (tm_report): one global sigma starved the lagging cycle (backward sat at
    # 17.8 deg while the mean was 12.5: its carrot was e^-2 and shrinking as the others improved).
    sig = getattr(env, "_tm_sigma_deg", None)
    sig_rad = torch.deg2rad(sig[k]) if sig is not None else math.radians(sigma_deg)
    return torch.exp(-err2 / sig_rad ** 2)


def tm_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, lib_file: str, gait_freq: float, sigma: float = 1.5) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lib = load_lib(lib_file, env.device)
    k = _cyc(env)
    qd_ref = _ref(lib["qd"], k, _tm_phase(env, lib, gait_freq)) * lib["clock_dir"][k].unsqueeze(1)
    err2 = ((asset.data.joint_vel[:, asset_cfg.joint_ids] - qd_ref) ** 2).mean(dim=1)
    vs = getattr(env, "_tm_vsigma", None)
    return torch.exp(-err2 / (vs[k] if vs is not None else sigma) ** 2)


def tm_feet(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, lib_file: str, gait_freq: float, sigma: float = 0.06) -> torch.Tensor:
    """exp(-mean_feet |p - p_ref|^2 / sigma^2): feet where the cycle puts them relative to the base. Joint-angle
    tracking alone lets short steps score well (v5-v7, v1a-d all crept at 1/3-1/2 speed); every published
    tracker also pays for end-effector / body positions, which is what pays for step LENGTH."""
    asset = env.scene[asset_cfg.name]
    lib = load_lib(lib_file, env.device)
    k = _cyc(env)
    K, N = lib["feet"].shape[:2]
    ref = _ref(lib["feet"].reshape(K, N, 6), k, _tm_phase(env, lib, gait_freq)).reshape(-1, 2, 3)
    err2 = ((_feet_rel(env, asset) - ref) ** 2).sum(dim=-1).mean(dim=-1)
    return torch.exp(-err2 / sigma ** 2)


# ---------------------------------------------------------------- termination
def tm_root_drift(env: "ManagerBasedRLEnv", lib_file: str, max_dist: float = 1.0, max_yaw: float = 1.0,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End the episode when the body has fallen `max_dist` m (or `max_yaw` rad) behind where the command would
    have taken it. Without this a policy is paid a full episode for creeping; published trackers terminate on
    drift from the reference (BeyondMimic 0.25 m, GMT an adaptive threshold). Called once per env step."""
    asset = env.scene[asset_cfg.name]
    lib = load_lib(lib_file, env.device)
    ref_xy, ref_yaw = _root_ref(env, asset)
    c = lib["cmd"][_cyc(env)]
    ref_yaw += c[:, 2] * env.step_dt
    cs, sn = torch.cos(ref_yaw), torch.sin(ref_yaw)
    ref_xy[:, 0] += (cs * c[:, 0] - sn * c[:, 1]) * env.step_dt
    ref_xy[:, 1] += (sn * c[:, 0] + cs * c[:, 1]) * env.step_dt
    dist = torch.norm(asset.data.root_pos_w[:, :2] - ref_xy, dim=1)
    yaw_err = wrap_to_pi(euler_xyz_from_quat(asset.data.root_quat_w)[2] - ref_yaw).abs()
    env._tm_drift = dist
    env._tm_yaw_drift = yaw_err
    return (dist > max_dist) | (yaw_err > max_yaw)


# ---------------------------------------------------------------- curriculum: logging + self-paced sigma
class tm_report(ManagerTermBase):
    def __call__(self, env: "ManagerBasedRLEnv", env_ids, asset_cfg: SceneEntityCfg, lib_file: str, gait_freq: float,
                 sigma_min_deg: float = 5.0, sigma_max_deg: float = 20.0, vel_sigma_min: float = 1.5, vel_sigma_max: float = 6.0):
        asset = env.scene[asset_cfg.name]
        lib = load_lib(lib_file, env.device)
        k = _cyc(env)
        phi = _tm_phase(env, lib, gait_freq)
        err = (asset.data.joint_pos[:, asset_cfg.joint_ids] - _ref(lib["q"], k, phi)) * 180.0 / math.pi
        rms_env = err.pow(2).mean(dim=1).sqrt()
        rmse = float(rms_env.mean())
        verr2_env = (asset.data.joint_vel[:, asset_cfg.joint_ids] - _ref(lib["qd"], k, phi) * lib["clock_dir"][k].unsqueeze(1)).pow(2).mean(dim=1)
        vrms = float(verr2_env.mean().sqrt())
        K = len(lib["names"])
        sig = getattr(env, "_tm_sigma_deg", None)
        if sig is None:
            sig = torch.full((K,), sigma_max_deg, device=env.device); env._tm_sigma_deg = sig
            env._tm_vsigma = torch.full((K,), vel_sigma_max, device=env.device)
        out = {"rmse_deg": rmse, "frac_within_5deg": float((rms_env < 5.0).float().mean()), "vel_rmse": vrms}
        for i, name in enumerate(lib["names"]):
            m = k == i
            if bool(m.any()):
                r_i = float(rms_env[m].mean())
                sig[i] = min(max(r_i, sigma_min_deg), sigma_max_deg)                      # each cycle paces itself
                env._tm_vsigma[i] = min(max(float(verr2_env[m].mean().sqrt()), vel_sigma_min), vel_sigma_max)
                out[f"rmse_{name}_deg"] = r_i
        out["sigma_deg"] = float(sig.mean())
        rmb = getattr(env, "_tm_rmse", None)
        if rmb is None:
            rmb = torch.full((K,), rmse, device=env.device); env._tm_rmse = rmb
        for i, name in enumerate(lib["names"]):
            if f"rmse_{name}_deg" in out:
                rmb[i] = out[f"rmse_{name}_deg"]
        if lib["feet"] is not None:
            Kf, Nf = lib["feet"].shape[:2]
            fe = (_feet_rel(env, asset) - _ref(lib["feet"].reshape(Kf, Nf, 6), k, phi).reshape(-1, 2, 3)).norm(dim=-1).mean(dim=-1)
            out["feet_err_cm"] = float(fe.mean()) * 100.0
            for i, name in enumerate(lib["names"]):
                m = k == i
                if bool(m.any()):
                    out[f"feet_err_{name}_cm"] = float(fe[m].mean()) * 100.0
        dr = getattr(env, "_tm_drift", None)
        if dr is not None:
            out["drift_m"] = float(dr.mean())
            out["yaw_drift_rad"] = float(env._tm_yaw_drift.mean())
        pr = getattr(env, "_tm_p", None)
        if pr is not None:
            for i, name in enumerate(lib["names"]):
                out[f"draw_p_{name}"] = float(pr[i])
                out[f"share_{name}"] = float((k == i).float().mean())
        return out
