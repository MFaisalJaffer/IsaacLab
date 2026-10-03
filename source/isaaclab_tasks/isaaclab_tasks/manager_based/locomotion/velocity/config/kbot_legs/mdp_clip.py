"""Time-indexed clip tracking (dataset expansion: backward, sideways, pivots, stops).

Unlike mdp_track (one repeating stride indexed by a phase clock), every env here follows ONE
labelled clip frame by frame: reference joints = clip[frame], where frame = t0 + steps since
reset, t0 drawn at reset (RSI). The velocity COMMAND the policy sees is the clip's own body
motion (smoothed base velocity + yaw rate), written into the command term every step, so the
policy learns "command -> motion" for motions the lineage never had (backward at 0.25, side-step,
pivot, stop). Its recordings then teach the AMP judge those motions, with matching commands.

Clip library: eval_watch/amp_refs/lafan1/clips/<label>_<n>.npz from amp_build_clips.py
(+ amp_clips_ground.py for base_z). Padded to a (C, Tmax, ...) tensor on first use.
"""
from __future__ import annotations

import glob
import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, yaw_quat

from .mdp_gait import _cmd_track_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LIBS: dict[tuple[str, str], dict] = {}


def load_library(clip_dir: str, device: str) -> dict:
    key = (clip_dir, str(device))
    if key in _LIBS:
        return _LIBS[key]
    files = sorted(glob.glob(os.path.join(clip_dir, "*.npz")))
    only = os.environ.get("KBOT_CLIP_ONLY")  # e.g. "side_left,backward": evaluation/rendering of specific labels
    if only:
        prefixes = tuple(x.strip() for x in only.split(",") if x.strip())
        files = [f for f in files if os.path.basename(f).startswith(prefixes)]
        print(f"[clip] KBOT_CLIP_ONLY={only}: {len(files)} clips")
    assert files, f"no clips in {clip_dir}"
    clips = [dict(np.load(f, allow_pickle=True)) for f in files]
    names = [str(c["joint_names"][0]) for c in clips]
    J = clips[0]["joint_pos"].shape[1]
    lens = np.array([c["joint_pos"].shape[0] for c in clips])
    Tmax = int(lens.max())
    C = len(clips)
    q = np.zeros((C, Tmax, J), np.float32); qd = np.zeros_like(q); cmd = np.zeros((C, Tmax, 3), np.float32); bz = np.zeros((C, Tmax), np.float32)
    labels = []
    for i, c in enumerate(clips):
        n = c["joint_pos"].shape[0]
        q[i, :n] = c["joint_pos"]; q[i, n:] = c["joint_pos"][-1]
        qd[i, :n] = c["joint_vel"]
        cmd[i, :n] = c["cmd"]; cmd[i, n:] = c["cmd"][-1]
        assert "base_z" in c, f"{files[i]}: run amp_clips_ground.py first"
        bz[i, :n] = c["base_z"]; bz[i, n:] = c["base_z"][-1]
        labels.append(str(c["label"]))
    lab_names = sorted(set(labels))
    lab_idx = torch.tensor([lab_names.index(l) for l in labels])
    counts = torch.bincount(lab_idx, minlength=len(lab_names)).float()
    weights = (1.0 / counts[lab_idx]).float()  # balance labels when sampling clips
    lib = {
        "q": torch.tensor(q, device=device), "qd": torch.tensor(qd, device=device), "cmd": torch.tensor(cmd, device=device),
        "base_z": torch.tensor(bz, device=device), "len": torch.tensor(lens, device=device), "fps": float(clips[0]["fps"]),
        "joint_names": [str(s) for s in clips[0]["joint_names"]], "labels": labels, "label_names": lab_names,
        "label_idx": lab_idx.to(device), "weights": (weights / weights.sum()).to(device), "files": files,
    }
    print(f"[clip] library: {C} clips, {Tmax / lib['fps']:.1f} s max, labels {dict(zip(lab_names, counts.int().tolist()))}")
    _LIBS[key] = lib
    return lib


def _state(env: "ManagerBasedRLEnv") -> dict:
    st = getattr(env, "_clip_state", None)
    if st is None:
        n = env.num_envs
        st = {"clip": torch.zeros(n, dtype=torch.long, device=env.device), "t0": torch.zeros(n, dtype=torch.long, device=env.device)}
        env._clip_state = st
    return st


def _frame(env: "ManagerBasedRLEnv", lib: dict) -> torch.Tensor:
    st = _state(env)
    f = st["t0"] + env.episode_length_buf.long()
    return torch.minimum(f, lib["len"][st["clip"]] - 1)


# ---------------------------------------------------------------- observation
def clip_phase_obs(env: "ManagerBasedRLEnv", clip_dir: str) -> torch.Tensor:
    """Same 4-vector layout as gait_phase_obs: progress through the clip as a phase (and its anti-phase)."""
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    f = _frame(env, lib).float()
    phi = 2.0 * math.pi * f / lib["len"][st["clip"]].float()
    return torch.stack([torch.cos(phi), torch.sin(phi), torch.cos(phi + math.pi), torch.sin(phi + math.pi)], dim=1)


def clip_ref_obs(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, clip_dir: str, offsets: tuple = (1, 10)) -> torch.Tensor:
    """Upcoming reference joints, relative to the current joints, at +k frames for k in offsets
    (N, J*len(offsets)). Without this the policy cannot know WHICH of the 98 clips it is in — the
    progress phase alone gave a flat 19 deg error (runs 12:46 and 13:02). Standard multi-clip
    tracking input (BeyondMimic-style); the recordings stay physics-only, so the AMP stage is
    unaffected by the tracker's observation design."""
    asset = env.scene[asset_cfg.name]
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    f = _frame(env, lib)
    q_now = asset.data.joint_pos[:, asset_cfg.joint_ids]
    outs = []
    for k in offsets:
        fk = torch.minimum(f + int(k), lib["len"][st["clip"]] - 1)
        outs.append(lib["q"][st["clip"], fk] - q_now)
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------- events
def clip_rsi(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, clip_dir: str, min_remaining_s: float = 2.0,
             height_margin: float = 0.01) -> None:
    """Reset into a random clip at a random frame: joints + joint velocities from the clip, base at
    the clip's ground-follow height, base velocity = the clip's command at that frame."""
    asset = env.scene[asset_cfg.name]
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    n = len(env_ids)
    cid = torch.multinomial(lib["weights"], n, replacement=True)
    length = lib["len"][cid]
    max_t0 = (length - int(min_remaining_s * lib["fps"])).clamp(min=1)
    t0 = (torch.rand(n, device=env.device) * max_t0.float()).long()
    st["clip"][env_ids] = cid
    st["t0"][env_ids] = t0
    q = lib["q"][cid, t0]
    qd = lib["qd"][cid, t0]
    asset.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    root = asset.data.root_state_w[env_ids].clone()
    root[:, 2] = env.scene.env_origins[env_ids, 2] + lib["base_z"][cid, t0] + height_margin
    c = lib["cmd"][cid, t0]
    vel_b = torch.zeros(n, 3, device=env.device); vel_b[:, :2] = c[:, :2]
    root[:, 7:10] = quat_apply(yaw_quat(root[:, 3:7]), vel_b)
    root[:, 10:12] = 0.0
    root[:, 12] = c[:, 2]
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    _write_commands(env, lib, env_ids)


def _write_commands(env: "ManagerBasedRLEnv", lib: dict, env_ids=None) -> None:
    term = env.command_manager.get_term("base_velocity")
    st = _state(env)
    f = _frame(env, lib)
    if env_ids is None:
        term.vel_command_b[:] = lib["cmd"][st["clip"], f]
    else:
        term.vel_command_b[env_ids] = lib["cmd"][st["clip"][env_ids], f[env_ids]]


def clip_command(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, clip_dir: str) -> None:
    """Interval event (every step): the velocity command IS the clip's body motion at this frame."""
    lib = load_library(clip_dir, env.device)
    _write_commands(env, lib)


# ---------------------------------------------------------------- rewards
def track_clip_pose(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, clip_dir: str, sigma_deg: float = 5.0,
                    cmd_gate_lin: float = 0.1, cmd_gate_ang: float = 0.05, command_name: str = "base_velocity") -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    q_ref = lib["q"][st["clip"], _frame(env, lib)]
    err2 = ((asset.data.joint_pos[:, asset_cfg.joint_ids] - q_ref) ** 2).mean(dim=1)
    r = torch.exp(-err2 / math.radians(sigma_deg) ** 2)
    if cmd_gate_lin > 0.0:
        r = r * _cmd_track_gate(env, asset, command_name, cmd_gate_lin, cmd_gate_ang)
    return r


def track_clip_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, clip_dir: str, sigma: float = 1.5) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    qd_ref = lib["qd"][st["clip"], _frame(env, lib)]
    err2 = ((asset.data.joint_vel[:, asset_cfg.joint_ids] - qd_ref) ** 2).mean(dim=1)
    return torch.exp(-err2 / sigma ** 2)


# ---------------------------------------------------------------- termination
def clip_end(env: "ManagerBasedRLEnv", clip_dir: str) -> torch.Tensor:
    """Episode ends (time-out style) when the clip runs out."""
    lib = load_library(clip_dir, env.device)
    st = _state(env)
    return (st["t0"] + env.episode_length_buf.long()) >= lib["len"][st["clip"]] - 1


# ---------------------------------------------------------------- curriculum: logging + self-paced sigma
class clip_report(ManagerTermBase):
    def __call__(self, env: "ManagerBasedRLEnv", env_ids, asset_cfg: SceneEntityCfg, clip_dir: str,
                 sigma_min_deg: float = 5.0, sigma_max_deg: float = 20.0):
        asset = env.scene[asset_cfg.name]
        lib = load_library(clip_dir, env.device)
        st = _state(env)
        q_ref = lib["q"][st["clip"], _frame(env, lib)]
        err = (asset.data.joint_pos[:, asset_cfg.joint_ids] - q_ref) * 180.0 / math.pi
        rms_env = err.pow(2).mean(dim=1).sqrt()
        rmse = float(rms_env.mean())
        sigma = min(max(rmse, sigma_min_deg), sigma_max_deg)
        rm = env.reward_manager
        if "track_clip_pose" in rm.active_terms:
            rm.get_term_cfg("track_clip_pose").params["sigma_deg"] = sigma
        out = {"rmse_deg": rmse, "sigma_deg": sigma, "frac_within_5deg": float((rms_env < 5.0).float().mean())}
        lab = lib["label_idx"][st["clip"]]
        for i, name in enumerate(lib["label_names"]):
            m = lab == i
            if bool(m.any()):
                out[f"rmse_{name}_deg"] = float(rms_env[m].mean())
        return out
