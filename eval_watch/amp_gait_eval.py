"""Evaluate an AMP checkpoint: does the walking look like the reference dataset?

Loads the checkpoint through KbotAmpRunner in the AMP task (same env vars as training),
runs N envs for T seconds with full DR (after a warm-up step + re-reset so the first
episode gets the real DR draw), and reports, for the envs commanded to walk:
survival, stance width, swing clearance, stride period, per-joint-family motion ranges
(p5..p95, deg) next to the SAME statistics computed from the dataset, hip-yaw offset,
speed tracking, torso tilt, and the discriminator's own mean style score. Standing envs
report quiet-stand tilt. --render_seconds renders env 0 (use --no_standers so env 0 walks).

Usage:
  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 KBOT_AMP_VX=0.30:0.50 ./isaaclab.sh -p eval_watch/amp_gait_eval.py \
      --checkpoint logs/rsl_rl/kbot_legs_amp/<run>/model_4999.pt --num_envs 256 --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--no_standers", action="store_true", help="every env walks (use for rendering env 0)")
parser.add_argument("--no_push", action="store_true", help="disable the velocity-kick and sustained-force push events (A/B the cost of pushes)")
parser.add_argument("--cmd", default="", help="fixed command vx:vy:wz for every env (spawn/corridor events off), e.g. 0:0:0 = stand, 0.3:0:0.3 = turn")
parser.add_argument("--render_seconds", type=float, default=0.0)
parser.add_argument("--tag", default="amp_gait_eval")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.render_seconds > 0
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle")


def fam_stats(q_deg: np.ndarray, names: list[str]) -> dict:
    """q_deg (..., J) -> per family: p5, p95 of |angle| pooled over L/R (mirrored sign)."""
    out = {}
    for fam in FAMILIES:
        cols = [i for i, n in enumerate(names) if fam in n]
        v = q_deg[..., cols].reshape(-1)
        out[fam] = {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                    "abs_p95": float(np.percentile(np.abs(v), 95))}
    return out


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.episode_length_s = args.seconds + 1.0
    if args.no_standers:
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if args.no_push:
        for name in ("push_robot", "sustained_push"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        for name in ("velocity_push_curriculum", "sustained_push_level"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)
    if args.cmd:
        vx, vy, wz = (float(x) for x in args.cmd.split(":"))
        c = env_cfg.commands.base_velocity
        c.ranges.lin_vel_x = (vx, vx); c.ranges.lin_vel_y = (vy, vy); c.ranges.ang_vel_z = (wz, wz)
        c.rel_standing_envs = 1.0 if (vx == 0 and vy == 0 and wz == 0) else 0.0
        c.resampling_time_range = (1000.0, 1000.0)
        for name in ("walk_at_spawn", "stand_corridor"):  # these rewrite commands at spawn / on transitions
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
    if args.render_seconds > 0:
        env_cfg.viewer.origin_type = "asset_root"; env_cfg.viewer.asset_name = "robot"; env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (960, 540); env_cfg.viewer.eye = (0.15, -2.15, 0.35); env_cfg.viewer.lookat = (0.0, 0.0, -0.3)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.render_seconds > 0 else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    disc = runner.alg.disc
    dataset = runner.alg.dataset
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    jn = robot.joint_names
    J = len(jn)
    feet = [i for i, n in enumerate(robot.body_names) if n.endswith("FOOT")]
    n = args.num_envs
    T = int(args.seconds / uenv.step_dt)

    with torch.inference_mode():  # warm-up + re-reset: first-episode DR gap (see amp_track_eval.py)
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, extras = env.reset()
    cmd = uenv.command_manager.get_command("base_velocity").clone()
    walking = (torch.norm(cmd[:, :3], dim=1) >= 0.1).cpu().numpy()
    print(f"[eval] commands after reset (first 4): {cmd[:4].cpu().numpy().round(3).tolist()}  walking envs {int(walking.sum())}/{n}")
    rec = {"q": torch.zeros(T, n, J, device=dev), "foot_rel": torch.zeros(T, n, 2, 3, device=dev), "v_b": torch.zeros(T, n, 3, device=dev),
           "tilt": torch.zeros(T, n, device=dev), "style": torch.zeros(T, n, device=dev), "yaw_rate": torch.zeros(T, n, device=dev)}
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    views = {"side": ((0.15, -2.15, 0.35), (0.0, 0.0, -0.3)), "rear34": ((-1.55, 1.25, 0.6), (0.0, 0.0, -0.35))}
    frames = {k: [] for k in views}
    n_render = int(args.render_seconds / uenv.step_dt)

    def capture():
        uenv.sim.render(); uenv.sim.render()
        return uenv.render(recompute=True)

    with torch.inference_mode():
        if n_render > 0:
            for _ in range(8):
                capture()
        for t in range(T):
            actions = policy(obs)
            obs, _, dones, infos = env.step(actions)
            fell |= dones.bool()
            rec["q"][t] = robot.data.joint_pos
            fp = robot.data.body_pos_w[:, feet] - robot.data.root_pos_w.unsqueeze(1)
            qq = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
            rec["foot_rel"][t] = quat_apply_inverse(qq, fp.reshape(-1, 3)).reshape(n, 2, 3)
            rec["v_b"][t] = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            rec["yaw_rate"][t] = robot.data.root_ang_vel_b[:, 2]
            rec["tilt"][t] = torch.asin(robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)) * 180 / math.pi
            rec["style"][t] = disc.predict_reward(infos["observations"]["amp"])
            if t < n_render and t % 2 == 0:
                for name, (eye, lookat) in views.items():
                    uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                    frames[name].append(capture())
    def write_render():
        if n_render <= 0:
            return
        import subprocess
        import imageio.v2 as imageio
        FF = os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.11", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-linux-x86_64-v7.0.2")
        for name in views:
            imageio.mimwrite(os.path.join(OUT, f"{args.tag}_{name}.mp4"), frames[name], fps=25, codec="libx264", quality=7, macro_block_size=None)
        gif = os.path.join(OUT, f"{args.tag}.gif")
        subprocess.run([FF, "-y", "-i", os.path.join(OUT, f"{args.tag}_side.mp4"), "-i", os.path.join(OUT, f"{args.tag}_rear34.mp4"), "-filter_complex",
                        "[0:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[a];[1:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[b];"
                        "[a][b]hstack,fps=10,split[s0][s1];[s0]palettegen=max_colors=32:stats_mode=diff[p];[s1][p]paletteuse=dither=none", gif],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        per = int(round(1.06 / uenv.step_dt / 2)); t0 = int(3.0 / uenv.step_dt / 2); rows = []
        for name in views:
            fr_ = frames[name]; idx = [t0 + int(k * per / 8) for k in range(8)]
            if idx[-1] < len(fr_):
                tiles = []
                for i in idx:
                    img = fr_[i]; hh = 360; ww = int(img.shape[1] * hh / img.shape[0])
                    ys = (np.arange(hh) * img.shape[0] / hh).astype(int); xs = (np.arange(ww) * img.shape[1] / ww).astype(int)
                    tiles.append(img[ys][:, xs])
                rows.append(np.concatenate(tiles, axis=1))
        if rows:
            imageio.imwrite(os.path.join(OUT, f"{args.tag}_filmstrip.png"), np.concatenate(rows, axis=0))
        print(f"[render] {gif} ({os.path.getsize(gif) / 1e6:.1f} MB) + filmstrip")

    ok = (~fell).cpu().numpy()
    R = {k: v.cpu().numpy() for k, v in rec.items()}
    skip = int(2.0 / uenv.step_dt)
    w = walking & ok
    s = (~walking) & ok
    stand = None
    if s.any():
        fs = R["foot_rel"][skip:, s]
        hz = fs[..., 2] - fs[..., 2].min(axis=2, keepdims=True)
        disp = np.linalg.norm(np.diff(fs[..., :2], axis=0), axis=-1).sum(axis=0) / (args.seconds - 2.0)  # m/s of foot travel in body frame
        stand = {"envs": int((~walking).sum()), "survival": float(ok[~walking].mean()),
                 "tilt_deg": {"mean": float(R["tilt"][skip:, s].mean()), "p90": float(np.percentile(R["tilt"][skip:, s], 90))},
                 "foot_lift_max_cm": float(hz.max() * 100), "foot_travel_mps_mean": float(disp.mean()),
                 "width_cm": float(np.median(np.abs(fs[:, :, 0, 1] - fs[:, :, 1, 1])) * 100),
                 "base_speed_abs_mps": float(np.linalg.norm(R["v_b"][skip:, s][..., :2], axis=-1).mean())}
    if not w.any():
        # either no env was commanded to walk, or every walking env fell before the scoring window
        res = {"checkpoint": args.checkpoint, "cmd": args.cmd, "standing": stand, "walking_envs": int(walking.sum()),
               "survival_walking": float(ok[walking].mean()) if walking.any() else None, "note": "no walking env survived" if walking.any() else "no walking envs"}
        print(json.dumps(res, indent=1))
        with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
        write_render()  # a stand-only test has no walking envs but can still be rendered
        env.close()
        return 0
    q = np.degrees(R["q"][skip:, w])
    fr = R["foot_rel"][skip:, w]
    z = fr[..., 2]
    h = z - z.min(axis=2, keepdims=True)
    both_down = (h[..., 0] < 0.01) & (h[..., 1] < 0.01)
    width = np.abs(fr[:, :, 0, 1] - fr[:, :, 1, 1])
    periods = []
    for e in range(w.sum()):
        left = h[:, e, 1]; air = False; last = None
        for t in range(len(left)):
            if left[t] > 0.02:
                air = True
            elif air and left[t] < 0.005:
                if last is not None:
                    periods.append((t - last) * uenv.step_dt)
                last = t; air = False
    v = R["v_b"][skip:, w]
    cmd_w = cmd[torch.tensor(w, device=dev)].cpu().numpy()
    # dataset statistics in the same units
    dq = np.degrees(dataset.x[: dataset.n_real, :J].cpu().numpy())  # q_t of every real transition
    res = {
        "checkpoint": args.checkpoint, "num_envs": n, "seconds": args.seconds, "walking_envs": int(walking.sum()),
        "survival_walking": float(ok[walking].mean()) if walking.any() else None, "survival_standing": float(ok[~walking].mean()) if (~walking).any() else None,
        "style_score_walking": float(R["style"][skip:, w].mean()), "style_score_dataset_selfcheck": float(disc.predict_reward(dataset.sample(8192)).mean()),
        "stance_width_cm": {"median": float(np.median(width[both_down]) * 100), "p90": float(np.percentile(width[both_down], 90) * 100)},
        "clearance_cm": float(np.median(h.max(axis=0).reshape(-1)) * 100),
        "stride_period_s": {"median": float(np.median(periods)) if periods else None, "p10": float(np.percentile(periods, 10)) if periods else None,
                            "p90": float(np.percentile(periods, 90)) if periods else None, "n": len(periods)},
        "speed": {"cmd_mean": float(np.abs(cmd_w[:, 0]).mean()), "achieved_fwd_mean": float(v[..., 0].mean()),
                  "tracking_err_mean": float(np.abs(v[..., 0].mean(axis=0) - cmd_w[:, 0]).mean()), "lateral_abs": float(np.abs(v[..., 1]).mean()),
                  "lateral_mean_signed": float(v[..., 1].mean()), "cmd_lateral_mean": float(cmd_w[:, 1].mean())},
        "yaw_rate_abs": float(np.abs(R["yaw_rate"][skip:, w]).mean()),
        "yaw": {"cmd_mean": float(cmd_w[:, 2].mean()), "achieved_mean_signed": float(R["yaw_rate"][skip:, w].mean()),
                "tracking_err_mean": float(np.abs(R["yaw_rate"][skip:, w].mean(axis=0) - cmd_w[:, 2]).mean())},
        "standing": stand, "cmd_override": args.cmd,
        "walk_tilt_deg": float(R["tilt"][skip:, w].mean()), "quiet_stand_tilt_deg": float(R["tilt"][skip:, s].mean()) if s.any() else None,
        "joint_ranges_deg": {"policy": fam_stats(q, jn), "dataset": fam_stats(dq, jn)},
        "dataset": {"stride_period_s": 1.06, "stance_width_cm": 30.0, "clearance_cm": 11.4},
    }
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_render()
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
