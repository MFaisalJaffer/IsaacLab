"""Evaluate the clip tracker per motion label and record its rollouts as an AMP dataset.

Runs Isaac-TrackClip-KbotLegs-v0 (RSI starts, clip commands) for T seconds. Per label: survival to
clip end, joint rmse vs the clip, achieved body velocity vs the clip's command (fwd / lateral / yaw).
--record writes (T, n_env, J) joint_pos/joint_vel at 50 Hz plus a `done` mask (episodes end at
clip end and restart in another clip; MotionDataset skips pairs that cross a reset) and per-step
labels/commands.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_clip_eval.py \
      --checkpoint <ckpt> --num_envs 256 --seconds 30 --record eval_watch/amp_refs/lafan1_tracked_kbot.npz --headless
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
parser.add_argument("--task", default="Isaac-TrackClip-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--record", default="")
parser.add_argument("--tag", default="amp_clip_eval")
parser.add_argument("--fall_window", type=int, default=25, help="frames before a fall marked invalid in the recording (tipping is not reference motion)")
parser.add_argument("--render_seconds", type=float, default=0.0, help="render env 0 for this many seconds (side + rear views) -> <tag>.gif")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.render_seconds > 0
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_clip  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if args.render_seconds > 0:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (960, 540)
        env_cfg.viewer.eye = (0.15, -2.15, 0.35)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.3)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.render_seconds > 0 else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    jn = robot.joint_names
    J = len(jn)
    n = args.num_envs
    T = int(args.seconds / uenv.step_dt)
    clip_dir = env_cfg.rewards.track_clip_pose.params["clip_dir"]
    lib = mdp_clip.load_library(clip_dir, dev)
    with torch.inference_mode():  # warm-up + re-reset (first-episode DR gap)
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
    R = {"q": torch.zeros(T, n, J, device=dev), "qd": torch.zeros(T, n, J, device=dev), "cmd": torch.zeros(T, n, 3, device=dev),
         "v": torch.zeros(T, n, 3, device=dev), "label": torch.zeros(T, n, dtype=torch.long, device=dev), "done": torch.zeros(T, n, dtype=torch.bool, device=dev),
         "err": torch.zeros(T, n, device=dev), "fell": torch.zeros(T, n, dtype=torch.bool, device=dev)}
    views = {"side": ((0.15, -2.15, 0.35), (0.0, 0.0, -0.3)), "rear34": ((-1.55, 1.25, 0.6), (0.0, 0.0, -0.35))}
    frames = {k: [] for k in views}
    n_render = int(args.render_seconds / uenv.step_dt)

    def capture():
        uenv.sim.render()
        uenv.sim.render()
        return uenv.render(recompute=True)

    with torch.inference_mode():
        if n_render > 0:
            for _ in range(8):
                capture()
        for t in range(T):
            st = uenv._clip_state
            R["label"][t] = lib["label_idx"][st["clip"]]
            q_ref = lib["q"][st["clip"], mdp_clip._frame(uenv, lib)]
            R["cmd"][t] = uenv.command_manager.get_command("base_velocity")
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if t < n_render and t % 2 == 0:
                for name, (eye, lookat) in views.items():
                    uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                    frames[name].append(capture())
            R["q"][t] = robot.data.joint_pos
            R["qd"][t] = robot.data.joint_vel
            vb = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            R["v"][t, :, :2] = vb[:, :2]
            R["v"][t, :, 2] = robot.data.root_ang_vel_b[:, 2]
            R["err"][t] = ((robot.data.joint_pos - q_ref) ** 2).mean(dim=1).sqrt() * 180 / math.pi
            R["done"][t] = dones.bool()
            R["fell"][t] = dones.bool() & ~uenv.termination_manager.get_term("clip_end")
    N = {k: v.cpu().numpy() for k, v in R.items()}
    res = {"checkpoint": args.checkpoint, "num_envs": n, "seconds": args.seconds, "labels": {}}
    for i, name in enumerate(lib["label_names"]):
        m = N["label"] == i
        if not m.any():
            continue
        v, c = N["v"][m], N["cmd"][m]
        ends = N["done"] & m
        falls = N["fell"] & m
        res["labels"][name] = {
            "steps": int(m.sum()), "episodes_ended": int(ends.sum()), "fell": int(falls.sum()), "fall_frac_of_ends": float(falls.sum() / max(ends.sum(), 1)),
            "rmse_deg": float(N["err"][m].mean()),
            "cmd_mean": [round(float(x), 3) for x in c.mean(0)], "achieved_mean": [round(float(x), 3) for x in v.mean(0)],
            "fwd_err": float(np.abs(v[:, 0] - c[:, 0]).mean()), "lat_err": float(np.abs(v[:, 1] - c[:, 1]).mean()), "yaw_err": float(np.abs(v[:, 2] - c[:, 2]).mean()),
        }
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    if args.record:
        # mask the window before every fall: MotionDataset drops pairs whose second frame is done, so marking
        # the whole window done removes the tipping frames while clip-end restarts stay single-frame cuts
        done_ext = N["done"].copy()
        T_, n_ = done_ext.shape
        for t_, e_ in zip(*np.nonzero(N["fell"])):
            done_ext[max(0, t_ - args.fall_window):t_ + 1, e_] = True
        print(f"[record] fall windows: {int(N['fell'].sum())} falls, {int(done_ext.sum() - N['done'].sum())} extra frames masked")
        np.savez(args.record, joint_names=np.array(jn), fps=1.0 / uenv.step_dt, joint_pos=N["q"], joint_vel=N["qd"], done=done_ext, done_raw=N["done"], label=N["label"],
                 label_names=np.array(lib["label_names"]), cmd=N["cmd"], base_vel=N["v"], layout="(T, n_env, J); pairs crossing done[t]=True are invalid")
        print(f"[record] {args.record}: {n} envs x {T} steps, {int(N['done'].sum())} episode boundaries")
    if n_render > 0:
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
        print(f"[render] {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
