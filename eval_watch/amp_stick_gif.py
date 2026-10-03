"""Stick-figure gif of a clip's legs from the physics puppet (no viewport renderer, so it runs next to
training): base held upright at the clip's ground-follow height, link origins read back, drawn as a side
view (x-z) and a rear view (y-z). --compare draws a second clip (e.g. the pre-fix version) alongside.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_stick_gif.py \
      --clip eval_watch/amp_refs/lafan1/clips_v3/backward_0.npz --compare eval_watch/amp_refs/lafan1/clips_v2/backward_0.npz \
      --out eval_watch/posture_backward_0.gif --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", required=True)
parser.add_argument("--compare", default="")
parser.add_argument("--labels", default="after fix (clips_v3),before (clips_v2)")
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--travel", action="store_true", help="move the figure by the clip's integrated body velocity (cmd) so stepping shows as travel")
parser.add_argument("--fps_out", type=float, default=12.5)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

CHAIN_L = ["floating_base_link", "KC_D_102L_L_Hip_Yoke_Drive", "RS03_5", "KC_D_301L_L_Femur_Lower_Drive", "KC_D_401L_L_Shin_Drive", "KB_D_501L_L_LEG_FOOT"]
CHAIN_R = ["floating_base_link", "KC_D_102R_R_Hip_Yoke_Drive", "RS03_4", "KC_D_301R_R_Femur_Lower_Drive", "KC_D_401R_R_Shin_Drive", "KB_D_501R_R_LEG_FOOT"]


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.physics_dt
    dev = uenv.device
    with torch.inference_mode():
        env.reset()
    J = len(robot.joint_names)
    bn = robot.body_names
    for n in CHAIN_L + CHAIN_R:
        assert n in bn, f"{n} not in {bn}"
    iL = [bn.index(n) for n in CHAIN_L]; iR = [bn.index(n) for n in CHAIN_R]
    zero = torch.zeros(1, J, device=dev); zero6 = torch.zeros(1, 6, device=dev)

    def trace(path):
        d = np.load(path, allow_pickle=True)
        q = np.asarray(d["joint_pos"], np.float32); fps = float(d["fps"])
        bz = np.asarray(d["base_z"], np.float32) if "base_z" in d else np.full(q.shape[0], 1.0, np.float32)
        n = min(q.shape[0], int(args.seconds * fps))
        step = max(1, int(round(fps / args.fps_out)))
        idx = np.arange(0, n, step)
        pts = np.zeros((len(idx), 2, len(CHAIN_L), 3), np.float32)
        with torch.inference_mode():
            for k, t in enumerate(idx):
                root = torch.tensor([[0.0, 0.0, float(bz[t]), 1.0, 0.0, 0.0, 0.0]], device=dev)
                robot.write_root_pose_to_sim(root); robot.write_root_velocity_to_sim(zero6)
                robot.write_joint_state_to_sim(torch.tensor(q[t:t + 1], device=dev), zero)
                uenv.sim.step(render=False); uenv.scene.update(dt)
                p = robot.data.body_pos_w[0].cpu().numpy()
                pts[k, 0] = p[iL]; pts[k, 1] = p[iR]
        trav = np.zeros((len(idx), 2), np.float32)
        if args.travel and "cmd" in d:
            c = np.asarray(d["cmd"], np.float32)[:n, :2]
            trav = np.cumsum(c, axis=0)[idx] / fps  # body-frame vx, vy integrated (yaw ignored)
        return (pts, trav), fps / step

    clips = [(args.clip, trace(args.clip))]
    if args.compare:
        clips.append((args.compare, trace(args.compare)))
    tmax = max(float(np.abs(c[1][0][1]).max()) for c in clips) if args.travel else 0.0
    labels = [s.strip() for s in args.labels.split(",")]
    n_fr = min(len(c[1][0][0]) for c in clips)
    frames = []
    for k in range(n_fr):
        fig, axes = plt.subplots(2, len(clips), figsize=((4.2 + 2.5 * tmax) * len(clips), 7.5), squeeze=False)
        for c, (path, ((pts, trav), _)) in enumerate(clips):
            for r, (ax_i, view) in enumerate(((0, "side view (x fwd)"), (1, "rear view (y left)"))):
                ax = axes[r][c]
                ax.axhline(0.0, color="k", lw=1)
                off = float(trav[k, ax_i]) if args.travel else 0.0
                for leg, col in ((0, "tab:blue"), (1, "tab:red")):
                    P = pts[k, leg]
                    ax.plot(P[:, ax_i] + off, P[:, 2], "-o", color=col, lw=2.5, ms=3)
                hip = pts[k, :, 1, :].mean(axis=0)
                ax.axvline(hip[ax_i] + off, color="gray", ls="--", lw=1)
                if args.travel:
                    ax.plot(trav[:k + 1, ax_i], np.zeros(k + 1) - 0.02, "-", color="gray", lw=1)
                lim = 0.6 + tmax
                ax.set_xlim(-lim, lim); ax.set_ylim(-0.05, 1.25); ax.set_aspect("equal")
                ax.set_title(f"{labels[c] if c < len(labels) else os.path.basename(path)}\n{view}", fontsize=9)
                ax.grid(alpha=0.3)
        fig.suptitle(f"{os.path.basename(args.clip)[:-4]}  t={k / clips[0][1][1]:.2f}s  (blue L, red R; dashed = hip)", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(img)
        plt.close(fig)
    imageio.mimwrite(args.out, frames, duration=1.0 / args.fps_out, loop=0)
    print(f"[out] {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.1f} MB")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
