"""Body velocity each clip IMPLIES on the K-Bot (kinematic): with the base held fixed, the stance
foot's motion in the base frame, negated, is the base velocity a perfectly tracking robot would
have (the stance foot is fixed to the ground). Compared with the clip's stored `cmd` (the retarget
source's root motion). If kin << cmd the labels ask for more speed than the joint trajectory can
deliver on our leg geometry, and a tracker that matches the joints will look 'slow'.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_clips_kinematic_speed.py --headless
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clips", default="/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1/clips")
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--out", default="/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1/clip_kinematic_speed.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


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
    fi = [[i for i, n in enumerate(bn) if "501R" in n][0], [i for i, n in enumerate(bn) if "501L" in n][0]]
    zero = torch.zeros(1, J, device=dev)
    zero6 = torch.zeros(1, 6, device=dev)
    root_hi = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=dev)

    def puppet(q):
        robot.write_root_pose_to_sim(root_hi); robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(q, zero); uenv.sim.step(render=False); uenv.scene.update(dt)

    res = {}
    by = {}
    with torch.inference_mode():
        for path in sorted(glob.glob(os.path.join(args.clips, "*.npz"))):
            d = dict(np.load(path, allow_pickle=True))
            fps = float(d["fps"]); lab = str(d["label"]); cmd = d["cmd"]
            q_all = torch.tensor(d["joint_pos"], device=dev)
            T = q_all.shape[0]
            rel = torch.zeros(T, 2, 3, device=dev); yaw = torch.zeros(T, 2, device=dev)
            for t in range(T):
                puppet(q_all[t:t + 1])
                rel[t] = robot.data.body_pos_w[0, fi] - robot.data.root_pos_w[0]
                qf = robot.data.body_quat_w[0, fi]  # (2, 4) wxyz
                yaw[t] = torch.atan2(2 * (qf[:, 0] * qf[:, 3] + qf[:, 1] * qf[:, 2]), 1 - 2 * (qf[:, 2] ** 2 + qf[:, 3] ** 2))
            p = rel.cpu().numpy(); y = np.unwrap(yaw.cpu().numpy(), axis=0)
            stance = np.argmin(p[:, :, 2], axis=1)
            idx = np.arange(T - 1)
            v_imp = -(p[1:] - p[:-1])[idx, stance[:-1]] * fps  # (T-1, 3)
            w_imp = -(y[1:] - y[:-1])[idx, stance[:-1]] * fps  # (T-1,)
            kin = [float(v_imp[:, 0].mean()), float(v_imp[:, 1].mean()), float(w_imp.mean())]
            c = [float(cmd[:, 0].mean()), float(cmd[:, 1].mean()), float(cmd[:, 2].mean())]
            # signed-agreement ratios over the clip's own frames (robust to sign changes within a clip)
            ratios = []
            for k in range(3):
                num = float((v_imp[:, k] if k < 2 else w_imp) @ cmd[:-1, k]); den = float(cmd[:-1, k] @ cmd[:-1, k])
                ratios.append(num / den if den > 0.05 * (T - 1) * (0.05 ** 2) and np.sqrt(den / (T - 1)) > 0.05 else float("nan"))
            res[os.path.basename(path)[:-4]] = {"label": lab, "cmd_mean": c, "kin_mean": kin, "ratio_vx_vy_wz": ratios}
            by.setdefault(lab, []).append(c + kin + ratios)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{'label':16s} {'cmd_vx':>7s} {'kin_vx':>7s} {'cmd_vy':>7s} {'kin_vy':>7s} {'cmd_wz':>7s} {'kin_wz':>7s} | {'ratio vx':>8s} {'vy':>6s} {'wz':>6s}")
    for lab, rows in sorted(by.items()):
        r = np.array(rows); m = r[:, :6].mean(axis=0); rt = np.nanmean(r[:, 6:9], axis=0)
        print(f"{lab:16s} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} {m[3]:7.3f} {m[4]:7.3f} {m[5]:7.3f} | {rt[0]:8.2f} {rt[1]:6.2f} {rt[2]:6.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
