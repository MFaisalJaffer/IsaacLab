"""Balance pass for a clip library: pivot both legs about the hips, frame by frame, so the stance foot
sits where our robot needs it (under the hip when stopped, a small lead proportional to speed when
walking — the proven stride keeps its stance foot 0.034 m behind the hip at 0.4 m/s). Knee shape is
kept; the ankles are re-levelled by the same angle so the soles stay as they were.

Why: the retargeted source stands with its hips extended ~7 deg and the pelvis tilted back, which on our
vertical-thigh robot puts the feet 11-13 cm BEHIND the hips at a standstill (user-spotted in the
stop_start gif), while its walking posture put them 5-16 cm IN FRONT. No constant angle offset fits both.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_clips_balance.py \
      --src eval_watch/amp_refs/lafan1/clips_v3 --dst eval_watch/amp_refs/lafan1/clips_v4 --headless
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--lead_per_mps", type=float, default=-0.085, help="target stance-foot x per m/s of forward speed (stride: -0.034 at 0.4)")
parser.add_argument("--max_lead", type=float, default=0.05)
parser.add_argument("--smooth_s", type=float, default=1.0, help="window over which the measured offset is averaged (one stride)")
parser.add_argument("--max_pivot_deg", type=float, default=20.0)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LEG = 0.96  # hip-to-foot (m), standing


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
    hi = [[i for i, n in enumerate(bn) if "102R" in n][0], [i for i, n in enumerate(bn) if "102L" in n][0]]
    zero = torch.zeros(1, J, device=dev); zero6 = torch.zeros(1, 6, device=dev)
    root_hi = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=dev)

    def measure(q_all):
        """per frame: stance-foot x relative to the hips (base frame), both feet z, foot pitch (deg)."""
        T = q_all.shape[0]
        sx = np.zeros(T); fz = np.zeros((T, 2)); fp = np.zeros((T, 2))
        with torch.inference_mode():
            for t in range(T):
                robot.write_root_pose_to_sim(root_hi); robot.write_root_velocity_to_sim(zero6)
                robot.write_joint_state_to_sim(q_all[t:t + 1], zero); uenv.sim.step(render=False); uenv.scene.update(dt)
                p = robot.data.body_pos_w[0]; qf = robot.data.body_quat_w[0, fi]
                feet = p[fi] - p[hi].mean(dim=0, keepdim=True)
                k = int(torch.argmin(feet[:, 2]))
                sx[t] = float(feet[k, 0]); fz[t] = feet[:, 2].cpu().numpy()
                fp[t] = torch.rad2deg(torch.asin((2 * (qf[:, 0] * qf[:, 2] - qf[:, 3] * qf[:, 1])).clamp(-1, 1))).cpu().numpy()
        return sx, fz, fp

    def smooth(x, k):
        k = max(1, int(k)); pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k - k // 2 - 1, x[-1])])
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    files = sorted(glob.glob(os.path.join(args.src, "*.npz")))
    assert files, args.src
    os.makedirs(args.dst, exist_ok=True)
    # ---- ankle sign: pivoting the legs by d must leave the soles level; test on the first clip
    d0 = np.load(files[0], allow_pickle=True); q0 = torch.tensor(d0["joint_pos"][:60], device=dev)
    _, _, fp_ref = measure(q0)
    best = None
    for s_ank in (1.0, -1.0):
        q = q0.clone(); d = np.radians(10.0)
        q[:, 0] += d; q[:, 1] -= d; q[:, 8] += s_ank * d; q[:, 9] -= s_ank * d
        _, _, fp = measure(q)
        err = float(np.abs(fp - fp_ref).mean())
        print(f"[ankle] sign {s_ank:+.0f}: sole pitch change {err:.2f} deg")
        if best is None or err < best[1]:
            best = (s_ank, err)
    s_ank = best[0]
    print(f"[ankle] using ankle sign {s_ank:+.0f} (hip pivot: left +, right -)")
    report = {}
    for path in files:
        d = dict(np.load(path, allow_pickle=True))
        q = np.asarray(d["joint_pos"], np.float64); fps = float(d["fps"]); T = q.shape[0]
        cmd = np.asarray(d["cmd"]); vx = smooth(cmd[:, 0].astype(np.float64), args.smooth_s * fps)
        sx, _, fp_before = measure(torch.tensor(q, dtype=torch.float32, device=dev))
        x_meas = smooth(sx, args.smooth_s * fps)
        x_t = np.clip(args.lead_per_mps * vx, -args.max_lead, args.max_lead)
        delta = np.arcsin(np.clip((x_t - x_meas) / LEG, -1, 1))
        delta = np.clip(delta, -np.radians(args.max_pivot_deg), np.radians(args.max_pivot_deg))
        q2 = q.copy()
        q2[:, 0] += delta; q2[:, 1] -= delta; q2[:, 8] += s_ank * delta; q2[:, 9] -= s_ank * delta
        sx2, fz2, fp_after = measure(torch.tensor(q2, dtype=torch.float32, device=dev))
        slow = np.abs(vx) < 0.08
        lab = str(d["label"])
        r = report.setdefault(lab, {"n": 0, "before_slow": [], "after_slow": [], "before_walk": [], "after_walk": [], "pivot_deg": [], "sole_change": []})
        r["n"] += 1
        if slow.any():
            r["before_slow"].append(float(sx[slow].mean())); r["after_slow"].append(float(sx2[slow].mean()))
        if (~slow).any():
            r["before_walk"].append(float(sx[~slow].mean())); r["after_walk"].append(float(sx2[~slow].mean()))
        r["pivot_deg"].append(float(np.degrees(np.abs(delta)).mean())); r["sole_change"].append(float(np.abs(fp_after - fp_before).mean()))
        d["joint_pos"] = q2.astype(np.float32)
        d["joint_vel"] = np.gradient(q2, 1.0 / fps, axis=0).astype(np.float32)
        d["balance_pivot_deg"] = np.degrees(delta).astype(np.float32)
        if "cmd_orig" in d:  # relabel again after the ground/kinematic passes on the new library
            d["cmd"] = d["cmd_orig"]; d.pop("cmd_orig"); d.pop("kin_ratio", None)
        d.pop("base_z", None)
        np.savez(os.path.join(args.dst, os.path.basename(path)), **d)
    print(f"{'label':16s} {'n':>3s} {'stand before':>12s} {'after':>7s} {'walk before':>12s} {'after':>7s} {'pivot':>6s} {'sole chg':>8s}   (stance-foot x rel. hip, m; + = in front)")
    for lab, r in sorted(report.items()):
        m = lambda k: (np.mean(r[k]) if r[k] else float("nan"))
        print(f"{lab:16s} {r['n']:3d} {m('before_slow'):12.3f} {m('after_slow'):7.3f} {m('before_walk'):12.3f} {m('after_walk'):7.3f} {m('pivot_deg'):6.1f} {m('sole_change'):8.2f}")
    json.dump({k: {kk: (float(np.mean(v)) if isinstance(v, list) and v else v) for kk, v in r.items()} for k, r in report.items()}, open(os.path.join(args.dst, "balance_report.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
