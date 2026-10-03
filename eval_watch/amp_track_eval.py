"""Evaluate a reference-tracking checkpoint (AMP_PLAN stage 2b verdict) and record rollouts.

Runs the policy in its own task (Isaac-Track-KbotLegs-v0, full DR unless --nodr), and
reports what the pinned probe could not: falls, tracking error while CARRYING the body,
stance torques against the motor continuous ratings, and the gait geometry actually
produced (width, clearance, cadence, speed). Optionally writes the rollouts of the
non-falling envs to an npz in the motion-file layout (joint_pos, joint_vel, fps,
joint_names, base velocities) -> the AMP dataset candidate.

Usage:
  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_track_eval.py \
      --checkpoint logs/rsl_rl/kbot_legs_track/<run>/model_2000.pt --headless --record out.npz
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
parser.add_argument("--task", default="Isaac-Track-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--nodr", action="store_true", help="null the randomisers (nominal plant)")
parser.add_argument("--record", default="", help="npz path for rollouts of the surviving envs")
parser.add_argument("--cmd", default="", help="fixed command vx:vy:wz for every env (e.g. 0.3:0:0.3 = turn, 0.5:0:0 = fast)")
parser.add_argument("--tag", default="amp_track_eval")
parser.add_argument("--render_seconds", type=float, default=0.0, help="render env 0 for this many seconds (two views) -> gif + filmstrip")
parser.add_argument("--stochastic", action="store_true", help="sample actions from the policy distribution (as in training) instead of the mean")
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
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_track  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

CONTINUOUS = {"_04": 7.5, "_03": 5.0, "_02": 5.0}
OUT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    if args.nodr:
        for name in ("add_limb_masses", "randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints",
                     "randomize_joint_properties", "randomize_joint_friction_ankles", "randomize_joint_play", "physics_material"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        if hasattr(env_cfg.curriculum, "series_k_band"):
            env_cfg.curriculum.series_k_band = None
    env_cfg.episode_length_s = args.seconds + 1.0
    env_cfg.observations.policy.enable_corruption = True  # as trained
    if args.cmd:
        vx, vy, wz = (float(x) for x in args.cmd.split(":"))
        c = env_cfg.commands.base_velocity
        c.ranges.lin_vel_x = (vx, vx); c.ranges.lin_vel_y = (vy, vy); c.ranges.ang_vel_z = (wz, wz)
        c.resampling_time_range = (1000.0, 1000.0)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
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
    cyc_cfg = env_cfg.rewards.track_ref_pose.params
    cyc = mdp_track.load_cycle(cyc_cfg["cycle_file"], dev)
    gait_freq = cyc_cfg["gait_freq"]
    feet = [i for i, n in enumerate(robot.body_names) if n.endswith("FOOT")]
    n = args.num_envs
    T = int(args.seconds / uenv.step_dt)

    # The construction-time reset happens BEFORE the actuators' lazily-allocated per-env
    # buffers exist, so DR events that write into them (ankle K_s, play) silently skip:
    # the first episode would run every env at the nominal K_s = 23, which the policy
    # never trains on (measured: 45% survival vs ~80% in steady state). Warm up one
    # step so the buffers exist, then reset again to get the steady-state distribution.
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
    rec = {k: torch.zeros(T, n, J, device=dev) for k in ("q", "qd", "tau", "q_ref")}
    rec["foot_rel"] = torch.zeros(T, n, 2, 3, device=dev)
    rec["v_b"] = torch.zeros(T, n, 3, device=dev)
    rec["yaw_rate"] = torch.zeros(T, n, device=dev)
    cmd0 = uenv.command_manager.get_command("base_velocity").clone()
    rec["tilt"] = torch.zeros(T, n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_at = torch.full((n,), T, device=dev)
    cause = {k: 0 for k in uenv.termination_manager.active_terms}
    cause_all = {k: 0 for k in uenv.termination_manager.active_terms}  # every episode end, steady state
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
            actions = runner.alg.policy.act(obs) if args.stochastic else policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()
            if t < n_render and t % 2 == 0:
                for name, (eye, lookat) in views.items():
                    uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                    frames[name].append(capture())
            newly = dones & ~fell
            for k in cause:
                cause[k] += int((uenv.termination_manager.get_term(k) & newly).sum())
            for k in cause_all:
                cause_all[k] += int((uenv.termination_manager.get_term(k) & dones).sum())
            fell_at[newly] = t
            fell |= dones
            phi_l = mdp_track._phase_l(uenv, gait_freq)
            rec["q_ref"][t] = mdp_track._ref_at_phase(cyc["q"], phi_l)
            rec["q"][t] = robot.data.joint_pos
            rec["qd"][t] = robot.data.joint_vel
            rec["tau"][t] = robot.data.applied_torque
            fp = robot.data.body_pos_w[:, feet] - robot.data.root_pos_w.unsqueeze(1)
            qq = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
            rec["foot_rel"][t] = quat_apply_inverse(qq, fp.reshape(-1, 3)).reshape(n, 2, 3)
            rec["v_b"][t] = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            rec["yaw_rate"][t] = robot.data.root_ang_vel_b[:, 2]
            rec["tilt"][t] = torch.asin(robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)) * 180 / math.pi
    # env.step resets fallen envs and keeps going; score only the never-fallen envs
    ok = (~fell).cpu().numpy()
    R = {k: v.cpu().numpy() for k, v in rec.items()}
    skip = int(2.0 / uenv.step_dt)
    err = np.degrees(R["q"][skip:] - R["q_ref"][skip:])  # (T', n, J)
    rmse_joint = np.sqrt((err[:, ok] ** 2).mean(axis=(0, 1)))
    tau = R["tau"][skip:, ok]
    trms = np.sqrt((tau ** 2).mean(axis=(0, 1)))
    tpk = np.abs(tau).max(axis=(0, 1))
    fr = R["foot_rel"][skip:, ok]
    z = fr[:, :, :, 2]
    ground = z.min(axis=2, keepdims=True)
    h = z - ground
    both_down = (h[..., 0] < 0.01) & (h[..., 1] < 0.01)
    width = np.abs(fr[:, :, 0, 1] - fr[:, :, 1, 1])
    v = R["v_b"][skip:, ok]
    # cadence from left-foot touchdowns
    tds = []
    for e in range(ok.sum()):
        left = h[:, e, 1]
        air = False
        last = None
        for t in range(len(left)):
            if left[t] > 0.02:
                air = True
            elif air and left[t] < 0.005:
                if last is not None:
                    tds.append((t - last) * uenv.step_dt)
                last = t
                air = False
    res = {
        "checkpoint": args.checkpoint, "num_envs": n, "seconds": args.seconds, "dr": not args.nodr, "stochastic": args.stochastic,
        "survival_frac": float(ok.mean()), "first_fall_cause_counts": cause, "all_episode_end_counts": cause_all,
        "fall_time_s_percentiles_10_25_50_75_90": [float(x) for x in np.percentile(fell_at.cpu().numpy()[~ok] * uenv.step_dt, [10, 25, 50, 75, 90])] if (~ok).any() else None, "fall_time_s_median": float(np.median(fell_at.cpu().numpy()[~ok]) * uenv.step_dt) if (~ok).any() else None,
        "rmse_deg": {jn[j]: float(rmse_joint[j]) for j in range(J)},
        "rmse_all_deg": float(np.sqrt((err[:, ok] ** 2).mean())),
        "speed_mps": {"mean": float(v[..., 0].mean()), "lateral_abs": float(np.abs(v[..., 1]).mean()), "cmd": float(cyc["speed"])},
        "yaw_rate_abs": float(np.abs(R["yaw_rate"][skip:, ok]).mean()),
        "yaw": {"cmd_mean": float(cmd0[torch.tensor(ok, device=dev), 2].mean()), "achieved_mean_signed": float(R["yaw_rate"][skip:, ok].mean()),
                "tracking_err_mean": float(np.abs(R["yaw_rate"][skip:, ok].mean(axis=0) - cmd0[torch.tensor(ok, device=dev), 2].cpu().numpy()).mean())},
        "cmd_override": args.cmd,
        "tilt_deg_mean": float(R["tilt"][skip:, ok].mean()),
        "stance_width_cm": {"median": float(np.median(width[both_down]) * 100), "p90": float(np.percentile(width[both_down], 90) * 100)},
        "clearance_cm": float(np.percentile(h.max(axis=0).reshape(-1), 50) * 100),
        "stride_period_s": {"median": float(np.median(tds)) if tds else None, "ref": float(cyc["period"]), "n": len(tds)},
        "torque_rms_over_continuous": {jn[j]: float(trms[j] / CONTINUOUS[jn[j][-3:]]) for j in range(J)},
        "torque_peak_nm": {jn[j]: float(tpk[j]) for j in range(J)},
    }
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    if n_render > 0:
        import imageio.v2 as imageio
        import subprocess
        FF = os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.11", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-linux-x86_64-v7.0.2")
        for name in views:
            mp4 = os.path.join(OUT, f"{args.tag}_{name}.mp4")
            imageio.mimwrite(mp4, frames[name], fps=25, codec="libx264", quality=7, macro_block_size=None)
        gif = os.path.join(OUT, f"{args.tag}.gif")
        subprocess.run([FF, "-y", "-i", os.path.join(OUT, f"{args.tag}_side.mp4"), "-i", os.path.join(OUT, f"{args.tag}_rear34.mp4"), "-filter_complex",
                        "[0:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[a];[1:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[b];"
                        "[a][b]hstack,fps=10,split[s0][s1];[s0]palettegen=max_colors=32:stats_mode=diff[p];[s1][p]paletteuse=dither=none", gif],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # filmstrip: one stride (period) from t = 3 s, 8 tiles per view
        per = int(round(cyc["period"] / uenv.step_dt / 2))  # in rendered frames (every 2nd step)
        t0 = int(3.0 / uenv.step_dt / 2)
        rows = []
        for name in views:
            fr = frames[name]
            idx = [t0 + int(k * per / 8) for k in range(8)]
            if idx[-1] < len(fr):
                tiles = []
                for i in idx:
                    img = fr[i]; h = 360; w = int(img.shape[1] * h / img.shape[0])
                    ys = (np.arange(h) * img.shape[0] / h).astype(int); xs = (np.arange(w) * img.shape[1] / w).astype(int)
                    tiles.append(img[ys][:, xs])
                rows.append(np.concatenate(tiles, axis=1))
        if rows:
            imageio.imwrite(os.path.join(OUT, f"{args.tag}_filmstrip.png"), np.concatenate(rows, axis=0))
        print(f"[render] {gif} ({os.path.getsize(gif) / 1e6:.1f} MB) + filmstrip")
    if args.record:
        fps = 1.0 / uenv.step_dt
        np.savez(args.record, joint_names=np.array(jn), fps=fps, joint_pos=R["q"][:, ok], joint_vel=R["qd"][:, ok],
                 base_lin_vel_b=R["v_b"][:, ok], base_yaw_rate=R["yaw_rate"][:, ok], q_ref=R["q_ref"][:, ok],
                 layout="(T, n_env, J): per-env episodes of one policy, no resets inside")
        print(f"[record] {args.record}: {int(ok.sum())} envs x {T} steps @ {fps:.0f} Hz")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
