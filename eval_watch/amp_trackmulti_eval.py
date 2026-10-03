"""Evaluate the multi-cycle tracker per cycle (forward / backward / side-steps / pivots), optionally
record its rollouts as an AMP dataset and render one robot.

Per cycle: survival (1 - falls / episodes), joint rmse vs the cycle, achieved body velocity (vx, vy,
yaw rate) vs the cycle's command. KBOT_TM_ONLY="backward" restricts the draw to some cycles (rendering).
--record writes (T, n_env, J) joint_pos / joint_vel at 50 Hz with a `done` mask (resets + the frames
before each fall), the cycle id and command per step.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_trackmulti_eval.py \
      --checkpoint <ckpt> --num_envs 256 --seconds 20 --tag tm_eval --headless
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
parser.add_argument("--task", default="Isaac-TrackMulti-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--record", default="")
parser.add_argument("--fall_window", type=int, default=25)
parser.add_argument("--tag", default="tm_eval")
parser.add_argument("--render_seconds", type=float, default=0.0, help="render env 0 (side + rear views) -> <tag>.gif")
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
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_track, mdp_trackmulti  # noqa: E402
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
    p = env_cfg.rewards.track_ref_pose.params
    lib = mdp_trackmulti.load_lib(p["lib_file"], dev)
    freq = p["gait_freq"]
    with torch.inference_mode():  # warm-up + re-reset (first-episode DR gap)
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
    R = {"q": torch.zeros(T, n, J, device=dev), "qd": torch.zeros(T, n, J, device=dev), "cmd": torch.zeros(T, n, 3, device=dev),
         "v": torch.zeros(T, n, 3, device=dev), "cyc": torch.zeros(T, n, dtype=torch.long, device=dev), "done": torch.zeros(T, n, dtype=torch.bool, device=dev),
         "err": torch.zeros(T, n, device=dev), "fell": torch.zeros(T, n, dtype=torch.bool, device=dev), "drift": torch.zeros(T, n, dtype=torch.bool, device=dev)}
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
            k = uenv._tm_cycle.clone()
            R["cyc"][t] = k
            q_ref = mdp_trackmulti._ref(lib["q"], k, mdp_trackmulti._tm_phase(uenv, lib, freq))
            R["cmd"][t] = uenv.command_manager.get_command("base_velocity")
            obs, _, dones, _ = env.step(policy(obs))
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
            ended = dones.bool() & ~uenv.termination_manager.get_term("time_out")
            if "root_drift" in uenv.termination_manager.active_terms:  # fell behind the command: not a fall
                R["drift"][t] = ended & uenv.termination_manager.get_term("root_drift")
                ended = ended & ~uenv.termination_manager.get_term("root_drift")
            R["fell"][t] = ended
    N = {k: v.cpu().numpy() for k, v in R.items()}
    res = {"checkpoint": args.checkpoint, "num_envs": n, "seconds": args.seconds, "cycles": {}}
    started = np.concatenate([np.ones((1, n), bool), N["done"][:-1]], axis=0)  # an episode starts at t=0 and after every done
    print(f"{'cycle':12s} {'episodes':>8s} {'falls':>6s} {'drift':>6s} {'surv':>6s} {'rmse':>6s} {'cmd vx/vy/wz':>22s} {'achieved vx/vy/wz':>22s}")
    for i, name in enumerate(lib["names"]):
        m = N["cyc"] == i
        if not m.any():
            continue
        ep = int((started & m).sum()); falls = int((N["fell"] & m).sum()); drifts = int((N["drift"] & m).sum())
        v, c = N["v"][m], N["cmd"][m]
        res["cycles"][name] = {"episodes": ep, "falls": falls, "drift_ends": drifts, "survival": float(1.0 - falls / max(ep, 1)), "rmse_deg": float(N["err"][m].mean()),
                               "cmd": [round(float(x), 3) for x in c.mean(0)], "achieved": [round(float(x), 3) for x in v.mean(0)]}
        r = res["cycles"][name]
        print(f"{name:12s} {ep:8d} {falls:6d} {drifts:6d} {r['survival']:6.2f} {r['rmse_deg']:6.1f} {r['cmd'][0]:7.2f}{r['cmd'][1]:7.2f}{r['cmd'][2]:7.2f}  {r['achieved'][0]:7.2f}{r['achieved'][1]:7.2f}{r['achieved'][2]:7.2f}")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    if args.record:
        done_ext = N["done"].copy()
        for t_, e_ in zip(*np.nonzero(N["fell"])):
            done_ext[max(0, t_ - args.fall_window):t_ + 1, e_] = True
        np.savez(args.record, joint_names=np.array(jn), fps=1.0 / uenv.step_dt, joint_pos=N["q"], joint_vel=N["qd"], done=done_ext, done_raw=N["done"],
                 cycle=N["cyc"], cycle_names=np.array(lib["names"]), cmd=N["cmd"], base_vel=N["v"], layout="(T, n_env, J); pairs crossing done[t]=True are invalid")
        print(f"[record] {args.record}: {n} envs x {T} steps, {int(N['fell'].sum())} falls, {int(done_ext.sum() - N['done'].sum())} pre-fall frames masked")
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
