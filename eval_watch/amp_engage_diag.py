"""Closer look at one engage condition: when robots fall, and what the stand looks like on the way there.

WARNING (2026-10-03): this script uses a stripped plant (no training randomization, NO ANKLE JOINT FRICTION, no
sensor noise). Policies were never trained on that plant; the slow drift it shows is not evidence about the trained
plant. Use amp_engage_test.py --plant trained (the default) for conclusions.

Same setup as amp_engage_test.py (rest at the zero pose, stand command and hard pin from tick 0, nominal plant,
ankle spring 52 with 1 deg play), every robot under the same torso moment. Prints fall times and, in 0.5 s bins,
the mean torso pitch/roll tilt, base horizontal speed, ankle actions and knee actions of the robots still up.

  ... ./isaaclab.sh -p eval_watch/amp_engage_diag.py --checkpoint <ckpt> --mx 0 --my 2.5 --headless
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--mx", type=float, default=0.0)
parser.add_argument("--my", type=float, default=2.5)
parser.add_argument("--rest", type=int, default=1, help="1 = spawn at rest at the zero pose; 0 = the training reset (joint offsets, base velocity)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> int:
    n = args.num_envs
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 2.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0; c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    for name in ("walk_at_spawn", "stand_corridor", "push_robot", "randomize_actuator_gains", "randomize_gains_small_joints",
                 "randomize_gains_04_joints", "add_limb_masses", "randomize_joint_properties", "randomize_imu_mount",
                 "randomize_joint_friction_ankles", "randomize_joint_play", "physics_material", "amp_unanswered", "amp_axis_bias"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "series_k_band", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    if args.rest:
        ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
    ev.sustained_push.interval_range_s = (0.02, 0.02)
    env_cfg.observations.policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    jn = list(robot.joint_names)
    ix = {k: jn.index(k) for k in jn}
    aL, aR, kL, kR = ix["dof_left_ankle_02"], ix["dof_right_ankle_02"], ix["dof_left_knee_04"], ix["dof_right_knee_04"]
    load = torch.tensor([[args.mx, args.my]], device=dev).expand(n, 2).clone()

    def set_plant():
        uenv._standing_moment_override = load
        for act in robot.actuators.values():
            shape = (n, len(act.joint_names))
            if getattr(act, "_series_k", 0.0) > 0.0:
                act._series_k_env = torch.full(shape, 52.0, device=dev)
                act._play = torch.full(shape, math.radians(1.0), device=dev)
            else:
                act._play = torch.zeros(shape, device=dev)

    T = int(args.seconds / uenv.step_dt)
    rec = {k: torch.zeros(T, n, device=dev) for k in ("gx", "gy", "vx", "vy", "aL", "aR", "kL", "kR", "qaL", "qaR", "z")}
    fell_at = torch.full((n,), float("inf"), device=dev)
    reason = {}
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        set_plant()
        obs, _ = env.reset()
        set_plant()
        x0 = robot.data.root_pos_w[:, :2].clone()
        for t in range(T):
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            newly = dones.bool() & torch.isinf(fell_at)
            if newly.any():
                fell_at[newly] = (t + 1) * uenv.step_dt
                tm = uenv.termination_manager
                for name in tm.active_terms:
                    k = int((tm.get_term(name) & newly).sum())
                    if k:
                        reason[name] = reason.get(name, 0) + k
            g = robot.data.projected_gravity_b
            vb = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            rec["gx"][t] = g[:, 0]; rec["gy"][t] = g[:, 1]; rec["vx"][t] = vb[:, 0]; rec["vy"][t] = vb[:, 1]
            rec["aL"][t] = a[:, aL]; rec["aR"][t] = a[:, aR]; rec["kL"][t] = a[:, kL]; rec["kR"][t] = a[:, kR]
            rec["qaL"][t] = robot.data.joint_pos[:, aL]; rec["qaR"][t] = robot.data.joint_pos[:, aR]; rec["z"][t] = robot.data.root_pos_w[:, 2]
    up = torch.isinf(fell_at)
    ft = fell_at[~up]
    print(f"\n[diag] {args.checkpoint} | torso moment ({args.mx:+.1f}, {args.my:+.1f}) Nm | start {'at rest' if args.rest else 'training reset'} | {n} robots, {args.seconds} s")
    print(f"[diag] fell {int((~up).sum())}/{n} ({100 * float((~up).float().mean()):.0f}%); reasons {reason}; fall times s: "
          f"{'none' if len(ft) == 0 else 'min %.1f median %.1f max %.1f' % (float(ft.min()), float(ft.median()), float(ft.max()))}")
    if len(ft):
        hist = torch.histc(ft, bins=int(args.seconds), min=0.0, max=args.seconds).int().tolist()
        print(f"[diag] falls per second of the test: {hist}")
    print(f"[diag] {'t (s)':>6s} {'up':>4s} | lean fwd/back (g_x) and side (g_y), deg | base speed x / y, m/s | ankle action L / R | knee action L / R | ankle angle L / R, deg | height")
    bins = int(args.seconds / 0.5)
    for b in range(bins):
        s, e = int(b * 0.5 / uenv.step_dt), int((b + 1) * 0.5 / uenv.step_dt)
        alive = fell_at > (b + 1) * 0.5
        if not alive.any():
            break
        m = lambda k: float(rec[k][s:e, alive].mean())
        print(f"[diag] {b * 0.5:6.1f} {int(alive.sum()):4d} | {math.degrees(math.asin(max(-1, min(1, m('gx'))))):+7.2f} {math.degrees(math.asin(max(-1, min(1, m('gy'))))):+7.2f}            | {m('vx'):+6.3f} {m('vy'):+6.3f}       | {m('aL'):+6.2f} {m('aR'):+6.2f}     | {m('kL'):+6.2f} {m('kR'):+6.2f}    | {math.degrees(m('qaL')):+6.1f} {math.degrees(m('qaR')):+6.1f}        | {m('z'):.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
