"""POSE RE-HOMING probe (2026-08-16) — does the robot return to its default
stand pose after surviving a push, or park in the deviated pose?

User/rig observation: stand commanded -> push -> protective maneuver -> robot
keeps standing in the NON-default end pose indefinitely.

Protocol: trained stand entry, baseline window, 10 s natural-drift control,
then per push direction: ramped 20 N x 1.5 s root force (training-faithful),
10 s free recovery. Metrics vs baseline: mean |q - q_default| over leg joints,
stance width, fore-aft foot stagger (body frame), at marks after push end.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--walk_in", type=int, default=150)
parser.add_argument("--decel_steps", type=int, default=75)
parser.add_argument("--decel_speed", type=float, default=0.12)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--push_n", type=float, default=20.0)
parser.add_argument("--push_s", type=float, default=1.5)
parser.add_argument("--ramp_s", type=float, default=0.3)
parser.add_argument("--recover_steps", type=int, default=500)
parser.add_argument("--ankle_play_deg", type=float, default=0.0)
parser.add_argument("--label", type=str, default="pose_rehome")
parser.add_argument("--out", type=str, default="eval_watch/pose_rehome.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for ev in ("push_robot", "sustained_push", "walk_at_spawn", "stand_corridor",
               "randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.episode_length_s = 300.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.ankle_play_deg > 0.0:
        _u = env.unwrapped
        with torch.inference_mode():
            env.step(torch.zeros(_u.num_envs, _u.action_manager.total_action_dim, device=_u.device))
        for _nm, _act in _u.scene["robot"].actuators.items():
            if getattr(_act, "_play", None) is not None and "ankle" in _nm:
                _act._play[:] = math.radians(args_cli.ankle_play_deg)
        print(f"[play] ankle band forced to {args_cli.ankle_play_deg} deg")

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    dev = uenv.device
    n = uenv.num_envs
    dt = uenv.step_dt
    names = list(robot.data.body_names)
    foot_ids = [names.index(f) for f in _FEET]

    def set_cmd(vx, standing):
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1:] = 0.0
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = standing

    def metrics():
        # pose error: mean |q - q_default| over all leg joints, deg
        pe = (robot.data.joint_pos - robot.data.default_joint_pos).abs().mean(dim=1) * 180 / math.pi
        fp_w = robot.data.body_pos_w[:, foot_ids, :] - robot.data.root_pos_w.unsqueeze(1)
        q = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        fp_b = quat_apply_inverse(q.reshape(-1, 4), fp_w.reshape(-1, 3)).reshape(n, 2, 3)
        width = (fp_b[:, 0, 1] - fp_b[:, 1, 1]).abs() * 100        # lateral, cm
        stagger = (fp_b[:, 0, 0] - fp_b[:, 1, 0]).abs() * 100      # fore-aft, cm
        return pe.cpu(), width.cpu(), stagger.cpu()

    def fmt(pe, w, st):
        # SURVIVOR-MASKED (2026-08-17): fallen envs contaminated the means in
        # the 10.2k battery (98% fallen -> geometry of a heap). Stats over
        # still-standing envs only, with the alive fraction shown.
        alive = (robot.data.root_pos_w[:, 2] >= 0.55).cpu()
        if alive.any():
            pe, w, st = pe[alive], w[alive], st[alive]
        return (f"pose_err={pe.mean():5.2f} deg  width={w.mean():4.1f} cm  "
                f"stagger={st.mean():4.1f} cm  [alive {alive.float().mean()*100:3.0f}%]")

    force = torch.zeros(n, 1, 3, device=dev)
    torque = torch.zeros(n, 1, 3, device=dev)
    push_steps = int(args_cli.push_s / dt)
    ramp_steps = max(1, int(args_cli.ramp_s / dt))

    lines = [f"{args_cli.label}: {args_cli.num_envs} envs, {args_cli.push_n:.0f} N x {args_cli.push_s}s ramped root push, "
             f"{args_cli.recover_steps*dt:.0f}s recovery, gains pinned"]
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.walk_in + args_cli.decel_steps + args_cli.settle):
            if t < args_cli.walk_in:
                set_cmd(0.25, False)
            elif t < args_cli.walk_in + args_cli.decel_steps:
                set_cmd(args_cli.decel_speed, False)
            else:
                set_cmd(0.0, True)
            obs, _, _, _ = env.step(policy(obs))
        base = metrics()
        lines.append(f"  BASELINE (settled stand):        {fmt(*base)}")

        # natural-drift control: 10 s quiet stand, no push
        for t in range(500):
            obs, _, _, _ = env.step(policy(obs))
        drift = metrics()
        lines.append(f"  +10s quiet (NO push, control):   {fmt(*drift)}")

        marks = [(0.5, 25), (1, 50), (2, 100), (5, 250), (10, 500)]
        for name, d in [("+Y (left)", (0.0, 1.0)), ("-Y (right)", (0.0, -1.0)), ("+X (front)", (1.0, 0.0))]:
            pre = metrics()
            for k in range(push_steps):
                s = min(1.0, (k + 1) / ramp_steps)
                force[:, 0, 0] = d[0] * args_cli.push_n * s
                force[:, 0, 1] = d[1] * args_cli.push_n * s
                robot.set_external_force_and_torque(force, torque, body_ids=[0])
                obs, _, _, _ = env.step(policy(obs))
            force.zero_()
            robot.set_external_force_and_torque(force, torque, body_ids=[0])
            peak = metrics()
            lines.append(f"  PUSH {name}: pre {fmt(*pre)}")
            lines.append(f"       push end (peak dev):        {fmt(*peak)}")
            for kk, (s_, i) in enumerate(marks):
                upto = i if kk == 0 else i - marks[kk - 1][1]
                for t in range(upto):
                    obs, _, _, _ = env.step(policy(obs))
                m = metrics()
                lines.append(f"       +{s_:>4}s after push:          {fmt(*m)}")
            h = robot.data.root_pos_w[:, 2]
            lines.append(f"       fallen: {(h < 0.55).float().mean()*100:.0f}%")

    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
