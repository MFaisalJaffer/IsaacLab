# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic: command the legs policy DEAD STRAIGHT (vx fixed, vy=0, wz=0, no
heading command) and measure base-yaw drift + xy path, to tell whether the
policy circles in Isaac itself or only in the MuJoCo viewer."""

import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--vx", type=float, default=0.6)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def yaw_of(quat):  # quat (w,x,y,z) -> yaw
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    task = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args_cli)

    # ---- force a fixed dead-straight command: no heading, no turn, no standing ----
    c = env_cfg.commands.base_velocity
    c.heading_command = False
    c.rel_standing_envs = 0.0
    c.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    c.ranges.lin_vel_y = (0.0, 0.0)
    c.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(c.ranges, "heading"):
        c.ranges.heading = (0.0, 0.0)
    # disable pushes so drift is purely the policy (also null the curriculum that
    # references push_robot, else it errors)
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    for ev in ("push_robot", "base_external_force_torque"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)

    log_root = f"logs/rsl_rl/{agent_cfg.experiment_name}"
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else \
        get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[eval] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obs, _ = env.get_observations()
    yaw0 = yaw_of(robot.data.root_quat_w).clone()
    pos0 = robot.data.root_pos_w[:, :2].clone()
    yaw_rate_sum = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    nsteps = 0
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            obs, _, _, _ = env.step(policy(obs))
            yaw_rate_sum += robot.data.root_ang_vel_w[:, 2]
            nsteps += 1

    yaw1 = yaw_of(robot.data.root_quat_w)
    pos1 = robot.data.root_pos_w[:, :2]
    dyaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))  # wrapped
    disp = pos1 - pos0
    mean_yaw_rate = yaw_rate_sum / nsteps

    lines = []
    lines.append("==== STRAIGHT-COMMAND EVAL (vx=%.2f, wz=0, %d steps) ====" % (args_cli.vx, args_cli.steps))
    lines.append("per-env  yaw_drift(deg)   mean_yaw_rate(rad/s)   disp_x(m)  disp_y(m)")
    for i in range(min(args_cli.num_envs, 16)):
        lines.append("  env%2d   %+8.1f         %+7.3f               %+6.2f    %+6.2f"
                     % (i, math.degrees(dyaw[i].item()), mean_yaw_rate[i].item(), disp[i, 0].item(), disp[i, 1].item()))
    n_pos = int((mean_yaw_rate > 0.05).sum().item())
    n_neg = int((mean_yaw_rate < -0.05).sum().item())
    lines.append("SUMMARY:")
    lines.append("  mean |yaw drift|  = %.1f deg   (~0 -> straight; large -> circles)"
                 % torch.rad2deg(dyaw.abs()).mean().item())
    lines.append("  mean yaw rate     = %+.3f rad/s   (|mean| large -> systematic turn)"
                 % mean_yaw_rate.mean().item())
    lines.append("  yaw-rate signs: %d envs turn +CCW, %d turn -CW, %d ~straight (of %d)"
                 % (n_pos, n_neg, args_cli.num_envs - n_pos - n_neg, args_cli.num_envs))
    lines.append("  mean disp        = (%.2f, %.2f) m  (want disp_x>>|disp_y|)" % (disp[:, 0].mean().item(), disp[:, 1].mean().item()))
    text = "\n".join(lines)
    with open("/home/faisal/IsaacLab/eval_straight_result.txt", "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
