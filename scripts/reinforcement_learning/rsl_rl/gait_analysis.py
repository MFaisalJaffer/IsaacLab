"""Quantitative gait analysis for the legs walker.

Runs the trained policy at a FIXED commanded velocity (no random command, pushes
disabled) and records every joint's position/velocity + foot contacts, then dumps
a .npz. A separate step analyzes amplitude, L/R symmetry, phase, knee usage, and
stance/flight fractions — so we KNOW what the gait does instead of eyeballing.
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--settle", type=int, default=100, help="warmup steps to discard")
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--out", type=str, default="/tmp/gait_trace.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task = args_cli.task
    task_name = task.split(":")[-1]
    ckpt = args_cli.checkpoint
    if ckpt is None:
        cks = glob.glob("logs/rsl_rl/kbot_legs_rough/*/model_*.pt")
        ckpt = max(cks, key=lambda p: int(p.split("model_")[1].split(".")[0]))
    print(f"[gait] checkpoint: {ckpt}")

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    # clean rollout: kill pushes (and the curriculum that drives them) so the gait
    # isn't disturbed
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    dev = env.unwrapped.device
    fixed = torch.tensor([args_cli.vx, args_cli.vy, args_cli.wz], device=dev)

    def force_cmd():
        # pin every env to the fixed walk command (override resample + standing)
        cmd_term.vel_command_b[:] = fixed
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = False

    # contact sensor for foot stance detection
    cs = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, n in enumerate(cs.body_names) if n.endswith("FOOT")]

    obs, _ = env.get_observations()
    force_cmd()
    jp, jv, fz, contacts, bh, bvx, bvyaw = [], [], [], [], [], [], []
    pgrav = []   # projected gravity in base frame -> torso tilt
    n = args_cli.steps
    for t in range(n):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        force_cmd()
        jp.append(robot.data.joint_pos.clone().cpu().numpy())
        jv.append(robot.data.joint_vel.clone().cpu().numpy())
        # foot heights + contact
        fz.append(robot.data.body_pos_w[:, :, 2].clone().cpu().numpy())
        f = cs.data.net_forces_w_history[:, :, foot_ids, :].norm(dim=-1).max(dim=1)[0]  # (N, nfeet)
        contacts.append((f > 1.0).float().cpu().numpy())
        bh.append(robot.data.root_pos_w[:, 2].clone().cpu().numpy())
        bvx.append(robot.data.root_lin_vel_b[:, 0].clone().cpu().numpy())
        bvyaw.append(robot.data.root_ang_vel_b[:, 2].clone().cpu().numpy())
        pgrav.append(robot.data.projected_gravity_b.clone().cpu().numpy())  # (N,3)

    np.savez(
        args_cli.out,
        joint_names=np.array(joint_names),
        foot_body_names=np.array([cs.body_names[i] for i in foot_ids]),
        joint_pos=np.array(jp), joint_vel=np.array(jv),
        contacts=np.array(contacts), base_height=np.array(bh),
        base_vx=np.array(bvx), base_vyaw=np.array(bvyaw), pgrav=np.array(pgrav),
        dt=float(env.unwrapped.step_dt), settle=args_cli.settle,
        cmd=np.array([args_cli.vx, args_cli.vy, args_cli.wz]),
    )
    print(f"[gait] wrote {args_cli.out}: {n} steps, {args_cli.num_envs} envs, joints={joint_names}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
