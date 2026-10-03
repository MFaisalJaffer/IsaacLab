"""Stand-pose diagnostic: WHERE is the standing policy relative to the default
pose, per joint? Quantifies the wide-lunge attractor seen in the 32k calm-run
render and computes the stand_pose kernel's actual payout/gradient reach.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--label", type=str, default="stand")
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--out", type=str, default="eval_watch/stand_pose_probe.txt")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    # commanded STAND everywhere, no pushes, no terminations (observe attractor)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    env_cfg.episode_length_s = 60.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    if args_cli.pin_gains:
        for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
            if getattr(env_cfg.events, ev, None) is not None:
                setattr(env_cfg.events, ev, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    q_def = robot.data.default_joint_pos[0]
    names = [n.replace("dof_", "") for n in robot.data.joint_names]

    obs, _ = env.get_observations()
    qs, tilts, tqs = [], [], []
    with torch.inference_mode():
        for t in range(args_cli.settle + args_cli.steps):
            cmd_term.vel_command_b[:, :] = 0.0
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = True
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args_cli.settle:
                continue
            qs.append(robot.data.joint_pos.cpu().clone())
            pg = robot.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
            tilts.append((torch.asin(pg) * 180 / math.pi).cpu())
            tq = torch.zeros(env.unwrapped.num_envs, 10, device=env.unwrapped.device)
            for a in robot.actuators.values():
                tq[:, a.joint_indices] = a.applied_effort.abs()
            tqs.append(tq.cpu())

    Q = torch.stack(qs)           # (T, n, 10)
    TL = torch.stack(tilts)       # (T, n)
    TQ = torch.stack(tqs)
    upright = TL.mean(dim=0) < 25.0
    nup = int(upright.sum())
    lines = [f"{args_cli.label}: upright {nup}/{Q.shape[1]} (mean-tilt<25deg); "
             f"upright tilt mean={TL[:, upright].mean():.1f} deg" if nup else
             f"{args_cli.label}: upright 0/{Q.shape[1]} — all fell/folded"]
    if nup:
        dq = (Q[:, upright, :] - q_def.cpu()).mean(dim=(0, 1))          # per-joint mean offset
        dqa = (Q[:, upright, :] - q_def.cpu()).abs().mean(dim=(0, 1))   # per-joint |offset|
        d2 = ((Q[:, upright, :] - q_def.cpu()) ** 2).sum(dim=2).mean()  # ||q-qdef||^2
        kern = torch.exp(-d2 / 0.5)
        tqm = TQ[:, upright, :].mean(dim=(0, 1))
        lines.append("  per-joint offset (signed / |abs|) rad:")
        for i, nm in enumerate(names):
            lines.append(f"    {nm:22s} {dq[i]:+.2f} / {dqa[i]:.2f}   |tau|={tqm[i]:.1f} Nm")
        lines.append(f"  ||q-q_default||^2 = {d2:.2f} rad^2  ->  stand_pose kernel exp(-d2/0.5) = {kern:.2e}"
                     f"  (pays ~0 AND ~0 gradient beyond ~1.5 rad^2)")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
