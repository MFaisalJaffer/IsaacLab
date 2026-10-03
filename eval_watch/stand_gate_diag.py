"""Why is Episode_Reward/stand_pose exactly 0? Break the term into its factors
under the REAL command manager (no forcing): how many envs are standing, of
those how many are calm, what the torque gate and pose kernel actually read.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default="eval_watch/stand_gate_diag.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)   # TRAINING config, untouched
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    rm = env.unwrapped.reward_manager
    sp_cfg = rm.get_term_cfg("stand_pose")
    thr = sp_cfg.params.get("stand_still_threshold", 0.1)
    vel_rel = sp_cfg.params.get("vel_release_threshold", 0.15)
    sens = sp_cfg.params.get("sensitivity", 0.5)
    dead = sp_cfg.params.get("torque_deadband", 12.0)
    tau_g = sp_cfg.params.get("torque_gate_tau", 4.0)
    print(f"[cfg] weight={sp_cfg.weight} thr={thr} vel_release={vel_rel} sens={sens} deadband={dead}")

    n_stand = n_calm = n_both = 0
    kern_sum = tg_sum = val_sum = 0.0
    cmdnorm_min = 1e9
    steps = 0
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            cn = torch.norm(cmd[:, :3], dim=1)
            standing = cn < thr
            calm = robot.data.root_lin_vel_w[:, :2].norm(dim=-1) < vel_rel
            pos_err = torch.sum(torch.square(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)
            kern = torch.exp(-pos_err / sens)
            hr = robot.data.applied_torque[:, 2:4].abs().max(dim=1)[0]
            tg = torch.exp(-(hr - dead).clamp(min=0.0) / tau_g)
            val = kern * (standing & calm).float() * tg
            if t % 100 == 0:
                print(f"  t={t:4d}  is_standing_env={int(cmd_term.is_standing_env.sum()):3d}/{args_cli.num_envs}"
                      f"  standing_now={int(standing.sum()):3d}  calm_now={int(calm.sum()):3d}"
                      f"  min|cmd|={float(cn.min()):.3f}")
            n_stand += int(standing.sum()); n_calm += int(calm.sum()); n_both += int((standing & calm).sum())
            cmdnorm_min = min(cmdnorm_min, float(cn.min()))
            if standing.any():
                kern_sum += float(kern[standing].mean()); tg_sum += float(tg[standing].mean())
            val_sum += float(val.mean())
            steps += 1
    N = args_cli.num_envs * steps
    lines = [
        f"steps={steps} envs={args_cli.num_envs}",
        f"  is_standing_env flagged: {int(cmd_term.is_standing_env.sum())}/{args_cli.num_envs} envs",
        f"  min |cmd| seen over run: {cmdnorm_min:.4f}   (threshold {thr})",
        f"  STANDING gate true: {n_stand/N*100:.1f}% of env-steps",
        f"  CALM gate true:     {n_calm/N*100:.1f}%",
        f"  BOTH true:          {n_both/N*100:.1f}%   <-- stand_pose can only pay here",
        f"  mean pose kernel (standing envs): {kern_sum/max(steps,1):.3f}",
        f"  mean torque gate  (standing envs): {tg_sum/max(steps,1):.3f}",
        f"  mean stand_pose term value: {val_sum/max(steps,1):.5f}  (x weight {sp_cfg.weight})",
    ]
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
