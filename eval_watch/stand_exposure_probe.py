"""STAND-EXPOSURE VERIFICATION (2026-08-12): with walk_at_spawn converting
spawn-drawn standers to walkers, standing should now enter ONLY via the 10 s
mid-episode resample. This measures whether that actually happens: fraction of
env-steps standing, the episode-age at which standing is active (must be
>~500 steps), stand onsets, and whether standing robots survive.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out", type=str, default="eval_watch/stand_exposure.txt")
parser.add_argument("--label", type=str, default="stand_exposure")
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    term = uenv.command_manager.get_term("base_velocity")

    stand_steps = 0
    gate_steps = 0                       # cmd_norm < 0.1 (what rewards see)
    total = 0
    age_young = 0                        # standing at age <= 10 (should be ~0)
    age_mid = 0                          # 11..499 (should be ~0)
    age_post = 0                         # >= 500 (the resample path)
    onsets = []                          # ages at flag flip false->true
    stand_heights = []
    prev_flag = term.is_standing_env.clone()
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            flag = term.is_standing_env
            age = uenv.episode_length_buf
            n = int(flag.sum())
            stand_steps += n
            cmd = uenv.command_manager.get_command("base_velocity")
            gate_steps += int((torch.norm(cmd[:, :3], dim=1) < 0.1).sum())
            total += uenv.num_envs
            if n:
                age_young += int((flag & (age <= 10)).sum())
                age_mid += int((flag & (age > 10) & (age < 500)).sum())
                age_post += int((flag & (age >= 500)).sum())
                stand_heights.append(robot.data.root_pos_w[flag, 2].mean().item())
            new = flag & ~prev_flag
            if new.any():
                onsets.extend(age[new].tolist())
            prev_flag = flag.clone()

    lines = [f"{args_cli.label}: {args_cli.num_envs} envs x {args_cli.steps} steps"]
    lines.append(f"  standing flag: {stand_steps/total*100:.2f}% of env-steps "
                 f"| reward-gate (cmd<0.1): {gate_steps/total*100:.2f}%")
    lines.append(f"  standing env-steps by episode age: <=10: {age_young}  11-499: {age_mid}  >=500: {age_post}")
    if onsets:
        o = sorted(onsets)
        lines.append(f"  stand onsets: n={len(onsets)}  age min={o[0]}  median={o[len(o)//2]}  max={o[-1]}")
    else:
        lines.append("  stand onsets: NONE OBSERVED")
    if stand_heights:
        lines.append(f"  mean base height while standing: {sum(stand_heights)/len(stand_heights):.2f} m "
                     f"(healthy stand ~1.0; collapsed <0.55)")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
