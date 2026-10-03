"""Measure how long the policy can STAND without falling. Commands zero velocity,
disables the episode time-out AND all fall-terminations so the env NEVER resets,
then tracks base height per env for a long window -> how long it stays upright."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=3000)  # 60s at 50Hz
parser.add_argument("--out", type=str, default="/tmp/stand_hold.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

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
    ckpt = args_cli.checkpoint
    if ckpt is None:
        cks = glob.glob("logs/rsl_rl/kbot_legs_rough/*/model_*.pt")
        ckpt = max(cks, key=lambda p: int(p.split("model_")[1].split(".")[0]))
    print(f"[hold] checkpoint: {ckpt}")

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    # kill pushes/curriculum
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    # NEVER reset: long episode + disable every termination so a fall persists
    env_cfg.episode_length_s = 10000.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task.split(":")[-1], args_cli)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    zero = torch.zeros(3, device=env.unwrapped.device)

    def force_stand():
        cmd_term.vel_command_b[:] = zero
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = True

    obs, _ = env.get_observations()
    force_stand()
    bh, con = [], []
    cs = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, n in enumerate(cs.body_names) if n.endswith("FOOT")]
    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        force_stand()
        bh.append(robot.data.root_pos_w[:, 2].clone().cpu().numpy())
        f = cs.data.net_forces_w_history[:, :, foot_ids, :].norm(dim=-1).max(dim=1)[0]
        con.append((f > 1.0).float().cpu().numpy())
    bh = np.array(bh); con = np.array(con)
    dt = float(env.unwrapped.step_dt)
    np.savez(args_cli.out, base_height=bh, contacts=con, dt=dt)

    # report: per env, first time base drops below 0.6 m (fell); else survived
    upright = bh > 0.6
    N = bh.shape[1]
    lines = [f"stand-hold: {args_cli.steps} steps = {args_cli.steps*dt:.0f}s, {N} envs, cmd=0, resets DISABLED"]
    fell_t = []
    for e in range(N):
        below = np.where(~upright[:, e])[0]
        if len(below):
            fell_t.append(below[0] * dt)
    if fell_t:
        lines.append(f"FELL: {len(fell_t)}/{N} envs fell. time-to-fall (s): min {min(fell_t):.1f} median {np.median(fell_t):.1f} max {max(fell_t):.1f}")
    lines.append(f"survived full {args_cli.steps*dt:.0f}s upright: {N-len(fell_t)}/{N} envs")
    lines.append(f"base height over window: mean {bh.mean():.2f}m  min {bh.min():.2f}m")
    both = (con.sum(2) == 2)
    lines.append(f"both-feet-down while 'standing': {both.mean()*100:.0f}%")
    out = "\n".join(lines)
    print(out)
    with open("/tmp/stand_hold_result.txt", "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
