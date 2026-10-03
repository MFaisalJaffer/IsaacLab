"""Foot-clearance probe: does the walk PICK UP its feet or DRAG them?

Rolls out a checkpoint at a fixed straight-walk command (flat, pinnable gains) and
reports, per foot, the swing-height distribution + swing/stance fractions. Reuses
walk_gap_probe's env-setup idiom. foot_z = foot-body world-z minus the planted
foot-body offset (0.05 m), so ~0 = planted, higher = lifted.

Verdict rule of thumb:
  picking up  : per-swing peak ~ feet_phase target (0.06 m now); swing frac ~40-50%
  dragging    : peak < ~0.02 m; foot ~always in contact (swing frac ~0)
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--settle", type=int, default=100)
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--foot_offset", type=float, default=0.05)
parser.add_argument("--label", type=str, default="base")
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--out", type=str, default="eval_watch/foot_clearance.txt")
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

_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
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
    names = list(robot.data.body_names)
    foot_ids = [names.index(f) for f in _FEET]
    # contact sensor for a stance cross-check
    try:
        cs = env.unwrapped.scene["contact_forces"]
        cf_ids = [cs.body_names.index(f) for f in _FEET]
    except Exception:
        cs = None; cf_ids = None

    obs, _ = env.get_observations()
    heights = []          # (steps, n, 2) foot z above planted offset
    contact_up = []       # (steps, n, 2) 1 if NOT in contact (swinging)
    with torch.inference_mode():
        for t in range(args_cli.settle + args_cli.steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args_cli.settle:
                continue
            fz = robot.data.body_pos_w[:, foot_ids, 2] - args_cli.foot_offset  # (n,2)
            heights.append(fz.clamp(min=0.0).cpu())
            if cs is not None:
                fmag = cs.data.net_forces_w[:, cf_ids, :].norm(dim=-1)  # (n,2)
                contact_up.append((fmag < 1.0).float().cpu())

    H = torch.stack(heights)                       # (T, n, 2)
    per_foot = H.permute(2, 0, 1).reshape(2, -1)   # (2, T*n)
    labels = ["L_foot", "R_foot"]
    lines = [f"{args_cli.label}: vx={args_cli.vx}  ({H.shape[0]} steps x {H.shape[1]} envs)  "
             f"feet_phase target this build = 0.06 m"]
    for i, lab in enumerate(labels):
        v = per_foot[i]
        p = lambda q: torch.quantile(v, q).item()
        # per-swing peak: mean of the top 10% heights = representative swing apex
        top = v.sort(descending=True).values[: max(1, v.numel() // 10)].mean().item()
        swing_frac = (v > 0.02).float().mean().item() * 100  # % time clearly off ground
        lines.append(f"  {lab}: swing-apex(mean top10%)={top*100:.1f} cm  "
                     f"p50={p(0.5)*100:.1f}  p90={p(0.9)*100:.1f}  p99={p(0.99)*100:.1f}  "
                     f"max={v.max()*100:.1f} cm  |  off-ground={swing_frac:.0f}% of time")
    if contact_up:
        C = torch.stack(contact_up)                # (T,n,2)
        cu = C.permute(2, 0, 1).reshape(2, -1).mean(dim=1) * 100
        lines.append(f"  contact-sensor swing%: L={cu[0]:.0f}  R={cu[1]:.0f}  "
                     f"(stance-duty L={100-cu[0]:.0f}%  R={100-cu[1]:.0f}%)")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
