"""STATIC-TILT SERVO probe (2026-08-16) — sim replication of the rig's
acceptance test (RIG->TRAINING report: stand policy does not regulate static
tilt; hardware baseline hip-roll differential ~0.002 rad at ~7 deg).

Kinematically holds the base at fixed roll tilts (feet planted, rates ~zero,
stand commanded) and measures the POLICY'S COMMAND response vs roll error.

Pass criteria (from the rig report):
  - lateral command magnitude increases monotonically with roll error
  - correlation(roll error, hip-roll differential) > 0.5
  - >= ~0.1 rad hip-roll differential at 10 deg tilt
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold_steps", type=int, default=150)
parser.add_argument("--measure_last", type=int, default=100)
parser.add_argument("--action_scale", type=float, default=0.5)
parser.add_argument("--z_correct", action="store_true",
                    help="lower base per tilt so BOTH feet stay loaded (0.20*sin|tilt|)")
parser.add_argument("--label", type=str, default="static_tilt")
parser.add_argument("--out", type=str, default="eval_watch/static_tilt.txt")
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
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for ev in ("push_robot", "sustained_push", "walk_at_spawn", "stand_corridor",
               "randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.episode_length_s = 120.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    dev = uenv.device
    n = uenv.num_envs

    _FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
    cs = uenv.scene["contact_forces"]
    cf_ids = [cs.body_names.index(f) for f in _FEET]
    tilts_deg = [-12, -10, -8, -5, -3, 0, 3, 5, 8, 10, 12]
    results = []
    contact_rows = []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for tdeg in tilts_deg:
            th = math.radians(tdeg)
            # roll tilt about x (body forward): quat (w, x, y, z)
            quat = torch.tensor([math.cos(th / 2), math.sin(th / 2), 0.0, 0.0],
                                device=dev).repeat(n, 1)
            pos = robot.data.default_root_state[:, :3].clone()
            pos[:, 0] = uenv.scene.env_origins[:, 0]
            pos[:, 1] = uenv.scene.env_origins[:, 1]
            pos[:, 2] = 1.00 * math.cos(th) - 0.005
            if args_cli.z_correct:
                pos[:, 2] -= 0.20 * abs(math.sin(th))
            diffs = []
            con = torch.zeros(2, device=dev)
            for k in range(args_cli.hold_steps):
                # kinematic hold: pin pose + zero rates every step
                root = torch.cat([pos, quat, torch.zeros(n, 6, device=dev)], dim=1)
                robot.write_root_pose_to_sim(root[:, :7])
                robot.write_root_velocity_to_sim(root[:, 7:])
                cmd_term.vel_command_b[:, :] = 0.0
                if hasattr(cmd_term, "is_standing_env"):
                    cmd_term.is_standing_env[:] = True
                act = policy(obs)
                obs, _, _, _ = env.step(act)
                if k >= args_cli.hold_steps - args_cli.measure_last:
                    # hip-roll command differential, action space -> rad
                    diffs.append(((act[:, 2] - act[:, 3]) * args_cli.action_scale).mean().item())
                    con += (cs.data.net_forces_w[:, cf_ids, :].norm(dim=-1) > 1.0).float().mean(dim=0)
            d = sum(diffs) / len(diffs)
            results.append((tdeg, d))
            con = (con / args_cli.measure_last * 100).tolist()
            contact_rows.append((tdeg, con[0], con[1]))

    # correlation + monotonicity + magnitude at 10 deg
    xs = [math.radians(t) for t, _ in results]
    ys = [d for _, d in results]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    corr = cov / (vx * vy) if vx * vy > 0 else 0.0
    d10 = next(abs(d) for t, d in results if t == 10)
    d0 = next(abs(d) for t, d in results if t == 0)
    lines = [f"{args_cli.label}: kinematic roll-tilt hold, {args_cli.num_envs} envs, "
             f"{args_cli.hold_steps} steps/tilt (measure last {args_cli.measure_last}), gains pinned",
             f"  {'tilt_deg':>9} {'hiproll_diff_cmd_rad':>21}"]
    for t, d in results:
        lines.append(f"  {t:>9} {d:>21.4f}")
    lines.append(f"  correlation(roll, differential) = {corr:+.3f}   (pass > 0.5)")
    lines.append(f"  |differential| at 10 deg = {d10:.4f} rad  (pass >= ~0.1; rig measured ~0.002)")
    lines.append(f"  |differential| at 0 deg  = {d0:.4f} rad (bias reference)")
    lines.append(f"  {'tilt':>6} {'L_contact%':>11} {'R_contact%':>11}   (roll + tilts toward L? uphill foot unloading = probe artifact)")
    for tdeg, cl, cr in contact_rows:
        lines.append(f"  {tdeg:>6} {cl:>11.0f} {cr:>11.0f}")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
