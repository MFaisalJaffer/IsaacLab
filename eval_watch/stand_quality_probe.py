"""QUIET-STAND QUALITY battery (lineage-2, 2026-08-13).

Stand commanded from t=0, NO pushes, walk_at_spawn nulled (it would convert
standers), terminations off. Measures: stance width, base height, tilt,
body stillness, foot fidgeting, net drift — and the A-FRAME DISCRIMINATOR:
sustained hip-roll torque (placement-wide stance is cheap; force-wide brace
stalls the 5 Nm-continuous hip rolls at 15-20 Nm and is thermally lethal).
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--walk_in", type=int, default=150,
                    help="steps of commanded WALKING before the stand (the trained entry path)")
parser.add_argument("--decel_steps", type=int, default=0,
                    help="corridor emulation: steps at decel_speed between walk and stand")
parser.add_argument("--decel_speed", type=float, default=0.12)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--ankle_play_deg", type=float, default=0.0,
                    help="FREE-PLAY validation: set this backlash band (deg) on the ankle actuators")
parser.add_argument("--label", type=str, default="stand_quality")
parser.add_argument("--out", type=str, default="eval_watch/stand_quality.txt")
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

_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    for ev in ("sustained_push", "walk_at_spawn"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
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

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    if args_cli.ankle_play_deg > 0.0:
        import math as _m
        band = _m.radians(args_cli.ankle_play_deg)
        env.step(torch.zeros(uenv.num_envs, 10, device=uenv.device))  # force lazy init
        for name, act in robot.actuators.items():
            if hasattr(act, "_play") and act._play is not None and "ankle" in name:
                act._play[:] = band
        print(f"[play] ankle backlash set to {args_cli.ankle_play_deg} deg on ankle actuators")
    cmd_term = uenv.command_manager.get_term("base_velocity")
    names = list(robot.data.body_names)
    foot_ids = [names.index(f) for f in _FEET]
    cs = uenv.scene["contact_forces"]
    cf_ids = [cs.body_names.index(f) for f in _FEET]

    W, H, P, R, G, C, TQ, ACT = [], [], [], [], [], [], [], []
    prof = []
    walk_h = 0.0
    start_pos = None
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.walk_in + args_cli.decel_steps + args_cli.settle + args_cli.steps):
            if t < args_cli.walk_in:
                # trained entry: walk first, then decelerate into the stand
                cmd_term.vel_command_b[:, 0] = 0.25
                cmd_term.vel_command_b[:, 1:] = 0.0
                if hasattr(cmd_term, "is_standing_env"):
                    cmd_term.is_standing_env[:] = False
            elif t < args_cli.walk_in + args_cli.decel_steps:
                # corridor phase: the trained 0.12 m/s deceleration window
                cmd_term.vel_command_b[:, 0] = args_cli.decel_speed
                cmd_term.vel_command_b[:, 1:] = 0.0
                if hasattr(cmd_term, "is_standing_env"):
                    cmd_term.is_standing_env[:] = False
            else:
                cmd_term.vel_command_b[:, :] = 0.0
                if hasattr(cmd_term, "is_standing_env"):
                    cmd_term.is_standing_env[:] = True
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            sw = args_cli.walk_in + args_cli.decel_steps
            if t == sw - 1:
                walk_h = robot.data.root_pos_w[:, 2].mean().item()
            if t >= sw:
                prof.append(robot.data.root_pos_w[:, 2].mean().item())
            if t < sw + args_cli.settle:
                continue
            if start_pos is None:
                start_pos = robot.data.root_pos_w[:, :2].clone()
            fp = robot.data.body_pos_w[:, foot_ids, :2]
            W.append((fp[:, 0, :] - fp[:, 1, :]).norm(dim=-1).cpu())
            H.append(robot.data.root_pos_w[:, 2].cpu())
            pg = robot.data.projected_gravity_b
            P.append((torch.asin(pg[:, 0].clamp(-1, 1)) * 180 / math.pi).cpu())
            R.append((torch.asin(pg[:, 1].clamp(-1, 1)) * 180 / math.pi).cpu())
            G.append(robot.data.root_ang_vel_b.norm(dim=-1).cpu())
            C.append((cs.data.net_forces_w[:, cf_ids, :].norm(dim=-1) > 1.0).cpu())
            TQ.append(robot.data.applied_torque[:, 2:4].abs().cpu())   # hip roll L/R
            ACT.append(act[:, 2:4].abs().cpu())
    drift = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1)
    W = torch.stack(W); H = torch.stack(H); P = torch.stack(P); R = torch.stack(R)
    G = torch.stack(G); C = torch.stack(C); TQ = torch.stack(TQ); ACT = torch.stack(ACT)
    dt = uenv.step_dt
    lifts = ((~C[1:]) & C[:-1]).float().sum() / (C.shape[0] * dt * C.shape[1] * 2) * 60
    dur = args_cli.steps * dt
    lines = [
        f"{args_cli.label}: {args_cli.num_envs} envs, {dur:.0f}s quiet stand (no pushes)",
        f"  stance width: mean={W.mean()*100:.1f} cm  p5={W.quantile(0.05)*100:.1f}  p95={W.quantile(0.95)*100:.1f}"
        f"   (lineage-1 stand ~40 cm; hips are ~18 cm apart)",
        f"  base height: mean={H.mean():.3f} m (healthy ~1.01; kill line 0.55)",
        f"  tilt: pitch mean={P.mean():+.1f} p95={P.abs().quantile(0.95):.1f} deg | roll mean={R.mean():+.1f} p95={R.abs().quantile(0.95):.1f} deg",
        f"  body gyro |w|: mean={G.mean():.2f} rad/s (rig-quiet ~0.1-0.3)",
        f"  both-feet-down: {(C.all(dim=2)).float().mean()*100:.0f}%   foot lifts/min: {lifts:.0f}",
        f"  net drift over {dur:.0f}s: mean={drift.mean()*100:.1f} cm  max={drift.max()*100:.1f} cm",
        f"  HIP-ROLL torque |tau|: mean={TQ.mean():.1f} Nm  p95={TQ.quantile(0.95):.1f}  max={TQ.max():.1f}"
        f"   (continuous rating 5.0; A-frame brace signature ~15-20 sustained)",
        f"  hip-roll |action|: mean={ACT.mean():.2f} (deadband-relevant; brace-era railed ~2.0)",
    ]
    lines.append(f"  TIMECOURSE — height at end of entry (post-corridor): {walk_h:.2f} m; after ZERO-cmd switch:")
    marks = [(0.5, 25), (1, 50), (2, 100), (3, 150), (4, 200), (6, 300), (8, 400), (10, 500), (12, 600)]
    lines.append("    " + "  ".join(f"+{s_}s:{prof[i]:.2f}" for s_, i in marks if i < len(prof)))
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
