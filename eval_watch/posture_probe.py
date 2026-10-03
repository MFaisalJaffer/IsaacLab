"""Is the robot WALKING WITH A BENT TORSO, or falling?

Decomposes torso tilt into PITCH (forward/back lean) vs ROLL (sideways), and
checks whether the lean is STEADY (a posture) or GROWING (a fall), while
measuring whether the robot actually translates and alternates its feet.
No alive-masking — reports the raw truth for every env.
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
parser.add_argument("--vx", type=float, default=0.2)
parser.add_argument("--label", type=str, default="posture")
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--gait_freq", type=float, default=None,
                    help="override the gait clock (Hz) — study knob; policy feels it via the phase obs")
parser.add_argument("--out", type=str, default="eval_watch/posture_probe.txt")
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
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    # EVAL HYGIENE (2026-08-05): the KBOT_ADAPT config carries the sustained_push
    # burst event — it MUST be nulled in probes or it contaminates measurements
    # (caught when a 'baseline' read 21 deg tilt: a training burst mid-probe).
    if getattr(env_cfg.events, "sustained_push", None) is not None:
        env_cfg.events.sustained_push = None
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    if getattr(env_cfg.curriculum, "sustained_push_level", None) is not None:
        env_cfg.curriculum.sustained_push_level = None   # curriculum writes into the (nulled) event
    env_cfg.episode_length_s = 60.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    if args_cli.gait_freq is not None:
        for _t in (env_cfg.observations.policy.gait_phase, env_cfg.rewards.feet_phase,
                   env_cfg.rewards.feet_alternation, env_cfg.rewards.knee_swing):
            _t.params["gait_freq"] = args_cli.gait_freq
            _t.params.pop("freq_map", None)   # force fixed-f for the study
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
    names = list(robot.data.body_names)
    foot_ids = [names.index(f) for f in _FEET]
    cs = env.unwrapped.scene["contact_forces"]
    cf_ids = [cs.body_names.index(f) for f in _FEET]

    P, R, VX, H, C = [], [], [], [], []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.settle + args_cli.steps):
            cmd_term.vel_command_b[:, 0] = args_cli.vx
            cmd_term.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = False
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args_cli.settle:
                continue
            pg = robot.data.projected_gravity_b
            # pg_x>0 => gravity points forward in body frame => torso pitched FORWARD
            P.append((torch.asin(pg[:, 0].clamp(-1, 1)) * 180 / math.pi).cpu())
            R.append((torch.asin(pg[:, 1].clamp(-1, 1)) * 180 / math.pi).cpu())
            VX.append(robot.data.root_lin_vel_b[:, 0].cpu())
            H.append(robot.data.root_pos_w[:, 2].cpu())
            C.append((cs.data.net_forces_w[:, cf_ids, :].norm(dim=-1) > 1.0).cpu())

    P = torch.stack(P); R = torch.stack(R); VX = torch.stack(VX); H = torch.stack(H); C = torch.stack(C)
    T = P.shape[0]
    first, last = slice(0, T // 4), slice(3 * T // 4, T)
    tot = (P.abs() + R.abs())
    # a "still walking" env: base height stays above 0.45 m for the whole window
    up = (H.min(dim=0).values > 0.45)
    lines = [
        f"{args_cli.label}: cmd_vx={args_cli.vx}  ({T} steps x {P.shape[1]} envs)",
        f"  PITCH (fwd+ / back-) deg: first-qtr mean={P[first].mean():+.1f}  last-qtr mean={P[last].mean():+.1f}"
        f"   p50={P.median():+.1f}  p95={P.abs().quantile(0.95):.1f}",
        f"  ROLL  (sideways)    deg: first-qtr mean={R[first].mean():+.1f}  last-qtr mean={R[last].mean():+.1f}"
        f"   |p95|={R.abs().quantile(0.95):.1f}",
        f"  -> steady lean if first~last; GROWING lean = falling",
        f"  base height m: start={H[first].mean():.2f}  end={H[last].mean():.2f}"
        f"   (spawn ~0.72; <0.45 = collapsed)",
        f"  envs still standing tall at end: {int(up.sum())}/{P.shape[1]}",
        f"  forward speed m/s: first-qtr={VX[first].mean():.3f}  last-qtr={VX[last].mean():.3f}"
        f"   (commanded {args_cli.vx})",
        f"  stance-duty: {C.float().mean()*100:.0f}%   both-feet-down: {(C.all(dim=2)).float().mean()*100:.0f}%"
        f"   airborne(both up): {(~C.any(dim=2)).float().mean()*100:.0f}%",
        f"  tilt>25deg fraction of steps: {(tot > 25).float().mean()*100:.0f}%"
        f"   | tilt>45: {(tot > 45).float().mean()*100:.0f}%",
    ]
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
