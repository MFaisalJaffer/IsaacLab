"""STEP-0 stride probe for the speed-scaled step_separation lever.

Measures, per commanded speed, the REALIZED fore-aft foot separation (heading
frame, same math as feet_alternation_reward) and realized cadence, vs the
natural stride v/f. Confirms/kills the overreach hypothesis: the fixed 0.30 m
target demands ~2.8x the natural stride at cmd 0.15 (hip_pitch's hottest point).
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
parser.add_argument("--settle", type=int, default=100)
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--gait_freq", type=float, default=1.4)
parser.add_argument("--label", type=str, default="base")
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--clock", type=float, default=None, help="override gait clock Hz (freq-map study)")
parser.add_argument("--out", type=str, default="eval_watch/stride_step0.txt")
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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
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
    if args_cli.clock is not None:
        for _t in (env_cfg.observations.policy.gait_phase, env_cfg.rewards.feet_phase,
                   env_cfg.rewards.feet_alternation, env_cfg.rewards.knee_swing):
            _t.params["gait_freq"] = args_cli.clock
            _t.params.pop("freq_map", None)
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

    obs, _ = env.get_observations()
    seps = []
    vxs = []
    with torch.inference_mode():
        for t in range(args_cli.settle + args_cli.steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args_cli.settle:
                continue
            rel = robot.data.body_pos_w[:, foot_ids, :] - robot.data.root_pos_w[:, None, :]
            yq = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, 2, -1)
            fr = quat_apply_inverse(yq.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)
            seps.append((fr[:, 0, 0] - fr[:, 1, 0]).cpu())    # heading-frame fore-aft sep
            vxs.append(robot.data.root_lin_vel_b[:, 0].mean().item())

    S = torch.stack(seps)                                      # (T, n)
    dt = env.unwrapped.step_dt
    vx_act = sum(vxs) / len(vxs)
    # oscillation amplitude per env: (p95 - p5)/2 of its own sep trace
    p95 = torch.quantile(S, 0.95, dim=0)
    p5 = torch.quantile(S, 0.05, dim=0)
    amp = ((p95 - p5) / 2)
    # realized cadence: sep zero-crossing rate / 2 (one gait cycle = 2 crossings)
    zc = ((S[1:] * S[:-1]) < 0).float().sum(dim=0)
    cad = zc / (2 * S.shape[0] * dt)
    natural = abs(vx_act) / args_cli.gait_freq
    line = (f"{args_cli.label}: cmd={args_cli.vx} act_vx={vx_act:.3f}  "
            f"sep amp mean={amp.mean()*100:.1f} cm (p10={amp.quantile(0.1)*100:.1f} p90={amp.quantile(0.9)*100:.1f})  "
            f"sep mean-offset={S.mean()*100:+.1f} cm  realized cadence={cad.mean():.2f} Hz  "
            f"| natural stride v/f={natural*100:.1f} cm  target(now)=30.0 cm  "
            f"overreach x{0.30/max(natural,1e-6):.2f}")
    print(line)
    with open(args_cli.out, "a") as f:
        f.write(line + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
