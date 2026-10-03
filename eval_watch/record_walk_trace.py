# Isaac walk-trace recorder for the HIL trace-diff (2026-07-17).
# Context: identical T-V limits + comparable saturation in both sims, yet Isaac
# walks 126% while the MuJoCo emulator limps at 43% — the sims disagree about
# dynamics AROUND actuator saturation. This records a fully-deterministic,
# fully-documented Isaac walk so the HIL can replay the SAME sequence at three
# levels (raw actions / PD targets / applied torques) and find the first
# joint+phase where trajectories diverge.
# Determinism: tv_randomization=0, delay pinned 15 ms (HIL-measured typical),
# stiction pinned to rig MJCF (ankle 0.1, hips/knees 0), mu pinned 1.0, limb
# mass randomization off, pushes off, terminations off, fixed seed.
import argparse

from isaaclab.app import AppLauncher

import sys
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=600, help="policy steps @50Hz (600=12s)")
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--delay_steps", type=int, default=3, help="pinned actuator delay (5ms units); 3=15ms HIL-typical")
parser.add_argument("--out", type=str, default="eval_watch/hil_trace_pkg/trace_isaac.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import numpy as np
import os
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
    env_cfg.seed = 42

    # fixed command, no stochastic events, no resets
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    env_cfg.episode_length_s = 120.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    # pin the physics to rig-matched, deterministic values
    env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.8, 0.8)
    env_cfg.events.randomize_joint_friction_ankles.params["friction_distribution_params"] = (0.1, 0.1)
    env_cfg.events.randomize_joint_friction_hips_knees.params["friction_distribution_params"] = (0.0, 0.0)
    if getattr(env_cfg.events, "add_limb_masses", None) is not None:
        env_cfg.events.add_limb_masses = None
    for name, act in env_cfg.scene.robot.actuators.items():
        act.min_delay = args_cli.delay_steps
        act.max_delay = args_cli.delay_steps
        act.tv_randomization = 0.0
    # nominal gains (gain-DR events postdate the first version of this script —
    # a trace must be recorded at the deployed kp/kd, not a DR draw)
    for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    sensor = uenv.scene.sensors["contact_forces"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    foot_ids_robot, foot_names = robot.find_bodies(".*FOOT")
    foot_ids_sensor = [sensor.body_names.index(n) for n in foot_names]
    act_term = uenv.action_manager.get_term("joint_pos")
    default_q = robot.data.default_joint_pos[0].cpu().numpy()

    rec = {k: [] for k in ("obs", "raw_action", "clipped_action", "joint_target",
                           "q", "qd", "applied_torque", "tv_limit", "root_pos",
                           "root_quat", "root_linvel", "root_angvel", "pgrav",
                           "foot_force")}
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            cmd_term.vel_command_b[:, 0] = args_cli.vx
            cmd_term.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = False
            raw = policy(obs)
            rec["obs"].append(obs.cpu().numpy().copy())
            rec["raw_action"].append(raw.cpu().numpy().copy())
            obs, _, _, _ = env.step(raw)
            # post-step captures (last physics substep of the decimation window)
            rec["clipped_action"].append(uenv.action_manager.action.cpu().numpy().copy())
            rec["joint_target"].append(act_term.processed_actions.cpu().numpy().copy())
            rec["q"].append(robot.data.joint_pos.cpu().numpy().copy())
            rec["qd"].append(robot.data.joint_vel.cpu().numpy().copy())
            tq = torch.zeros_like(robot.data.joint_pos)
            tl = torch.zeros_like(robot.data.joint_pos)
            for aname, act in robot.actuators.items():
                ids = act.joint_indices
                tq[:, ids] = act.applied_effort
                tl[:, ids] = act.tv_motoring_limit
            rec["applied_torque"].append(tq.cpu().numpy().copy())
            rec["tv_limit"].append(tl.cpu().numpy().copy())
            rec["root_pos"].append(robot.data.root_pos_w.cpu().numpy().copy())
            rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy().copy())
            rec["root_linvel"].append(robot.data.root_lin_vel_b.cpu().numpy().copy())
            rec["root_angvel"].append(robot.data.root_ang_vel_b.cpu().numpy().copy())
            rec["pgrav"].append(robot.data.projected_gravity_b.cpu().numpy().copy())
            rec["foot_force"].append(
                sensor.data.net_forces_w[:, foot_ids_sensor, :].cpu().numpy().copy())

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(args_cli.out,
                        joint_names=np.array(robot.data.joint_names),
                        foot_names=np.array(foot_names),
                        default_joint_pos=default_q,
                        **{k: np.array(v) for k, v in rec.items()})
    vx_act = np.array(rec["root_linvel"])[100:, :, 0].mean()
    print(f"[trace] wrote {args_cli.out}  steps={args_cli.steps} envs={args_cli.num_envs}"
          f"  achieved vx={vx_act:.3f} (cmd {args_cli.vx})")
    with open(os.path.join(os.path.dirname(args_cli.out), "trace_summary.txt"), "w") as f:
        f.write(f"ckpt={resume_path}\nsteps={args_cli.steps} envs={args_cli.num_envs} "
                f"vx_cmd={args_cli.vx} vx_achieved={vx_act:.3f}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
