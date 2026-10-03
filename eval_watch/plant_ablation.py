"""PLANT ABLATION (2026-08-21) — which new plant parameter kills the policy?

The rig ran checkpoint 2026-08-16_20-40-23/model_39000 in THEIR sim with the
real hardware configs: it stands and walks. Our sim with OUR implementation of
those same configs: 100% collapse during the walk-in entry. One implementation
is wrong; this isolates which parameter is responsible.

Arms (one per process; --arm):
  old        every plant parameter at the pre-handoff values (control)
  kd         ONLY the deploy kd (hp2.5/hr1.5/yaw1/knee1/ankle0.5)
  friction   ONLY the fitted Coulomb (hips 2-5, knee .3-.9, yaw .2-.6 Nm)
  viscous    ONLY the viscous b (hp/hr 1.0, knee 0.2)
  delay      ONLY the measured per-family latencies
  all        everything (= the eval that said "collapse")
Protocol: trained stand entry (walk -> decel -> settle), then 2 s hold.
Reports alive%, tilt, and whether it kept translating while walking.
"""
import argparse
import functools
import sys

print = functools.partial(print, flush=True)  # noqa: A001
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--arm", type=str, required=True,
                    choices=["old", "kd", "friction", "viscous", "delay", "all"])
parser.add_argument("--out", type=str, default="eval_watch/plant_ablation.txt")
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

# pre-handoff (old) plant
_OLD_KD = {"hip_pitch": 5.0, "hip_roll": 5.0, "hip_yaw": 3.0, "knee": 5.0, "ankle": 3.0}
_OLD_FRICTION = {"randomize_joint_friction_hip_pitch_roll": (0.0, 0.05),
                 "randomize_joint_friction_knees": (0.0, 0.05),
                 "randomize_joint_friction_yaws": (0.0, 0.05)}
_OLD_DELAY = (0, 8)


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    arm = args_cli.arm

    # ---- build the requested plant ----
    keep_kd = arm in ("kd", "all")
    keep_friction = arm in ("friction", "all")
    keep_viscous = arm in ("viscous", "all")
    keep_delay = arm in ("delay", "all")

    for jn, acfg in env_cfg.scene.robot.actuators.items():
        if not keep_kd:
            for k, v in _OLD_KD.items():
                if k in jn:
                    acfg.damping = {jn: v}
        if not keep_viscous:
            acfg.viscous_b = 0.0
        if not keep_delay:
            acfg.min_delay, acfg.max_delay = _OLD_DELAY
    if not keep_friction:
        for ev_name, rng in _OLD_FRICTION.items():
            ev = getattr(env_cfg.events, ev_name, None)
            if ev is not None:
                ev.params["friction_distribution_params"] = rng

    # probe hygiene
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for ev in ("push_robot", "sustained_push", "walk_at_spawn", "stand_corridor",
               "randomize_actuator_gains", "randomize_gains_small_joints",
               "randomize_gains_04_joints", "randomize_joint_play"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum",
               "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
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
    cmd = uenv.command_manager.get_term("base_velocity")

    def set_cmd(vx, standing):
        cmd.vel_command_b[:, 0] = vx
        cmd.vel_command_b[:, 1:] = 0.0
        if hasattr(cmd, "is_standing_env"):
            cmd.is_standing_env[:] = standing

    def tilt_deg():
        pg = robot.data.projected_gravity_b
        return torch.asin(pg[:, :2].norm(dim=1).clamp(-1, 1)) * 180 / math.pi

    obs, _ = env.get_observations()
    marks = {}
    with torch.inference_mode():
        x0 = robot.data.root_pos_w[:, 0].clone()
        for t in range(150 + 75 + 150 + 100):
            if t < 150:
                set_cmd(0.25, False)
            elif t < 225:
                set_cmd(0.12, False)
            else:
                set_cmd(0.0, True)
            obs, _, _, _ = env.step(policy(obs))
            if t in (149, 224, 374, 474):
                alive = (robot.data.root_pos_w[:, 2] >= 0.55)
                marks[t] = (alive.float().mean().item() * 100,
                            tilt_deg()[alive].mean().item() if alive.any() else float("nan"),
                            (robot.data.root_pos_w[:, 0] - x0)[alive].mean().item() if alive.any() else float("nan"))
    line = (f"ARM {arm:9s} | end-of-walk: alive {marks[149][0]:3.0f}% tilt {marks[149][1]:5.1f} deg "
            f"walked {marks[149][2]*100:5.1f} cm | post-decel: alive {marks[224][0]:3.0f}% tilt {marks[224][1]:5.1f} "
            f"| settled: alive {marks[374][0]:3.0f}% tilt {marks[374][1]:5.1f} "
            f"| +2s stand: alive {marks[474][0]:3.0f}% tilt {marks[474][1]:5.1f}")
    print(line)
    with open(args_cli.out, "a") as f:
        f.write(line + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
