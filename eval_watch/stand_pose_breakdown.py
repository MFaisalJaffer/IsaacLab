"""Per-joint deviation from the DEFAULT pose at a settled stand command."""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--ankle_play_deg", type=float, default=0.3)
cli_args.add_rsl_rl_args(parser); AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app

import gymnasium as gym, math, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
acfg = cli_args.parse_rsl_rl_cfg(a.task.split(":")[-1], a)
cfg.commands.base_velocity.rel_standing_envs = 1.0
cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
cfg.episode_length_s = 120.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)

env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
obs, _ = env.get_observations()
with torch.inference_mode():
    for t in range(150+75+300):           # walk -> decel -> stand, settle 6 s
        v = 0.25 if t < 150 else (0.12 if t < 225 else 0.0)
        cmd.vel_command_b[:, 0] = v; cmd.vel_command_b[:, 1:] = 0.0
        if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = (v == 0.0)
        obs, _, _, _ = env.step(policy(obs))
    dev = (robot.data.joint_pos - robot.data.default_joint_pos) * 180 / math.pi
    print(f"{'joint':<26}{'mean dev':>10}{'p95|dev|':>10}")
    for i, nm in enumerate(robot.data.joint_names):
        print(f"{nm:<26}{dev[:, i].mean():>+10.1f}{dev[:, i].abs().quantile(0.95):>10.1f}")
    print(f"\nmean |dev| over all joints: {dev.abs().mean():.2f} deg")
env.close(); simulation_app.close()
