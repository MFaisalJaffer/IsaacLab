"""WALK STANCE WIDTH — lateral foot separation during gait, body frame.

User observation (2026-09-22): the robot walks with its legs far apart. Nothing
in the reward set prices lateral foot placement while WALKING (stance_geometry
is stand-gated; feet_phase is foot HEIGHT; feet_alternation is fore-aft). This
measures it so the proposal can be sized against a number.
"""
import argparse, functools, math, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--vx", type=float, default=0.3)
p.add_argument("--label", default="walkwidth")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
acfg = cli_args.parse_rsl_rl_cfg(a.task.split(":")[-1], a)
cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor","randomize_joint_play",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
for jn, ac in cfg.scene.robot.actuators.items():
    if "ankle" in jn: ac.series_k = 52.0
cfg.episode_length_s = 200.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
bn = list(robot.data.body_names); fid = [bn.index(f) for f in _FEET]
jn = list(robot.data.joint_names)
hr = [jn.index(j) for j in jn if "hip_roll" in j]
n = u.num_envs

def width_body():
    fp = robot.data.body_pos_w[:, fid, :] - robot.data.root_pos_w.unsqueeze(1)
    q = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    fb = quat_apply_inverse(q.reshape(-1, 4), fp.reshape(-1, 3)).reshape(n, 2, 3)
    return (fb[:, 0, 1] - fb[:, 1, 1]).abs()

obs, _ = env.get_observations()
# reference: foot separation at the DEFAULT joint pose (what "hip width" means for this robot)
with torch.inference_mode():
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))
    u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    w_ref = width_body().mean().item()
    env.reset(); obs, _ = env.get_observations()
    def run(vx, standing, steps):
        W, HR = [], []
        for k in range(steps):
            cmd.vel_command_b[:, 0] = vx; cmd.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = standing
            obs_, _, _, _ = env.step(policy(obs if k == 0 else obs_))
            if k >= steps // 3:
                alive = robot.data.root_pos_w[:, 2] >= 0.55
                W.append(width_body()[alive].cpu()); HR.append(robot.data.joint_pos[alive][:, hr].abs().mean(1).cpu())
        return torch.cat(W), torch.cat(HR)
    obs_ = obs
    wS, hS = run(0.0, True, 300)
    wW, hW = run(a.vx, False, 500)
print(f"{a.label}: foot separation at DEFAULT pose = {w_ref*100:.1f} cm  (the anatomical hip width)")
print(f"  STAND  width mean {wS.mean()*100:5.1f} cm  p90 {wS.quantile(0.9)*100:5.1f}  |hip_roll| mean {hS.mean()*57.3:4.1f} deg")
print(f"  WALK {a.vx:.1f} width mean {wW.mean()*100:5.1f} cm  p90 {wW.quantile(0.9)*100:5.1f}  |hip_roll| mean {hW.mean()*57.3:4.1f} deg")
print(f"  walk width / anatomical = {wW.mean().item()/w_ref:.2f}x")
env.close(); simulation_app.close()
