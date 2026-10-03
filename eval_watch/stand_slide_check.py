"""SLIDE-vs-STEP check: when the feet return to nominal stance, do they LIFT
and step, or scuff along the ground? Measures, during the re-home window:
  - stance width trajectory
  - cumulative in-contact foot travel (= sliding; the Goodhart path)
  - foot-lift events (= stepping; the intended path)
Two cases: natural settle after the trained entry, and after a 20 N push.
"""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--ankle_play_deg", type=float, default=0.3)
p.add_argument("--label", type=str, default="slide_check")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app

import gymnasium as gym, math, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
acfg = cli_args.parse_rsl_rl_cfg(a.task.split(":")[-1], a)
cfg.commands.base_velocity.rel_standing_envs = 1.0
cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
cfg.episode_length_s = 300.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)

env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
names = list(robot.data.body_names); fid = [names.index(f) for f in _FEET]
cs = u.scene["contact_forces"]; cfid = [cs.body_names.index(f) for f in _FEET]
n = u.num_envs; dt = u.step_dt

if a.ankle_play_deg > 0:
    with torch.inference_mode():
        env.step(torch.zeros(n, u.action_manager.total_action_dim, device=u.device))
    for nm, act in robot.actuators.items():
        if getattr(act, "_play", None) is not None and "ankle" in nm:
            act._play[:] = math.radians(a.ankle_play_deg)

def set_cmd(vx, standing):
    cmd.vel_command_b[:, 0] = vx; cmd.vel_command_b[:, 1:] = 0.0
    if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = standing

def width_cm():
    fp_w = robot.data.body_pos_w[:, fid, :] - robot.data.root_pos_w.unsqueeze(1)
    q = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    fp_b = quat_apply_inverse(q.reshape(-1,4), fp_w.reshape(-1,3)).reshape(n,2,3)
    return (fp_b[:,0,1]-fp_b[:,1,1]).abs()*100

def window(steps, label, push=None):
    """run `steps`, accumulating in-contact foot travel and lift events"""
    slide = torch.zeros(n, 2, device=u.device); lifts = torch.zeros(n, 2, device=u.device)
    steps_ = torch.zeros(n, 2, device=u.device)      # REAL steps: foot clears >1 cm
    airborne = torch.zeros(n, 2, dtype=torch.bool, device=u.device)
    prev_contact = (cs.data.net_forces_w[:, cfid, :].norm(dim=-1) > 1.0)
    w0 = width_cm().mean().item()
    global obs
    force = torch.zeros(n,1,3, device=u.device); tq = torch.zeros(n,1,3, device=u.device)
    with torch.inference_mode():
        for k in range(steps):
            if push is not None and k < int(1.5/dt):
                sc = min(1.0, (k+1)/int(0.3/dt))
                force[:,0,1] = push*sc
                robot.set_external_force_and_torque(force, tq, body_ids=[0])
            elif push is not None and k == int(1.5/dt):
                force.zero_(); robot.set_external_force_and_torque(force, tq, body_ids=[0])
            set_cmd(0.0, True)
            obs2 = policy(obs)
            vel = robot.data.body_lin_vel_w[:, fid, :2].norm(dim=-1)
            contact = (cs.data.net_forces_w[:, cfid, :].norm(dim=-1) > 1.0)
            slide += vel * contact.float() * dt * 100      # cm travelled while planted
            lifts += (prev_contact & ~contact).float()
            # a REAL step = foot rises >1 cm above its planted height (contact
            # loss alone counts flicker, which at 0.3 deg play is frequent)
            fz = robot.data.body_pos_w[:, fid, 2]
            high = fz > (0.05 + 0.01)                 # planted foot body z ~0.05
            steps_ += (high & ~airborne).float()
            airborne = high
            prev_contact = contact
            obs, _, _, _ = env.step(obs2)
    w1 = width_cm().mean().item()
    alive = (robot.data.root_pos_w[:,2] >= 0.55)
    print(f"{label}: width {w0:.1f} -> {w1:.1f} cm | planted-foot travel {slide[alive].sum(dim=1).mean():.1f} cm "
          f"| REAL steps/robot {steps_[alive].sum(dim=1).mean():.2f} | contact-breaks {lifts[alive].sum(dim=1).mean():.2f} "
          f"| alive {alive.float().mean()*100:.0f}%")

obs, _ = env.get_observations()
with torch.inference_mode():
    for t in range(150+75):
        set_cmd(0.25 if t < 150 else 0.12, False)
        obs, _, _, _ = env.step(policy(obs))
print(f"--- {a.label} ---")
window(300, "A settle-from-walk (6 s)")
window(250, "B quiet hold  (5 s)")
window(500, "C after 20N push (10 s)", push=20.0)
env.close(); simulation_app.close()
