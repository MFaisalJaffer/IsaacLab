"""COMMAND SMOOTHNESS across regimes.

The rig's hardware complaint was jerky commands: nothing below ~3 deg tilt,
then everything at once past 4. We now price jerk in training (action-rate
taxes only 75%-discounted during disturbances) so this measures what the robot
would actually feel: ||action_t - action_{t-1}|| per 20 ms control step, in
    (a) quiet stand   (b) walking 0.3 m/s   (c) the 1 s after a 20 N push
plus the worst per-joint command rate, since a single buzzing joint is what
gets heard and felt on hardware.
"""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--label", type=str, default="jerk")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, math, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
acfg = cli_args.parse_rsl_rl_cfg(a.task.split(":")[-1], a)
cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
cfg.episode_length_s = 200.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
jn = list(robot.data.joint_names); n = u.num_envs; dt = u.step_dt
obs, _ = env.get_observations()

def run(steps, vx, standing, push=None, collect_from=0):
    global obs
    da, per_joint, prev = [], [], None
    force = torch.zeros(n,1,3, device=u.device); tq = torch.zeros(n,1,3, device=u.device)
    with torch.inference_mode():
        for k in range(steps):
            cmd.vel_command_b[:, 0] = vx; cmd.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = standing
            if push is not None and k < int(1.5/dt):
                sc = min(1.0, (k+1)/int(0.3/dt)); force[:,0,1] = push*sc
                robot.set_external_force_and_torque(force, tq, body_ids=[0])
            elif push is not None and k == int(1.5/dt):
                force.zero_(); robot.set_external_force_and_torque(force, tq, body_ids=[0])
            act = policy(obs)
            if prev is not None and k >= collect_from:
                d = (act - prev)
                alive = robot.data.root_pos_w[:, 2] >= 0.55
                if alive.any():
                    da.append(d[alive].norm(dim=1).cpu()); per_joint.append(d[alive].abs().mean(0).cpu())
            prev = act.clone()
            obs, _, _, _ = env.step(act)
    if not da: return None
    D = torch.cat(da); PJ = torch.stack(per_joint).mean(0)
    return D.mean().item(), D.quantile(0.95).item(), D.max().item(), PJ

print(f"{a.label}: command-rate ||da|| per 20 ms step (action units; scale 0.5 -> rad)")
# warm the gait, then quiet stand
run(150, 0.25, False); run(75, 0.12, False)
for name, args_ in (("QUIET STAND      ", (400, 0.0, True, None, 100)),
                    ("WALK 0.3 m/s     ", (400, 0.30, False, None, 100)),
                    ("AFTER 20 N PUSH  ", (100, 0.0, True, 20.0, 0))):
    res = run(*args_)
    if res is None: print(f"{name}: all fell"); continue
    m, p95, mx, pj = res
    worst = int(torch.argmax(pj))
    print(f"{name}: mean {m:.4f}  p95 {p95:.4f}  max {mx:.4f}   worst joint {jn[worst]} {pj[worst]:.4f}")
print("JERK-DONE")
env.close(); simulation_app.close()
