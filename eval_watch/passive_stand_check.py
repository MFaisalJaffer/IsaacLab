import faulthandler, functools, sys
faulthandler.enable()
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app

import gymnasium as gym, math, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=8)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
import os
if os.environ.get("PC_NOFRICTION")=="1":
    for fe in ("randomize_joint_friction_hip_pitch_roll","randomize_joint_friction_knees",
               "randomize_joint_friction_yaws","randomize_joint_friction_ankles",
               "randomize_joint_friction_hips_knees"):
        if getattr(cfg.events, fe, None) is not None: setattr(cfg.events, fe, None)
for cu in ("sustained_push_level","velocity_push_curriculum","ankle_play_level","plant_friction_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
cfg.events.reset_base.params["pose_range"]["z"] = (0.05, 0.05)
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=cfg, render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]
if os.environ.get("PC_OLDGAINS")=="1":
    _old_kp={"hip_pitch":150.,"hip_roll":150.,"hip_yaw":60.,"knee":150.,"ankle":60.}
    _old_kd={"hip_pitch":5.,"hip_roll":5.,"hip_yaw":3.,"knee":5.,"ankle":3.}
    for _nm,_a in robot.actuators.items():
        for _k in _old_kp:
            if _k in _nm:
                _a.stiffness[:] = _old_kp[_k]; _a.damping[:] = _old_kd[_k]
    print("[arm] OLD gains restored on live actuators")
if os.environ.get("PC_NOVISC")=="1":
    for _nm,_a in robot.actuators.items():
        if hasattr(_a,"_viscous_b"): _a._viscous_b = 0.0
    print("[arm] viscous zeroed")
env.reset()
act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
with torch.inference_mode():
    for k in range(300):   # 6 s zero-action PD hold at default pose
        env.step(act)
        if k in (49, 149, 299):
            pg = robot.data.projected_gravity_b
            tilt = torch.asin(pg[:, :2].norm(dim=1).clamp(-1,1)) * 180 / math.pi
            h = robot.data.root_pos_w[:, 2]
            print(f"t={0.02*(k+1):.1f}s  tilt mean {tilt.mean():.1f} p95 {tilt.quantile(0.95):.1f} deg   height mean {h.mean():.2f} m")
print("PASSIVE-CHECK-COMPLETE")
env.close()
simulation_app.close()
