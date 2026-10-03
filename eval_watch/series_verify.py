"""Rig handoff #3 §6.2 verification: static ratio and torque balance.
Hold the ankle at an angle with the PD commanding zero; the rotor must settle
at q_m = q*K_s/(K_s+kp) => q/q_m = (kp+K_s)/K_s = 3.6 at kp 60, K_s 23.
A ratio of ~1.0 means the spring path is not running (§6.1)."""
import functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, math, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=cfg, render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]
env.reset()
jn = list(robot.data.joint_names); aidx = [i for i,n in enumerate(jn) if "ankle" in n]
act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
with torch.inference_mode():
    for _ in range(60): env.step(act)          # settle, zero command
    # hold the ankle joints at a fixed angle by writing joint state each step
    q_hold = math.radians(3.0)
    for k in range(200):
        js = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
        js[:, aidx] = q_hold; jv[:, aidx] = 0.0
        robot.write_joint_state_to_sim(js, jv)
        env.step(act)
    for nm, ac in robot.actuators.items():
        if "ankle" in nm and getattr(ac, "_rotor_pos", None) is not None:
            qm = ac._rotor_pos.mean().item()
            ratio = q_hold / qm if abs(qm) > 1e-6 else float("inf")
            tau_s = ac.applied_effort.mean().item() if ac.applied_effort is not None else float("nan")
            print(f"{nm}: q={math.degrees(q_hold):.2f} deg  q_m={math.degrees(qm):.3f} deg  "
                  f"RATIO q/q_m={ratio:.2f} (expect ~3.6)  tau_s={tau_s:+.3f} Nm")
print("SERIES-VERIFY-DONE")
env.close(); simulation_app.close()
