"""Rig #3 §3 validation protocol, frame-independent: position hold, NO policy,
release upright and see if it stands and which way it tips.
KBOT_COM_X = extra body-frame x offset (m) applied on top of the current event."""
import argparse, functools, os, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, math, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

extra = float(os.environ.get("KBOT_COM_X", "0.0"))
rigid = os.environ.get("PB_RIGID") == "1"
cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=16)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
# deterministic COM: correction only, no randomization, plus the sweep offset
cfg.events.correct_torso_com.params["com_range"] = {
    "x": (0.0, 0.0), "y": (-0.0243 + extra, -0.0243 + extra), "z": (0.0, 0.0)}
cfg.events.reset_base.params["pose_range"]["z"] = (0.02, 0.02)
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.episode_length_s = 40.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=cfg, render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]; n = u.num_envs; dt = u.step_dt
if rigid:
    for _nm,_a in robot.actuators.items():
        if hasattr(_a,"_series_k"): _a._series_k = 0.0
    print("[rigid] series zeroed for frame validation")
env.reset()
hold = torch.zeros(n, u.action_manager.total_action_dim, device=u.device)
fell = torch.full((n,), float("nan")); sign = torch.zeros(n)
with torch.inference_mode():
    for k in range(int(20.0/dt)):
        u.command_manager.get_term("base_velocity").vel_command_b[:, :] = 0.0
        env.step(hold)
        pitch = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1,1)).cpu()
        down = (robot.data.root_pos_w[:, 2] < 0.55).cpu()
        new = down & torch.isnan(fell)
        fell[new] = k*dt; sign[new] = torch.sign(pitch[new])
alive = torch.isnan(fell)
fwd = (sign > 0).sum().item(); bwd = (sign < 0).sum().item()
med = fell[~alive].median().item() if (~alive).any() else float("nan")
print(f"COM_X extra {extra*1000:+.0f} mm | stood 20s: {alive.sum().item()}/{n} "
      f"| median fall {med:.2f}s | fell forward {fwd}, backward {bwd}")
print("PASSIVE-DONE")
env.close(); simulation_app.close()
