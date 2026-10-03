"""Where is the whole-robot COM relative to the ankle axis, in WORLD frame?
The rig's correction is +11.6 mm world-FORWARD. Isaac's randomize_rigid_body_com
applies its offset in the BODY frame, and the rig notes the torso body is
quat-rotated ~90 deg about z -> a body-frame +x may not be world-forward."""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, torch, os
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=2)
_x = float(os.environ.get("KBOT_COM_X", "0.0"))
cfg.events.correct_torso_com.params["com_range"] = {"x": (0.0, 0.0), "y": (-0.0243 + _x, -0.0243 + _x), "z": (0.0, 0.0)}
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=cfg, render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]
env.reset()
with torch.inference_mode():
    for _ in range(30): env.step(torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device))
    masses = robot.root_physx_view.get_masses().to(u.device)          # (n, nbody)
    coms_b = robot.root_physx_view.get_coms().to(u.device)            # (n, nbody, 7) local
    names = list(robot.data.body_names)
    ti = names.index("Torso_Side_Right")
    print(f"torso local COM offset applied (body frame): {coms_b[0, ti, :3].tolist()}")
    # world COM: body world pos + rotated local com offset
    from isaaclab.utils.math import quat_apply
    bq = robot.data.body_quat_w[0]; bp = robot.data.body_pos_w[0]
    com_w = bp + quat_apply(bq, coms_b[0, :, :3])
    m = masses[0]
    whole = (com_w * m.unsqueeze(1)).sum(0) / m.sum()
    ank = [i for i, nm in enumerate(names) if "FOOT" in nm]
    # ankle axis x ~ foot body x (axis sits 7.7 cm from heel, foot body is at the axis)
    ank_x = robot.data.body_pos_w[0, ank, 0].mean()
    print(f"total mass {m.sum():.2f} kg")
    print(f"[body-frame x offset applied: {_x*1000:.0f} mm]")
    print(f"whole-robot COM world xyz = {[round(v,4) for v in whole.tolist()]}")
    print(f"whole-robot COM world x = {whole[0]:.4f}  ankle-axis world x = {ank_x:.4f}")
    print(f"  COM - ankle axis = {(whole[0]-ank_x)*1000:+.1f} mm   (rig target: ~0, i.e. ON the axis)")
    print(f"  COM - ankle axis, LATERAL y = {(whole[1]-robot.data.body_pos_w[0, ank, 1].mean())*1000:+.1f} mm")
print("COMCHECK-DONE")
env.close(); simulation_app.close()
