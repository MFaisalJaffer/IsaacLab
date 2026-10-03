"""Rig handoff §4 (2026-08-24): how much of the standing balance moment is
carried by the CONTACT PATCH vs by the ankle actuator?

Both engines can report this without agreeing on an encoding (MuJoCo's explicit
rolling coefficient vs PhysX's implicit multi-point patch):
    gravity moment about the ankle axis  M_g   = m*g*(com_x - ankle_x)
    ankle-carried moment                 M_ank = sum(tau_s) over both ankles
    contact-carried moment               M_con = M_g - M_ank      (equilibrium)
    CoP excursion from the ankle axis    d_cop = M_con / F_vertical
"""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--label", type=str, default="contact_moment")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, math, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
acfg = cli_args.parse_rsl_rl_cfg(a.task.split(":")[-1], a)
cfg.commands.base_velocity.rel_standing_envs = 1.0
cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints",
           "add_base_mass","add_limb_masses","randomize_joint_play"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
if getattr(cfg.events, "correct_torso_com", None) is not None:
    cfg.events.correct_torso_com.params["com_range"] = {"x": (0.0,0.0), "y": (-0.0243,-0.0243), "z": (0.0,0.0)}
for _jn, _ac in cfg.scene.robot.actuators.items():
    _ac.tv_randomization = 0.0
    _ac.play_range = (math.radians(0.3), math.radians(0.3))
cfg.episode_length_s = 60.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)

env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
names = list(robot.data.body_names); feet = [i for i,n in enumerate(names) if "FOOT" in n]
cs = u.scene["contact_forces"]; cf = [cs.body_names.index(n) for n in names if "FOOT" in n]
obs, _ = env.get_observations()
Mg, Mank, Fz, TL, TR, TSUM = [], [], [], [], [], []
with torch.inference_mode():
    for k in range(1500):
        cmd.vel_command_b[:, :] = 0.0
        if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = True
        obs, _, _, _ = env.step(policy(obs))
        if k < 400: continue                      # settle
        alive = robot.data.root_pos_w[:, 2] >= 0.55
        if not alive.any(): continue
        m = robot.root_physx_view.get_masses().to(u.device)
        coms_b = robot.root_physx_view.get_coms().to(u.device)[:, :, :3]
        com_w = robot.data.body_pos_w + quat_apply(
            robot.data.body_quat_w.reshape(-1,4), coms_b.reshape(-1,3)).reshape(coms_b.shape)
        whole = (com_w * m.unsqueeze(2)).sum(1) / m.sum(1, keepdim=True)
        ank_x = robot.data.body_pos_w[:, feet, 0].mean(1)
        mg = (m.sum(1) * 9.81 * (whole[:, 0] - ank_x))
        # MIRRORED AXES (rig 2026-08-25 §5): left/right ankle joint axes are
        # opposed in world (our asset's action clips are mirrored: left
        # (-2.514, 0.454) vs right (-0.454, 2.514)). The physical pitch moment
        # is therefore tau_L - tau_R, NOT the raw sum we reported.
        tauL = sum(ac.applied_effort.sum(dim=1) for nm, ac in robot.actuators.items()
                   if "ankle" in nm and "left" in nm)
        tauR = sum(ac.applied_effort.sum(dim=1) for nm, ac in robot.actuators.items()
                   if "ankle" in nm and "right" in nm)
        tau = tauL - tauR                      # physical, mirrored-axis aware
        TL.append(tauL[alive].mean().item()); TR.append(tauR[alive].mean().item())
        TSUM.append((tauL + tauR)[alive].mean().item())
        fz = cs.data.net_forces_w[:, cf, 2].sum(dim=1)
        Mg.append(mg[alive].mean().item()); Mank.append(tau[alive].mean().item()); Fz.append(fz[alive].mean().item())
mg = sum(Mg)/len(Mg); mank = sum(Mank)/len(Mank); fz = sum(Fz)/len(Fz)
mcon = mg - mank
print(f"{a.label}: standing balance-moment split (nominal plant, {a.num_envs} envs)")
print(f"  gravity moment about ankle axis  M_g   = {mg:+.3f} Nm")
print(f"  ankle-carried (sum of tau_s)     M_ank = {mank:+.3f} Nm")
print(f"  CONTACT-carried                  M_con = {mcon:+.3f} Nm   ({abs(mcon)/max(abs(mg),1e-6)*100:.0f}% of gravity moment)")
tl=sum(TL)/len(TL); tr=sum(TR)/len(TR); tsum=sum(TSUM)/len(TSUM)
print(f"  per-ankle: tau_L = {tl:+.3f} Nm   tau_R = {tr:+.3f} Nm")
print(f"  raw sum (what we WRONGLY reported) = {tsum:+.3f} Nm ; physical (L-R) = {mank:+.3f} Nm")
print(f"  vertical contact force           F_z   = {fz:.1f} N")
print(f"  CoP EXCURSION from ankle axis    d_cop = {mcon/max(fz,1e-6)*1000:+.1f} mm  (foot is 212 mm long, axis 77 mm from heel)")
print("CONTACT-MOMENT-DONE")
env.close(); simulation_app.close()
