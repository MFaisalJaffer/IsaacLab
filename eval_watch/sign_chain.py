"""END-TO-END SIGN-CHAIN VERIFICATION.

The chain is: applied_effort -> joint axis in world -> moment on the foot ->
contact CoP -> whole-body equilibrium. Rather than argue any single link, test
the INVARIANT that closes the whole chain:

    for a static robot, the load-weighted mean CoP must lie under the COM.

Per foot (static, so moments about its ankle axis sum to zero):
    contact_moment_i = -tau_i_world - m_foot*g*(foot_com_x - ankle_x)
    CoP_i            = contact_moment_i / Fz_i        (offset from ankle axis)
tau_i_world is computed BOTH ways (aligned: +tau; mirrored: tau*axis_y_sign),
and we report which convention closes |mean CoP - COM_x|. That is the answer,
and it needs no assumption about which link carries the sign.
"""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
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
           "randomize_joint_play","add_base_mass","add_limb_masses"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
if getattr(cfg.events, "correct_torso_com", None) is not None:
    cfg.events.correct_torso_com.params["com_range"] = {"x": (0.0,0.0), "y": (-0.0243,-0.0243), "z": (0.0,0.0)}
for _jn, _ac in cfg.scene.robot.actuators.items():
    _ac.tv_randomization = 0.0
cfg.episode_length_s = 40.0
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
bn = list(robot.data.body_names); jn = list(robot.data.joint_names)
LF, RF = bn.index("KB_D_501L_L_LEG_FOOT"), bn.index("KB_D_501R_R_LEG_FOOT")
LJ, RJ = jn.index("dof_left_ankle_02"), jn.index("dof_right_ankle_02")
cs = u.scene["contact_forces"]; cLF, cRF = cs.body_names.index(bn[LF]), cs.body_names.index(bn[RF])
# measured axis y-signs (eval_watch/axisclamp2.log): left +0.733, right -0.915
AX_L, AX_R = +1.0, -1.0

acc = {k: [] for k in ("comx","ankx","fzL","fzR","tauL","tauR","fcomLx","fcomRx","mf")}
obs, _ = env.get_observations()
with torch.inference_mode():
    for k in range(1200):
        cmd.vel_command_b[:, :] = 0.0
        if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = True
        obs, _, _, _ = env.step(policy(obs))
        if k < 500: continue
        alive = robot.data.root_pos_w[:, 2] >= 0.55
        if not alive.any(): continue
        m = robot.root_physx_view.get_masses().to(u.device)
        coms_b = robot.root_physx_view.get_coms().to(u.device)[:, :, :3]
        com_w = robot.data.body_pos_w + quat_apply(
            robot.data.body_quat_w.reshape(-1,4), coms_b.reshape(-1,3)).reshape(coms_b.shape)
        whole = (com_w * m.unsqueeze(2)).sum(1) / m.sum(1, keepdim=True)
        acc["comx"].append(whole[alive,0].mean().item())
        acc["ankx"].append(robot.data.body_pos_w[alive][:,[LF,RF],0].mean().item())
        fz = cs.data.net_forces_w[:, [cLF,cRF], 2]
        acc["fzL"].append(fz[alive,0].mean().item()); acc["fzR"].append(fz[alive,1].mean().item())
        tl = [ac.applied_effort for nm,ac in robot.actuators.items() if nm=="dof_left_ankle_02"][0]
        tr = [ac.applied_effort for nm,ac in robot.actuators.items() if nm=="dof_right_ankle_02"][0]
        acc["tauL"].append(tl[alive].mean().item()); acc["tauR"].append(tr[alive].mean().item())
        acc["fcomLx"].append(com_w[alive,LF,0].mean().item()); acc["fcomRx"].append(com_w[alive,RF,0].mean().item())
        acc["mf"].append(m[alive][:,LF].mean().item())
M = {k: sum(v)/len(v) for k,v in acc.items()}
g = 9.81
print(f"standing-state means: com_x {M['comx']:.4f}  ankle_x {M['ankx']:.4f}  "
      f"Fz L/R {M['fzL']:.1f}/{M['fzR']:.1f} N  tau L/R {M['tauL']:+.2f}/{M['tauR']:+.2f} Nm  foot mass {M['mf']:.3f} kg")
print(f"COM offset from ankle axis: {(M['comx']-M['ankx'])*1000:+.1f} mm  <- the CoP must land here\n")
for label, sL, sR in (("ALIGNED  (tau as-is)", +1.0, +1.0), ("MIRRORED (axis-signed)", AX_L, AX_R)):
    copL = (-(M['tauL']*sL) - M['mf']*g*(M['fcomLx']-M['ankx'])) / max(M['fzL'],1e-6)
    copR = (-(M['tauR']*sR) - M['mf']*g*(M['fcomRx']-M['ankx'])) / max(M['fzR'],1e-6)
    mean_cop = (copL*M['fzL'] + copR*M['fzR']) / max(M['fzL']+M['fzR'],1e-6)
    err = (mean_cop - (M['comx']-M['ankx'])) * 1000
    print(f"{label}: CoP L {copL*1000:+7.1f} mm  R {copR*1000:+7.1f} mm  "
          f"load-weighted mean {mean_cop*1000:+7.1f} mm  |CLOSURE ERROR {err:+8.1f} mm|")
print("\nThe convention with the small closure error is the physical one.")
print("SIGN-CHAIN-DONE")
env.close(); simulation_app.close()
