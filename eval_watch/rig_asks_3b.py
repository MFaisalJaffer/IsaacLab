"""Rig handoff #3B: §1 cmd=(0,0,0) stand survival + §3.1 spring loading."""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip
p = argparse.ArgumentParser()
p.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--secs", type=float, default=45.0)
p.add_argument("--label", type=str, default="rig3b")
p.add_argument("--delay_steps", type=int, default=-1,
               help=">=0 pins EVERY joint's command delay to this many 5 ms steps. Rig REPLY4: hardware "
                    "onset is 10-20 ms on all ten joints; training had 0-5 ms on the hips.")
p.add_argument("--series_k", type=float, default=None,
               help="pin ankle K_s (Nm/rad). Since lineage 9 randomises it log-uniform "
                    "20-120, a single-point probe must SAY which point it ran.")
p.add_argument("--nominal", action="store_true",
               help="rig ask 2026-08-24: ALL domain randomization off, every parameter at its "
                    "centre value — isolates population-averaging bias from real plant difference")
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
# EXACTLY the rig's ask: flat, cmd=0 from t=0, no pushes, no resets, no DR
cfg.commands.base_velocity.rel_standing_envs = 1.0
cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level","ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
if a.nominal:
    # every stochastic element pinned to its centre
    for ev in ("add_base_mass", "add_limb_masses", "randomize_rigid_body_material",
               "randomize_joint_friction_ankles", "randomize_joint_friction_hip_pitch_roll",
               "randomize_joint_friction_knees", "randomize_joint_friction_yaws",
               "randomize_joint_play", "base_com"):
        if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
    # COM: correction only, zero spread (local -y = world forward)
    if getattr(cfg.events, "correct_torso_com", None) is not None:
        cfg.events.correct_torso_com.params["com_range"] = {
            "x": (0.0, 0.0), "y": (-0.0243, -0.0243), "z": (0.0, 0.0)}
    # floor: midpoint of our bracket (we have no measured value from the rig)
    if getattr(cfg.events, "physics_material", None) is not None:
        cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.7)
        cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    # actuator stochastics: T-V randomization off, play pinned at the measured 0.3 deg
    for _jn, _actcfg in cfg.scene.robot.actuators.items():
        _actcfg.tv_randomization = 0.0
        _actcfg.play_range = (math.radians(0.3), math.radians(0.3))
    print("[nominal] all DR disabled; COM/friction/play/TV pinned at centre")
if a.series_k is not None:
    for _jn, _actcfg in cfg.scene.robot.actuators.items():
        if "ankle" in _jn:
            _actcfg.series_k = a.series_k
    print(f"[plant] ankle K_s pinned at {a.series_k} Nm/rad")
if a.delay_steps >= 0:
    for _jn, _actcfg in cfg.scene.robot.actuators.items():
        _actcfg.min_delay, _actcfg.max_delay = a.delay_steps, a.delay_steps
    print(f"[plant] command delay pinned at {a.delay_steps} steps ({5*a.delay_steps} ms) on every joint")
cfg.episode_length_s = a.secs + 10
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make(a.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=acfg.clip_actions)
r = OnPolicyRunner(env, acfg.to_dict(), log_dir=None, device=acfg.device)
r.load(retrieve_file_path(a.checkpoint)); policy = r.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped; robot = u.scene["robot"]; cmd = u.command_manager.get_term("base_velocity")
n = u.num_envs; dt = u.step_dt
fell_at = torch.full((n,), float("nan"))
worst = torch.zeros(n)          # rig ask: worst |tilt| per survivor over the window
obs, _ = env.get_observations()
defl_log, tau_log = [], []
with torch.inference_mode():
    for k in range(int(a.secs/dt)):
        cmd.vel_command_b[:, :] = 0.0
        if hasattr(cmd, "is_standing_env"): cmd.is_standing_env[:] = True
        obs, _, _, _ = env.step(policy(obs))
        tilt_now = (torch.asin(robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(-1,1))
                    * 180/math.pi).cpu()
        worst = torch.maximum(worst, tilt_now)
        down = (robot.data.root_pos_w[:, 2] < 0.55).cpu()
        newly = down & torch.isnan(fell_at)
        fell_at[newly] = k * dt
        if 100 < k < 400:      # §3.1 sample while standing
            for nm, ac in robot.actuators.items():
                if "ankle" in nm and getattr(ac, "_rotor_pos", None) is not None:
                    defl_log.append((ac._rotor_pos - robot.data.joint_pos[:, ac.joint_indices]).abs().mean().item())
                    tau_log.append(ac.applied_effort.abs().mean().item())
alive = torch.isnan(fell_at)
med = fell_at[~alive].median().item() if (~alive).any() else float("nan")
print(f"{a.label}: cmd=(0,0,0) flat stand, {a.secs:.0f}s, {n} envs")
print(f"  SURVIVED full {a.secs:.0f}s: {alive.float().mean()*100:.0f}%  |  median fall time of the rest: {med:.2f}s")
pitch = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1,1))
alive_now = robot.data.root_pos_w[:, 2] >= 0.55
if alive_now.any():
    print(f"  standing pitch (alive): mean {math.degrees(pitch[alive_now].mean().item()):+.2f} deg")
if defl_log:
    print(f"  SPRING LOAD (rig expects 2-4 deg): |rotor-joint| mean {math.degrees(sum(defl_log)/len(defl_log)):.2f} deg"
          f"  max {math.degrees(max(defl_log)):.2f} deg  |tau_s| mean {sum(tau_log)/len(tau_log):.2f} Nm")
w = worst[alive]
if alive.any():
    q = [round(w.quantile(x).item(),1) for x in (0.5,0.9,1.0)]
    print(f"  WORST |tilt| across surviving stands: median {q[0]} deg  p90 {q[1]}  max {q[2]}  "
          f"(rig survivors ranged 8.1-19.4)")
    print(f"  survivors with worst-tilt >8 deg: {(w>8).float().mean()*100:.0f}%")
print("RIG3B-DONE")
env.close(); simulation_app.close()
