"""Measure the ankle joint-axis convention DIRECTLY, no inference.

Method: hold the base fixed, apply a known +torque to ONE ankle at a time with
everything else passive, and watch which way that foot's body rotates in WORLD.
If the two ankles rotate the body OPPOSITE ways for the same signed joint
torque, the axes are mirrored (rig's MJCF convention) and the physical pitch
moment is tau_L - tau_R. If they rotate the SAME way, axes are same-signed and
the moment is tau_L + tau_R.
"""
import argparse, functools, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, math, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=4)
for ev in ("push_robot","sustained_push","walk_at_spawn","stand_corridor",
           "randomize_actuator_gains","randomize_gains_small_joints","randomize_gains_04_joints",
           "randomize_joint_play","add_base_mass","add_limb_masses"):
    if getattr(cfg.events, ev, None) is not None: setattr(cfg.events, ev, None)
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
for t in ("time_out","base_contact","base_height","bad_orientation"):
    if getattr(cfg.terminations, t, None) is not None: setattr(cfg.terminations, t, None)
env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=cfg, render_mode=None)
u = env.unwrapped; robot = u.scene["robot"]; n = u.num_envs; dev = u.device
env.reset()
jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
L = jn.index("dof_left_ankle_02"); R = jn.index("dof_right_ankle_02")
LF = bn.index("KB_D_501L_L_LEG_FOOT"); RF = bn.index("KB_D_501R_R_LEG_FOOT")

def pitch_of(body_idx):
    """foot pitch about world-y, from its rotation matrix' x-axis z-component"""
    from isaaclab.utils.math import matrix_from_quat
    m = matrix_from_quat(robot.data.body_quat_w[:, body_idx])
    return torch.asin((-m[:, 2, 0]).clamp(-1, 1)) * 180 / math.pi

results = {}
with torch.inference_mode():
    for label, jidx, bidx in (("LEFT", L, LF), ("RIGHT", R, RF)):
        env.reset()
        # lift the base clear of the ground and PIN it, so only the driven
        # ankle moves and contact cannot mask the rotation
        for _ in range(5):
            root = robot.data.default_root_state.clone()
            root[:, 2] += 1.2
            robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:13])
            env.step(torch.zeros(n, u.action_manager.total_action_dim, device=dev))
        p0 = pitch_of(bidx).mean().item()
        q0 = robot.data.joint_pos[:, jidx].mean().item()
        eff = torch.zeros(n, len(jn), device=dev)
        eff[:, jidx] = 3.0                    # +3 Nm on this ankle only
        for _ in range(40):
            root = robot.data.default_root_state.clone(); root[:, 2] += 1.2
            robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:13])
            robot.set_joint_effort_target(eff)
            robot.write_data_to_sim()
            u.sim.step(); robot.update(u.physics_dt)
        p1 = pitch_of(bidx).mean().item()
        q1 = robot.data.joint_pos[:, jidx].mean().item()
        dq = math.degrees(q1 - q0)
        results[label] = (p1 - p0, dq)
        # THE CONTROL I OMITTED FIRST TIME (rig's protocol reports it): did the
        # commanded torque actually move the JOINT in the + direction? If dq is
        # ~0 the effort never reached the joint (the TV actuator overrides
        # effort targets) and the foot motion is just gravity — which would
        # make "both feet rotate the same way" an artefact, not a measurement.
        print(f"{label} ankle: +3.0 Nm -> joint dq {dq:+.2f} deg | foot pitch "
              f"{p0:+.2f} -> {p1:+.2f} ({p1-p0:+.2f} deg)")
dqL, dqR = results["LEFT"][1], results["RIGHT"][1]
if abs(dqL) < 1.0 or abs(dqR) < 1.0:
    print(f"\nINVALID: joint barely moved (dq L {dqL:+.2f}, R {dqR:+.2f}) — the torque did not "
          f"reach the joint, so foot pitch reflects gravity, not the axis. Test says nothing.")
    print("AXIS-CHECK-DONE"); env.close(); simulation_app.close(); raise SystemExit
same = (results["LEFT"][0] * results["RIGHT"][0]) > 0
print(f"\nVERDICT: same-signed +torque rotates the two feet "
      f"{'the SAME way -> axes ALIGNED -> pitch moment = tau_L + tau_R' if same else 'OPPOSITE ways -> axes MIRRORED -> pitch moment = tau_L - tau_R'}")
print("AXIS-CHECK-DONE")
env.close(); simulation_app.close()
