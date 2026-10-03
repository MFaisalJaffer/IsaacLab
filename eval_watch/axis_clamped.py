"""Ankle axis convention, rig's clamped protocol (RIG_AXIS_MEASURED.md §3).

1. CLAMP every non-test joint by writing qpos/qvel directly each step — exact
   and unconditionally stable. Do NOT hold them with a stiff PD, and do NOT try
   to out-muscle the actuator: our TVCurveActuator's position PD overrode an
   effort target entirely in the previous attempt (joint dq 0.2-0.4 deg, so the
   foot motion was gravity and the "ALIGNED" verdict was an artefact).
   Here the TEST joint is driven by POSITION (which the actuator honours).
2. Report three controls and refuse the result if any fails:
     dq        - did the test joint actually move?          (want >> 1 deg)
     leak      - did anything clamped move?                 (want ~0)
     pitch/dq  - is the foot following the joint alone?     (want ~ +-1)
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
from isaaclab.utils.math import matrix_from_quat

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
IDX = {"right_ankle_02": jn.index("dof_right_ankle_02"),
       "left_ankle_02":  jn.index("dof_left_ankle_02")}
FOOT = {"right_ankle_02": bn.index("KB_D_501R_R_LEG_FOOT"),
        "left_ankle_02":  bn.index("KB_D_501L_L_LEG_FOOT")}

def foot_pitch(b):
    m = matrix_from_quat(robot.data.body_quat_w[:, b])
    return torch.asin((-m[:, 2, 0]).clamp(-1, 1)) * 180 / math.pi

rows = []
AXES = {}
with torch.inference_mode():
    for name, j in IDX.items():
        env.reset()
        b = FOOT[name]
        # base pinned well clear of the ground; all joints at home
        root = robot.data.default_root_state.clone(); root[:, 2] += 1.5
        q_hold = robot.data.default_joint_pos.clone()
        v_zero = torch.zeros_like(robot.data.joint_vel)
        for _ in range(10):
            robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:13])
            robot.write_joint_state_to_sim(q_hold, v_zero)
            u.sim.step(); robot.update(u.physics_dt)
        p0 = foot_pitch(b).mean().item(); q0 = q_hold[:, j].mean().item()
        R0 = matrix_from_quat(robot.data.body_quat_w[:, b]).clone()
        # drive the TEST joint by POSITION (+15 deg), CLAMP everything else
        target = q_hold.clone(); target[:, j] += math.radians(15.0)
        for _ in range(120):
            robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(root[:, 7:13])
            js = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
            keep = [k for k in range(len(jn)) if k != j]
            js[:, keep] = q_hold[:, keep]; jv[:, keep] = 0.0          # CLAMP (exact)
            robot.write_joint_state_to_sim(js, jv)
            robot.set_joint_position_target(target)                    # actuator honours this
            robot.write_data_to_sim()
            u.sim.step(); robot.update(u.physics_dt)
        dq = math.degrees(robot.data.joint_pos[:, j].mean().item() - q0)
        dpitch = foot_pitch(b).mean().item() - p0
        # DIRECT axis measurement: with only this joint free, the foot's net
        # rotation IS the joint rotation, so its axis-angle axis in world IS
        # the joint axis. This avoids the pitch-extraction projection, which
        # under-reads when the leg carries any yaw (left ratio came out 0.74).
        R1 = matrix_from_quat(robot.data.body_quat_w[:, b])
        dR = torch.bmm(R1, R0.transpose(1, 2))
        ang = torch.arccos(((dR[:, 0,0]+dR[:, 1,1]+dR[:, 2,2] - 1) / 2).clamp(-1, 1))
        ax = torch.stack([dR[:, 2,1]-dR[:, 1,2], dR[:, 0,2]-dR[:, 2,0], dR[:, 1,0]-dR[:, 0,1]], dim=1)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
        axm = ax.mean(dim=0); angm = math.degrees(ang.mean().item())
        print(f"   -> rotation axis (world xyz) = [{axm[0]:+.3f} {axm[1]:+.3f} {axm[2]:+.3f}]  "
              f"angle {angm:.2f} deg   (axis*angle vs dq: {angm/max(abs(dq),1e-9):.2f})")
        AXES[name] = axm.clone()
        leak = math.degrees((robot.data.joint_pos[:, [k for k in range(len(jn)) if k != j]]
                             - q_hold[:, [k for k in range(len(jn)) if k != j]]).abs().max().item())
        ratio = dpitch / dq if abs(dq) > 1e-6 else float("nan")
        rows.append((name, dq, dpitch, ratio, leak))
        print(f"{name:18s} dq {dq:+8.3f} deg | foot pitch d {dpitch:+8.3f} | pitch/dq {ratio:+.2f} | leak {leak:.4f} deg")

ok = all(abs(r[1]) > 1.0 and r[4] < 0.05 and 0.8 < abs(r[3]) < 1.25 for r in rows)
if not ok:
    print("\nINVALID — a control failed (need |dq|>1 deg, leak<0.05 deg, |pitch/dq| in 0.8-1.25)")
else:
    same = (rows[0][3] * rows[1][3]) > 0
    print(f"\nVERDICT: pitch/dq signs {'MATCH' if same else 'OPPOSE'} -> our USD ankles are "
          f"{'ALIGNED (physical moment = tau_L + tau_R)' if same else 'MIRRORED (physical moment = tau_L - tau_R)'}")
if len(AXES) == 2:
    l, r = AXES["left_ankle_02"], AXES["right_ankle_02"]
    dot = float((l * r).sum())
    print(f"\nDIRECT AXIS TEST: left.right = {dot:+.3f}  -> ankles are "
          f"{'ALIGNED (moment = tau_L + tau_R)' if dot > 0 else 'MIRRORED (moment = tau_L - tau_R)'}")
print("AXIS-CLAMPED-DONE")
env.close(); simulation_app.close()
