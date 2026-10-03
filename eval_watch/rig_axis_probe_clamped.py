#!/usr/bin/env python3
"""Definitive ankle-axis probe, v4 — other joints held by KINEMATIC CLAMP.

v3 held them with a stiff PD (kp 4000) which diverged at dt=0.002 explicit Euler:
NaN in QACC, "leg moved 279213 deg", and a garbage verdict. A clamp is exact and
unconditionally stable, and locking those joints is exactly the intent.

Controls reported so that a failed perturbation cannot be mistaken for a result
(training's retracted test moved the joints only 0.2-0.4 deg and read gravity):
  dq        did the test ankle actually move?
  leak      did any clamped joint move? (must be ~0)
  pitch/dq  is the foot following the ankle and nothing else? (want ~ +-1)
"""
import io, math
import numpy as np
import mujoco
from mujoco_scenes.mjcf import load_mjmodel

SRC = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/robot.mjcf"
PIN = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/_axisprobe4.mjcf"
JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02", "dof_left_hip_pitch_04",
          "dof_left_hip_roll_04", "dof_left_hip_yaw_03", "dof_left_knee_04",
          "dof_left_ankle_02"]
FOOT = [("dof_right_ankle_02", "KB_D_501R_R_LEG_FOOT"),
        ("dof_left_ankle_02", "KB_D_501L_L_LEG_FOOT")]
TAU, STEPS = 0.2, 200

src = io.open(SRC, encoding="utf-8").read()
p = src.replace('<freejoint name="floating_base" />', "")
p = p.replace('<body name="base" pos="0.00000000 0.00000000 1.00947904"',
              '<body name="base" pos="0.00000000 0.00000000 2.50000000"')
io.open(PIN, "w", encoding="utf-8").write(p)
m = load_mjmodel(PIN, "smooth")
IDX = {}
for n in JOINTS:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    IDX[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])


def foot_pitch(d, bid):
    fx = d.xmat[bid].reshape(3, 3)[:, 0]
    return math.degrees(math.atan2(-fx[2], math.hypot(fx[0], fx[1])))


print("+%.2f Nm on ONE ankle; all other joints kinematically clamped; base pinned.\n" % TAU)
print("%-16s %11s %13s %10s %10s" % ("test joint", "dq (deg)", "foot pitch d", "pitch/dq", "leak"))
out = {}
for jn, bn in FOOT:
    d = mujoco.MjData(m)
    qa, dof = IDX[jn]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
    others = [n2 for n2 in JOINTS if n2 != jn]

    def clamp(vals):
        for n2 in others:
            q2, v2 = IDX[n2]
            d.qpos[q2] = vals[n2]; d.qvel[v2] = 0.0

    vals = {n2: 0.0 for n2 in others}
    for _ in range(400):
        d.qfrc_applied[:] = 0.0
        clamp(vals); mujoco.mj_step(m, d)
    q0, p0 = float(d.qpos[qa]), foot_pitch(d, bid)
    for _ in range(STEPS):
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[dof] = TAU
        clamp(vals); mujoco.mj_step(m, d)
    dq = math.degrees(float(d.qpos[qa]) - q0)
    dp = foot_pitch(d, bid) - p0
    leak = math.degrees(max(abs(float(d.qpos[IDX[n2][0]]) - vals[n2]) for n2 in others))
    out[jn] = (dq, dp)
    print("%-16s %+10.3f %+12.3f %+9.2f %9.4f" %
          (jn.replace("dof_", ""), dq, dp, dp / dq if abs(dq) > 1e-9 else float("nan"), leak))

dqR, dpR = out["dof_right_ankle_02"]
dqL, dpL = out["dof_left_ankle_02"]
print()
print("CONTROL  ankle actually moved:  R %+.2f  L %+.2f deg" % (dqR, dqL))
print("RESULT   foot pitch sign:       R %+.0f  L %+.0f" % (np.sign(dpR), np.sign(dpL)))
same_dq = (np.sign(dqR) == np.sign(dqL))
print("         (same applied torque gave %s joint-motion signs)"
      % ("SAME" if same_dq else "OPPOSITE"))
print("VERDICT: %s" % ("MIRRORED (physical = tau_L - tau_R)" if (dpR * dpL) < 0
                       else "ALIGNED (physical = tau_L + tau_R)"))
print("\nhardware URDF declares: right ankle axis '0 0 1', left ankle axis '0 0 -1'")
