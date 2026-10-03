#!/usr/bin/env python3
"""Attribute the damping in the emulator's bare-plant swing, one factor at a time.

Standalone MuJoCo, no stack. Reference = freejoint removed, Euler, other joints
kinematically clamped: hip_pitch swings at ~102 deg/s with a 1.333 s period
(matches Isaac's 1.360). Through the emulator with PIN_WELD=1 it peaked at 13.6.
Something between those two configurations eats ~85% of the swing velocity.
Candidates, added one per rung so the culprit is unambiguous:

  A  reference: no freejoint, Euler, clamp
  B  freejoint + WELD (relpose = world-in-base = 0 0 -Z), Euler, clamp
  C  as B but weld relpose written as +Z  (what the first patch did -- if MuJoCo's
     relpose is body2-relative-to-body1 this drags the base to z=-Z, into the floor)
  D  as B + IMPLICITFAST integrator (the emulator forces this)
  E  as D + other joints PD-held (kp/kd as deployed) instead of clamped
  F  as E + ankle two-mass series spring (K_s 52, b 0.07, rotor J = K/(2pi 15)^2)

Every rung reports the CONTROL that matters here: base z drift and base tilt over
the run. A pin that lets the base move by more than a millimetre is not a pin.
"""
import io
import math

import mujoco
import numpy as np
from mujoco_scenes.mjcf import load_mjmodel

SRC = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/robot.mjcf"
DIR = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/"
JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
KP = dict(zip(JOINTS, [150, 150, 60, 150, 60, 150, 150, 60, 150, 60]))
KD = dict(zip(JOINTS, [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]))
TEST = "dof_right_hip_pitch_04"
Z, START, SECS, DT = 1.60, 15.0, 6.0, 0.002
KS, BS = 52.0, 0.07
JM = KS / (2 * math.pi * 15.0) ** 2

src = io.open(SRC, encoding="utf-8").read()


def build(weld=None, freejoint=True):
    p = src
    if not freejoint:
        p = p.replace('<freejoint name="floating_base" />', "")
        p = p.replace('<body name="base" pos="0.00000000 0.00000000 1.00947904"',
                      '<body name="base" pos="0.00000000 0.00000000 %.6f"' % Z)
    if weld is not None:
        w = ('<weld name="base_weld" body1="base" active="true" solref="0.005 1" '
             'relpose="0 0 %.6f 1 0 0 0"/>' % weld)
        p = p.replace("</mujoco>", "<equality>%s</equality></mujoco>" % w)
    path = DIR + "_ladder.mjcf"
    io.open(path, "w", encoding="utf-8").write(p)
    m = load_mjmodel(path, "smooth")
    m.opt.timestep = DT
    return m


def run(m, integrator, hold, series):
    m.opt.integrator = integrator
    d = mujoco.MjData(m)
    idx = {}
    for n in JOINTS:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        idx[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
    free = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    if free:
        d.qpos[0:3] = [0, 0, Z]; d.qpos[3:7] = [1, 0, 0, 0]
    qa, dof = idx[TEST]
    d.qpos[qa] = math.radians(START)
    mujoco.mj_forward(m, d)
    others = [o for o in JOINTS if o != TEST]
    ank = [o for o in others if "ankle" in o]
    rot = {o: (float(d.qpos[idx[o][0]]), 0.0) for o in ank}      # rotor pos, vel
    z0 = float(d.qpos[2]) if free else Z
    ys, vs, zs, tilt = [], [], [], []
    for _ in range(int(SECS / DT)):
        d.qfrc_applied[:] = 0.0
        for o in others:
            q2, v2 = idx[o]
            if hold == "clamp":
                d.qpos[q2] = 0.0; d.qvel[v2] = 0.0
            else:
                q, qd = float(d.qpos[q2]), float(d.qvel[v2])
                if series and o in ank:
                    qm, qdm = rot[o]
                    defl = qm - q
                    ts = KS * defl + BS * (qdm - qd)
                    ctrl = KP[o] * (0.0 - qm) + KD[o] * (0.0 - qdm)
                    qdm += (ctrl - ts) / JM * DT; qm += qdm * DT
                    rot[o] = (qm, qdm)
                    d.qfrc_applied[v2] = ts
                else:
                    d.qfrc_applied[v2] = KP[o] * (0.0 - q) + KD[o] * (0.0 - qd)
        mujoco.mj_step(m, d)
        ys.append(math.degrees(float(d.qpos[qa]))); vs.append(math.degrees(float(d.qvel[dof])))
        if free:
            zs.append(float(d.qpos[2]))
            w, x, y, zq = d.qpos[3:7]
            tilt.append(math.degrees(2 * math.acos(min(1.0, abs(w)))))
    rev = sum(1 for i in range(1, len(vs)) if vs[i] * vs[i - 1] < 0)
    per = 2.0 * SECS / rev if rev >= 2 else float("nan")
    bz = (max(zs) - min(zs)) * 1000 if zs else 0.0
    bt = max(tilt) if tilt else 0.0
    zend = zs[-1] if zs else Z
    return max(abs(v) for v in vs), per, rev, bz, bt, zend


EU, IF = mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
rungs = [
    ("A  no-freejoint  Euler  clamp",          build(None, freejoint=False), EU, "clamp", False),
    ("B  weld(-Z)      Euler  clamp",          build(-Z), EU, "clamp", False),
    ("C  weld(+Z)      Euler  clamp  [1st patch]", build(+Z), EU, "clamp", False),
    ("D  weld(-Z)      IMPLICITFAST clamp",    build(-Z), IF, "clamp", False),
    ("E  weld(-Z)      IMPLICITFAST PD-held",  build(-Z), IF, "pd", False),
    ("F  weld(-Z)      IMPLICITFAST PD + ankle series", build(-Z), IF, "pd", True),
]
print("hip_pitch released from %.0f deg, zero torque, %.0f s   (Isaac: 1.360 s, 63.5 deg/s)\n" % (START, SECS))
print("%-44s %10s %9s %5s %11s %9s %9s" % ("rung", "peak|qd|", "period", "rev", "base dz mm", "tilt deg", "z end"))
for name, m, integ, hold, series in rungs:
    pk, per, rev, bz, bt, zend = run(m, integ, hold, series)
    print("%-44s %8.1f°/s %8s %5d %11.2f %9.3f %9.3f%s"
          % (name, pk, ("%.3f s" % per) if per == per else "-", rev, bz, bt, zend,
             "   <- base NOT held" if bz > 1.0 else ""))
