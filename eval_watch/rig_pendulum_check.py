#!/usr/bin/env python3
"""Is our base pin the reason the bare plant will not swing? Test it directly.

Our emulator "pins" the base by rewriting root qpos and ZEROING root qvel every
physics step, immediately before mj_step (virtual_motor_node.py:575). That is not a
weld -- it removes momentum from the system every step and damps whatever hangs off
the base. Training found the identical bug on their side (their Bug 1) and fixing it
moved rise times up to 40 ms and hip-roll leak 3.88 -> 0.24 deg.

This bypasses the emulator entirely: pure MuJoCo, freejoint REMOVED from the XML so
the base is genuinely fixed, zero applied torque, all other joints held by kinematic
clamp (exact and stable -- a stiff PD diverges at this timestep).

Prediction if the diagnosis is right: hip_pitch swings freely with a ~1.36 s period
and ~69 deg/s peak from a 15 deg release, matching training. Yaw must NOT move at all
(its axis is vertical -> no gravity moment).
"""
import io
import math

import mujoco
import numpy as np
from mujoco_scenes.mjcf import load_mjmodel

SRC = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/robot.mjcf"
PIN = "/root/ros2_ws/ksim-kbot/ksim_kbot/kscale-assets/kbot-v2-legs/_pendulum.mjcf"
JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
TEST = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04",
        "dof_right_hip_yaw_03", "dof_right_knee_04"]
START, SECS = 15.0, 6.0

src = io.open(SRC, encoding="utf-8").read()
p = src.replace('<freejoint name="floating_base" />', "")          # genuine fix, not an override
p = p.replace('<body name="base" pos="0.00000000 0.00000000 1.00947904"',
              '<body name="base" pos="0.00000000 0.00000000 2.50000000"')   # clear of the floor
io.open(PIN, "w", encoding="utf-8").write(p)
m = load_mjmodel(PIN, "smooth")
IDX = {}
for n in JOINTS:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    IDX[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
dt = float(m.opt.timestep)
N = int(SECS / dt)

print("freejoint REMOVED (true fix, not a per-step override) | zero torque | dt %.4f s\n" % dt)
print("%-24s %10s %11s %11s %10s %s"
      % ("joint", "travel", "peak|qd|", "period", "reversals", "note"))
for jn in TEST:
    d = mujoco.MjData(m)
    qa, dof = IDX[jn]
    others = [o for o in JOINTS if o != jn]
    d.qpos[qa] = math.radians(START)
    mujoco.mj_forward(m, d)
    ys, vs = [], []
    for _ in range(N):
        for o in others:                       # kinematic clamp on everything else
            q2, v2 = IDX[o]
            d.qpos[q2] = 0.0
            d.qvel[v2] = 0.0
        d.qfrc_applied[:] = 0.0                # BARE: no torque at all
        mujoco.mj_step(m, d)
        ys.append(math.degrees(float(d.qpos[qa])))
        vs.append(math.degrees(float(d.qvel[dof])))
    rev = sum(1 for i in range(1, len(vs)) if vs[i] * vs[i - 1] < 0)
    # period from zero crossings of velocity (two per cycle)
    per = (2.0 * SECS / rev) if rev >= 2 else float("nan")
    note = "swings" if rev >= 2 else "no swing (expected for a vertical axis)"
    print("%-24s %9.2f° %10.1f°/s %10s %9.0f%% %s"
          % (jn.replace("dof_", ""), ys[-1] - ys[0], max(abs(v) for v in vs),
             ("%.3f s" % per) if per == per else "-", 100.0 * rev / len(vs), note))

print("\ntraining's bare-plant numbers, for comparison:")
print("  hip_pitch  period 1.360 s   peak 63.5 deg/s")
print("  hip_roll   period 1.330 s   peak 66.3 deg/s")
print("  knee       period 1.101 s   peak 22.8 deg/s")
print("  hip_yaw    no swing (vertical axis)")
