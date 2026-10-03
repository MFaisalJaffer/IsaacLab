#!/usr/bin/env python3
"""Control for the bare-plant arm: is the torque ACTUALLY zero during the release?

Training's challenge: with a 1.36 s period and a 15 deg release a free pendulum must
reach ~69 deg/s, but our hip_pitch peaked at 6.4 deg/s with 0% velocity reversals --
an overdamped settle, not a swing. And the MJCF has damping 0.0 / frictionloss 0.0015
on every joint, so with FRICTION_SCALE=0 there is nothing left that could damp it.

The control I never ran: verify the commanded gains actually reached the plant.
Reads applied torque straight off the vcan MIT feedback during the release.

    |tau| ~ 0        -> genuinely bare; the slow motion is something else
    |tau| significant -> gains never reached zero, and the "swing" was the PD
                         pulling the joint to its target, not gravity

Also reports yaw, whose axis is VERTICAL ([0,0,1]) so it carries no gravity moment
and must not swing at all. Ours reported -105 deg at 68 deg/s, which cannot be
gravity -- most likely release transient coasting with nothing to stop it.
"""
import math
import subprocess
import threading
import time

import numpy as np
import rclpy
from odrive_mit_example.msg import LegCmd
from sensor_msgs.msg import JointState

JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
NODE = [3, 4, 5, 6, 7, 13, 14, 15, 16, 17]
KP = [150.0, 150.0, 60.0, 150.0, 60.0, 150.0, 150.0, 60.0, 150.0, 60.0]
KD = [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]
TEST = [0, 2]                       # right hip_pitch (gravity moment), right hip_yaw (none)
START, DUR, RATE = 15.0, 6.0, 100.0
T_MIN, T_MAX = -50.0, 50.0
TS = (T_MAX - T_MIN) / 4095.0

torque = {}


def canreader():
    p = subprocess.Popen(["candump", "-L", "vcan0", "vcan1"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        f = line.split()
        if len(f) < 3:
            continue
        try:
            cid_s, data_s = f[2].split("#")
            cid = int(cid_s, 16)
        except ValueError:
            continue
        if (cid & 0x1F) != 0x008:
            continue
        raw = bytes.fromhex(data_s)
        if len(raw) != 6:
            continue
        torque[cid >> 5] = (((raw[4] & 0x0F) << 8) | raw[5]) * TS + T_MIN


threading.Thread(target=canreader, daemon=True).start()
rclpy.init()
n = rclpy.create_node("bareplant_check")
pub = n.create_publisher(LegCmd, "/leg_impedance_controller/command", 10)
st = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: st.update({"n": list(m.name), "p": list(m.position),
                                           "v": list(m.velocity)}), 50)
import socket
_sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sk.sendto(b"RESET", ("127.0.0.1", 9994))
time.sleep(1.0)
_sk.sendto(b"HOMEMODE", ("127.0.0.1", 9994))
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
IDX = [st["n"].index(j) for j in JOINTS]


def send(p, kp, kd):
    c = LegCmd()
    c.position_des = [float(x) for x in p]
    c.velocity_des = [0.0] * 10
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in kp]
    c.kd_scale = [float(x) for x in kd]
    pub.publish(c)


print("BARE-PLANT CONTROL  (torque read off vcan, not inferred)\n")
print("%-22s %10s %10s %11s %10s %9s" %
      ("joint", "travel", "peak|qd|", "mean|tau|", "max|tau|", "reversals"))
for ji in TEST:
    jn = JOINTS[ji]
    p = [0.0] * 10
    p[ji] = math.radians(START)
    e = time.time() + 3.0
    while time.time() < e:
        send(p, KP, KD)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
    kp0, kd0 = list(KP), list(KD)
    kp0[ji] = 0.0
    kd0[ji] = 0.0
    ys, vs, ts_ = [], [], []
    end = time.time() + DUR
    while time.time() < end:
        send(p, kp0, kd0)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        ys.append(math.degrees(st["p"][IDX[ji]]))
        vs.append(math.degrees(st["v"][IDX[ji]]))
        tq = torque.get(NODE[ji])
        if tq is not None:
            ts_.append(abs(tq))
    rev = sum(1 for i in range(1, len(vs)) if vs[i] * vs[i - 1] < 0)
    print("%-22s %9.2f° %9.1f°/s %10.4f %10.4f %8.0f%%"
          % (jn.replace("dof_", ""), ys[-1] - ys[0], max(abs(v) for v in vs),
             np.mean(ts_) if ts_ else float("nan"),
             max(ts_) if ts_ else float("nan"),
             100.0 * rev / max(len(vs) - 1, 1)))
    e = time.time() + 1.0
    while time.time() < e:
        send([0.0] * 10, KP, KD)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)

print("\nexpected for a genuinely bare pendulum released from %.0f deg:" % START)
print("  hip_pitch ~69 deg/s peak (1.36 s period), |tau| ~ 0, reversals a few %%")
print("  hip_yaw   NO motion at all (axis is vertical, no gravity moment)")
rclpy.shutdown()
