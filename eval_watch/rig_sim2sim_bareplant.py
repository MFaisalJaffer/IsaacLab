#!/usr/bin/env python3
"""BARE-PLANT swing battery through the full stack: gravity + inertia only.

Per joint: drive to the release angle with full PD, then set kp=kd=0 on THAT joint
while the others stay PD-held, and record the free swing. Period = 2*pi*sqrt(I/mgl)
involves no actuator, so this is the cleanest inertia comparison available.

Preconditions, each VERIFIED at start (a bare plant that is not bare produced a
confident wrong number twice already):
  FRICTION_SCALE=0   -- read from the live emulator command line
  PIN_WELD=1         -- read from /tmp/vmotor.log (per-step pin override damps everything)
  wire torque ~0     -- vcan MIT feedback on the released joint, reported per row
  base tilt          -- from /imu, reported per row (a fallen base has huge tilt)

HOLD_SCALE env (default 1.0) multiplies the gains on the HELD joints. Our standalone
ladder showed PD-holding the other joints (vs clamping) lengthens hip_pitch period
1.333 -> 1.500 s; Isaac's PD-held 1.360 sits near our RIGID value. Sweeping
HOLD_SCALE tells how much of that gap is hold compliance on our side.
"""
import json
import math
import os
import re
import socket
import subprocess
import threading
import time

import numpy as np
import rclpy
from odrive_mit_example.msg import LegCmd
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState

JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
NODE = [3, 4, 5, 6, 7, 13, 14, 15, 16, 17]
KP = [150.0, 150.0, 60.0, 150.0, 60.0, 150.0, 150.0, 60.0, 150.0, 60.0]
KD = [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]
SIGN = [+1, +1, +1, -1, +1, +1, +1, +1, +1, -1]      # knees one-sided; ankles opposite
HOLD = float(os.environ.get("HOLD_SCALE", "1.0"))
START, DUR, RATE = 15.0, 6.0, 100.0
T_MIN, T_MAX = -50.0, 50.0
TS = (T_MAX - T_MIN) / 4095.0
OUT = "/tmp/sim2sim_bare"

# ---------------- preconditions ----------------
ps = subprocess.run(["bash", "-lc", "ps -eo args --no-headers | grep [v]irtual_motor_node | head -1"],
                    capture_output=True, text=True).stdout
mfs = re.search(r"friction-scale ([0-9.]+)", ps)
fs = float(mfs.group(1)) if mfs else float("nan")
weld = False
try:
    weld = "PIN_WELD=1" in open("/tmp/vmotor.log").read()
except OSError:
    pass
print("preconditions: friction-scale=%s  PIN_WELD=%s  HOLD_SCALE=%.2f" % (fs, weld, HOLD))
if not (fs == 0.0 and weld):
    raise SystemExit("!! refusing to run: need FRICTION_SCALE=0 and PIN_WELD=1 (a bare plant that "
                     "is not bare gives a confident wrong number)")

torque = {}


def canreader():
    p = subprocess.Popen(["candump", "-L", "vcan0", "vcan1"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        f = line.split()
        if len(f) < 3:
            continue
        try:
            cid_s, data_s = f[2].split("#"); cid = int(cid_s, 16)
        except ValueError:
            continue
        if (cid & 0x1F) != 0x008:
            continue
        raw = bytes.fromhex(data_s)
        if len(raw) == 6:
            torque[cid >> 5] = (((raw[4] & 0x0F) << 8) | raw[5]) * TS + T_MIN


threading.Thread(target=canreader, daemon=True).start()
os.makedirs(OUT, exist_ok=True)
rclpy.init()
n = rclpy.create_node("bareplant")
pub = n.create_publisher(LegCmd, "/leg_impedance_controller/command", 10)
st, imu = {}, {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: st.update({"n": list(m.name), "p": list(m.position), "v": list(m.velocity)}), 50)


def on_imu(m):
    o = m.orientation
    gz = -(1 - 2 * (o.x * o.x + o.y * o.y))
    imu["tilt"] = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))


n.create_subscription(Imu, "/imu", on_imu, qos_profile_sensor_data)
sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sk.sendto(b"RESET", ("127.0.0.1", 9994)); time.sleep(1.0)
sk.sendto(b"HOMEMODE", ("127.0.0.1", 9994))
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
IDX = [st["n"].index(j) for j in JOINTS]


def send(p, kp, kd):
    c = LegCmd()
    c.position_des = [float(x) for x in p]; c.velocity_des = [0.0] * 10
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in kp]; c.kd_scale = [float(x) for x in kd]
    pub.publish(c)


def spin(dur, p, kp, kd, rec=None, ji=None):
    e = time.time() + dur
    while time.time() < e:
        send(p, kp, kd)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        if rec is not None:
            rec.append((math.degrees(st["p"][IDX[ji]]), math.degrees(st["v"][IDX[ji]]),
                        torque.get(NODE[ji], float("nan")), imu.get("tilt", float("nan"))))


print("\n%-22s %8s %10s %9s %5s %9s %9s %7s" %
      ("joint", "travel", "peak|qd|", "period", "rev%", "mean|tau|", "max|tau|", "tilt"))
results = {}
kph = [k * HOLD for k in KP]; kdh = [k * HOLD for k in KD]
for ji, jn in enumerate(JOINTS):
    p = [0.0] * 10
    p[ji] = math.radians(START * SIGN[ji])
    spin(3.0, p, KP, KD)                                 # drive to release point at deployed gains
    kp0, kd0 = list(kph), list(kdh)
    kp0[ji] = 0.0; kd0[ji] = 0.0
    rec = []
    spin(DUR, p, kp0, kd0, rec, ji)
    ys = np.array([r[0] for r in rec]); vs = np.array([r[1] for r in rec])
    taus = np.array([abs(r[2]) for r in rec if r[2] == r[2]])
    tilts = np.array([r[3] for r in rec if r[3] == r[3]])
    rev = int(np.sum(vs[1:] * vs[:-1] < 0))
    per = 2.0 * DUR / rev if rev >= 3 else float("nan")
    m = {"release_deg": START * SIGN[ji], "travel_deg": round(float(ys[-1] - ys[0]), 3),
         "peak_qd_deg_s": round(float(np.max(np.abs(vs))), 2),
         "reversal_frac": round(rev / max(len(vs) - 1, 1), 4),
         "period_s": (round(per, 4) if per == per else None),
         "mean_abs_tau_Nm": round(float(taus.mean()), 4) if len(taus) else None,
         "max_abs_tau_Nm": round(float(taus.max()), 4) if len(taus) else None,
         "tilt_max_deg": round(float(tilts.max()), 3) if len(tilts) else None,
         "swings": bool(rev >= 3)}
    results[jn] = m
    print("%-22s %7.2f° %8.1f°/s %9s %5.0f %9.4f %9.4f %6.2f°%s"
          % (jn.replace("dof_", ""), m["travel_deg"], m["peak_qd_deg_s"],
             ("%.3f s" % per) if per == per else "-", 100 * m["reversal_frac"],
             m["mean_abs_tau_Nm"] or 0, m["max_abs_tau_Nm"] or 0, m["tilt_max_deg"] or 0,
             "" if (m["tilt_max_deg"] or 0) < 1.0 else "  <- BASE MOVED"))
    spin(1.0, [0.0] * 10, KP, KD)

json.dump({"schema": "sim2sim_bareplant_v1", "release_deg": START, "dur_s": DUR, "rate_hz": RATE,
           "hold_scale": HOLD, "held_kp": kph, "held_kd": kdh, "joint_order": JOINTS,
           "protocol": "test joint kp=kd=0; OTHER joints PD-held at deployed gains x HOLD_SCALE; "
                       "friction off (verified); base held by MuJoCo weld (PIN_WELD=1, verified)",
           "results": results}, open("%s/metrics.json" % OUT, "w"), indent=1)
print("\nwrote %s/metrics.json  (hold_scale %.2f)" % (OUT, HOLD))
print("Isaac (PD-held): hip_pitch 1.360 s / 63.5 deg/s | hip_roll 1.330 / 66.3 | knee 1.101 / 22.8 | yaw no swing")
rclpy.shutdown()
