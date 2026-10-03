#!/usr/bin/env python3
"""BATTERY 3 — constant-velocity friction sweep. The friction comparison we still lack.

After Test B was retracted we had NO comparison of the dynamic Coulomb friction that
is actually in both plants. Test A measures STATIC friction (both sides ~zero, and
neither sim has stiction). The bare-plant arm switches friction OFF. So the 4 Nm that
a policy actually feels has never been compared.

METHOD. Drive a joint at constant velocity and measure the torque it takes. Run each
speed in BOTH directions: gravity and inertia are identical in the two directions
while Coulomb friction FLIPS SIGN, so

    tau_friction(v) = (tau_forward - tau_backward) / 2      <- gravity cancels exactly
    tau_gravity     = (tau_forward + tau_backward) / 2      <- reported as a cross-check

Fit tau_friction against |v|:
    intercept  -> Coulomb  tau_c   (the 4.0 / 0.6 / 0.4 Nm we configured)
    slope      -> viscous  b

TORQUE IS MEASURED, NOT INFERRED. The emulator writes real torque into the vcan MIT
feedback (`encode_mit_fb(nid,q,qd,tau)`) and `_tau` returns ctrl+friction, so the wire
carries the friction term. Same 12-bit decode as the hardware drives.

CONTROL: velocity must actually be achieved. If the joint cannot track the commanded
rate the torque means something else, so we report tracking error and drop any point
that missed. (Three tests this week produced confident numbers from motion that never
happened.)
"""
import json
import math
import os
import socket
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
NOMINAL = {"hip_pitch": 4.0, "hip_roll": 4.0, "hip_yaw": 0.4, "knee": 0.6, "ankle": 0.1}
SPEEDS = [5.0, 10.0, 20.0, 40.0]          # deg/s
SPAN = 12.0                                 # deg: sweep -SPAN..+SPAN about zero
RATE = 100.0
T_MIN, T_MAX = -50.0, 50.0
TS = (T_MAX - T_MIN) / 4095.0
OUT = "/tmp/sim2sim_fricsweep_lo"

# ---- background vcan torque reader -------------------------------------------
torque = {}
_stop = threading.Event()


def canreader():
    p = subprocess.Popen(["candump", "-L", "vcan0", "vcan1"], stdout=subprocess.PIPE, text=True)
    try:
        for line in p.stdout:
            if _stop.is_set():
                break
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
            if len(raw) != 6:                       # 8-byte frames are commands
                continue
            t_raw = ((raw[4] & 0x0F) << 8) | raw[5]
            torque[cid >> 5] = t_raw * TS + T_MIN
    finally:
        p.terminate()


os.makedirs(OUT, exist_ok=True)
threading.Thread(target=canreader, daemon=True).start()
rclpy.init()
n = rclpy.create_node("fricsweep")
pub = n.create_publisher(LegCmd, "/leg_impedance_controller/command", 10)
st = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: st.update({"n": list(m.name), "p": list(m.position),
                                           "v": list(m.velocity)}), 50)
_sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sk.sendto(b"RESET", ("127.0.0.1", 9994))
time.sleep(1.0)
_sk.sendto(b"HOMEMODE", ("127.0.0.1", 9994))
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
IDX = [st["n"].index(j) for j in JOINTS]


def send(pos_rad, vel_rad):
    c = LegCmd()
    c.position_des = [float(x) for x in pos_rad]
    c.velocity_des = [float(x) for x in vel_rad]
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in KP]
    c.kd_scale = [float(x) for x in KD]
    pub.publish(c)


def ramp(ji, v_deg, dur_settle=None):
    """Drive joint ji from -SPAN*sign to +SPAN*sign at |v_deg|/s.

    Returns mean CONTROL torque (kp*pos_err + kd*vel_err), not the net wire torque.
    At constant velocity the net applied torque just balances gravity whatever the
    friction -- friction sits INSIDE it (out = ctrl + friction) and the PD pays for
    it by building tracking error. So the friction signal is in ctrl, not in `out`.
    Measured directly: 1.88 deg of lag x kp 150 = 4.9 Nm, i.e. the 4 Nm friction.

        forward :  ctrl = g + |f|
        backward:  ctrl = g - |f|      ->  |f| = (ctrl_fwd - ctrl_bwd)/2
    """
    sgn = 1.0 if v_deg > 0 else -1.0
    start, stop = -SPAN * sgn, SPAN * sgn
    # move to start and settle
    p = [0.0] * 10
    p[ji] = math.radians(start)
    e = time.time() + 1.2
    while time.time() < e:
        send(p, [0.0] * 10)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
    dur = abs(stop - start) / abs(v_deg)
    # margins must scale with duration: a fixed 0.4 s skipped EVERY sample at 40 deg/s
    # (0.6 s sweep), which is why the first run reported "no torque frames".
    if dur_settle is None:
        dur_settle = min(0.4, 0.25 * dur)
    ctrls, taus, errs, vels = [], [], [], []
    t_start = time.time()
    while True:
        el = time.time() - t_start
        if el > dur:
            break
        tgt = start + sgn * abs(v_deg) * el
        p[ji] = math.radians(tgt)
        vv = [0.0] * 10
        vv[ji] = math.radians(v_deg)
        send(p, vv)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        if el < dur_settle or el > dur - dur_settle:
            continue                                    # skip accel / decel ends
        q = math.degrees(st["p"][IDX[ji]])
        qd = math.degrees(st["v"][IDX[ji]])
        perr = math.radians(tgt - q)
        verr = math.radians(v_deg - qd)
        ctrls.append(KP[ji] * perr + KD[ji] * verr)
        taus.append(torque.get(NODE[ji], float("nan")))
        errs.append(q - tgt)
        vels.append(qd)
    good = [t for t in taus if t == t]
    return (float(np.mean(ctrls)) if ctrls else float("nan"),
            float(np.mean(errs)) if errs else float("nan"),
            float(np.mean(vels)) if vels else float("nan"),
            float(np.mean(good)) if good else float("nan"))


print("BATTERY 3 — constant-velocity friction sweep")
print("torque MEASURED off vcan MIT feedback; gravity removed by +/- differencing\n")
print("%-22s %6s %9s %9s %10s %9s %8s" %
      ("joint", "speed", "ctrl_fwd", "ctrl_bwd", "FRICTION", "gravity", "vel ok"))
results = {}
for ji, jn in enumerate(JOINTS):
    rows, vs, fr = [], [], []
    for v in SPEEDS:
        cp, ep, vp, tp = ramp(ji, +v)
        cn, en, vn, tn = ramp(ji, -v)
        if cp != cp or cn != cn:
            print("%-22s %5.0f  no samples" % (jn.replace("dof_", ""), v))
            continue
        f_tau = 0.5 * (cp - cn)                 # gravity cancels, friction doubles
        g_tau = 0.5 * (cp + cn)
        vach = 0.5 * (abs(vp) + abs(vn))
        ok = vach > 0.6 * v                     # control: did it actually move at speed?
        rows.append({"speed_deg_s": v, "ctrl_fwd_Nm": round(cp, 4), "ctrl_bwd_Nm": round(cn, 4),
                     "friction_Nm": round(f_tau, 4), "gravity_Nm": round(g_tau, 4),
                     "net_wire_tau_fwd": round(tp, 4) if tp == tp else None,
                     "track_err_fwd_deg": round(ep, 3), "track_err_bwd_deg": round(en, 3),
                     "vel_achieved_deg_s": round(vach, 2), "control_pass": bool(ok)})
        print("%-22s %5.0f° %8.3f %9.3f %10.3f %9.3f %7.0f%%%s"
              % (jn.replace("dof_", "") if v == SPEEDS[0] else "", v, cp, cn, f_tau, g_tau,
                 100 * vach / v, "" if ok else "  <- DROPPED"))
        if ok:
            vs.append(v)
            fr.append(abs(f_tau))
    m = {"points": rows}
    if len(vs) >= 2:
        slope, icpt = np.polyfit(vs, fr, 1)
        nom = [v for k, v in NOMINAL.items() if k in jn]
        m.update({"coulomb_Nm": round(float(icpt), 4),
                  "viscous_Nm_per_deg_s": round(float(slope), 6),
                  "nominal_Fc_Nm": nom[0] if nom else None})
        print("%-24s  -> Coulomb %.3f Nm (configured %.1f), viscous %.4f Nm/(deg/s)\n"
              % ("", icpt, nom[0] if nom else float("nan"), slope))
    results[jn] = m

json.dump({"schema": "sim2sim_frictionsweep_v1", "speeds_deg_s": SPEEDS, "span_deg": SPAN,
           "kp": KP, "kd": KD, "joint_order": JOINTS,
           "method": "tau_friction = (tau_fwd - tau_bwd)/2 at matched |v|; gravity cancels",
           "torque_source": "MEASURED from vcan MIT feedback 0x008 (12-bit, +-50 Nm)",
           "results": results}, open("%s/metrics.json" % OUT, "w"), indent=1)
_stop.set()
print("wrote %s/metrics.json" % OUT)
rclpy.shutdown()
