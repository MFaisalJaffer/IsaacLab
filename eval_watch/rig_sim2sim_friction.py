#!/usr/bin/env python3
"""Friction / stiction battery — the two probes the step test is blind to.

The +/-20 deg step battery cannot see friction: at kp 150 a 20 deg step commands
52 Nm of restoring torque, so even 4 Nm of Coulomb friction is a 7% effect, and our
smoothed friction (-fc*tanh(qd/0.02)) vanishes at qd=0 so it leaves no steady-state
error at all. Our hips carry Fc=4.0 Nm and the step test reported frac 0.998.

TEST A — AMPLITUDE LADDER. Same step, amplitudes 0.5 .. 20 deg. Steady-state error
vs amplitude separates the mechanisms by shape, not magnitude:
    constant error      -> Coulomb friction; the intercept IS tau_c / kp
    proportional error  -> compliance / series spring
A straight-line fit gives both at once: intercept = friction, slope = compliance.

TEST B — FREE DECAY. kp=kd=0 on one joint, displace, release, let gravity swing it.
Envelope shape identifies the mechanism, and the actuator is switched OFF entirely
so this isolates the PLANT (no TV actuator, no MIT emulation in the loop):
    linear decay      -> Coulomb friction
    exponential decay -> viscous damping
Log-envelope curvature separates them; we report both fits and which wins.

Base LIFTED clear of the floor (PIN_Z) -- "base pinned" is not "no contact": at the
standing pin height the feet clear the floor plane by 3.7 mm and ~1.6 deg of ankle
rotation plants them, which silently contaminated an entire earlier dataset.
"""
import json
import math
import os
import socket
import time

import numpy as np
import rclpy
from odrive_mit_example.msg import LegCmd
from sensor_msgs.msg import JointState

JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
KP = [150.0, 150.0, 60.0, 150.0, 60.0, 150.0, 150.0, 60.0, 150.0, 60.0]
KD = [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]
LADDER = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SIGN = {  # direction each joint may travel (knees are one-sided)
    "dof_right_hip_pitch_04": +1, "dof_right_hip_roll_04": -1, "dof_right_hip_yaw_03": +1,
    "dof_right_knee_04": -1, "dof_right_ankle_02": +1,
    "dof_left_hip_pitch_04": +1, "dof_left_hip_roll_04": +1, "dof_left_hip_yaw_03": +1,
    "dof_left_knee_04": +1, "dof_left_ankle_02": -1,
}
DECAY_START = 15.0     # deg, release point for the free-decay test
RATE = 100.0
OUT = "/tmp/sim2sim_fric"

os.makedirs(OUT, exist_ok=True)
rclpy.init()
n = rclpy.create_node("sim2sim_fric")
pub = n.create_publisher(LegCmd, "/leg_impedance_controller/command", 10)
st = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: st.update({"n": list(m.name), "p": list(m.position),
                                           "v": list(m.velocity)}), 50)
_sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sk.sendto(b"RESET", ("127.0.0.1", 9994))
time.sleep(1.0)
_sk.sendto(b"HOMEMODE", ("127.0.0.1", 9994))     # physics live, base held. NOT RESET alone.
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
if "n" not in st:
    raise SystemExit("no /joint_states")
IDX = [st["n"].index(j) for j in JOINTS]


def send(tgt_rad, kp=None, kd=None):
    c = LegCmd()
    c.position_des = [float(x) for x in tgt_rad]
    c.velocity_des = [0.0] * 10
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in (kp if kp is not None else KP)]
    c.kd_scale = [float(x) for x in (kd if kd is not None else KD)]
    pub.publish(c)


def pos():
    return [math.degrees(st["p"][i]) for i in IDX]


def vel():
    return [math.degrees(st["v"][i]) for i in IDX]


def hold(tgt, dur, kp=None, kd=None, rec=None, ji=None, tag=""):
    end = time.time() + dur
    while time.time() < end:
        send(tgt, kp, kd)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        if rec is not None:
            rec.append((time.time(), tag, pos()[ji], vel()[ji]))


def preflight():
    import urllib.request
    try:
        d = json.load(urllib.request.urlopen("http://127.0.0.1:8080/state.json", timeout=3))
        return (d.get("preflight", {}) or {}).get("state")
    except Exception:
        return None


print("waiting for controller to ENGAGE...")
eng, t0 = False, time.time()
while time.time() - t0 < 40.0:
    send([0.0] * 10)
    rclpy.spin_once(n, timeout_sec=0.01)
    if time.time() - t0 > 3.0 and preflight() == "ENGAGED":
        eng = True
        break
if not eng:
    raise SystemExit("!! never ENGAGED (%s) - aborting" % preflight())
print("  ENGAGED\n")

results = {}

# ---------------- TEST A: amplitude ladder ----------------
print("=== A. AMPLITUDE LADDER  (steady-state error vs step size) ===")
print("%-24s %7s %10s %11s" % ("joint", "amp", "reached", "err(deg)"))
for ji, jn in enumerate(JOINTS):
    errs, amps = [], []
    for a_mag in LADDER:
        a = a_mag * SIGN[jn]
        hold([0.0] * 10, 1.0)
        rec = []
        tgt = [0.0] * 10
        tgt[ji] = math.radians(a)
        hold(tgt, 2.0, rec=rec, ji=ji, tag="step")
        tail = [r[2] for r in rec[-30:]]
        got = sum(tail) / len(tail)
        err = abs(a) - abs(got)          # positive = fell short
        errs.append(err)
        amps.append(abs(a))
        print("%-24s %6.1f° %9.3f° %10.4f°" % (jn.replace("dof_", "") if a_mag == LADDER[0] else "",
                                               abs(a), abs(got), err))
    A = np.array(amps)
    E = np.array(errs)
    slope, icpt = np.polyfit(A, E, 1)
    # intercept is in DEGREES; kp is Nm/rad -> convert or the answer is 57x too big
    tau_c = math.radians(icpt) * KP[ji]
    # our MIT feedback is 16-bit over +-12.5 rad => 0.0219 deg quantum. Any intercept
    # within a couple of LSB is the quantiser, not friction: report it as a BOUND.
    lsb = math.degrees(25.0 / 65535.0)
    quantised = abs(icpt) < 2.5 * lsb
    results[jn] = {"ladder_amp_deg": amps, "ladder_err_deg": [round(e, 5) for e in errs],
                   "fit_slope": round(float(slope), 6),
                   "fit_intercept_deg": round(float(icpt), 5),
                   "implied_coulomb_Nm": round(float(tau_c), 4),
                   "intercept_in_lsb": round(abs(icpt) / lsb, 2),
                   "quantisation_limited": bool(quantised),
                   "implied_series_frac": round(float(slope), 5)}
    print("%-24s  fit: err = %.5f*amp %+.5f°  -> Coulomb %s%.4f Nm, compliance %.3f%% of amp\n"
          % ("", slope, icpt, "<" if quantised else "", abs(tau_c), 100 * slope)
          + ("%-24s  (intercept %.2f LSB - QUANTISATION-LIMITED, treat as upper bound)\n"
             % ("", abs(icpt) / lsb) if quantised else ""))

# ---------------- TEST B: free decay (actuator OFF) ----------------
print("=== B. FREE DECAY  (kp=kd=0; plant only, no actuator) ===")
print("%-24s %8s %9s %10s %10s %s" % ("joint", "peaks", "lin_r2", "exp_r2", "halflife", "verdict"))
ZERO = [0.0] * 10
for ji, jn in enumerate(JOINTS):
    start = DECAY_START * SIGN[jn]
    tgt = [0.0] * 10
    tgt[ji] = math.radians(start)
    hold(tgt, 2.5)                                    # drive to release point
    kp0, kd0 = list(KP), list(KD)
    kp0[ji] = 0.0
    kd0[ji] = 0.0
    rec = []
    t_rel = time.time()
    end = t_rel + 6.0
    while time.time() < end:
        send(tgt, kp0, kd0)                           # test joint limp, others held
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        rec.append((time.time() - t_rel, pos()[ji], vel()[ji]))
    y = np.array([r[1] for r in rec])
    t = np.array([r[0] for r in rec])
    # envelope from successive local extrema
    pk = []
    for i in range(2, len(y) - 2):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
            pk.append((t[i], abs(y[i] - y[-1])))
    pk = [p for p in pk if p[1] > 0.05]
    m = {"decay_start_deg": start, "n_peaks": len(pk)}
    if len(pk) >= 3:
        tp = np.array([p[0] for p in pk])
        ap = np.array([p[1] for p in pk])
        lin = np.polyfit(tp, ap, 1)
        r2l = 1 - np.sum((ap - np.polyval(lin, tp)) ** 2) / max(np.sum((ap - ap.mean()) ** 2), 1e-12)
        ex = np.polyfit(tp, np.log(np.maximum(ap, 1e-6)), 1)
        r2e = 1 - np.sum((np.log(np.maximum(ap, 1e-6)) - np.polyval(ex, tp)) ** 2) / \
            max(np.sum((np.log(np.maximum(ap, 1e-6)) - np.log(np.maximum(ap, 1e-6)).mean()) ** 2), 1e-12)
        hl = math.log(2) / abs(ex[0]) if abs(ex[0]) > 1e-9 else float("inf")
        verdict = "COULOMB (linear)" if r2l > r2e + 0.05 else \
                  ("VISCOUS (exponential)" if r2e > r2l + 0.05 else "mixed/ambiguous")
        m.update({"linear_r2": round(float(r2l), 4), "exp_r2": round(float(r2e), 4),
                  "halflife_s": round(float(hl), 3), "envelope_verdict": verdict})
        print("%-24s %8d %9.3f %10.3f %9.2fs  %s"
              % (jn.replace("dof_", ""), len(pk), r2l, r2e, hl, verdict))
    else:
        m["envelope_verdict"] = "no oscillation (overdamped or joint did not swing)"
        print("%-24s %8d %9s %10s %10s  %s"
              % (jn.replace("dof_", ""), len(pk), "-", "-", "-", m["envelope_verdict"]))
    m["settled_deg"] = round(float(y[-1]), 4)
    m["residual_from_zero_deg"] = round(float(abs(y[-1])), 4)
    results[jn].update(m)
    hold([0.0] * 10, 1.0)

json.dump({"schema": "sim2sim_friction_v1", "ladder_deg": LADDER,
           "decay_start_deg": DECAY_START, "kp": KP, "kd": KD, "joint_order": JOINTS,
           "rate_hz": RATE, "base": "pinned+lifted (PIN_Z)", "contact": "none",
           "note": "ladder intercept*kp = Coulomb torque; ladder slope = compliance fraction; "
                   "decay envelope linear=Coulomb, exponential=viscous",
           "results": results}, open("%s/metrics.json" % OUT, "w"), indent=1)
print("\nwrote %s/metrics.json" % OUT)
rclpy.shutdown()
