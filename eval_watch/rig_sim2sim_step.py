#!/usr/bin/env python3
"""SIM2SIM step-response battery v2 — one joint at a time, limit-aware, full stack.

Runs the COMPLETE deployed stack (LegCmd -> ros2_control -> vcan -> emulator), so
the MIT actuator model, series compliance (K_s=52), play band and friction are all
in the loop. The Isaac side must likewise run its full stack INCLUDING the TV
actuator: we are comparing what a policy actually experiences, not the bare plant.

Base PINNED, no ground contact, gravity on.

AMPLITUDES ARE HARDCODED, not derived from each sim's own limits. Joint ranges are
strongly asymmetric and two are one-sided (knee), so a uniform +/-20 deg is invalid;
but deriving locally would let a USD/MJCF limit difference silently make the two
sides run different tests. Same numbers both sides, always.

CONTROLS:
  reach  -- did the test joint get to target? GATES the result. A frozen or
            limit-blocked joint yields flat lines that read as perfect agreement.
            (Twice already: emulator HELD freezes joints entirely -- use HOMEMODE,
            never RESET alone; and controller preflight drops to LIMP without a
            command stream.)
  leak   -- how far the HELD joints moved. NOT a failure: with kp 150 the hold is
            not rigid, so leak is real coupled dynamics and a quantity both sims
            should reproduce. Compare it; do not discard it.
"""
import json
import math
import os
import socket
import time

import rclpy
from odrive_mit_example.msg import LegCmd
from sensor_msgs.msg import JointState

JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
KP = [150.0, 150.0, 60.0, 150.0, 60.0, 150.0, 150.0, 60.0, 150.0, 60.0]
KD = [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]

# (positive_deg, negative_deg); 0 = that direction is outside the joint's range.
# From MJCF limits, 0.6 margin, capped at 20. Mirrored L/R as the hardware is.
AMP = {
    "dof_right_hip_pitch_04": (20.0, -20.0),   # [-127,  60]
    "dof_right_hip_roll_04":  (7.0, -20.0),    # [-130,  12]
    "dof_right_hip_yaw_03":   (20.0, -20.0),   # [ -90,  90]
    "dof_right_knee_04":      (0.0, -20.0),    # [-155,   0]  one-sided
    "dof_right_ankle_02":     (20.0, -7.0),    # [ -13,  72]
    "dof_left_hip_pitch_04":  (20.0, -20.0),   # [ -60, 127]
    "dof_left_hip_roll_04":   (20.0, -7.0),    # [ -12, 130]
    "dof_left_hip_yaw_03":    (20.0, -20.0),   # [ -90,  90]
    "dof_left_knee_04":       (20.0, 0.0),     # [   0, 155]  one-sided
    "dof_left_ankle_02":      (7.0, -20.0),    # [ -72,  13]
}
HOLD_S, SETTLE_S, RATE = 2.0, 1.5, 100.0
OUT = "/tmp/sim2sim"

os.makedirs(OUT, exist_ok=True)
rclpy.init()
n = rclpy.create_node("sim2sim_step")
pub = n.create_publisher(LegCmd, "/leg_impedance_controller/command", 10)
state = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: state.update({"n": list(m.name), "p": list(m.position),
                                              "v": list(m.velocity)}), 50)

# HOMEMODE = physics live, base held (virtual_motor_node.py:252). NOT RESET alone:
# RESET leaves the emulator HELD where "the joints are frozen and nothing can move
# them" (:255). Never GO -- that frees the base and the robot falls.
_sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sk.sendto(b"RESET", ("127.0.0.1", 9994))
time.sleep(1.0)
_sk.sendto(b"HOMEMODE", ("127.0.0.1", 9994))

t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
if "n" not in state:
    raise SystemExit("no /joint_states")
IDX = [state["n"].index(j) for j in JOINTS]


def send(t_rad):
    c = LegCmd()
    c.position_des = [float(x) for x in t_rad]
    c.velocity_des = [0.0] * 10
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in KP]
    c.kd_scale = [float(x) for x in KD]
    pub.publish(c)


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
    raise SystemExit("!! never ENGAGED (%s) - aborting rather than logging flat lines" % preflight())
print("  ENGAGED after %.1f s\n" % (time.time() - t0))


def run_phase(ji, tgt_deg, dur, rec, f, tstart):
    tgt = [0.0] * 10
    tgt[ji] = math.radians(tgt_deg)
    end = time.time() + dur
    while time.time() < end:
        send(tgt)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        p = [math.degrees(state["p"][i]) for i in IDX]
        v = [math.degrees(state["v"][i]) for i in IDX]
        el = time.time() - tstart
        rec.append((el, tgt_deg, p, v))
        f.write("%.4f\t%.3f\t" % (el, tgt_deg) + "\t".join("%.5f" % x for x in p)
                + "\t" + "\t".join("%.5f" % x for x in v) + "\n")


def metrics(t, y, y0, tgt):
    step = tgt - y0
    if abs(step) < 1e-6:
        return {}

    def frac(fr):
        want = y0 + fr * step
        for i, v in enumerate(y):
            if (step > 0 and v >= want) or (step < 0 and v <= want):
                return t[i]
        return None
    t10, t90 = frac(0.10), frac(0.90)
    peak = max(y, key=lambda v: (v - y0) * (1 if step > 0 else -1))
    tol = 0.02 * abs(step)
    settle = None
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - tgt) > tol:
            settle = t[i + 1] if i + 1 < len(t) else None
            break
        settle = t[i]
    tail = y[max(0, len(y) - 20):]
    return {"rise_10_90_ms": None if (t10 is None or t90 is None) else round(1000 * (t90 - t10), 1),
            "overshoot_pct": round(100.0 * ((peak - y0) / step - 1.0), 2),
            "settle_ms": None if settle is None else round(1000 * settle, 1),
            "steady_state_err_deg": round(sum(tail) / len(tail) - tgt, 4)}


print("SIM2SIM v2 | full stack | base pinned | no contact | %.0f Hz\n" % RATE)
print("%-26s %16s %10s %s" % ("joint", "reached", "leak", "reach control"))
results = {}
for ji, jn in enumerate(JOINTS):
    amps = [a for a in AMP[jn] if abs(a) > 1e-6]
    f = open("%s/%s.tsv" % (OUT, jn), "w", buffering=1)
    f.write("# t_s\ttarget_deg\t" + "\t".join("pos_" + j for j in JOINTS)
            + "\t" + "\t".join("vel_" + j for j in JOINTS) + "\n")
    rec, tstart = [], time.time()
    run_phase(ji, 0.0, 1.0, rec, f, tstart)
    for a in amps:
        run_phase(ji, a, HOLD_S, rec, f, tstart)
        run_phase(ji, 0.0, SETTLE_S, rec, f, tstart)
    f.close()
    send([0.0] * 10)

    m = {"amplitudes_deg": AMP[jn]}
    reach_ok = True
    reached = []
    for a in amps:
        seg = [r for r in rec if abs(r[1] - a) < 1e-6]
        if not seg:
            continue
        got = seg[-1][2][ji]
        reached.append(got)
        err = abs(got - a)
        if err > max(2.0, 0.15 * abs(a)):
            reach_ok = False
        lbl = "pos" if a > 0 else "neg"
        tt = [r[0] - seg[0][0] for r in seg]
        yy = [r[2][ji] for r in seg]
        m[lbl] = metrics(tt, yy, yy[0], a)
        m[lbl]["target_deg"] = a
        m[lbl]["reached_deg"] = round(got, 3)
        m[lbl]["peak_vel_deg_s"] = round(max(abs(r[3][ji]) for r in seg), 2)
    leak, leak_j = 0.0, ""
    for r in rec:
        for k in range(10):
            if k != ji and abs(r[2][k]) > leak:
                leak, leak_j = abs(r[2][k]), JOINTS[k]
    m["max_leak_deg"] = round(leak, 3)
    m["worst_leak_joint"] = leak_j
    m["reach_control_pass"] = reach_ok
    results[jn] = m
    print("%-26s %16s %9.2f  %s"
          % (jn, "/".join("%+.1f" % r for r in reached), leak,
             "OK" if reach_ok else "!! FAIL - exclude"))

json.dump({"schema": "sim2sim_step_v2", "rate_hz": RATE, "kp": KP, "kd": KD,
           "joint_order": JOINTS, "amplitudes_deg": AMP,
           "hold_s": HOLD_S, "settle_s": SETTLE_S,
           "base": "pinned", "contact": "none", "gravity_z": -9.81,
           "stack": "FULL: LegCmd -> ros2_control -> vcan -> virtual_motor_node "
                    "(MIT actuator, series K_s=52 Nm/rad, b=0.07, play 0.3 deg, friction 1.0)",
           "results": results}, open("%s/metrics.json" % OUT, "w"), indent=1)
print("\nwrote %s/metrics.json + 10 tsv" % OUT)
rclpy.shutdown()
