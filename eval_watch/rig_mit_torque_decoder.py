#!/usr/bin/env python3
"""Read REAL measured ankle torque straight off the CAN bus. Passive: listens only.

The driver decodes MIT feedback torque into torque_estimate_ but exports
torque_target_ on the effort interface, so /joint_states shows NaN. The frame
itself is on the wire regardless, so decode it here rather than patch and rebuild
a driver under a live robot.

MIT feedback 0x008, 6 bytes, big-endian (from odrive_hardware_interface.cpp:469-483):
  byte0 motor_id | b1-2 pos(16) | b3,b4[7:4] vel(12) | b4[3:0],b5 torque(12)
Command frames share the arbitration ID but are 8 bytes, so filter on DLC == 6.

Reports measured torque against kp*enc from the control law. Agreement means the
drive delivers commanded stiffness; a shortfall means it does not, and every K_s
derived from the control law is scaled wrong.
"""
import math
import subprocess
import sys
import time

NODE = int(sys.argv[1]) if len(sys.argv) > 1 else 17     # 17 = left ankle (can0)
BUS = sys.argv[2] if len(sys.argv) > 2 else "can0"
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
KP = 60.0

P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -65.0, 65.0
T_MIN, T_MAX = -50.0, 50.0
PS = (P_MAX - P_MIN) / 65535.0
VS = (V_MAX - V_MIN) / 4095.0
TS = (T_MAX - T_MIN) / 4095.0
WANT = (NODE << 5) | 0x008

print("listening %s for node %d (arb id 0x%03X), DLC 6 only, %.0f s" % (BUS, NODE, WANT, SECS))
print("PASSIVE - candump transmits nothing\n")
print("%7s %11s %12s %12s %10s" % ("t", "pos(deg)", "vel(deg/s)", "TORQUE(Nm)", "kp*enc"))

proc = subprocess.Popen(["candump", "-L", BUS], stdout=subprocess.PIPE, text=True)
t0 = time.time()
nxt = 0.0
n = 0
peak_t = 0.0
peak_p = 0.0
ref = None
try:
    for line in proc.stdout:
        if time.time() - t0 > SECS:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            cid_s, data_s = parts[2].split("#")
        except ValueError:
            continue
        if int(cid_s, 16) != WANT:
            continue
        raw = bytes.fromhex(data_s)
        if len(raw) != 6:                     # 8-byte frames are commands, not feedback
            continue
        p_raw = (raw[1] << 8) | raw[2]
        v_raw = (raw[3] << 4) | ((raw[4] >> 4) & 0x0F)
        t_raw = ((raw[4] & 0x0F) << 8) | raw[5]
        pos = p_raw * PS + P_MIN
        vel = v_raw * VS + V_MIN
        trq = t_raw * TS + T_MIN
        n += 1
        if ref is None:
            ref = pos
        enc = math.degrees(pos - ref)
        if abs(trq) > abs(peak_t):
            peak_t, peak_p = trq, enc
        el = time.time() - t0
        if el >= nxt:
            nxt += 1.0
            print("%6.1fs %+10.2f %+11.1f %+11.3f %+9.3f"
                  % (el, enc, math.degrees(vel), trq, -KP * math.radians(enc)))
finally:
    proc.terminate()

print("\n%d feedback frames decoded" % n)
if n:
    print("peak measured torque %+.3f Nm at encoder %+.2f deg" % (peak_t, peak_p))
    if abs(peak_p) > 0.5:
        implied = abs(peak_t / math.radians(peak_p))
        print("implied stiffness at peak: %.1f Nm/rad  (commanded kp = %.0f)" % (implied, KP))
        print("  -> %s" % ("drive IS delivering commanded kp" if 0.75 * KP < implied < 1.35 * KP
                           else "MISMATCH - drive is not delivering commanded kp"))
