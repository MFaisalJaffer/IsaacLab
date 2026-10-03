#!/usr/bin/env python3
"""Re-examine our free-decay result using training's diagnostic.

Their challenge: a Coulomb-damped pendulum loses 4*tau_c/k amplitude per cycle.
With Fc=4.0 Nm on a hip against gravity's restoring stiffness, a 15 deg swing
should die almost at once -- yet we reported 39 peaks and a 2.01 s half-life.
Both cannot be true.

Confirmed in the code: _tau adds `-fc*tanh(qd/0.02) - bv*qd` AFTER the PD term,
unconditionally. Setting kp=kd=0 zeroes ctrl but leaves friction fully live. So
our "actuator off" arm never removed friction, and the 39 "peaks" are most likely
numerical dither, not oscillation: at 4 Nm on a ~0.19 kgm^2 leg, one 2 ms step
changes qd by ~0.04 rad/s, twice the 0.02 rad/s tanh width, so the term can
overshoot zero velocity every step.

Discriminator (theirs): fraction of steps where qd REVERSES SIGN.
  ~100%  -> dither / limit cycle, not a swing. Envelope fits are meaningless.
  few %  -> genuine pendulum oscillation.
Also report net excursion: a real swing travels far, a dithering joint creeps.
"""
import math
import sys
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
TEST = [0, 2, 3]                    # right hip_pitch, hip_yaw, knee
START, DUR, RATE = 15.0, 6.0, 100.0
LABEL = sys.argv[1] if len(sys.argv) > 1 else "arm"

rclpy.init()
n = rclpy.create_node("decay_diag")
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


def send(t_rad, kp, kd):
    c = LegCmd()
    c.position_des = [float(x) for x in t_rad]
    c.velocity_des = [0.0] * 10
    c.feedforward_torque = [0.0] * 10
    c.kp_scale = [float(x) for x in kp]
    c.kd_scale = [float(x) for x in kd]
    pub.publish(c)


def hold(t, dur, kp, kd):
    e = time.time() + dur
    while time.time() < e:
        send(t, kp, kd)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)


print("ARM: %s   (release from %.0f deg, gains 0 on the test joint)\n" % (LABEL, START))
print("%-24s %10s %12s %12s %10s" % ("joint", "reversals", "net travel", "peak |qd|", "verdict"))
for ji in TEST:
    jn = JOINTS[ji]
    tgt = [0.0] * 10
    tgt[ji] = math.radians(START)
    hold(tgt, 3.0, KP, KD)                       # drive to release point
    kp0, kd0 = list(KP), list(KD)
    kp0[ji] = 0.0
    kd0[ji] = 0.0
    ys, vs = [], []
    end = time.time() + DUR
    while time.time() < end:
        send(tgt, kp0, kd0)
        rclpy.spin_once(n, timeout_sec=1.0 / RATE)
        ys.append(math.degrees(st["p"][IDX[ji]]))
        vs.append(math.degrees(st["v"][IDX[ji]]))
    rev = sum(1 for i in range(1, len(vs))
              if vs[i] * vs[i - 1] < 0 and abs(vs[i]) > 1e-9 and abs(vs[i - 1]) > 1e-9)
    frac = 100.0 * rev / max(len(vs) - 1, 1)
    travel = ys[-1] - ys[0]
    verdict = ("DITHER (limit cycle)" if frac > 40 else
               "swing" if abs(travel) > 5 else "barely moved")
    print("%-24s %8.0f%% %10.2f° %11.1f°/s %s"
          % (jn.replace("dof_", ""), frac, travel, max(abs(v) for v in vs), verdict))
    hold([0.0] * 10, 1.0, KP, KD)
rclpy.shutdown()
