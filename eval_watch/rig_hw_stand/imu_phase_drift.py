"""Passive probe: how fast does the IMU's message timing slide against a 50 Hz clock on this computer
(the observation feeder ticks every 20.000 ms of local time)?  Fits the sensor's period from the driver's
stamps; the phase of an IMU message inside the feeder's tick then drifts by (period - 20 ms) per message.
usage: imu_phase_drift.py [topic=/imu/data] [secs=120]"""
import sys, time, rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
topic = sys.argv[1] if len(sys.argv) > 1 else "/imu/data"; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
rclpy.init(); n = Node("imu_phase_drift"); st = []
n.create_subscription(Imu, topic, lambda m: st.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), qos_profile_sensor_data)
t0 = time.monotonic()
while time.monotonic() - t0 < secs: rclpy.spin_once(n, timeout_sec=0.05)
t = np.array(st); t -= t[0]; k = np.round(t / 0.02)                      # message index on a 20 ms grid
k = np.cumsum(np.r_[0, np.maximum(1, np.round(np.diff(t) / 0.02))])     # robust to a missed message
P, c = np.polyfit(k, t, 1); res = 1000 * (t - (P * k + c))
drift_ms_per_s = (P - 0.02) * 1000 / P
print("%s: %d messages in %.0f s | sensor period %.4f ms (fit) | against a 20.000 ms tick the IMU phase slides %+.3f ms per second"
      % (topic, len(t), t[-1], 1000 * P, drift_ms_per_s))
print("   -> crosses the whole 20 ms tick in %s | jitter of each message about its own cadence: rms %.2f ms, 1-99%%: %+.1f..%+.1f ms"
      % ("%.0f s" % abs(20.0 / drift_ms_per_s) if abs(drift_ms_per_s) > 1e-4 else "more than an hour", res.std(), np.percentile(res, 1), np.percentile(res, 99)))
half = len(t) // 2
P1 = np.polyfit(k[:half], t[:half], 1)[0]; P2 = np.polyfit(k[half:], t[half:], 1)[0]
print("   period first half %.4f ms, second half %.4f ms" % (1000 * P1, 1000 * P2))
n.destroy_node(); rclpy.shutdown()
