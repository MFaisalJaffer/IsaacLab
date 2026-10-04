"""Passive probe (subscribe only): what rate does the BNO08x really deliver new samples at?
Listens to a topic for SECS and counts messages vs DISTINCT accel / gyro / quat samples.
The accelerometer is noisy enough to change on every genuine report even when the robot is still.
usage: imu_rate_probe.py [topic=/imu/data] [secs=10]"""
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
topic = sys.argv[1] if len(sys.argv) > 1 else "/imu/data"; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
rclpy.init(); n = Node("imu_rate_probe"); rows = []
def cb(m):
    a, g, q = m.linear_acceleration, m.angular_velocity, m.orientation
    rows.append((time.monotonic(), (a.x, a.y, a.z), (g.x, g.y, g.z), (q.x, q.y, q.z, q.w)))
n.create_subscription(Imu, topic, cb, qos_profile_sensor_data)
t0 = time.monotonic()
while time.monotonic() - t0 < secs: rclpy.spin_once(n, timeout_sec=0.05)
if len(rows) < 3: print("no messages on", topic); sys.exit(1)
dur = rows[-1][0] - rows[0][0]
print("%s: %d messages in %.1f s = %.1f msg/s" % (topic, len(rows), dur, (len(rows) - 1) / dur))
for name, k in (("accel", 1), ("gyro", 2), ("quat", 3)):
    new = [i for i in range(1, len(rows)) if rows[i][k] != rows[i - 1][k]]
    gaps = [rows[b][0] - rows[a][0] for a, b in zip(new, new[1:])]
    gaps.sort()
    med = gaps[len(gaps) // 2] if gaps else float("nan"); p90 = gaps[int(0.9 * len(gaps))] if gaps else float("nan")
    print("  %-5s %4d distinct samples = %5.1f new samples/s | gap between new samples: median %.0f ms, 90th pct %.0f ms"
          % (name, len(new) + 1, len(new) / dur, 1000 * med, 1000 * p90))
n.destroy_node(); rclpy.shutdown()
