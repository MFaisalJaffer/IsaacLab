"""Passive probe (subscribe only) of the IMU topic's timing: message rate, spacing of the driver's own header
stamps (how regular the hand-over is), gaps, and how many messages carry a new gyro / orientation value.
usage: imu_timing_probe.py [topic=/imu/data] [secs=20]"""
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
topic = sys.argv[1] if len(sys.argv) > 1 else "/imu/data"; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
rclpy.init(); n = Node("imu_timing_probe"); rows = []
def cb(m):
    g, q = m.angular_velocity, m.orientation
    rows.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(), (g.x, g.y, g.z), (q.x, q.y, q.z, q.w)))
n.create_subscription(Imu, topic, cb, qos_profile_sensor_data)
t0 = time.monotonic()
while time.monotonic() - t0 < secs: rclpy.spin_once(n, timeout_sec=0.05)
st = [r[0] for r in rows]; d = sorted(1000 * (b - a) for a, b in zip(st, st[1:])); k = len(d)
tr = sorted(1000 * (r[1] - r[0]) for r in rows)
big = [(round(st[i + 1] - st[0], 1), round(1000 * (st[i + 1] - st[i]))) for i in range(len(st) - 1) if st[i + 1] - st[i] > 0.03]
print("%s: %d messages in %.1f s = %.1f msg/s | spacing of driver stamps: median %.2f ms, 1-99%%: %.1f..%.1f, max %.1f | gaps over 30 ms: %d %s"
      % (topic, len(rows), st[-1] - st[0], (len(rows) - 1) / (st[-1] - st[0]), d[k // 2], d[k // 100], d[-k // 100 - 1], d[-1], len(big), big[:8]))
print("   stamp -> received here: median %.1f ms | new gyro value in %.0f%% of messages, new orientation in %.0f%% (a still robot repeats honestly)"
      % (tr[len(tr) // 2], 100.0 * sum(1 for a, b in zip(rows, rows[1:]) if a[2] != b[2]) / max(1, len(rows) - 1), 100.0 * sum(1 for a, b in zip(rows, rows[1:]) if a[3] != b[3]) / max(1, len(rows) - 1)))
n.destroy_node(); rclpy.shutdown()
