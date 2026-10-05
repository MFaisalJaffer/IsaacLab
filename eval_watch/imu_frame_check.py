"""Which way do the axes of the policy's IMU inputs point on the real robot?  (ASK_RIG_IMU_FRAME.md)

Reads one episode recorded by the rig's runner (npz with `data`: t, dt, obs27[jpos10, jvel10, quat4, gyro3],
obs43[pgrav3, cmd3, jpos10, jvel10, gyro3, last_act10, phase4], ...). The operator makes three moves by hand, in this
order, each from upright and back, holding the end for about 2 s:

    A  tilt the torso FORWARD (chest toward the toes), 8 deg or more
    B  tilt the torso to the robot's LEFT, 8 deg or more
    C  turn the torso to the robot's LEFT about the vertical (counter-clockwise seen from above), 20 deg or more

The script finds the largest excursion of each of three signals — gravity y, gravity z, and the angle turned about
x (the integral of gyro x) — and prints when it happened, its sign, and the sign of the gyro while moving there.
Next to each it prints what the simulator's imu frame gives for the same move (imu x = down, y = backward,
z = left; eval_watch/mirror_check.py). It also checks how the policy's gravity and gyro are built from the IMU.

usage: imu_frame_check.py <ep.npz> [t1] [t2]
"""
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
r = z["data"]
t = r[:, 0] - r[0, 0]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else float(t[-1])
m = (t >= t1) & (t <= t2)
t = t[m]; q = r[m, 22:26]; gyro27 = r[m, 26:29]; pg = r[m, 29:32]; gyro = r[m, 55:58]
dt = float(np.median(np.diff(t)))
print(f"{sys.argv[1].split('/')[-1]}  {t[0]:.1f}-{t[-1]:.1f} s, {len(t)} ticks")

# ---- 1. how the policy's inputs are built from the IMU
w, x, y, zz = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
g_sensor = np.stack([-2 * (x * zz - y * w), -2 * (y * zz + x * w), -(1 - 2 * (x * x + y * y))], 1)   # R(q)^T (0, 0, -1)
print(f"1. policy gravity = gravity in the IMU's own frame from its quaternion (w, x, y, z): max difference {np.abs(g_sensor - pg).max():.1e}"
      f"  | policy gyro = the IMU gyro unchanged: max difference {np.abs(gyro27 - gyro).max():.1e}")
print(f"   upright (first second): gravity {np.round(pg[t < t[0] + 1.0].mean(0), 3)}   (simulator: [1, 0, 0])")

# ---- 2. are gravity and gyro one right-handed frame?  d g/dt = -(w x g)
k = np.ones(9) / 9
sm = lambda v: np.convolve(v, k, mode="same")
G = np.stack([sm(pg[:, i]) for i in range(3)], 1); W = np.stack([sm(gyro[:, i]) for i in range(3)], 1)
dG = np.gradient(G, t, axis=0); pred = -np.cross(W, G)
out = []
for i, n in ((1, "y"), (2, "z")):
    a, b = pred[10:-10, i], dG[10:-10, i]
    out.append(f"gravity {n}: slope {np.polyfit(a, b, 1)[0]:+.2f}, correlation {np.corrcoef(a, b)[0, 1]:+.2f}" if b.std() > 2e-3 else f"gravity {n}: too little motion")
print("2. gravity against gyro (slope +1 = one consistent right-handed frame): " + " | ".join(out))

# ---- 3. the three moves
lean_y = np.degrees(np.arcsin(np.clip(G[:, 1], -1, 1))); lean_z = np.degrees(np.arcsin(np.clip(G[:, 2], -1, 1)))
turn_x = np.degrees(np.cumsum(W[:, 0]) * dt)
n0 = max(int(1.0 / dt), 1)
rows = []
for name, sig, need, rate, sim_sig, sim_rate, move in (
        ("gravity y", lean_y, 4.0, W[:, 2], "-", "+", "A  forward tilt"),
        ("gravity z", lean_z, 4.0, W[:, 1], "+", "+", "B  tilt to the left"),
        ("angle turned about x", turn_x, 10.0, W[:, 0], "-", "-", "C  turn to the left")):
    d = sig - sig[:n0].mean()
    i = int(np.abs(d).argmax())
    if abs(d[i]) < need:
        rows.append((None, name, f"no move found (largest change {d[i]:+.1f} deg, {need:.0f} needed)", move, sim_sig, sim_rate)); continue
    j = i
    while j > 0 and abs(d[j]) > 0.15 * abs(d[i]):       # back to where the move started
        j -= 1
    rt = float(rate[j:i + 1].mean()) if i > j else 0.0
    which = "gyro z" if name == "gravity y" else ("gyro y" if name == "gravity z" else "gyro x")
    rows.append((t[i], name, f"{'+' if d[i] > 0 else '-'} ({d[i]:+.1f} deg at {t[i]:.1f} s), {which} on the way there {'+' if rt > 0 else '-'} ({rt:+.3f} rad/s)", move, sim_sig, sim_rate))
print("3. largest excursion of each signal   | measured                                                        | simulator, for the move")
for when, name, txt, move, s1, s2 in rows:
    print(f"   {name:22s} | {txt:64s} | {s1} with its gyro {s2}   <- move {move}")
found = sorted((w_, n_) for w_, n_, *_ in rows if w_ is not None)
if len(found) >= 2:
    print("   order in time: " + " -> ".join(n_ for _, n_ in found) + "   (moves done in the order A, B, C must give: gravity y -> gravity z -> angle turned about x)")
