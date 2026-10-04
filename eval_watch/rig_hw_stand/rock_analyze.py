"""Characterize a standing 'rocking' episode: axis, frequency, growth, which joints drive it.
usage: rock_analyze.py <ep.npz> [t_start] [t_end]   (window in seconds; default 0.6 .. end-0.1)"""
import numpy as np, sys
np.set_printoptions(precision=2, suppress=True, linewidth=200)
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
o43 = r[:, 29:72]; a = r[:, 72:82]; pg = o43[:, 0:3]; jp = o43[:, 6:16]; gy = o43[:, 26:29]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1] - 0.1
m = (t >= t1) & (t <= t2); fs = 1.0 / np.median(np.diff(t))
names = ["LhP", "RhP", "LhR", "RhR", "Lyaw", "Ryaw", "Lkn", "Rkn", "Lank", "Rank"]
tilt = np.degrees(np.arccos(np.clip(pg[:, 0], -1, 1)))
py = np.degrees(np.arcsin(np.clip(pg[:, 1], -1, 1))); pz = np.degrees(np.arcsin(np.clip(pg[:, 2], -1, 1)))
print("%s: %d ticks %.2f s; window %.2f-%.2f s (%d ticks)" % (sys.argv[1].split("/")[-1], len(t), t[-1], t1, t2, int(m.sum())))
print("torso angle about policy-y (pitch-like): mean %+.2f deg, p2p %.2f | about policy-z (roll-like): mean %+.2f, p2p %.2f | total tilt max %.1f"
      % (py[m].mean(), np.ptp(py[m]), pz[m].mean(), np.ptp(pz[m]), tilt[m].max()))
def dom(x):
    x = x - x.mean(); n = len(x)
    if n < 16: return float("nan"), 0.0
    F = np.abs(np.fft.rfft(x * np.hanning(n))); f = np.fft.rfftfreq(n, 1 / fs); k = np.argmax(F[1:]) + 1
    return f[k], F[k] / (F[1:].sum() + 1e-12)
fy, cy = dom(py[m]); fz, cz = dom(pz[m])
print("dominant frequency: pitch-like %.2f Hz (%.0f%% of spectrum), roll-like %.2f Hz (%.0f%%)" % (fy, 100 * cy, fz, 100 * cz))
# envelope growth of the pitch-like angle: successive extrema
x = py[m] - py[m].mean(); tt = t[m]
ext = [i for i in range(1, len(x) - 1) if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0 and abs(x[i]) > 0.15]
print("pitch-like extrema (t, deg):", [(round(float(tt[i]), 2), round(float(x[i]), 2)) for i in ext][:14])
print("ANKLES: encoder p2p L %.1f / R %.1f deg | body pitch p2p %.2f deg -> ankle/body ratio %.1f | action p2p L %.2f / R %.2f (R/L %.1f)" % (
    np.degrees(np.ptp(jp[m, 8])), np.degrees(np.ptp(jp[m, 9])), np.ptp(py[m]), np.degrees(max(np.ptp(jp[m, 8]), np.ptp(jp[m, 9]))) / max(np.ptp(py[m]), 1e-6),
    np.ptp(a[m, 8]), np.ptp(a[m, 9]), np.ptp(a[m, 9]) / max(np.ptp(a[m, 8]), 1e-6)))
print("action p2p per joint in window:", dict(zip(names, np.ptp(a[m], axis=0).round(2))))
print("joint  p2p (deg)  per joint in window:", dict(zip(names, np.degrees(np.ptp(jp[m], axis=0)).round(2))))
# who leads: correlation of each joint action with the pitch-like angle (zero lag) and best lag
for j in (8, 9, 0, 1, 2, 3):
    aj = a[m, j] - a[m, j].mean(); best = (0, 0.0)
    for lag in range(-12, 13):
        if lag >= 0: c = np.corrcoef(aj[lag:], x[:len(x) - lag])[0, 1] if len(x) - lag > 8 else 0
        else: c = np.corrcoef(aj[:lag], x[-lag:])[0, 1] if len(x) + lag > 8 else 0
        if abs(c) > abs(best[1]): best = (lag, c)
    print("  action %-4s vs pitch-like angle: best corr %+.2f at lag %+d ticks (%+.0f ms; + = action lags angle)" % (names[j], best[1], best[0], best[0] * 1000 / fs))
if "wire" in z.files and len(z["wire"]):
    w = z["wire"]; t0 = r[0, 0]; node = w[:, 1].astype(int); kind = w[:, 2].astype(int)
    nm = {17: "Lank", 7: "Rank", 14: "LhR", 4: "RhR", 13: "LhP", 3: "RhP"}
    row = []
    for n in (17, 7, 13, 3, 14, 4):
        f = w[(node == n) & (kind == 1)]; mm = (f[:, 0] - t0 >= t1) & (f[:, 0] - t0 <= t2)
        if mm.sum() > 4: row.append("%s tau mean %+.1f p2p %.1f" % (nm[n], f[mm, 5].mean(), np.ptp(f[mm, 5])))
    print("drive torque in window: " + " | ".join(row))
