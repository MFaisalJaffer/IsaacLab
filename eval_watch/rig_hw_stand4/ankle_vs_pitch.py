"""Per-ankle torque against torso lean, for episodes where the drives HOLD a fixed pose (shadow runs) or any episode.
Torque = kp x (target - encoder), expressed as 'push that resists a forward lean' for both ankles (axes are mirrored).
Prints, per ankle: straight-line stiffness (Nm per rad of torso pitch), torque share, and the mean torque in pitch bins.
usage: ankle_vs_pitch.py <ep.npz> [t1] [t2]"""
import numpy as np, sys, json
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0; t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]; m = (t >= t1) & (t <= t2)
fr = r[m, 29:72]; pitch = np.degrees(np.arcsin(np.clip(fr[:, 1], -1, 1))); jp = fr[:, 6:16]; tg = 0.5 * r[m, 82:92]
live = "MIRROR" not in str(z["meta"]) and np.abs(np.diff(jp[:, 6:8], axis=0)).sum() >= 0
# shadow runs: the drives hold zero, the logged actions never reach them
hold = len(sys.argv) > 4 and sys.argv[4] == "hold"
TL = 60.0 * ((0 if hold else tg[:, 8]) - jp[:, 8]) * -1.0      # + = left ankle resists a forward lean
TR = 60.0 * ((0 if hold else tg[:, 9]) - jp[:, 9])             # + = right ankle resists a forward lean
print("%s  %.1f-%.1f s (%s) | pitch range %+.2f..%+.2f deg" % (sys.argv[1].split("/")[-1], t1, t2, "drives hold zero" if hold else "policy targets", pitch.min(), pitch.max()))
for n, T in (("LEFT ", TL), ("RIGHT", TR)):
    A = np.c_[np.radians(pitch), np.ones(len(pitch))]; k, b = np.linalg.lstsq(A, T, rcond=None)[0]; res = T - A @ np.r_[k, b]
    print("  %s ankle: torque %+.2f..%+.2f Nm (swing %.2f) | stiffness vs torso pitch %5.1f Nm/rad (R2 %.2f) | encoder swing %.2f deg" % (
        n, T.min(), T.max(), np.ptp(T), k, 1 - res.var() / max(T.var(), 1e-12), np.degrees(np.ptp(jp[:, 8 if n == "LEFT " else 9]))))
print("  share of the torque swing carried by the left ankle: %.0f%%" % (100 * np.ptp(TL) / (np.ptp(TL) + np.ptp(TR))))
edges = np.arange(np.floor(pitch.min()), np.ceil(pitch.max()) + 0.01, 1.0)
print("  pitch bin (deg)   n    left Nm   right Nm")
for a, b in zip(edges[:-1], edges[1:]):
    s = (pitch >= a) & (pitch < b)
    if s.sum() >= 10: print("  %+5.1f..%+5.1f  %5d   %+6.2f    %+6.2f" % (a, b, s.sum(), TL[s].mean(), TR[s].mean()))
