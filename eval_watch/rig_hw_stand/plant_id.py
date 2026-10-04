"""Does the torso respond to the measured ankle torques like a rigid body on flat feet?
Model: I*alpha = c*tau_ankle(t - lag) + G*theta   (theta = pitch-like angle, alpha from the gyro)
Fits c/I, G/I and the best lag by least squares on a standing window. Compare hardware vs sim.
usage: plant_id.py <ep.npz> <t1> <t2> [label]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]
t1, t2 = float(sys.argv[2]), float(sys.argv[3]); label = sys.argv[4] if len(sys.argv) > 4 else sys.argv[1].split("/")[-1]
t = r[:, 0] - t0; o43 = r[:, 29:72]; pg = o43[:, 0:3]; gy = o43[:, 26:29]
theta = np.arcsin(np.clip(pg[:, 1], -1, 1))                      # pitch-like angle, rad
# which gyro axis is d(theta)/dt ?
dth = np.gradient(theta, t); m = (t >= t1) & (t <= t2)
cz = [np.corrcoef(dth[m], gy[m, k])[0, 1] for k in range(3)]; ax = int(np.argmax(np.abs(cz))); sgn = np.sign(cz[ax])
omega = sgn * gy[:, ax]; alpha = np.gradient(omega, t)
node = w[:, 1].astype(int); kind = w[:, 2].astype(int)
def fb(n):
    f = w[(node == n) & (kind == 1)]; return f[:, 0] - t0, f[:, 3], f[:, 5]
tl, pl, taul = fb(17); tr, pr, taur = fb(7)
tauL = np.interp(t, tl, taul); tauR = np.interp(t, tr, taur)
enc = 0.5 * (np.interp(t, tl, pl) - np.interp(t, tr, pr))       # mean ankle encoder (mirrored), rad
best = None
for lag in range(0, 7):                                          # ticks of 20 ms
    for name, tau in (("tauL-tauR", tauL - tauR), ("tauL+tauR", tauL + tauR)):
        ts = np.roll(tau, lag); mm = m.copy(); mm[:lag + 1] = False
        A = np.c_[ts[mm], theta[mm], np.ones(mm.sum())]; y = alpha[mm]
        x, res, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ x
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best[0]: best = (r2, lag, name, x)
r2, lag, name, x = best
print("%s  window %.2f-%.2f s (%d ticks)" % (label, t1, t2, int(m.sum())))
print("  pitch rate = %+d * gyro[%d] (corr %.2f) | theta p2p %.2f deg | alpha rms %.1f rad/s^2" % (sgn, ax, cz[ax], np.degrees(np.ptp(theta[m])), alpha[m].std()))
print("  best fit: alpha = %.3f * (%s)[t - %d ms] + %.1f * theta   (R^2 = %.2f)" % (x[0], name, lag * 20, x[1], r2))
print("  -> effective inertia I = 1/|c| = %.2f kg m^2 ; gravity term G/I = %.1f 1/s^2 (rigid: mgh/I ~ 86/I)" % (1 / max(abs(x[0]), 1e-9), x[1]))
tt = (tauL - tauR)[m]
print("  ankle torque (L-R) mean %+.1f Nm, p2p %.1f Nm | ankle encoder p2p %.1f deg vs body pitch p2p %.1f deg" % (tt.mean(), np.ptp(tt), np.degrees(np.ptp(enc[m])), np.degrees(np.ptp(theta[m]))))
# spring check: encoder - body angle vs torque  -> series stiffness seen in this episode
defl = enc[m] - theta[m] * np.sign(np.corrcoef(enc[m], theta[m])[0, 1])
k, b = np.polyfit(defl - defl.mean(), 0.5 * (tauL - tauR)[m] - 0.5 * (tauL - tauR)[m].mean(), 1)
print("  per-ankle torque vs (encoder - body) deflection: slope %.1f Nm/rad (series stiffness seen), corr %.2f" % (abs(k), np.corrcoef(defl, 0.5 * (tauL - tauR)[m])[0, 1]))
