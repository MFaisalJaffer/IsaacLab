"""!!! FAILED VALIDATION 2026-10-04 — DO NOT USE THE NUMBER.  On a sim episode with a true +40 ms IMU delay
this returned -18 ms (the spring/play/damping phase is absorbed by the fit).  Kept as a record of the attempt.

IMU latency at the policy input, measured against the CAN bus, from one standing episode npz.
With both feet flat the torso pitch is fixed by each leg's chain:  pitch = hip + knee + (ankle encoder
- spring deflection), and the spring deflection is ankle torque / K.  Hip, knee, ankle encoder and ankle
torque all come from the CAN wire tap (time-stamped on the bus); the pitch is what the policy was fed.
For each shift L we fit  pitch(t) = a*hip + b*knee + c*ankle_enc + d*ankle_torque + e  with the CAN
signals taken at t - L, and keep the L with the smallest residual.  Done per leg and for both legs.
Validate on sim episodes where the IMU delay is known before trusting the hardware number.
usage: imu_latency.py <ep.npz> [t1] [t2]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]
t = r[:, 0] - t0; pg = r[:, 29:32]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]
theta = np.arcsin(np.clip(pg[:, 1], -1, 1)); m = (t >= t1) & (t <= t2)
node = w[:, 1].astype(int); kind = w[:, 2].astype(int)
def fb(n):
    f = w[(node == n) & (kind == 1)]
    return f[:, 0] - t0, np.unwrap(f[:, 3]), f[:, 5]
legs = {"L": (13, 16, 17), "R": (3, 6, 7)}
sig = {}
for k, (h, kn, an) in legs.items():
    th, ph, _ = fb(h); tk, pk, _ = fb(kn); ta, pa, tau = fb(an)
    sig[k] = ((th, ph), (tk, pk), (ta, pa), (ta, tau))
def design(which, L):
    cols = []
    for k in which:
        for tt, x in sig[k]: cols.append(np.interp(t[m] - L, tt, x))
    cols.append(np.ones(m.sum())); return np.array(cols).T
y = theta[m]
print("%s  window %.2f-%.2f s (%d ticks), pitch p2p %.2f deg" % (sys.argv[1].split("/")[-1], t1, t2, int(m.sum()), np.degrees(np.ptp(y))))
shifts = np.arange(-40, 141, 2)
for label, which in (("left leg", "L"), ("right leg", "R"), ("both legs", "LR")):
    errs = []
    for L in shifts:
        A = design(which, L / 1000.0); x = np.linalg.lstsq(A, y, rcond=None)[0]
        errs.append(np.sqrt(np.mean((A @ x - y) ** 2)))
    errs = np.array(errs); i = int(np.argmin(errs))
    ok = shifts[errs <= errs[i] * 1.10]                      # shifts within 10 % of the best residual
    print("  %-9s IMU pitch is %3d ms behind the CAN bus (within 10%% of best fit: %d..%d ms) | residual %.1f%% of p2p, at 0 ms it is %.1f%%"
          % (label, shifts[i], ok.min(), ok.max(), 100 * errs[i] / np.ptp(y), 100 * errs[np.where(shifts == 0)[0][0]] / np.ptp(y)))
