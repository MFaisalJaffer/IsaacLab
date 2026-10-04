"""Per-tick age of the joint position the policy was fed: for every policy tick, find the CAN feedback
frame whose value the observation equals (affine map fitted once), and report tick time minus frame time.
Only meaningful while the joint moves (values distinguishable).  usage: joint_age_per_tick.py <ep.npz> <node> [t1] [t2]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]; t = r[:, 0] - t0; n = int(sys.argv[2])
t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2; t2 = float(sys.argv[4]) if len(sys.argv) > 4 else t[-1]
f = w[(w[:, 1] == n) & (w[:, 2] == 1)]; tf = f[:, 0] - t0; pf = (f[:, 3] + np.pi) % (2 * np.pi) - np.pi
jp = r[:, 2:12]; m = (t >= t1) & (t <= t2)
col = max(range(10), key=lambda c: abs(np.corrcoef(np.interp(t[m] - 0.014, tf, pf), jp[m, c])[0, 1]) if np.ptp(jp[m, c]) > 1e-4 else 0)
a, b = np.polyfit(np.interp(t[m] - 0.014, tf, pf), jp[m, col], 1); a = np.sign(a)        # same units: slope is +-1
b = np.median(jp[m, col] - a * np.interp(t[m] - 0.014, tf, pf))
ages = []
for i in np.flatnonzero(m):
    k = np.flatnonzero((tf <= t[i] + 0.002) & (tf >= t[i] - 0.08))
    if len(k) < 3: continue
    err = np.abs(a * pf[k] + b - jp[i, col]); j = k[np.argmin(err)]
    moving = np.ptp(pf[k]) > 0.004
    if moving and err.min() < 0.0006 and np.sort(err)[1] > 2 * max(err.min(), 1e-4): ages.append(1000 * (t[i] - tf[j]))
ages = np.array(ages)
if len(ages) < 5: print("node %d: too few unambiguous ticks (%d)" % (n, len(ages))); sys.exit()
print("node %d: %d ticks matched to one CAN frame | age ms: median %.1f, 5th-95th pct %.1f..%.1f, min %.1f, max %.1f | histogram (5 ms bins from 0): %s"
      % (n, len(ages), np.median(ages), np.percentile(ages, 5), np.percentile(ages, 95), ages.min(), ages.max(), np.histogram(ages, bins=np.arange(0, 45, 5))[0]))
