"""Observation latency from one episode npz: how old is the joint position the policy was fed,
relative to the same motion on the CAN bus?  Both streams are stamped with the policy server's
clock (wire tap thread at frame reception; obs row at packet reception).
Method: for each joint with enough motion, resample the CAN feedback position to 1 kHz and find
the shift that best matches the policy-side jpos samples (least squares over shifts 0..120 ms).
usage: obs_latency.py <ep.npz> [t1] [t2]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]
t = r[:, 0] - t0; o27 = r[:, 2:29]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]
node = w[:, 1].astype(int); kind = w[:, 2].astype(int)
# HIL order of obs27 jpos: find by matching each node's fb trace to the 10 jpos columns
names = {13: "L hip pitch", 14: "L hip roll", 15: "L hip yaw", 16: "L knee", 17: "L ankle",
         3: "R hip pitch", 4: "R hip roll", 5: "R hip yaw", 6: "R knee", 7: "R ankle"}
m = (t >= t1) & (t <= t2); res = []
for n in sorted(names):
    f = w[(node == n) & (kind == 1)]
    if len(f) < 20: continue
    tf = f[:, 0] - t0; pf = ((f[:, 3] + np.pi) % (2 * np.pi)) - np.pi           # strip any 2*pi wrap offset
    # which obs column is this joint? best |corr| at zero shift
    col = max(range(10), key=lambda c: abs(np.corrcoef(np.interp(t[m], tf, pf), o27[m, c])[0, 1]) if np.ptp(o27[m, c]) > 1e-4 else 0)
    y = o27[m, col]; amp = np.ptp(y)
    if amp < 0.03: continue                                                     # need real motion (> ~1.7 deg)
    sg = np.sign(np.corrcoef(np.interp(t[m], tf, pf), y)[0, 1])
    best = (1e9, 0)
    for sh in range(0, 121, 2):                                                 # ms
        e = np.mean((sg * np.interp(t[m] - sh / 1000.0, tf, pf) - y - np.mean(sg * np.interp(t[m] - sh / 1000.0, tf, pf) - y)) ** 2)
        if e < best[0]: best = (e, sh)
    res.append((names[n], best[1], np.degrees(amp), np.sqrt(best[0]) / amp))
    print("%-12s motion %5.1f deg p2p | policy saw the CAN position %3d ms late (fit residual %.1f%% of p2p)" % (names[n], np.degrees(amp), best[1], 100 * np.sqrt(best[0]) / amp))
if res:
    good = [x[1] for x in res if x[3] < 0.08]
    print("=> joint observation latency (CAN bus -> policy input): median %s ms over %d joints with clean fits" % (np.median(good) if good else "n/a", len(good)))
print("policy tick period: median %.1f ms | CAN feedback period: median %.1f ms" % (1000 * np.median(np.diff(t)), 1000 * np.median(np.diff(w[(node == 7) & (kind == 1)][:, 0])) if ((node == 7) & (kind == 1)).sum() > 2 else float("nan")))
