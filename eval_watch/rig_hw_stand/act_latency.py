"""Action latency from one episode npz: how long after the policy received an observation does the
command computed from it appear on the CAN bus?  Both stamped with the policy server's clock.
Method: the wire command position of each joint is a held, delayed copy of (offset + gain * action).
For each shift L, pair every wire command at time tw with the policy row active at tw - L and fit a
line; the L with the smallest residual is the latency (obs received -> command on the wire).
usage: act_latency.py <ep.npz> [t1] [t2]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]
t = r[:, 0] - t0; act = r[:, 82:92]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]
node = w[:, 1].astype(int); kind = w[:, 2].astype(int)
names = {13: "L hip pitch", 14: "L hip roll", 15: "L hip yaw", 16: "L knee", 17: "L ankle",
         3: "R hip pitch", 4: "R hip roll", 5: "R hip yaw", 6: "R knee", 7: "R ankle"}
res = []
for n in sorted(names):
    c = w[(node == n) & (kind == 0)]
    if len(c) < 20: continue
    tw = c[:, 0] - t0; pw = c[:, 3]; m = (tw >= t1 + 0.15) & (tw <= t2); tw = tw[m]; pw = pw[m]
    if len(tw) < 20 or np.ptp(pw) < 0.02: continue
    def held(L):                                    # action row active at tw - L
        i = np.clip(np.searchsorted(t, tw - L, side="right") - 1, 0, len(t) - 1); return act[i]
    a0 = held(0.02); col = max(range(10), key=lambda k: abs(np.corrcoef(a0[:, k], pw)[0, 1]) if np.ptp(a0[:, k]) > 1e-6 else 0)
    best = (1e9, 0)
    for L in range(0, 101):                         # ms
        x = held(L / 1000.0)[:, col]; A = np.c_[x, np.ones(len(x))]
        e = np.sqrt(np.mean((A @ np.linalg.lstsq(A, pw, rcond=None)[0] - pw) ** 2))
        if e < best[0] - 1e-12: best = (e, L)
    res.append((best[1], best[0] / np.ptp(pw)))
    print("%-12s command swing %5.1f deg | on the wire %3d ms after the observation arrived (fit residual %.1f%% of swing)"
          % (names[n], np.degrees(np.ptp(pw)), best[1], 100 * best[0] / np.ptp(pw)))
good = [x[0] for x in res if x[1] < 0.08]
if good: print("=> action latency (obs received -> command on CAN): median %.0f ms over %d joints with clean fits" % (np.median(good), len(good)))
print("inference time: median %.1f ms" % np.median(r[:, 1]))
c = w[(node == 7) & (kind == 0)]
if len(c) > 2: print("command frames on the wire every %.1f ms (median)" % (1000 * np.median(np.diff(c[:, 0]))))
