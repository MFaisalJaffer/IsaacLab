"""Stability margin from a kick: after a short pitch-moment pulse, how fast does the torso rocking die?
Detrends the policy-side pitch with a 1 s moving average, lists the swing extrema after the pulse, and
reports frequency and the average ratio between successive half-swings (<1 decays, >1 grows).
usage: ringdown.py <ep.npz> <t_pulse_end> [t_end]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
tp = float(sys.argv[2]); te = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]
th = np.degrees(np.arcsin(np.clip(r[:, 30], -1, 1)))
n = max(3, int(round(1.0 / np.median(np.diff(t)))) | 1); k = np.ones(n) / n
slow = np.convolve(np.pad(th, n // 2, mode="edge"), k, mode="valid"); x = th - slow
m = (t >= tp) & (t <= te); tt = t[m]; xx = x[m]
ext = [i for i in range(1, len(xx) - 1) if (xx[i] - xx[i - 1]) * (xx[i + 1] - xx[i]) < 0 and abs(xx[i]) > 0.05]
# keep alternating-sign extrema only (largest of each same-sign run)
alt = []
for i in ext:
    if alt and np.sign(xx[i]) == np.sign(xx[alt[-1]]):
        if abs(xx[i]) > abs(xx[alt[-1]]): alt[-1] = i
    else: alt.append(i)
amps = np.abs(xx[alt]); pre = (t >= max(0.6, tp - 3)) & (t < tp - 0.4)
print("%s: pitch before the kick: p2p %.2f deg (1-4 Hz part rms %.3f deg)" % (sys.argv[1].split("/")[-1], np.ptp(th[pre]), x[pre].std()))
print("  kick peak %.2f deg | swings after the kick (t s, deg): %s" % (np.abs(x[(t >= tp - 0.5) & (t <= tp + 0.3)]).max(), " ".join("%.2f:%+.2f" % (tt[i], xx[i]) for i in alt[:12])))
if len(alt) >= 3:
    per = 2 * np.median(np.diff(tt[alt[:8]])); ratios = amps[1:8] / amps[:7]
    print("  rocking %.2f Hz | half-swing ratio %.2f (geometric mean of %d) -> %s; swings above 0.2 deg: %d; rms 1-4 s after the kick %.3f deg"
          % (1 / per, np.exp(np.mean(np.log(ratios))), len(ratios), "DECAYS" if np.exp(np.mean(np.log(ratios))) < 0.95 else "SUSTAINS/GROWS",
             int((amps > 0.2).sum()), x[(t >= tp + 1) & (t <= tp + 4)].std()))
else:
    print("  fewer than 3 swings above 0.05 deg after the kick -> well damped; rms 1-4 s after the kick %.3f deg" % x[(t >= tp + 1) & (t <= tp + 4)].std())
