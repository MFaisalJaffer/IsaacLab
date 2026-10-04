"""How fresh is each sensor channel at the policy input?  A channel that is bit-identical to the
previous tick is an old sample sent again.  Reports, per channel: share of repeated ticks, how many
distinct samples per second actually arrived, the hold-length histogram, and the mean extra age the
holding alone adds.  Use a window where the robot is moving (a still robot can repeat honestly).
usage: imu_staleness.py <ep.npz> [t1] [t2]"""
import numpy as np, sys
f = sys.argv[1]; z = np.load(f, allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]; o = r[:, 2:29]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t2 = float(sys.argv[3]) if len(sys.argv) > 3 else t[-1]
m = (t >= t1) & (t <= t2); o = o[m]; tt = t[m]; dt = float(np.median(np.diff(tt))); dur = tt[-1] - tt[0]
print("%s  window %.2f-%.2f s, %d ticks, tick %.1f ms" % (f.split("/")[-1], t1, t2, len(tt), 1000 * dt))
for name, sl in (("IMU quat", slice(20, 24)), ("IMU gyro", slice(24, 27)), ("joint pos", slice(0, 10)), ("joint vel", slice(10, 20))):
    x = o[:, sl]; new = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
    b = np.flatnonzero(new); runs = np.diff(np.r_[b, len(x)])
    hist = " ".join("%dx:%d" % (k, (runs == k).sum()) for k in range(1, 6) if (runs == k).sum()) + (" 6+x:%d" % (runs >= 6).sum() if (runs >= 6).sum() else "")
    age = sum(n * (n - 1) / 2.0 for n in runs) / len(x) * dt
    print("  %-9s repeated on %4.1f%% of ticks | %5.1f distinct samples/s | holds %s | mean extra age %4.1f ms, worst hold %3.0f ms"
          % (name, 100 * (1 - new.mean()), new.sum() / dur, hist, 1000 * age, 1000 * dt * (runs.max() - 1)))
