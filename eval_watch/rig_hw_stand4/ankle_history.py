"""Left vs right ankle across hardware episodes, using only quantities that do not depend on the IMU zero:
each ankle's torque (60 x (target - encoder); 'hold' episodes: target = 0) in the sense 'resists a backward lean',
and the straight-line relation left = a x right + b over the episode (a = stiffness ratio, b = offset at right = 0).
usage: ankle_history.py label=ep.npz[:hold][@t1:t2] ..."""
import numpy as np, sys
print("%-34s %6s | left Nm mean (min..max)   | right Nm mean (min..max)  | left = a x right + b        | swing share L" % ("episode", "secs"))
for arg in sys.argv[1:]:
    lbl, rest = arg.split("=", 1); win = None
    if "@" in rest: rest, w = rest.split("@"); win = [float(x) for x in w.split(":")]
    hold = rest.endswith(":hold"); f = rest[:-5] if hold else rest
    z = np.load(f, allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
    m = (t >= win[0]) & (t <= win[1]) if win else np.ones(len(t), bool)
    jp = r[m, 35:45]; tg = 0.5 * r[m, 82:92]
    TL = -60.0 * ((0 if hold else tg[:, 8]) - jp[:, 8]); TR = 60.0 * ((0 if hold else tg[:, 9]) - jp[:, 9])
    if np.ptp(TR) > 0.3:
        A = np.c_[TR, np.ones(len(TR))]; a, b = np.linalg.lstsq(A, TL, rcond=None)[0]; r2 = 1 - (TL - A @ np.r_[a, b]).var() / max(TL.var(), 1e-12)
        fit = "a %.2f  b %+.2f Nm  (R2 %.2f)" % (a, b, r2)
    else: fit = "too little motion to fit   "
    print("%-34s %6.1f | %+5.2f (%+5.2f..%+5.2f)     | %+5.2f (%+5.2f..%+5.2f)     | %s | %3.0f%%" % (
        lbl, t[m][-1] - t[m][0], TL.mean(), TL.min(), TL.max(), TR.mean(), TR.min(), TR.max(), fit, 100 * np.ptp(TL) / max(np.ptp(TL) + np.ptp(TR), 1e-9)))
