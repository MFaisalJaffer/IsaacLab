"""One line per episode: how hard is the stand rocking?  Slow drift (a 1 s moving average) is removed,
so the numbers are the fast (>1 Hz) part only: torso pitch rms and peak frequency, ankle encoder rms
(from the CAN wire tap), ankle torque p2p.
usage: rock_strength.py <label>=<ep.npz>[@t1:t2] ..."""
import numpy as np, sys
def hp(x, n): return x - np.convolve(np.pad(x, n // 2, mode="edge"), np.ones(n) / n, mode="valid")
print("%-44s %-21s %-9s %-21s %s" % ("run", "pitch rms/p2p (deg)", "peak Hz", "ankle enc rms L/R", "ankle torque p2p L/R (Nm)"))
for a in sys.argv[1:]:
    lab, f = a.split("=", 1); win = None
    if "@" in f: f, win = f.split("@"); win = [float(x) for x in win.split(":")]
    z = np.load(f, allow_pickle=True); r = z["data"]; w = z["wire"]; t0 = r[0, 0]; t = r[:, 0] - t0
    a0, b0 = win if win else (0.6, t[-1] - 0.3)
    x = hp(np.degrees(np.arcsin(np.clip(r[:, 30], -1, 1))), 51); m = (t >= a0) & (t <= b0)
    xs = x[m] - x[m].mean(); dt = np.median(np.diff(t)); F = np.abs(np.fft.rfft(xs * np.hanning(len(xs)), 4096)); fr = np.fft.rfftfreq(4096, dt)
    band = (fr >= 1.0) & (fr <= 6.0); fpk = fr[band][np.argmax(F[band])]
    o = []
    for nd in (17, 7):
        f1 = w[(w[:, 1] == nd) & (w[:, 2] == 1)]; tf = f1[:, 0] - t0; mm = (tf >= a0) & (tf <= b0)
        o.append((hp(np.degrees(np.unwrap(f1[:, 3])), 101)[mm].std(), np.ptp(f1[mm, 5])))
    print("%-44s %5.2f / %4.2f          %4.2f      %5.2f / %5.2f         %4.1f / %4.1f" % (lab, x[m].std(), np.ptp(x[m]), fpk, o[0][0], o[1][0], o[0][1], o[1][1]))
