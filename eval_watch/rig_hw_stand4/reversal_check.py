"""Training's check (reply to the stand #4 note, section 2): after the torso reverses direction, does the ankle encoder
follow the torso MORE closely than while sliding (towards 1:1 = the gap is locked, foot + gap + rotor move as one piece
against the drive's kp) or LESS (towards 0 = the rotor is stuck)?  Drives hold zero, torso rocked by hand.
For every stroke between two torso reversals: slope d(encoder)/d(torso pitch) in windows of distance travelled since
the reversal; the chain stiffness that slope implies (kp x s / (1 - s)); and whether the torque creeps while the torso
is held still.  NOTE: a 1 deg first window hides the structure (it averages a dead start with a locked stretch).
usage: reversal_check.py <shadow_ep.npz> [t1] [t2]"""
import numpy as np, sys
KP = 60.0
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0; t2 = float(sys.argv[3]) if len(sys.argv) > 3 else 62.0; m = (t >= t1) & (t <= t2)
fr = r[m, 29:72]; tt = t[m]; pitch = np.degrees(np.arcsin(np.clip(fr[:, 1], -1, 1))); jp = np.degrees(fr[:, 6:16])
ps = np.convolve(pitch, np.ones(5) / 5, mode="same")
HYST = 0.4; rev = []; d = 0; ext = 0          # a reversal = running extreme, confirmed once the torso has come back HYST deg
for i in range(1, len(ps)):
    if d >= 0 and ps[i] > ps[ext]: ext = i; d = 1 if d == 0 and ps[i] - ps[0] > HYST else d
    elif d <= 0 and ps[i] < ps[ext]: ext = i; d = -1 if d == 0 and ps[0] - ps[i] > HYST else d
    if d == 1 and ps[ext] - ps[i] > HYST: rev.append((ext, -1)); d = -1; ext = i
    elif d == -1 and ps[i] - ps[ext] > HYST: rev.append((ext, +1)); d = 1; ext = i
print("%s  %.0f-%.0f s | torso reversals at (s, deg): %s" % (sys.argv[1].split("/")[-1], t1, t2, " ".join("%.1f:%+.1f" % (tt[i], pitch[i]) for i, _ in rev)))
W = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1, 2), (2, 4), (4, 8), (8, 14)]
for name, e in (("RIGHT", -jp[:, 9]), ("LEFT", jp[:, 8])):
    print("\n%s ankle | torso travel since reversal | encoder deg per torso deg: median (each stroke) | implied chain stiffness" % name)
    for a, b in W:
        sl = []
        for j, (i0, sgn) in enumerate(rev):
            i1 = rev[j + 1][0] if j + 1 < len(rev) else len(pitch) - 1
            if abs(pitch[i1] - pitch[i0]) < 2.5: continue
            seg = np.arange(i0, i1 + 1); dp = sgn * (ps[seg] - ps[i0]); s = seg[(dp >= a) & (dp < b)]
            if len(s) >= 5 and np.ptp(pitch[s]) > 0.6 * (b - a): sl.append(np.polyfit(pitch[s], e[s], 1)[0])
        if sl:
            q = float(np.median(sl)); print("   %4.1f-%-4.1f deg | %.2f  (%s) | %s" % (a, b, q, " ".join("%.2f" % v for v in sl), "%.0f Nm/rad" % (KP * q / (1 - q)) if 0.1 < q < 0.95 else "-"))
print("\ntorso held still (within 0.25 deg for over 1.2 s): torque at start -> end of the hold, right | left (Nm)")
i = 0; N = len(pitch)
while i < N - 60:
    j = i + 1
    while j < N and abs(ps[j] - ps[i]) < 0.25: j += 1
    if (j - i) * 0.02 > 1.2:
        tr = KP * np.radians(-jp[i:j, 9]); tl = KP * np.radians(jp[i:j, 8]); k = 8
        print("   %5.1f-%5.1f s at %+5.1f deg: %+5.2f -> %+5.2f | %+5.2f -> %+5.2f" % (tt[i], tt[j - 1], ps[i:j].mean(), tr[:k].mean(), tr[-k:].mean(), tl[:k].mean(), tl[-k:].mean()))
        i = j
    else: i += 5
