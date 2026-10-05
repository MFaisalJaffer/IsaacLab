"""Time course of a stand in bins: torso pitch/roll (deg, from the policy-frame gravity: y = pitch, z = roll), gyro rms per axis,
mean joint angles (deg) and mean policy targets (deg = action x 0.5 rad).
usage: timecourse.py <ep.npz> [bin_s]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]; B = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
fr = r[:, 29:72]; pg = fr[:, 0:3]; pitch = np.degrees(np.arcsin(np.clip(pg[:, 1], -1, 1))); roll = np.degrees(np.arcsin(np.clip(pg[:, 2], -1, 1)))
gy = fr[:, 26:29]; jp = np.degrees(fr[:, 6:16]); a = np.degrees(0.5 * r[:, 82:92])
N = ["LhP", "RhP", "LhR", "RhR", "Lyaw", "Ryaw", "Lkn", "Rkn", "Lank", "Rank"]
print("  t(s)    pitch  roll | gyro rms yaw/roll/pitch | joint deg: " + " ".join("%6s" % n for n in N) + " | target deg: " + " ".join("%6s" % n for n in N))
for t0 in np.arange(0, t[-1], B):
    m = (t >= t0) & (t < t0 + B)
    if m.sum() < 5: continue
    print(" %4.0f-%-3.0f %+5.2f %+5.2f |   %.3f %.3f %.3f     |            " % (t0, t0 + B, pitch[m].mean(), roll[m].mean(), *np.sqrt((gy[m] ** 2).mean(0)))
          + " ".join("%+6.2f" % v for v in jp[m].mean(0)) + " |             " + " ".join("%+6.2f" % v for v in a[m].mean(0)))
