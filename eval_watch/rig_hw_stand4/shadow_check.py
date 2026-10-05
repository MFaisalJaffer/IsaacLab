"""Shadow-run check (run in kbot-zed): finds the window where the torso was moved by hand, reports sensor freshness
there, tick timing, what the policy saw (gravity, phase, joints) and what it would have commanded.
usage: shadow_check.py <ep.npz>"""
import numpy as np, sys, json, subprocess
f = sys.argv[1]; z = np.load(f, allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]; m = json.loads(str(z["meta"]))
d = np.diff(t) * 1000
print("%s | %d ticks, %.1f s | tick gap median %.1f ms, 99%% %.1f, max %.0f, gaps over 40 ms: %d" % (str(m.get("ckpt", "?")).split("/")[-1], len(t), t[-1], np.median(d), np.percentile(d, 99), d.max(), (d > 40).sum()))
gy = r[:, 26:29]; mag = np.linalg.norm(gy, axis=1); k = 25; sm = np.convolve(mag, np.ones(k) / k, mode="same"); mv = sm > 0.03
best = (0, 0); i = 0
while i < len(mv):
    if mv[i]:
        j = i
        while j < len(mv) and mv[j]: j += 1
        if j - i > best[1] - best[0]: best = (i, j)
        i = j
    else: i += 1
fr = r[:, 29:72]; pg = fr[:, 0:3]; tilt = np.degrees(np.arccos(np.clip(pg[:, 0] / np.linalg.norm(pg, axis=1), -1, 1)))
print("gravity in policy frame: mean [%.3f %.3f %.3f] | tilt mean %.2f, max %.2f deg | gyro rms %.3f, peak %.2f rad/s" % (*pg.mean(0), tilt.mean(), tilt.max(), np.sqrt((mag ** 2).mean()), mag.max()))
ph = np.unique(np.round(fr[:, 39:43], 3), axis=0); print("gait phase values seen:", ph.tolist()[:4], "| command:", np.unique(np.round(fr[:, 3:6], 3), axis=0).tolist()[:3])
print("joint readings: max |pos| %.2f deg, max |vel| %.2f rad/s" % (np.degrees(np.abs(fr[:, 6:16]).max()), np.abs(fr[:, 16:26]).max()))
a = r[:, 82:92]; N = ["LhP", "RhP", "LhR", "RhR", "Lyaw", "Ryaw", "Lkn", "Rkn", "Lank", "Rank"]
def rng(lbl, s):
    if s.sum() < 5: return
    print("  actions %-18s " % lbl + " ".join("%s %+.2f..%+.2f" % (n, a[s, i].min(), a[s, i].max()) for i, n in enumerate(N)))
rng("first second:", t < 1.0); rng("whole run:", t >= 0); rng("last 2 s:", t > t[-1] - 2)
if best[1] - best[0] >= 50:
    t1, t2 = t[best[0]], t[best[1] - 1]; rng("while moved by hand:", (t >= t1) & (t <= t2))
    print("hand motion found %.1f-%.1f s (gyro rms there %.3f rad/s, tilt swing %.2f deg)" % (t1, t2, np.sqrt((mag[best[0]:best[1]] ** 2).mean()), np.ptp(tilt[best[0]:best[1]])))
    print(subprocess.run(["python3", "/root/walker_test/imu_staleness.py", f, "%.2f" % t1, "%.2f" % t2], capture_output=True, text=True).stdout.strip())
else:
    print("!! no hand motion found (longest moving stretch %.1f s) - freshness cannot be judged on a still robot" % ((best[1] - best[0]) * 0.02))
print("whole run, for reference:"); print(subprocess.run(["python3", "/root/walker_test/imu_staleness.py", f], capture_output=True, text=True).stdout.strip())
