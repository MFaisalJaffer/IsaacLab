"""Cut the edited LAFAN1 clips (our joint order/signs, 50 Hz) into labelled motion segments = the
clip library for the time-indexed tracker (dataset expansion: backward, sideways, pivots, stops).

For each segment: joints, joint velocities, body-frame base velocity (vx, vy) and yaw rate from the
clip's own root path (the tracker's COMMAND), the label, and the time scale applied (backward
segments faster than --max_back are stretched so our robot is never asked to back up faster than
it walks). Output: eval_watch/amp_refs/lafan1/clips/<label>_<n>.npz + clip_library.json.
Body height per frame is added by amp_clips_ground.py (one Isaac puppet pass).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", default="/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1")
p.add_argument("--max_per_label", type=int, default=8)
p.add_argument("--max_back", type=float, default=0.35, help="max backward speed (m/s) after time scaling")
p.add_argument("--max_side", type=float, default=0.30, help="max sideways speed (m/s) after time scaling")
p.add_argument("--margin", type=float, default=0.3, help="seconds kept before/after each segment")
p.add_argument("--edit_suffix", default="_kbot_edit.npz", help="which edited clips to cut (library v2: _kbot_edit2.npz)")
p.add_argument("--out", default="clips", help="output sub-directory under --src")
a = p.parse_args()
OUT = os.path.join(a.src, a.out)
os.makedirs(OUT, exist_ok=True)

# label -> (mask function on smoothed vx, vy, wz, speed, knee_ok ; min seconds)
CATS = {
    "backward": (lambda vx, vy, wz, sp, ok: (vx < -0.12) & ok, 2.0),
    # lateral-dominant stepping (humans side-step with some forward drift): |vy| > |vx| and |vy| > 0.12
    "side_left": (lambda vx, vy, wz, sp, ok: (vy > 0.12) & (np.abs(vy) > np.abs(vx)) & (np.abs(wz) < 0.6) & ok, 1.2),
    "side_right": (lambda vx, vy, wz, sp, ok: (vy < -0.12) & (np.abs(vy) > np.abs(vx)) & (np.abs(wz) < 0.6) & ok, 1.2),
    "turn_in_place": (lambda vx, vy, wz, sp, ok: (sp < 0.15) & (np.abs(wz) >= 0.3) & ok, 1.5),
    "forward_turn": (lambda vx, vy, wz, sp, ok: (vx > 0.15) & (vx < 0.55) & (np.abs(wz) >= 0.3) & ok, 2.0),
    "forward_straight": (lambda vx, vy, wz, sp, ok: (vx > 0.15) & (vx < 0.55) & (np.abs(wz) < 0.25) & ok, 2.0),
}


def segs(mask, fps, min_s):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            if (i - start) / fps >= min_s:
                out.append((start, i))
            start = None
    return out


cands = {k: [] for k in CATS}
cands["stop_start"] = []
cands["start_stop"] = []
for path in sorted(glob.glob(os.path.join(a.src, "walk*" + a.edit_suffix))):
    d = np.load(path, allow_pickle=True)
    fps = float(d["fps"]); q = d["joint_pos"]; v = d["base_lin_vel_b"]; w = d["base_yaw_rate"]
    T = q.shape[0]
    k = 25
    sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
    vx, vy, wz = sm(v[:, 0]), sm(v[:, 1]), sm(w)
    sp = np.hypot(vx, vy)
    knee = np.degrees(np.maximum(q[:, 6], -q[:, 7]))
    ok = knee < 95
    for lab, (fn, mn) in CATS.items():
        for s, e in segs(fn(vx, vy, wz, sp, ok), fps, mn):
            cands[lab].append((e - s, path, s, e))
    still = (sp < 0.06) & (np.abs(wz) < 0.15) & ok
    for s, e in segs(still, fps, 1.0):
        f = min(T - 1, e + int(3 * fps))
        if f > e and (vx[e:f] > 0.15).mean() > 0.7:
            cands["stop_start"].append((f - s, path, s, f))
        b = max(0, s - int(3 * fps))
        if s > b and (vx[b:s] > 0.15).mean() > 0.7:
            cands["start_stop"].append((e - b, path, b, e))

PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
MIRROR_LABEL = {"side_left": "side_right", "side_right": "side_left"}
index = []
for lab, lst in cands.items():
    lst.sort(key=lambda t: -t[0])
    for n, (ln, path, s, e) in enumerate(lst[: a.max_per_label]):
        d = np.load(path, allow_pickle=True)
        fps = float(d["fps"])
        m = int(a.margin * fps)
        s0, e0 = max(0, s - m), min(d["joint_pos"].shape[0], e + m)
        q = d["joint_pos"][s0:e0].astype(np.float64); v = d["base_lin_vel_b"][s0:e0].astype(np.float64); w = d["base_yaw_rate"][s0:e0].astype(np.float64)
        scale = 1.0
        if lab == "backward":
            vb = -np.percentile(-v[:, 0], 90)  # 90th pct backward speed (negative)
            if -vb > a.max_back:
                scale = -vb / a.max_back  # stretch time by this factor
        elif lab in ("side_left", "side_right"):
            vs = np.percentile(np.abs(v[:, 1]), 90)
            if vs > a.max_side:
                scale = vs / a.max_side
        if scale != 1.0:
            t_in = np.arange(q.shape[0]) / fps * scale
            t_out = np.arange(0, t_in[-1], 1 / fps)
            q = np.stack([np.interp(t_out, t_in, q[:, j]) for j in range(q.shape[1])], 1)
            v = np.stack([np.interp(t_out, t_in, v[:, j]) for j in range(2)], 1) / scale
            w = np.interp(t_out, t_in, w) / scale
        qd = np.gradient(q, 1 / fps, axis=0)
        k = 25
        cmd = np.stack([np.convolve(v[:, 0], np.ones(k) / k, mode="same"), np.convolve(v[:, 1], np.ones(k) / k, mode="same"),
                        np.convolve(w, np.ones(k) / k, mode="same")], 1)
        name = f"{lab}_{n}"
        np.savez(os.path.join(OUT, name + ".npz"), joint_names=d["joint_names"], fps=fps, joint_pos=q.astype(np.float32), joint_vel=qd.astype(np.float32),
                 cmd=cmd.astype(np.float32), base_lin_vel_b=v.astype(np.float32), base_yaw_rate=w.astype(np.float32), label=lab,
                 source=f"{os.path.basename(path)}[{s0 / fps:.1f}:{e0 / fps:.1f}s] x{scale:.2f}")
        index.append({"name": name, "label": lab, "frames": int(q.shape[0]), "seconds": round(q.shape[0] / fps, 2), "time_scale": round(scale, 3),
                      "cmd_mean": [round(float(x), 3) for x in cmd.mean(0)], "source": f"{os.path.basename(path)}[{s0 / fps:.1f}:{e0 / fps:.1f}]"})
        # mirrored copy: left<->right joints negated, lateral velocity and yaw rate flipped
        qm = -q[:, PERM]; qdm = -qd[:, PERM]
        cmdm = cmd * np.array([1.0, -1.0, -1.0]); vm = v * np.array([1.0, -1.0]); wm = -w
        mlab = MIRROR_LABEL.get(lab, lab); mname = f"{mlab}_{n}m"
        np.savez(os.path.join(OUT, mname + ".npz"), joint_names=d["joint_names"], fps=fps, joint_pos=qm.astype(np.float32), joint_vel=qdm.astype(np.float32),
                 cmd=cmdm.astype(np.float32), base_lin_vel_b=vm.astype(np.float32), base_yaw_rate=wm.astype(np.float32), label=mlab,
                 source=f"MIRROR of {name}")
        index.append({"name": mname, "label": mlab, "frames": int(q.shape[0]), "seconds": round(q.shape[0] / fps, 2), "time_scale": round(scale, 3),
                      "cmd_mean": [round(float(x), 3) for x in cmdm.mean(0)], "source": f"mirror of {name}"})
json.dump(index, open(os.path.join(a.src, "clip_library.json" if a.out == "clips" else f"clip_library_{a.out}.json"), "w"), indent=1)
by = {}
for c in index:
    by.setdefault(c["label"], []).append(c)
print(f"{'label':17s} {'n':>2s} {'total s':>8s}  cmd_mean (vx, vy, wz) of longest")
for lab, lst in by.items():
    print(f"{lab:17s} {len(lst):2d} {sum(c['seconds'] for c in lst):8.1f}  {lst[0]['cmd_mean']}  (x{lst[0]['time_scale']})")
print(f"{len(index)} clips -> {OUT}")
