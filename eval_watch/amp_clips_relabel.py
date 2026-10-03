"""Relabel each clip's command with what its joints kinematically deliver on OUR robot.

The stored `cmd` is the retarget source's root motion; amp_clips_kinematic_speed.py measures, per clip
and component, the signed-agreement ratio kinematic/command (stance-foot motion with the base fixed).
Here cmd[:, k] *= ratio_k (nan -> 1, clamped to [lo, hi]); clips whose PRIMARY component (vx for
forward/backward/stop labels, vy for side, wz for turn_in_place) delivers < --min_ratio are moved to
<clips>/_dropped (glitched retargets such as 5 rad/s yaw spikes). Original command kept as `cmd_orig`.

  python eval_watch/amp_clips_relabel.py --clips eval_watch/amp_refs/lafan1/clips_v2 \
      --ratios eval_watch/amp_refs/lafan1/clip_kinematic_speed_v2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil

import numpy as np

PRIMARY = {"backward": 0, "forward_straight": 0, "forward_turn": 0, "start_stop": 0, "stop_start": 0, "side_left": 1, "side_right": 1, "turn_in_place": 2}

p = argparse.ArgumentParser()
p.add_argument("--clips", required=True)
p.add_argument("--ratios", required=True)
p.add_argument("--min_ratio", type=float, default=0.15)
p.add_argument("--lo", type=float, default=0.05)
p.add_argument("--hi", type=float, default=1.5)
a = p.parse_args()

ratios = json.load(open(a.ratios))
dropped_dir = os.path.join(a.clips, "_dropped")
os.makedirs(dropped_dir, exist_ok=True)
rows = []
for path in sorted(glob.glob(os.path.join(a.clips, "*.npz"))):
    name = os.path.basename(path)[:-4]
    d = dict(np.load(path, allow_pickle=True))
    if "cmd_orig" in d:
        print(f"{name}: already relabelled, skipping"); continue
    lab = str(d["label"])
    r = ratios[name]["ratio_vx_vy_wz"]
    r = [1.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in r]
    prim = PRIMARY[lab]
    if r[prim] < a.min_ratio:
        shutil.move(path, os.path.join(dropped_dir, os.path.basename(path)))
        rows.append((name, lab, r, "DROPPED")); continue
    rc = [min(max(x, a.lo), a.hi) for x in r]
    cmd = d["cmd"].astype(np.float32)
    d["cmd_orig"] = cmd.copy()
    d["cmd"] = (cmd * np.array(rc, dtype=np.float32)[None, :]).astype(np.float32)
    d["kin_ratio"] = np.array(rc, dtype=np.float32)
    np.savez(path, **d)
    rows.append((name, lab, rc, "ok"))
by = {}
for name, lab, r, st in rows:
    by.setdefault(lab, []).append((r, st))
print(f"{'label':16s} {'n':>3s} {'dropped':>7s} {'ratio vx':>8s} {'vy':>6s} {'wz':>6s}  (mean of kept)")
for lab, lst in sorted(by.items()):
    kept = np.array([r for r, st in lst if st == "ok"])
    nd = sum(1 for _, st in lst if st != "ok")
    m = kept.mean(axis=0) if len(kept) else [float("nan")] * 3
    print(f"{lab:16s} {len(lst):3d} {nd:7d} {m[0]:8.2f} {m[1]:6.2f} {m[2]:6.2f}")
for name, lab, r, st in rows:
    if st != "ok":
        print(f"  dropped {name} ({lab}) ratios {[round(x, 2) for x in r]}")
json.dump({n: {"label": l, "ratio": r, "status": s} for n, l, r, s in rows}, open(os.path.join(a.clips, "relabel.json"), "w"), indent=1)
