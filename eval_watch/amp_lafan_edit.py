"""Edit the converted LAFAN1 clips for our body: hip-roll scale, joint-limit clamp, walking-only segments.

Input : eval_watch/amp_refs/lafan1/<clip>_kbot.npz   (amp_lafan_to_kbot.py: our joint order/signs, 50 Hz)
Output: eval_watch/amp_refs/lafan1/<clip>_kbot_edit.npz  + segments.json  + edit_summary.json

Edits (joint space only — the discriminator never sees the base):
  * hip roll  x ROLL_SCALE (human/G1 pelvic sway +-18..40 deg on our joints -> ~+-7 deg like our reference)
  * hip yaw   x YAW_SCALE  (keep most of it: it carries the turning)
  * clamp every joint to our URDF limits minus MARGIN deg (no reference should lean on a hard stop)
  * velocities recomputed from the edited angles
Segments: 1-s windows are labelled walk_straight / walk_turn / excluded from the ORIGINAL base path
(forward speed in band, |yaw rate| threshold) and the knee range (deep crouches are not walking).
"""
import glob, json, os
import numpy as np

D = "/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1"
ROLL_SCALE, YAW_SCALE, MARGIN = 0.3, 0.8, 3.0
# Library v2 (2026-10-01): side-steps live in hip roll, so a flat x0.3 removed 80% of their lateral motion.
# KBOT_EDIT_SIDE_ROLL=1.0 blends the roll scale from ROLL_SCALE (|vy|=0) to that value (|vy|>=0.15 m/s).
SIDE_ROLL = float(os.environ.get("KBOT_EDIT_SIDE_ROLL", "0"))
EDIT_SUFFIX = os.environ.get("KBOT_EDIT_SUFFIX", "_kbot_edit.npz")
TAG = "" if EDIT_SUFFIX == "_kbot_edit.npz" else EDIT_SUFFIX.replace("_kbot_edit", "").replace(".npz", "")
# our URDF limits (deg), joint order [Lhp,Rhp, Lhr,Rhr, Lhy,Rhy, Lk,Rk, La,Ra]
LIM = np.radians(np.array([[-60, 127], [-127, 60], [-12, 130], [-130, 12], [-90, 90], [-90, 90], [0, 155], [-155, 0], [-72, 13], [-13, 72]], dtype=np.float64))
BAND = (0.08, 0.60)       # forward speed band (m/s) for 'walking'
TURN = 0.3                # |yaw rate| (rad/s) that makes a window 'walk_turn'
KNEE_MAX = 95.0           # deg; windows with deeper knee flexion are not walking (crouch/step-over)

summary, segments = {}, {}
for path in sorted(glob.glob(D + "/walk*_kbot.npz")):
    name = os.path.basename(path).replace("_kbot.npz", "")
    d = np.load(path, allow_pickle=True)
    fps = float(d["fps"]); q = d["joint_pos"].astype(np.float64); vb = d["base_lin_vel_b"]; yr = d["base_yaw_rate"]
    T = q.shape[0]
    qe = q.copy()
    if SIDE_ROLL > 0:
        vy_s = np.convolve(np.abs(vb[:, 1]), np.ones(25) / 25, mode="same")
        roll_scale = ROLL_SCALE + (SIDE_ROLL - ROLL_SCALE) * np.clip(vy_s / 0.15, 0.0, 1.0)
        qe[:, 2:4] *= roll_scale[:, None]
    else:
        qe[:, 2:4] *= ROLL_SCALE
    qe[:, 4:6] *= YAW_SCALE
    lo = LIM[:, 0] + np.radians(MARGIN); hi = LIM[:, 1] - np.radians(MARGIN)
    clipped = np.mean((qe < lo) | (qe > hi), axis=0)
    qe = np.clip(qe, lo, hi)
    qde = np.gradient(qe, 1 / fps, axis=0)
    # window labels on the ORIGINAL base motion
    k = 25
    sm = lambda a: np.convolve(a, np.ones(k) / k, mode="same")
    v_s, yr_s = sm(vb[:, 0]), sm(yr)
    knee = np.degrees(np.maximum(q[:, 6], -q[:, 7]))
    labels = np.full(T, "excluded", dtype=object)
    W = int(fps)
    for s in range(0, T - W, W):
        e = s + W
        if (v_s[s:e] > BAND[0]).all() and (v_s[s:e] < BAND[1]).all() and knee[s:e].max() < KNEE_MAX:
            labels[s:e] = "walk_turn" if np.abs(yr_s[s:e]).mean() >= TURN else "walk_straight"
    segs = []
    start, cur = 0, labels[0]
    for t in range(1, T + 1):
        if t == T or labels[t] != cur:
            segs.append((round(start / fps, 2), round(t / fps, 2), cur)); start, cur = t, (labels[t] if t < T else None)
    keep = labels != "excluded"
    np.savez(path.replace("_kbot.npz", EDIT_SUFFIX), joint_names=d["joint_names"], fps=fps, joint_pos=qe.astype(np.float32), joint_vel=qde.astype(np.float32),
             base_lin_vel_b=d["base_lin_vel_b"], base_yaw_rate=d["base_yaw_rate"], base_pos=d["base_pos"], base_yaw=d["base_yaw"], labels=labels.astype(str),
             edits=f"hip_roll x{ROLL_SCALE}, hip_yaw x{YAW_SCALE}, clamp to kbot limits -{MARGIN} deg", source=str(d["source"]))
    segments[name] = [s for s in segs if s[2] != "excluded"]
    deg = np.degrees(qe)
    summary[name] = {"dur_s": round(T / fps, 1), "walk_straight_s": round(float(np.sum(labels == "walk_straight") / fps), 1), "walk_turn_s": round(float(np.sum(labels == "walk_turn") / fps), 1),
                     "excluded_s": round(float(np.sum(~keep) / fps), 1), "hip_roll_abs_p95_deg": round(float(np.percentile(np.abs(deg[:, 2:4]), 95)), 1),
                     "clipped_frac_per_joint_max": round(float(clipped.max()), 3), "knee_p95_walk_deg": round(float(np.percentile(knee[keep], 95)), 1) if keep.any() else None}
json.dump(summary, open(D + f"/edit_summary{TAG}.json", "w"), indent=1); json.dump(segments, open(D + f"/segments{TAG}.json", "w"), indent=1)
tot = {k: sum(s[k] for s in summary.values()) for k in ("walk_straight_s", "walk_turn_s", "excluded_s")}
print(f"{'clip':16s} {'straight s':>10s} {'turn s':>7s} {'excl s':>7s} {'roll95':>7s} {'clip%':>6s} {'knee95':>7s}")
for n, s in summary.items():
    print(f"{n:16s} {s['walk_straight_s']:10.1f} {s['walk_turn_s']:7.1f} {s['excluded_s']:7.1f} {s['hip_roll_abs_p95_deg']:7.1f} {100*s['clipped_frac_per_joint_max']:6.1f} {str(s['knee_p95_walk_deg']):>7s}")
print("TOTAL walking kept: straight %.0f s, turning %.0f s, excluded %.0f s" % (tot["walk_straight_s"], tot["walk_turn_s"], tot["excluded_s"]))
