"""Build K-Bot reference-motion files from a replayed clip (output of amp_replay_clip.py).

Produces, under eval_watch/amp_refs/:
  <tag>_kbot.npz    the whole clip in OUR joint order/signs at 50 Hz (for AMP later)
  <tag>_cycle.npz   one canonical stride: phase-averaged over clean strides inside
                    --window, then L/R-symmetrised with the same mirror the rsl_rl
                    symmetry loss uses (swap L/R, negate). Indexed by the LEFT-leg
                    phase exactly as mdp_gait._phase defines it (phi_l = 0 at reset,
                    phi_r = phi_l + pi), so a tracking env can look up q_ref(phi_l).

Stride segmentation: left-foot touchdown = the left foot drops below 5 mm above the
frame's lowest foot after having been higher than 2 cm (the same "lowest foot is the
planted foot" convention the replay used to move the base).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--frames", default="eval_watch/amp_ref_asimov_frames.npz")
p.add_argument("--tag", default="asimov_walk")
p.add_argument("--window", default="12:32", help="seconds of steady straight walking to average over")
p.add_argument("--n_phase", type=int, default=50, help="samples per cycle in the output")
p.add_argument("--out_dir", default="eval_watch/amp_refs")
a = p.parse_args()

PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]  # symmetry.py: swap L/R, negate


def mirror(x: np.ndarray) -> np.ndarray:
    return -x[..., PERM]


d = np.load(a.frames, allow_pickle=True)
fps = float(d["fps"])
q, qd = d["joint_pos"], d["joint_vel"]
foot = d["foot_rel"]              # (T, 2, 3)  index 0 = right foot, 1 = left foot (replay's fi order)
base, yaw = d["base"], d["base_yaw"]
jn = [str(s) for s in d["joint_names"]]
T = q.shape[0]
os.makedirs(a.out_dir, exist_ok=True)

# ---- whole clip in our joint space ----
v_w = np.gradient(base[:, :2], 1.0 / fps, axis=0)
c, s = np.cos(yaw), np.sin(yaw)
v_b = np.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1]], 1)  # base-frame (fwd, left)
yaw_rate = np.gradient(np.unwrap(yaw), 1.0 / fps)
np.savez(os.path.join(a.out_dir, f"{a.tag}_kbot.npz"), joint_names=np.array(jn), fps=fps, joint_pos=q, joint_vel=qd,
         base_lin_vel_b=v_b, base_yaw_rate=yaw_rate, base_pos=base, base_yaw=yaw)

# ---- stride segmentation on the left foot ----
z = foot[:, :, 2]
h = z - z.min(axis=1, keepdims=True)  # height above the planted foot
lo, hi = (float(x) for x in a.window.split(":"))
t0, t1 = int(lo * fps), int(hi * fps)
left = h[:, 1]
touch = []
airborne = False
for t in range(t0, t1):
    if left[t] > 0.02:
        airborne = True
    elif airborne and left[t] < 0.005:
        touch.append(t)
        airborne = False
touch = np.array(touch)
periods = np.diff(touch) / fps
print(f"[strides] {len(touch) - 1} strides in {lo:.0f}-{hi:.0f} s, period {periods.mean():.3f} +- {periods.std():.3f} s "
      f"(min {periods.min():.2f}, max {periods.max():.2f})")
keep = np.abs(periods - np.median(periods)) < 0.15 * np.median(periods)
print(f"[strides] keeping {keep.sum()} strides within 15% of the median period")

# ---- phase-average ----
N = a.n_phase
cyc = np.zeros((N, q.shape[1]))
cyc_z = np.zeros(N)
stride_len = []
n_used = 0
for i in range(len(touch) - 1):
    if not keep[i]:
        continue
    s0, s1 = touch[i], touch[i + 1]
    src_t = np.linspace(s0, s1, N, endpoint=False)
    for j in range(q.shape[1]):
        cyc[:, j] += np.interp(src_t, np.arange(T), q[:, j])
    cyc_z += np.interp(src_t, np.arange(T), base[:, 2])
    stride_len.append(np.linalg.norm(base[s1, :2] - base[s0, :2]))
    n_used += 1
cyc /= n_used
cyc_z /= n_used
period = float(np.median(periods[keep]))
speed = float(np.mean(stride_len) / period)

# ---- symmetrise (right leg = mirrored left leg half a cycle later) ----
half = N // 2
cyc_sym = 0.5 * (cyc + mirror(np.roll(cyc, -half, axis=0)))
cyc_z = 0.5 * (cyc_z + np.roll(cyc_z, -half))
asym = np.degrees(np.abs(cyc - cyc_sym).max())
# ---- phase 0 = frame nearest the standing (all-zero) pose ----
k0 = int(np.argmin((cyc_sym ** 2).sum(axis=1)))
cyc_sym = np.roll(cyc_sym, -k0, axis=0)
cyc_z = np.roll(cyc_z, -k0)
cyc_qd = np.gradient(np.concatenate([cyc_sym[-2:], cyc_sym, cyc_sym[:2]]), period / N, axis=0)[2:-2]

# ---- diagnostics on the averaged window ----
seg = slice(t0, t1)
both_down = (h[seg, 0] < 0.01) & (h[seg, 1] < 0.01)
width_cm = float(np.median(np.abs(foot[seg, 0, 1] - foot[seg, 1, 1])[both_down]) * 100)
clear_cm = [float(h[seg, f].max() * 100) for f in range(2)]
np.savez(os.path.join(a.out_dir, f"{a.tag}_cycle.npz"), joint_names=np.array(jn), cycle_q=cyc_sym, cycle_qd=cyc_qd, cycle_base_z=cyc_z,
         period_s=period, speed_mps=speed, n_strides=n_used, phase0_index_in_raw=k0, max_asymmetry_deg=asym,
         stance_width_cm=width_cm, clearance_cm=np.array(clear_cm))
summary = {
    "period_s": period, "speed_mps": speed, "strides_used": n_used, "max_LR_asymmetry_removed_deg": float(asym),
    "stance_width_cm": width_cm, "clearance_cm": clear_cm,
    "base_z_range_m": [float(cyc_z.min()), float(cyc_z.max())],
    "cycle_range_deg": {jn[j]: [float(np.degrees(cyc_sym[:, j].min())), float(np.degrees(cyc_sym[:, j].max()))] for j in range(len(jn))},
    "pose_at_phase0_deg": {jn[j]: float(np.degrees(cyc_sym[0, j])) for j in range(len(jn))},
}
print(json.dumps(summary, indent=1))
with open(os.path.join(a.out_dir, f"{a.tag}_cycle_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(f"[out] {a.out_dir}/{a.tag}_kbot.npz  {a.out_dir}/{a.tag}_cycle.npz")
