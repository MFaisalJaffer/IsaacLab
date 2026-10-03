"""Assemble the multi-cycle tracker's library from the single-cycle files (after amp_cycle_ground.py):
eval_watch/amp_refs/multicycle_v1.npz with names, cycle_q/qd (K, N, J), cycle_base_z (K, N), cmd (K, 3),
weights, period_s. Commands are each cycle's measured kinematic velocity, cleaned to its primary component
(forward/backward: vx; side-steps: vy; pivots: wz) and made equal in magnitude across mirror pairs.

  python eval_watch/amp_build_multicycle.py
"""
import numpy as np

R = "/home/faisal/IsaacLab/eval_watch/amp_refs"
ITEMS = [("forward", "asimov_walk_cycle.npz", 0.20, 0), ("backward", "asimov_walk_cycle_rev.npz", 0.20, 0),
         ("side_left", "sidestep_left_cycle.npz", 0.15, 1), ("side_right", "sidestep_right_cycle.npz", 0.15, 1),
         ("pivot_left", "pivot_left_cycle.npz", 0.15, 2), ("pivot_right", "pivot_right_cycle.npz", 0.15, 2)]
qs, qds, bzs, raw, names, ws, dirs, feet = [], [], [], [], [], [], [], []
P = None; jn = None
for name, f, w, _ in ITEMS:
    d = np.load(f"{R}/{f}", allow_pickle=True)
    P = P or float(d["period_s"])
    assert abs(float(d["period_s"]) - P) < 1e-6 and d["cycle_q"].shape == (50, 10), f
    jn = jn if jn is not None else d["joint_names"]
    assert list(d["joint_names"]) == list(jn)
    vb = [float(x) for x in d["vel_b"]] if "vel_b" in d else [float(d["speed_mps"]), 0.0]
    raw.append([vb[0], vb[1], float(d["wz"]) if "wz" in d else 0.0])
    if name == "backward":  # the forward stride with the gait clock running backward (same poses, same clock mapping)
        f0 = np.load(f"{R}/asimov_walk_cycle.npz", allow_pickle=True)
        qs.append(f0["cycle_q"]); qds.append(f0["cycle_qd"]); bzs.append(f0["cycle_base_z"]); dirs.append(-1.0); feet.append(f0["cycle_feet_rel"])
    else:
        qs.append(d["cycle_q"]); qds.append(d["cycle_qd"]); bzs.append(d["cycle_base_z"]); dirs.append(1.0); feet.append(d["cycle_feet_rel"])
    names.append(name); ws.append(w)
raw = np.array(raw, np.float32)
cmd = np.zeros_like(raw)
for i, (_, _, _, k) in enumerate(ITEMS):
    cmd[i, k] = raw[i, k]
for a, b in ((2, 3), (4, 5)):  # mirror pairs: same magnitude
    k = ITEMS[a][3]; m = 0.5 * (abs(cmd[a, k]) + abs(cmd[b, k]))
    cmd[a, k] = np.sign(cmd[a, k]) * m; cmd[b, k] = np.sign(cmd[b, k]) * m
np.savez(f"{R}/multicycle_v1.npz", names=np.array(names), cycle_q=np.stack(qs).astype(np.float32), cycle_qd=np.stack(qds).astype(np.float32),
         cycle_base_z=np.stack(bzs).astype(np.float32), cmd=cmd, weights=np.array(ws, np.float32), clock_dir=np.array(dirs, np.float32), feet=np.stack(feet).astype(np.float32), period_s=P, joint_names=jn)
for n, c, r in zip(names, cmd, raw):
    print(f"  {n:12s} cmd vx {c[0]:6.3f} vy {c[1]:6.3f} wz {c[2]:6.3f}   (measured {r.round(3).tolist()})")
print("clock_dir:", dirs)
print("wrote multicycle_v1.npz")
