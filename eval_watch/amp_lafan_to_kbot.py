"""LAFAN1 (Unitree G1 retarget, 30 fps CSV) -> K-Bot leg joints at 50 Hz, with turning analysis.

CSV row = root xyz + quat (x y z w) + 29 G1 joints (README order). We take the 12 leg joints,
drop ankle roll, map signs by joint-axis convention (G1 axes are +y/+x/+z/+y/+y in the pelvis
frame; ours from the in-sim test: right hip_pitch + = +y, left + = -y; roll + = +x both;
yaw + = +z both; right knee/ankle + = -y, left + = +y), time-scale by the Froude ratio of leg
lengths (G1 0.63 m vs ours 0.65 m -> x1.015, i.e. native), resample to 50 Hz and write
eval_watch/amp_refs/lafan1/<clip>_kbot.npz in the motion-file layout (joint_pos, joint_vel,
fps, joint_names, base_lin_vel_b, base_yaw_rate). Prints per-clip speed/turn statistics and
the turning segments (|yaw rate| >= 0.3 rad/s for >= 1 s) that pilot 1's dataset lacks.
"""
import csv, glob, json, math, os, sys
import numpy as np

D = "/home/faisal/IsaacLab/eval_watch/lafan1_g1"
OUT = "/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1"
OUR = ["dof_left_hip_pitch_04", "dof_right_hip_pitch_04", "dof_left_hip_roll_04", "dof_right_hip_roll_04", "dof_left_hip_yaw_03",
       "dof_right_hip_yaw_03", "dof_left_knee_04", "dof_right_knee_04", "dof_left_ankle_02", "dof_right_ankle_02"]
# G1 CSV columns after the 7 root values: L hip_pitch, roll, yaw, knee, ankle_pitch, ankle_roll, R hip_pitch, roll, yaw, knee, ankle_pitch, ankle_roll
G1_COL = {"left_hip_pitch": 7, "left_hip_roll": 8, "left_hip_yaw": 9, "left_knee": 10, "left_ankle": 11,
          "right_hip_pitch": 13, "right_hip_roll": 14, "right_hip_yaw": 15, "right_knee": 16, "right_ankle": 17}
MAP = [("left_hip_pitch", -1), ("right_hip_pitch", +1), ("left_hip_roll", +1), ("right_hip_roll", +1), ("left_hip_yaw", +1),
       ("right_hip_yaw", +1), ("left_knee", +1), ("right_knee", -1), ("left_ankle", +1), ("right_ankle", -1)]
FPS_IN, FPS_OUT, TIME_SCALE = 30.0, 50.0, 1.015  # clip plays 1.5% slower on our (longer) legs


def quat_to_yaw(q):  # xyzw
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_rpy(q):  # xyzw -> roll (about x), pitch (about y, + = nose down), yaw (ZYX)
    x, y, z, w = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


# POSTURE FIX (2026-10-01, KBOT_LAFAN_POSTURE=1): the puppet showed the hips sitting BEHIND the feet
# (stance foot 5-16 cm in front of the hip vs 3 cm behind in the proven stride), so a tracker had to
# over-step to balance. Two causes, both folded into the joints here:
#   * the G1's thigh is tilted 9.1 deg backward at its zero pose (knee origin 0.053 m behind the hip
#     pitch axis over 0.332 m; URDF), ours hangs vertical -> hip AND knee flexion reduced by that angle;
#   * the mocap pelvis pitch/roll was dropped (only yaw kept) -> folded into hip pitch/roll per frame so
#     the legs keep their orientation relative to gravity under our upright base.
POSTURE = os.environ.get("KBOT_LAFAN_POSTURE", "0") == "1"
G1_THIGH_TILT = np.arctan2(0.0533, 0.3319)  # rad, from g1_29dof_rev_1_0.urdf


summary = {}
for path in sorted(glob.glob(D + "/walk*.csv")):
    name = os.path.basename(path)[:-4]
    rows = np.array([[float(v) for v in r] for r in csv.reader(open(path))], dtype=np.float64)
    T_in = rows.shape[0]
    q_g1 = np.stack([rows[:, G1_COL[j]] * s for j, s in MAP], axis=1)  # (T, 10) our order, our signs, radians
    root_p, root_q = rows[:, 0:3], rows[:, 3:7]
    yaw = np.unwrap(quat_to_yaw(root_q))
    if POSTURE:
        r_p, p_p, _ = quat_to_rpy(root_q)
        # our signs: left hip pitch + = flexion, right hip pitch - = flexion; knee: left + / right - = flexion
        q_g1[:, 0] -= G1_THIGH_TILT + p_p; q_g1[:, 1] += G1_THIGH_TILT + p_p
        q_g1[:, 6] -= G1_THIGH_TILT;       q_g1[:, 7] += G1_THIGH_TILT
        q_g1[:, 2] += r_p;                 q_g1[:, 3] += r_p
    t_in = np.arange(T_in) / FPS_IN * TIME_SCALE
    t_out = np.arange(0, t_in[-1], 1 / FPS_OUT)
    q = np.stack([np.interp(t_out, t_in, q_g1[:, j]) for j in range(10)], 1)
    p = np.stack([np.interp(t_out, t_in, root_p[:, k]) for k in range(3)], 1)
    yw = np.interp(t_out, t_in, yaw)
    qd = np.gradient(q, 1 / FPS_OUT, axis=0)
    v_w = np.gradient(p[:, :2], 1 / FPS_OUT, axis=0)
    c, s = np.cos(yw), np.sin(yw)
    v_b = np.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1]], 1)
    yaw_rate = np.gradient(yw, 1 / FPS_OUT)
    # smooth the rates a little (mocap jitter) for the statistics only
    k = 15
    sm = lambda a: np.convolve(a, np.ones(k) / k, mode="same")
    vb_s, yr_s = sm(v_b[:, 0]), sm(yaw_rate)
    turning = np.abs(yr_s) >= 0.3
    segs, start = [], None
    for i, t in enumerate(turning):
        if t and start is None: start = i
        if (not t or i == len(turning) - 1) and start is not None:
            if i - start >= FPS_OUT: segs.append((start / FPS_OUT, i / FPS_OUT, float(np.mean(yr_s[start:i])), float(np.mean(vb_s[start:i]))))
            start = None
    np.savez(os.path.join(OUT, f"{name}_kbot.npz"), joint_names=np.array(OUR), fps=FPS_OUT, joint_pos=q.astype(np.float32), joint_vel=qd.astype(np.float32),
             base_lin_vel_b=v_b.astype(np.float32), base_yaw_rate=yaw_rate.astype(np.float32), base_pos=p.astype(np.float32), base_yaw=yw.astype(np.float32),
             source="LAFAN1 G1 retarget (lvhaidong/LAFAN1_Retargeting_Dataset), CC BY-NC-ND 4.0", time_scale=TIME_SCALE,
             posture_fix=("thigh-tilt 9.1 deg + pelvis pitch/roll folded into hips" if POSTURE else "none"))
    deg = np.degrees(q)
    summary[name] = {"dur_s": round(len(t_out) / FPS_OUT, 1), "fwd_speed_p50": round(float(np.percentile(vb_s, 50)), 2), "fwd_speed_p90": round(float(np.percentile(vb_s, 90)), 2),
                     "frac_in_our_band_0.1-0.55": round(float(np.mean((vb_s > 0.1) & (vb_s < 0.55))), 2), "frac_turning": round(float(turning.mean()), 2),
                     "turn_segments_n": len(segs), "turn_segments_s": round(sum(b - a for a, b, _, _ in segs), 1),
                     "knee_p5_p95": [round(float(np.percentile(deg[:, 6], 5)), 1), round(float(np.percentile(deg[:, 6], 95)), 1)],
                     "hip_roll_abs_p95": round(float(np.percentile(np.abs(deg[:, 2:4]), 95)), 1), "segments": [(round(a, 1), round(b, 1), round(w, 2), round(v, 2)) for a, b, w, v in segs[:6]]}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(f"{'clip':16s} {'dur':>6s} {'v50':>5s} {'v90':>5s} {'in-band':>7s} {'turn%':>6s} {'#segs':>5s} {'seg s':>6s} {'knee p5..p95':>14s} {'roll95':>6s}")
for n, s in summary.items():
    print(f"{n:16s} {s['dur_s']:6.1f} {s['fwd_speed_p50']:5.2f} {s['fwd_speed_p90']:5.2f} {s['frac_in_our_band_0.1-0.55']:7.2f} {s['frac_turning']:6.2f} {s['turn_segments_n']:5d} {s['turn_segments_s']:6.1f} {str(s['knee_p5_p95']):>14s} {s['hip_roll_abs_p95']:6.1f}")
print("example turning segments (start s, end s, yaw rate rad/s, fwd m/s):", summary["walk1_subject1"]["segments"])
