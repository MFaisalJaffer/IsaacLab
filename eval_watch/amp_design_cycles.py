"""Clean one-period reference cycles for the multi-cycle tracker (option 2, 2026-10-01).

All cycles share the proven stride's period (1.06 s) and sample count (50), so ONE gait clock serves
every direction and the tracker's observation is exactly the main walker's (command + clock).

  * forward        — asimov_walk_cycle.npz as is (0.41 m/s).
  * backward       — the same stride reversed in time.
  * side_left      — designed on the stride's double-support posture: left leg lifts and abducts, plants,
                     weight shifts (rolls swap), right leg lifts and closes. Small roll (pitch-only ankles).
  * pivot_left     — designed turn in place (counter-clockwise): the pelvis rotates steadily, each planted
                     foot's hip yaw runs +a -> -a through its stance and swings back while lifted; the tiny
                     fore/aft foot offsets that a rotating pelvis implies go into hip pitch.
  * side_right / pivot_right — left/right mirrors, shifted half a period (mirror symmetry of the clock).

Signs (measured in sim): left hip pitch + = flexion, right - = flexion; left knee + / right - = flexion;
left roll + / right - = abduction; hip yaw + = counter-clockwise for both legs.
cycle_base_z and the kinematic velocity are filled by eval_watch/amp_cycle_ground.py (puppet pass).

  python eval_watch/amp_design_cycles.py [--roll_deg 6 --knee_deg 20 --hip_deg 7 --yaw_deg 6]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

A = "/home/faisal/IsaacLab/eval_watch/amp_refs"
PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]  # L/R mirror: swap pairs, negate

p = argparse.ArgumentParser()
p.add_argument("--roll_deg", type=float, default=12.0, help="side-step abduction amplitude (user 21:55: use more hip roll; was 6)")
p.add_argument("--knee_deg", type=float, default=45.0, help="swing knee flexion bump (20 deg gave 2 cm clearance)")
p.add_argument("--hip_deg", type=float, default=22.0, help="swing hip flexion bump (about half the knee: foot stays under the hip)")
p.add_argument("--stand_x", type=float, default=0.072, help="how far the stride's double-support posture holds the feet BEHIND the hips (m); removed for zero-speed cycles")
p.add_argument("--yaw_deg", type=float, default=12.0, help="pivot: hip yaw half-range a, stance runs +a -> -a (user 21:55: use more hip yaw; was 6)")
p.add_argument("--stance_w", type=float, default=0.30, help="stance width (m) for the pivot's fore/aft offsets")
a = p.parse_args()

src = np.load(os.path.join(A, "asimov_walk_cycle.npz"), allow_pickle=True)
jn = src["joint_names"]; q = src["cycle_q"].astype(np.float64); bz = src["cycle_base_z"].astype(np.float64)
P = float(src["period_s"]); v = float(src["speed_mps"]); N = q.shape[0]
LEG = 0.96


def circ_gradient(x: np.ndarray, dt: float) -> np.ndarray:
    pad = np.concatenate([x[-2:], x, x[:2]])
    return np.gradient(pad, dt, axis=0)[2:-2]


def save(name: str, qs: np.ndarray, base_z: np.ndarray, vel: tuple, derived: str) -> None:
    qd = circ_gradient(qs, P / N)
    np.savez(os.path.join(A, f"{name}.npz"), joint_names=jn, cycle_q=qs.astype(np.float32), cycle_qd=qd.astype(np.float32),
             cycle_base_z=base_z.astype(np.float32), period_s=P, speed_mps=float(vel[0]), vel_b=np.array(vel[:2], np.float32),
             wz=float(vel[2]), n_strides=0, derived=derived)
    print(f"[{name}] {N} samples, period {P:.2f} s, nominal vel (vx {vel[0]:.3f}, vy {vel[1]:.3f}, wz {vel[2]:.3f}); "
          f"ranges deg hipP {np.degrees(qs[:, 0]).min():.0f}..{np.degrees(qs[:, 0]).max():.0f} roll {np.degrees(qs[:, 2]).min():.0f}..{np.degrees(qs[:, 2]).max():.0f} "
          f"yaw {np.degrees(qs[:, 4]).min():.0f}..{np.degrees(qs[:, 4]).max():.0f} knee {np.degrees(qs[:, 6]).min():.0f}..{np.degrees(qs[:, 6]).max():.0f}")


def mirror_shift(qs: np.ndarray) -> np.ndarray:
    """left/right mirror, shifted half a period: ref_mirror(phi) = M ref(phi - pi)."""
    return np.roll(-qs[:, PERM], N // 2, axis=0)


# ---------------------------------------------------------------- backward = time reversal of the stride
save("asimov_walk_cycle_rev", q[::-1].copy(), bz[::-1].copy(), (-v, 0.0, 0.0), "time reversal of asimov_walk_cycle.npz")

# ---------------------------------------------------------------- shared pieces for the designed cycles
i0 = int(np.argmin(np.abs(q[:, 6]) + np.abs(q[:, 7])))  # double support: both knees most extended
q0 = q[i0]; q0s = 0.5 * (q0 + (-q0[PERM]))              # symmetrised standing posture
# the stride posture leans forward (stance foot 7 cm behind the hip): right for walking, wrong for cycles that
# do not travel forward. Pivot both legs about the hips so the feet sit under them; ankles re-levelled (+ sign).
dlt = np.arcsin(a.stand_x / LEG)
q0s[0] += dlt; q0s[1] -= dlt; q0s[8] += dlt; q0s[9] -= dlt
print("[base posture deg]", np.degrees(q0s).round(1).tolist(), f"(stride frame {i0}, pivoted {np.degrees(dlt):.1f} deg to stand over the feet)")
u = np.arange(N) / N
K, H = np.radians(a.knee_deg), np.radians(a.hip_deg)
SW_L, SW_R = (0.00, 0.30), (0.50, 0.80)   # swing windows, half a period apart like the stride


def bump(ph, lo, hi):
    x = np.clip((ph - lo) / (hi - lo), 0.0, 1.0)
    return np.sin(np.pi * x) * ((ph >= lo) & (ph < hi))


def ramp(ph, lo, hi):
    x = np.clip((ph - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lifts(qs: np.ndarray) -> None:
    qs[:, 6] += K * bump(u, *SW_L); qs[:, 0] += H * bump(u, *SW_L)   # left: knee + / hip + = flexion
    qs[:, 7] -= K * bump(u, *SW_R); qs[:, 1] -= H * bump(u, *SW_R)   # right: knee - / hip - = flexion


bz0 = np.full(N, float(bz.max()))

# ---------------------------------------------------------------- side-step left
R = np.radians(a.roll_deg)
qs = np.tile(q0s, (N, 1)); lifts(qs)
qs[:, 2] += R * (ramp(u, 0.05, 0.30) - ramp(u, 0.30, 0.50))   # left abducts in swing, returns as the weight shifts
qs[:, 3] -= R * (ramp(u, 0.30, 0.50) - ramp(u, 0.55, 0.80))   # right ends up abducted, closes in its swing
vy = LEG * np.sin(R) / P
save("sidestep_left_cycle", qs, bz0, (0.0, vy, 0.0), f"designed side-step: roll {a.roll_deg} deg, knee {a.knee_deg}, hip {a.hip_deg}")
save("sidestep_right_cycle", mirror_shift(qs), bz0, (0.0, -vy, 0.0), "mirror of sidestep_left_cycle (half-period shift)")

# ---------------------------------------------------------------- pivot left (counter-clockwise)
Y = np.radians(a.yaw_deg)


def saw(ph, sw):
    """hip yaw offset of a foot: swings -Y -> +Y while lifted (window sw), then +Y -> -Y linearly through stance."""
    lo, hi = sw
    x = np.mod(ph - lo, 1.0); d = hi - lo
    return np.where(x < d, -Y + 2 * Y * ramp(x, 0.0, d), Y - 2 * Y * (x - d) / (1.0 - d))


psi_l, psi_r = saw(u, SW_L), saw(u, SW_R)
qs = np.tile(q0s, (N, 1)); lifts(qs)
qs[:, 4] += psi_l; qs[:, 5] += psi_r
# a pelvis rotated past a planted foot sees that foot fore/aft: x_left = -(w/2) sin(psi), x_right = +(w/2) sin(psi)
qs[:, 0] += np.arcsin(np.clip(-(a.stance_w / 2) * np.sin(psi_l) / LEG, -1, 1))   # left: + = flexion (foot ahead)
qs[:, 1] -= np.arcsin(np.clip((a.stance_w / 2) * np.sin(psi_r) / LEG, -1, 1))    # right: - = flexion
wz = 2 * Y / ((1.0 - (SW_L[1] - SW_L[0])) * P)
save("pivot_left_cycle", qs, bz0, (0.0, 0.0, wz), f"designed pivot: hip yaw +-{a.yaw_deg} deg, knee {a.knee_deg}, hip {a.hip_deg}")
save("pivot_right_cycle", mirror_shift(qs), bz0, (0.0, 0.0, -wz), "mirror of pivot_left_cycle (half-period shift)")
