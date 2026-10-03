#!/usr/bin/env python3
"""Side-by-side: our MuJoCo rig (real weld) vs Isaac (fix_root_link), all three batteries.

Isaac numbers are hardcoded from training's corrected reply of 2026-08-26 (their
Bug-1 fix). Ankle rows are motor-side on both. Their friction fit spanned 10-80 deg/s
and is contaminated the same way ours was (the 40/80 points never reach steady
state), so hip Coulomb is flagged as not-yet-comparable; yaw is clean on both.

usage: compare_sim2sim.py step.json fric_lo.json [bare.json]
"""
import json
import math
import sys

TRAIN_STEP = {  # (rise_ms, overshoot_pct) per direction; frac; leak
    "dof_right_hip_pitch_04": {"pos": (95, 25.1), "neg": (100, 25.2), "frac": 0.917, "leak": 2.00},
    "dof_left_hip_pitch_04":  {"pos": (95, 25.1), "neg": (95, 24.9),  "frac": 0.937, "leak": 2.00},
    "dof_right_hip_roll_04":  {"pos": (75, 18.7), "neg": (80, 31.6),  "frac": 0.970, "leak": 0.24},
    "dof_left_hip_roll_04":   {"pos": (80, 31.9), "neg": (70, 20.0),  "frac": 0.942, "leak": 0.24},
    "dof_right_hip_yaw_03":   {"pos": (25, 0.1),  "neg": (25, 0.1),   "frac": 0.999, "leak": 0.20},
    "dof_left_hip_yaw_03":    {"pos": (25, 0.1),  "neg": (25, 0.1),   "frac": 0.999, "leak": 0.20},
    "dof_right_knee_04":      {"neg": (22.5, 17.7), "frac": 0.995, "leak": 3.6},
    "dof_left_knee_04":       {"pos": (22.5, 17.7), "frac": 0.995, "leak": 3.6},
    "dof_right_ankle_02":     {"pos": (20, 26.2), "neg": (20, 26.2), "frac": 1.000, "leak": 0.7},
    "dof_left_ankle_02":      {"pos": (20, 26.2), "neg": (20, 26.2), "frac": 1.000, "leak": 0.7},
}
TRAIN_FRIC = {  # coulomb Nm (their linear fit), viscous Nm/(deg/s)
    "hip_pitch": (2.90, 0.0375, "FIT over 10-80: contaminated, not comparable"),
    "hip_roll":  (3.06, 0.0375, "FIT over 10-80: contaminated, not comparable"),
    "hip_yaw":   (0.400, 0.0052, "clean"),
    "knee":      (None, None, "dropped both sides (velocity control fails)"),
    "ankle":     (0.056, None, "series path - uninterpretable both sides"),
}
TRAIN_BARE = {"hip_pitch": (1.360, 63.5), "hip_roll": (1.333, 66.3), "knee": (1.098, 22.8),
              "hip_yaw": (None, None), "ankle": (None, None)}
KP_YAW = 60.0


def fam(j):
    for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle"):
        if k in j:
            return k


def pct(a, b):
    return 100.0 * (a / b - 1.0) if (a is not None and b) else float("nan")


step = json.load(open(sys.argv[1]))
fric = json.load(open(sys.argv[2]))
bare = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None

print("=" * 96)
print("BATTERY 1 - STEP RESPONSE   (rig = real weld, PIN_Z 1.6 | isaac = fix_root_link)   rig / isaac")
print("=" * 96)
print("%-18s %-4s %18s %18s %16s %14s" % ("joint", "dir", "rise ms", "overshoot %", "frac", "leak deg"))
rise_dev, over_dev, frac_dev, leak_dev = [], [], [], []
for j in step["joint_order"]:
    r = step["results"][j]; t = TRAIN_STEP[j]
    for d in ("pos", "neg"):
        if d not in r or d not in t:
            continue
        m = r[d]; tr, to = t[d]
        rise = m.get("rise_10_90_ms"); over = m.get("overshoot_pct")
        fr = m["reached_deg"] / m["target_deg"]
        rp = pct(rise, tr); op = (over - to) if over is not None else float("nan")
        rise_dev.append(rp); over_dev.append(op); frac_dev.append(fr - t["frac"])
        print("%-18s %-4s %7s / %-6s %+4.0f%%  %6.1f / %-5.1f %+5.1f  %6.3f / %-6.3f %+6.3f  %5.2f / %-5.2f"
              % (j.replace("dof_", "")[:18], d, ("%.0f" % rise) if rise else "-", tr, rp,
                 over, to, op, fr, t["frac"], fr - t["frac"], r["max_leak_deg"], t["leak"]))
    leak_dev.append(r["max_leak_deg"] - t["leak"])
ok = lambda v, tol: all(abs(x) <= tol for x in v if x == x)
print("\n  rise time : rig slower by %+.0f%% .. %+.0f%% (mean %+.0f%%)  -> %s"
      % (min(rise_dev), max(rise_dev), sum(rise_dev) / len(rise_dev),
         "within +-20%" if ok(rise_dev, 20) else "OUTSIDE +-20% on some joints"))
print("  overshoot : max |delta| %.1f pts -> %s" % (max(abs(x) for x in over_dev), "within 5 pts" if ok(over_dev, 5) else "OUTSIDE 5 pts"))
print("  frac      : max |delta| %.3f     -> %s" % (max(abs(x) for x in frac_dev), "within 0.03" if ok(frac_dev, 0.03) else "OUTSIDE 0.03"))
print("  leak      : max |delta| %.2f deg -> %s" % (max(abs(x) for x in leak_dev), "within 1 deg" if ok(leak_dev, 1.0) else "OUTSIDE 1 deg"))

print("\n" + "=" * 96)
print("BATTERY 3 - FRICTION (rig = low-speed 5-40 sweep, real weld)   rig / isaac")
print("=" * 96)
print("%-18s %22s %26s   %s" % ("joint", "Coulomb Nm", "viscous Nm/(deg/s)", "comparability"))
for j in fric["joint_order"]:
    r = fric["results"][j]; f = fam(j); tc, tv, note = TRAIN_FRIC[f]
    c = r.get("coulomb_Nm"); v = r.get("viscous_Nm_per_deg_s")
    lo = next((p["friction_Nm"] for p in r["points"] if p["control_pass"]), None)
    print("%-18s %7s / %-6s  (lowest-speed pt %s)   %7s / %-7s   %s"
          % (j.replace("dof_", "")[:18], ("%.3f" % c) if c is not None else "-", tc if tc is not None else "-",
             ("%.2f" % lo) if lo is not None else "-",
             ("%.4f" % v) if v is not None else "-", tv if tv is not None else "-", note))
ry = fric["results"]["dof_right_hip_yaw_03"].get("viscous_Nm_per_deg_s")
if ry:
    print("\n  LATENCY from yaw viscous slope (configured viscous = 0, so slope = kp * delay):")
    print("    rig   : %.4f Nm/(deg/s) * 57.3 / %g = %.1f ms" % (ry, KP_YAW, 1000 * ry * 57.2958 / KP_YAW))
    print("    isaac : 0.0052 * 57.3 / 60 = %.1f ms   (their configured yaw delay: 1-3 steps @ 5 ms)"
          % (1000 * 0.0052 * 57.2958 / KP_YAW))

if bare:
    print("\n" + "=" * 96)
    print("BARE PLANT - free swing, others PD-held x%.1f   rig / isaac" % bare["hold_scale"])
    print("=" * 96)
    print("%-18s %20s %22s %8s %10s" % ("joint", "period s", "peak deg/s", "tau", "tilt"))
    for j in bare["joint_order"]:
        r = bare["results"][j]; tp, tk = TRAIN_BARE[fam(j)]
        per = r.get("period_s"); pk = r.get("peak_qd_deg_s")
        print("%-18s %7s / %-6s %+5s  %7.1f / %-5s %+5s  %7.4f %8.2f°"
              % (j.replace("dof_", "")[:18], ("%.3f" % per) if per else "-", tp if tp else "-",
                 ("%.0f%%" % pct(per, tp)) if (per and tp) else "", pk, tk if tk else "-",
                 ("%.0f%%" % pct(pk, tk)) if (pk and tk) else "",
                 r.get("mean_abs_tau_Nm") or 0, r.get("tilt_max_deg") or 0))
    print("\n  rig rigid-leg analytic period (mass model) 1.345 s; clamped 1.333 s.")
    print("  Isaac PD-held 1.360 sits on our RIGID value -> ask: how were THEIR other joints held?")
