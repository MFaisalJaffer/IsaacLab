"""Windowed means of the scalars rsl_rl prints per iteration (training log on stdout).

  python3 eval_watch/train_trend.py <log> [--window 100] [--start 0] [--keys "Mean reward,Mean episode length,..."]

Default keys cover the obstacle course run; any "name: value" line of the iteration block can be listed.
"""
import argparse
import re

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--window", type=int, default=100)
p.add_argument("--start", type=int, default=0)
p.add_argument("--keys", default="Mean reward,Mean episode length,Mean action noise std,Mean value_function loss,Episode_Reward/track_lin_vel_xy_exp,"
               "Episode_Reward/ref_foot_lift,Episode_Reward/feet_stumble,Episode_Reward/leg_contact,Curriculum/obstacle_levels/height_cm_platform,"
               "Curriculum/obstacle_levels/cross_platform,Curriculum/obstacle_levels/height_cm_beam,Curriculum/obstacle_levels/cross_beam,"
               "Episode_Termination/time_out,Episode_Termination/bad_orientation,Episode_Termination/base_height,Episode_Termination/root_drift,Iteration time")
a = p.parse_args()
t = re.sub(r"\x1b\[[0-9;]*m", "", open(a.log, errors="replace").read())
parts = re.split(r"Learning iteration (\d+)/\d+", t)
its = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
keys = [k.strip() for k in a.keys.split(",") if k.strip()]


def val(b, key):
    m = re.search(r"^\s*" + re.escape(key) + r":\s+(-?[0-9.]+(?:e-?\d+)?)", b, re.M)
    return float(m.group(1)) if m else float("nan")


ks = sorted(its)
if not ks:
    raise SystemExit("no iterations in the log yet")
wins = [(s, [k for k in ks if s <= k < s + a.window]) for s in range(a.start - a.start % a.window, ks[-1] + 1, a.window)]
wins = [(s, w) for s, w in wins if w]
short = lambda k: k.replace("Episode_Reward/", "R/").replace("Episode_Termination/", "end/").replace("Curriculum/obstacle_levels/", "obst/").replace("Mean ", "")
print(f"iterations {ks[0]}..{ks[-1]}")
print(f"{'':30s}" + "".join(f"{str(s) + '+':>9s}" for s, _ in wins))
for key in keys:
    row = [sum(val(its[k], key) for k in w) / len(w) for _, w in wins]
    if all(v != v for v in row):
        continue
    print(f"{short(key)[:30]:30s}" + "".join(f"{v:9.3f}" for v in row))
