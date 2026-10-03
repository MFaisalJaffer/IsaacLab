#!/bin/bash
# Gifs of a stage-1b checkpoint, heights picked from its own test result:
#   stairs at the tallest riser it climbs (>= 80% cross) and at the first riser where fewer than half cross,
#   platform and beam at the tallest height they climb.
# Waits for the autopilot's crossing test of that checkpoint, then renders (lock shared with the tests).
#   bash eval_watch/obstacle_render_auto_s1b.sh <N>
# -> eval_watch/obst_s1b_<N>_gif_<kind><level>.gif ; level k = (2 + 2k) cm
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
N=${1:?checkpoint iteration}
until grep -q -E "model_$N: crossing" eval_watch/OBSTACLE_S1B_STATUS.md; do sleep 15; done
PAIRS=$(python3 - "eval_watch/obst_s1b_$N.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["kinds"]
def climbs(rows):            # index of the tallest height that >= 80% cross, lower ones passing too (0 if none)
    k = 0
    for i, x in enumerate(rows):
        if x["crossed"] < 0.8:
            break
        k = i
    return k
def fails(rows):             # first height where fewer than half cross (None if it crosses them all)
    for i, x in enumerate(rows):
        if x["crossed"] < 0.5:
            return i
    return None
out = [f"stairs:{climbs(r['stairs'])}"]
f = fails(r["stairs"])
if f is not None and f != climbs(r["stairs"]):
    out.append(f"stairs:{f}")
out += [f"platform:{climbs(r['platform'])}", f"beam:{climbs(r['beam'])}"]
print(" ".join(out))
PY
)
echo "rendering stage-1b model_$N: $PAIRS"
OBST_ENV_FILE=eval_watch/obstacle_env_v2.sh OBST_RENDER_MIX=flat:0.10,platform:0.30,beam:0.30,stairs:0.30 bash eval_watch/obstacle_render.sh "logs/autopilot/obst_s1b_$N.pt" "obst_s1b_${N}_gif" $PAIRS
