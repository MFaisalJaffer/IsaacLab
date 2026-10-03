#!/bin/bash
# Gifs of an obstacle-run checkpoint, heights picked from its own test result:
#   platform at the tallest height it climbs (>= 80% cross), platform at the first height where fewer than half
#   cross (where it fails), beam at the tallest height it climbs.
# Waits for the autopilot's test of that checkpoint, then renders (lock shared with the tests).
#   bash eval_watch/obstacle_render_auto.sh <run: map|blind> <N>
# -> eval_watch/obst_s1_<run>_<N>_gif_<kind><level>.gif ; level k = (2 + 2k) cm
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
RUN=${1:?run}; N=${2:?checkpoint iteration}
until grep -q -E "$RUN model_$N: score" eval_watch/OBSTACLE_S1_STATUS.md; do sleep 15; done
PAIRS=$(python3 - "eval_watch/obst_s1_${RUN}_$N.json" <<'PY'
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
out = [f"platform:{climbs(r['platform'])}"]
f = fails(r["platform"])
if f is not None and f != climbs(r["platform"]):
    out.append(f"platform:{f}")
out.append(f"beam:{climbs(r['beam'])}")
print(" ".join(out))
PY
)
echo "rendering $RUN model_$N: $PAIRS"
[ "$RUN" = "blind" ] && export KBOT_OBST_BLIND=1
bash eval_watch/obstacle_render.sh "logs/autopilot/obst_s1_${RUN}_$N.pt" "obst_s1_${RUN}_${N}_gif" $PAIRS
