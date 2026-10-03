#!/bin/bash
# Detached watcher for the multi-cycle tracker v2 (user away). Evaluates checkpoints as they appear, writes one
# line each to eval_watch/TRACKER_V2_STATUS.md and keeps the best one in archive_anchors/trackmulti_v2_best.pt.
# It never stops or launches a training run: evaluation only.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1
RUN="$1"
ST=$IL/eval_watch/TRACKER_V2_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
score() {  # prints "<score> | <summary>": score = mean over directions of survival x fraction of commanded speed achieved
python3 - "$1" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))["cycles"]
P = {"forward": 0, "backward": 0, "side_left": 1, "side_right": 1, "pivot_left": 2, "pivot_right": 2}
tot, parts = 0.0, []
for name, v in r.items():
    k = P[name]; c = v["cmd"][k]; a = v["achieved"][k]
    frac = min(1.0, max(0.0, a / c)) if abs(c) > 1e-6 else 0.0
    tot += v["survival"] * frac
    parts.append(f"{name} {a:+.2f}/{c:+.2f} surv {v['survival']:.2f} rmse {v['rmse_deg']:.1f}")
print(f"{tot / max(len(r), 1):.3f} | " + "; ".join(parts))
PY
}

echo "# TRACKER v2 watch $(date '+%Y-%m-%d %H:%M') — run $(basename "$RUN") (achieved/commanded: m/s for forward, backward, side; rad/s for pivot)" >> "$ST"
BEST=0
for N in 400 800 1200 1600 2000 2600 3200 4000 5000 6000 7000 8000 9000 9999; do
    CK="$RUN/model_$N.pt"; idle=0
    while [ ! -f "$CK" ]; do
        if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
        if [ "$idle" -ge 20 ]; then log "training process gone; newest checkpoint $(ls -t "$RUN"/model_*.pt | head -1 | xargs basename)"; exit 0; fi
        sleep 30
    done
    sleep 20
    cp "$CK" "$LOGD/tm_v2_$N.pt"
    env KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 KBOT_TM_ADAPT=0 ./isaaclab.sh -p eval_watch/amp_trackmulti_eval.py --checkpoint "$LOGD/tm_v2_$N.pt" --num_envs 256 --seconds 20 --tag "tm_eval_v2_$N" --headless > "$LOGD/tm_eval_v2_$N.log" 2>&1
    OUT=$(score "eval_watch/tm_eval_v2_$N.json")
    S=${OUT%% |*}; [ -n "$S" ] || { log "model_$N: evaluation failed (see $LOGD/tm_eval_v2_$N.log)"; continue; }
    log "model_$N: score $OUT"
    if python3 -c "import sys; sys.exit(0 if float('$S') > float('$BEST') else 1)"; then
        BEST=$S; cp "$CK" archive_anchors/trackmulti_v2_best.pt
        log "  -> new best (score $S): archive_anchors/trackmulti_v2_best.pt = model_$N"
    fi
done
log "DONE: all scheduled checkpoints evaluated; best score $BEST"
