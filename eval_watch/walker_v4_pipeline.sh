#!/bin/bash
# WALKER v4 pipeline (user go 2026-10-02 08:45): the main walker, started from the multi-cycle tracker v2.
# Detached; progress in eval_watch/WALKER_V4_STATUS.md.
#   0. wait for the tracker run + its watcher to finish (one training at a time)
#   1. freeze the tracker's best checkpoint
#   2. record its motion (all six directions) as the judge's dataset
#   3. launch the walker: tracker weights, 10-frame history, signed clock, judge on forward/turn + the new
#      recording, commands over forward/backward/lateral/yaw (60% single-direction), tight velocity rewards,
#      fell-behind termination, pushes frozen (skills first, hardening later)
#   4. test checkpoints per direction, keep the best in archive_anchors/walker_v4_best.pt
# The run is named ..._walker_v4 so it is easy to find on TensorBoard (port 6007, logs/rsl_rl/kbot_legs_amp).
# It never stops a training run. Lineage 11 stays paused.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
ST=$IL/eval_watch/WALKER_V4_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
WARM=archive_anchors/trackmulti_v2_best_final.pt
ITERS=${WALKER_ITERS:-4000}
WALK_ENV="KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5 KBOT_AMP_MOTION_FILES=$DS_TURN,$DS_MULTI"
TRAIN_ENV="$WALK_ENV KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0 KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_INIT_STD=0.25"
c="train_am"; d="p.py"; PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
jget() { python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    r = r[k]
print(r)
PY
}

echo "# WALKER v4 pipeline $(date '+%Y-%m-%d %H:%M')" >> "$ST"
log "waiting for the tracker run and its watcher to finish"

# ---------------------------------------------------------------- 0. tracker done
idle=0
while true; do
    grep -q -E "DONE: all scheduled|training process gone" eval_watch/TRACKER_V2_STATUS.md && [ "$(gpu_n)" = "0" ] && break
    if [ "$(gpu_n)" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
    [ "$idle" -ge 12 ] && break   # 6 minutes with an idle GPU: the tracker and its tests are over
    sleep 30
done

# ---------------------------------------------------------------- 1. best tracker checkpoint
[ -f archive_anchors/trackmulti_v2_best.pt ] || { log "no tracker best checkpoint — stopping"; exit 1; }
cp archive_anchors/trackmulti_v2_best.pt "$WARM"
log "tracker finished; best checkpoint frozen as $WARM ($(grep 'new best' eval_watch/TRACKER_V2_STATUS.md | tail -n 1 | sed 's/.*-> //'))"

# ---------------------------------------------------------------- 2. record the judge's dataset
env KBOT_TM_ADAPT=0 KBOT_TM_DRIFT=0 ./isaaclab.sh -p eval_watch/amp_trackmulti_eval.py --checkpoint "$WARM" --num_envs 256 --seconds 40 --record "$DS_MULTI" --tag walker_v4_record --headless > "$LOGD/walker_v4_record.log" 2>&1
python3 - "$DS_MULTI" <<'PY' > "$LOGD/walker_v4_ds_check.txt" 2>&1
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("md", "/home/faisal/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/kbot_legs/amp/motion_dataset.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
ds = md.MotionDataset([sys.argv[1]], device="cpu", fps_expected=50.0, mirror=True)
d = np.load(sys.argv[1], allow_pickle=True)
names = [str(s) for s in d["cycle_names"]]; share = {n: round(float((d["cycle"] == i).mean()), 2) for i, n in enumerate(names)}
print(ds.describe(), "| share per direction:", share, "| pre-fall frames masked:", int(d["done"].sum() - d["done_raw"].sum()))
PY
[ -f "$DS_MULTI" ] || { log "recording FAILED — see $LOGD/walker_v4_record.log"; exit 1; }
log "dataset recorded: $(tail -n 1 "$LOGD/walker_v4_ds_check.txt" | cut -c1-360)"
log "recording per direction: $(grep -a -E '^(forward|backward|side_|pivot_)' "$LOGD/walker_v4_record.log" | awk '{printf "%s surv %s achieved %s/%s/%s; ", $1, $5, $10, $11, $12}')"

# ---------------------------------------------------------------- 3. launch the walker
[ "$(gpu_n)" = "0" ] || sleep 60
env $TRAIN_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$WARM" --run_name walker_v4 > "$LOGD/walker_v4.log" 2>&1 < /dev/null &
sleep 240
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_walker_v4/ 2>/dev/null | head -1)
echo "$LOGD/walker_v4.log" > logs/amp_pilot/CURRENT; echo "$LOGD/walker_v4.log" > logs/track_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
    log "WALKER v4 launched: $ITERS iterations; on TensorBoard (port 6007) as kbot_legs_amp/$(basename "$RUN") — $(grep -a -m1 '\[amp-env\] v4' "$LOGD/walker_v4.log" | cut -c1-260)"
else
    log "walker launch FAILED — see $LOGD/walker_v4.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/walker_v4.log" | cut -c1-200)"; exit 1
fi

# ---------------------------------------------------------------- 4. per-direction tests, best checkpoint
score_one() {  # $1 name, $2 command "vx:vy:wz", $3 result file -> prints "<score 0..1>|<text>"
python3 - "$1" "$2" "$3" <<'PY' 2>/dev/null
import json, os, sys
name, cmd, path = sys.argv[1], [float(x) for x in sys.argv[2].split(":")], sys.argv[3]
if not os.path.isfile(path):
    print(f"0|{name} TEST FAILED (no result file);"); raise SystemExit
r = json.load(open(path))
if name == "stand":
    st = r.get("standing")
    if not st:
        print("0|stand surv 0.00 (every robot fell);")
    else:
        print(f"{st['survival']:.3f}|stand surv {st['survival']:.2f} tilt {st['tilt_deg']['mean']:.1f} deg;")
    raise SystemExit
sv = r.get("survival_walking")
sp, yw = r.get("speed") or {}, r.get("yaw") or {}
fw, lt, yr = sp.get("achieved_fwd_mean"), sp.get("lateral_mean_signed"), yw.get("achieved_mean_signed")
if sv is None or fw is None or lt is None or yr is None:
    print(f"0|{name} surv {0.0 if sv is None else sv:.2f} (no robot finished the test);"); raise SystemExit
k = {"fwd": 0, "back": 0, "side": 1, "pivot": 2, "turn": 0}[name]
frac = sv * min(1.0, max(0.0, (fw, lt, yr)[k] / cmd[k]))
print(f"{frac:.3f}|{name} {fw:+.2f} {lt:+.2f} {yr:+.2f} surv {sv:.2f};")
PY
}
test_ck() {  # $1 checkpoint file, $2 label -> logs a line, sets SC (mean over the six tests of survival x share of the commanded speed)
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=walker_v4_${lab}_$name
        rm -f "eval_watch/$tag.json"
        env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$ck" --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    SC=$(python3 -c "print(round($tot / $n, 3))")
    log "$lab: score $SC | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3) $parts"
}
BEST=0
for N in 400 800 1200 1600 2000 2600 3200 $((ITERS - 1)); do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/walker_v4_$N.pt"
    test_ck "$LOGD/walker_v4_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" archive_anchors/walker_v4_best.pt; log "  -> new best (score $SC): archive_anchors/walker_v4_best.pt = model_$N"
    fi
done
log "DONE: best score $BEST. Pushes are still frozen; hardening is the next decision."
