#!/bin/bash
# OBSTACLE COURSE, STAGE 1 (eval_watch/OBSTACLE_PERCEPTION_PROPOSAL.md) — detached autopilot.
# Question: can the walker climb what it sees? Two runs from the same start (walker v5 + an unconnected map
# input), same course, same settings, one after the other:
#   map    the policy sees the height map
#   blind  the map is zeroed (KBOT_OBST_BLIND=1) — the control that shows what the map buys
# Per run: launch -> test checkpoints -> keep the best in archive_anchors/obstacle_s1_<run>_best.pt.
# Tests per checkpoint (they wait for free GPU memory and retry; lock shared with render jobs):
#   * obstacle test: 1200 robots spread over every kind x height, forward 0.35 m/s, random heading, 20 s
#   * (map run) the same with the map zeroed — does the policy use the map?
#   * at 1200 and at the end: the walker's own six-direction test on an all-flat course (did flat walking survive?)
# Status: eval_watch/OBSTACLE_S1_STATUS.md. TensorBoard (port 6007): kbot_legs_obstacle/<time>_obst_s1_<run>.
# It never stops a training. OBST_RUNS="map" runs only the first; OBST_DRY=<ckpt> tests one checkpoint and exits.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
source eval_watch/obstacle_env.sh
ST=${OBST_STATUS:-$IL/eval_watch/OBSTACLE_S1_STATUS.md}
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
LOCK=$LOGD/gpu_side_job.lock
TASK=Isaac-Velocity-Obstacle-KbotLegs-AMP-v0
WARM=archive_anchors/obstacle_s1_warm_from_walker_v5.pt
ITERS=${OBST_ITERS:-3000}
RUNS=${OBST_RUNS:-"map blind"}
TEST_ENVS=${OBST_TEST_ENVS:-1200}
c="train_am"; d="p.py"; PAT_AMP="$c$d"
# the walker's six-direction test is run the way walker v5's was: without the training-only command and termination settings
FLAT_TEST="env -u KBOT_AMP_AXIS_P -u KBOT_AMP_VEL_STD -u KBOT_AMP_VEL_W -u KBOT_AMP_YAW_STD -u KBOT_AMP_YAW_W -u KBOT_AMP_DRIFT -u KBOT_AMP_DRIFT_YAW -u KBOT_AMP_ENTROPY -u KBOT_AMP_LIFT_W KBOT_OBST_FLAT_ONLY=1"
eval "$(sed -n '/^score_one() /,/^}/p' eval_watch/walker_v5_pipeline.sh)"   # the walker's scorer: survival x speed share x step height

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() { local n=0; while [ "$(free_mb)" -lt "$1" ]; do n=$((n + 1)); [ "$n" -ge 90 ] && return 1; sleep 10; done; return 0; }

side_job() {  # $1 = result file; rest = command. One GPU side job at a time, only with room, retried if it leaves no result.
    local result="$1" try; shift
    rm -f "$result"
    for try in 1 2 3; do
        ( flock -w 1800 9 || exit 1; wait_room 3500; "$@" ) 9> "$LOCK"
        [ -f "$result" ] && return 0
        sleep 30
    done
    return 1
}

obst_test() {  # $1 checkpoint, $2 tag, $3 "map" | "blind" -> sets SC and TEXT
    local ck="$1" tag="$2" run="$3" bl=""
    [ "$run" = "blind" ] && bl="KBOT_OBST_BLIND=1"
    side_job "eval_watch/$tag.json" bash -c "env $bl ./isaaclab.sh -p eval_watch/obstacle_eval.py --checkpoint '$ck' --num_envs $TEST_ENVS --tag '$tag' --headless > '$LOGD/$tag.log' 2>&1"
    if [ "$run" = "map" ]; then   # the same checkpoint with the map zeroed
        side_job "eval_watch/${tag}_nomap.json" bash -c "./isaaclab.sh -p eval_watch/obstacle_eval.py --checkpoint '$ck' --num_envs $TEST_ENVS --blind --tag '${tag}_nomap' --headless > '$LOGD/${tag}_nomap.log' 2>&1"
        OUT=$(python3 eval_watch/obstacle_summary.py "eval_watch/$tag.json" "eval_watch/${tag}_nomap.json" 2>/dev/null)
    else
        OUT=$(python3 eval_watch/obstacle_summary.py "eval_watch/$tag.json" 2>/dev/null)
    fi
    [ -n "$OUT" ] || OUT="0|TEST FAILED (no result file)"
    SC=${OUT%%|*}; TEXT=${OUT#*|}
}

flat_test() {  # $1 checkpoint, $2 tag prefix, $3 "map" | "blind" -> logs the walker's six-direction line
    local ck="$1" pre="$2" run="$3" bl="" tot=0 parts="" n=0 out name cmd tag
    [ "$run" = "blind" ] && bl="KBOT_OBST_BLIND=1"
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=${pre}_flat_$name
        side_job "eval_watch/$tag.json" bash -c "$FLAT_TEST $bl ./isaaclab.sh -p eval_watch/amp_gait_eval.py --task $TASK --checkpoint '$ck' --num_envs 256 --seconds 20 --cmd=$cmd --tag '$tag' --headless > '$LOGD/$tag.log' 2>&1"
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    log "  flat walking (walker's own test; walker v5 scored 0.876): score $(python3 -c "print(round($tot / $n, 3))") |$parts"
}

if [ -n "${OBST_DRY:-}" ]; then   # test one checkpoint with the real functions and exit
    obst_test "$OBST_DRY" obst_s1_dry "${OBST_DRY_RUN:-map}"; log "DRY model: score $SC | $TEXT"
    [ "${OBST_DRY_FLAT:-0}" = "1" ] && flat_test "$OBST_DRY" obst_s1_dry "${OBST_DRY_RUN:-map}"
    exit 0
fi

echo "# OBSTACLE COURSE stage 1 — $(date '+%Y-%m-%d %H:%M') (runs: $RUNS; $ITERS iterations each; crossing shares listed by height 2,4,...,20 cm)" >> "$ST"
[ -f "$WARM" ] || { log "missing $WARM — stopping"; exit 1; }
log "start point (walker v5, no map yet): $(python3 eval_watch/obstacle_summary.py eval_watch/obst_s1_baseline_walker_v5.json 2>/dev/null | cut -d'|' -f2-)"

for RUN_NAME in $RUNS; do
    BL=""; [ "$RUN_NAME" = "blind" ] && BL="KBOT_OBST_BLIND=1"
    n=0
    while [ "$(gpu_n)" != "0" ]; do
        n=$((n + 1)); [ "$n" -ge 40 ] && { log "$RUN_NAME: the GPU is still busy after 20 minutes — not launching"; exit 1; }
        sleep 30
    done
    env $BL setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=$TASK --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$WARM" --run_name "obst_s1_$RUN_NAME" > "$LOGD/obst_s1_$RUN_NAME.log" 2>&1 < /dev/null &
    sleep 240
    RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_obstacle/*_obst_s1_"$RUN_NAME"/ 2>/dev/null | head -1)
    echo "$LOGD/obst_s1_$RUN_NAME.log" > logs/amp_pilot/CURRENT
    if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
        log "RUN '$RUN_NAME' launched: $ITERS iterations; TensorBoard (port 6007) run kbot_legs_obstacle/$(basename "$RUN") — $(grep -a -m1 '\[obstacle-env\]' "$LOGD/obst_s1_$RUN_NAME.log" | sed 's/kinds per column.*start level/start level/' | cut -c1-260)"
    else
        log "run '$RUN_NAME' launch FAILED — see $LOGD/obst_s1_$RUN_NAME.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/obst_s1_$RUN_NAME.log" | cut -c1-200)"; exit 1
    fi
    BEST=-1
    for N in 400 800 1200 1600 2000 2400 $((ITERS - 1)); do
        CK="$RUN/model_$N.pt"
        while [ ! -f "$CK" ]; do
            pgrep -f "$PAT_AMP" >/dev/null || break
            sleep 30
        done
        [ -f "$CK" ] || { log "$RUN_NAME: training process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
        sleep 20; cp "$CK" "$LOGD/obst_s1_${RUN_NAME}_$N.pt"
        obst_test "$LOGD/obst_s1_${RUN_NAME}_$N.pt" "obst_s1_${RUN_NAME}_$N" "$RUN_NAME"
        LV=$(grep -a -E "Curriculum/obstacle_levels/height_cm_(platform|beam)" "$LOGD/obst_s1_$RUN_NAME.log" | tail -n 2 | sed 's/\x1b\[[0-9;]*m//g' | awk '{printf "%s cm ", $NF}')
        log "$RUN_NAME model_$N: score $SC | $TEXT | training heights now (platform, beam): $LV"
        if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
            BEST=$SC; cp "$CK" "archive_anchors/obstacle_s1_${RUN_NAME}_best.pt"; log "  -> new best (score $SC): archive_anchors/obstacle_s1_${RUN_NAME}_best.pt = model_$N"
        fi
        if [ "$N" = "1200" ] || [ "$N" = "$((ITERS - 1))" ]; then flat_test "$LOGD/obst_s1_${RUN_NAME}_$N.pt" "obst_s1_${RUN_NAME}_$N" "$RUN_NAME"; fi
    done
    log "run '$RUN_NAME' DONE: best score $BEST"
    while pgrep -f "$PAT_AMP" >/dev/null; do sleep 20; done
done
log "ALL DONE ($RUNS)."
