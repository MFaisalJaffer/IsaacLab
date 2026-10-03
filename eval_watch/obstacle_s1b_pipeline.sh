#!/bin/bash
# OBSTACLE COURSE, STAGE 1b — detached autopilot (user go 2026-10-03: "one more run before stage 2 and add
# more like stairs to practice"). One run from stage 1's iteration-1600 checkpoint (all flat skills intact),
# with eval_watch/obstacle_env_v2.sh:
#   * stairs added (3 risers up, landing, 3 down); mix 40% flat / 20% platform / 15% beam / 25% stairs
#   * action noise capped at 0.2
#   * falling behind the command no longer demotes a robot that got onto its obstacle; 20% of episodes replay
#     a lower height; 25 s episodes
# Tests at EVERY checkpoint (stage 1 lost side-stepping unseen between two flat tests):
#   * obstacle test: 1200 robots over every kind x height (10% flat control, 30% each obstacle kind), 26 s
#   * the walker's own six-direction test on an all-flat course
# A checkpoint is ELIGIBLE only if every flat direction scores >= 0.5; the best is the eligible checkpoint with
# the highest crossing score -> archive_anchors/obstacle_s1b_best.pt. Status: eval_watch/OBSTACLE_S1B_STATUS.md.
# It never stops a training. OBST_DRY=<ckpt> tests one checkpoint and exits.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
source eval_watch/obstacle_env_v2.sh
ST=${OBST_STATUS:-$IL/eval_watch/OBSTACLE_S1B_STATUS.md}
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
LOCK=$LOGD/gpu_side_job.lock
TASK=Isaac-Velocity-Obstacle-KbotLegs-AMP-v0
START=archive_anchors/obstacle_s1_map_1600_allskills.pt
ITERS=${OBST_ITERS:-3000}
TEST_ENVS=${OBST_TEST_ENVS:-1200}
TEST_MIX="KBOT_OBST_MIX=flat:0.10,platform:0.30,beam:0.30,stairs:0.30"
NAME=obst_s1b
c="train_am"; d="p.py"; PAT_AMP="$c$d"
FLAT_TEST="env -u KBOT_AMP_AXIS_P -u KBOT_AMP_VEL_STD -u KBOT_AMP_VEL_W -u KBOT_AMP_YAW_STD -u KBOT_AMP_YAW_W -u KBOT_AMP_DRIFT -u KBOT_AMP_DRIFT_YAW -u KBOT_AMP_ENTROPY -u KBOT_AMP_LIFT_W KBOT_OBST_FLAT_ONLY=1"
DIRS="fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0"
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

obst_test() {  # $1 checkpoint, $2 tag, $3 extra args -> sets SC and TEXT
    local ck="$1" tag="$2" extra="${3:-}"
    side_job "eval_watch/$tag.json" bash -c "env $TEST_MIX ./isaaclab.sh -p eval_watch/obstacle_eval.py --checkpoint '$ck' --num_envs $TEST_ENVS --seconds 26 $extra --tag '$tag' --headless > '$LOGD/$tag.log' 2>&1"
    OUT=$(python3 eval_watch/obstacle_summary.py "eval_watch/$tag.json" 2>/dev/null)
    [ -n "$OUT" ] || OUT="0|TEST FAILED (no result file)"
    SC=${OUT%%|*}; TEXT=${OUT#*|}
}

flat_score() {  # $1 tag prefix -> sets FLAT_SC (mean), FLAT_MIN (weakest direction), FLAT_WORST (its name), FLAT_TEXT
    local pre="$1" tot=0 n=0 out name cmd v
    FLAT_MIN=9; FLAT_WORST=""; FLAT_TEXT=""
    for pair in $DIRS; do
        name=${pair%%:*}; cmd=${pair#*:}
        out=$(score_one "$name" "$cmd" "eval_watch/${pre}_flat_$name.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        v=${out%%|*}; FLAT_TEXT="$FLAT_TEXT ${out#*|}"
        tot=$(python3 -c "print($tot + $v)"); n=$((n + 1))
        if python3 -c "import sys; sys.exit(0 if float('$v') < float('$FLAT_MIN') else 1)"; then FLAT_MIN=$v; FLAT_WORST=$name; fi
    done
    FLAT_SC=$(python3 -c "print(round($tot / $n, 3))")
}

flat_test() {  # $1 checkpoint, $2 tag prefix -> runs the six tests, then flat_score
    local ck="$1" pre="$2" name cmd tag
    for pair in $DIRS; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=${pre}_flat_$name
        side_job "eval_watch/$tag.json" bash -c "$FLAT_TEST ./isaaclab.sh -p eval_watch/amp_gait_eval.py --task $TASK --checkpoint '$ck' --num_envs 256 --seconds 20 --cmd=$cmd --tag '$tag' --headless > '$LOGD/$tag.log' 2>&1"
    done
    flat_score "$pre"
}

if [ -n "${OBST_DRY:-}" ]; then   # test one checkpoint with the real functions and exit
    obst_test "$OBST_DRY" ${NAME}_dry; log "DRY model: crossing $SC | $TEXT"
    if [ "${OBST_DRY_FLAT:-0}" = "1" ]; then flat_test "$OBST_DRY" ${NAME}_dry; else flat_score "${OBST_DRY_FLAT_PREFIX:-${NAME}_dry}"; fi
    log "DRY flat: score $FLAT_SC, weakest $FLAT_WORST $FLAT_MIN |$FLAT_TEXT"
    exit 0
fi

echo "# OBSTACLE COURSE stage 1b — $(date '+%Y-%m-%d %H:%M') ($ITERS iterations from $START; crossing shares listed by height 2,4,...,20 cm; for stairs that is the riser height)" >> "$ST"
[ -f "$START" ] || { log "missing $START — stopping"; exit 1; }
log "start point on the new course: $(python3 eval_watch/obstacle_summary.py eval_watch/obst_s1b_baseline_start1600.json 2>/dev/null | cut -d'|' -f2-)"
log "for reference, stage 1's best climber on the new course: $(python3 eval_watch/obstacle_summary.py eval_watch/obst_s1b_baseline_stage1final.json 2>/dev/null | cut -d'|' -f2-)"

n=0
while [ "$(gpu_n)" != "0" ]; do
    n=$((n + 1)); [ "$n" -ge 40 ] && { log "the GPU is still busy after 20 minutes — not launching"; exit 1; }
    sleep 30
done
env KBOT_OBST_INIT_LEVEL=2 setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=$TASK --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$START" --run_name "$NAME" > "$LOGD/$NAME.log" 2>&1 < /dev/null &
sleep 240
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_obstacle/*_"$NAME"/ 2>/dev/null | head -1)
echo "$LOGD/$NAME.log" > logs/amp_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
    log "RUN launched: $ITERS iterations; TensorBoard (port 6007) run kbot_legs_obstacle/$(basename "$RUN") — $(grep -a -m1 '\[obstacle-env\]' "$LOGD/$NAME.log" | cut -c1-330); $(grep -a -m1 'action noise capped' "$LOGD/$NAME.log" | cut -c1-80)"
else
    log "launch FAILED — see $LOGD/$NAME.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/$NAME.log" | cut -c1-200)"; exit 1
fi

BEST=-1; BEST_N=""; TOP=-1; TOP_N=""
for N in 400 800 1200 1600 2000 2400 $((ITERS - 1)); do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "training process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/${NAME}_$N.pt"
    obst_test "$LOGD/${NAME}_$N.pt" "${NAME}_$N"
    LV=$(grep -a -E "Curriculum/obstacle_levels/height_cm_(platform|beam|stairs)" "$LOGD/$NAME.log" | tail -n 3 | sed 's/\x1b\[[0-9;]*m//g' | awk '{printf "%.1f ", $NF}')
    NOISE=$(grep -a "Mean action noise std" "$LOGD/$NAME.log" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk '{print $NF}')
    log "model_$N: crossing $SC | $TEXT | training heights (platform beam stairs): $LV cm, noise $NOISE"
    flat_test "$LOGD/${NAME}_$N.pt" "${NAME}_$N"
    if python3 -c "import sys; sys.exit(0 if float('$FLAT_MIN') >= 0.5 else 1)"; then
        log "  flat walking: score $FLAT_SC (walker v5 0.876), weakest $FLAT_WORST $FLAT_MIN -> ELIGIBLE |$FLAT_TEXT"
        if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
            BEST=$SC; BEST_N=$N; cp "$CK" archive_anchors/obstacle_s1b_best.pt; log "  -> new best (crossing $SC, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_$N"
        fi
    else
        log "  flat walking: score $FLAT_SC (walker v5 0.876), weakest $FLAT_WORST $FLAT_MIN -> NOT ELIGIBLE: a flat skill is below 0.5 |$FLAT_TEXT"
    fi
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$TOP') else 1)"; then TOP=$SC; TOP_N=$N; fi
done
if [ -n "$BEST_N" ]; then   # does the best checkpoint use the map?
    obst_test "$LOGD/${NAME}_$BEST_N.pt" "${NAME}_${BEST_N}_nomap" "--blind"
    log "best model_$BEST_N with the map zeroed: crossing $SC | $TEXT"
fi
log "DONE: best eligible checkpoint model_${BEST_N:-none} (crossing $BEST); highest crossing of any checkpoint: model_$TOP_N ($TOP)."
