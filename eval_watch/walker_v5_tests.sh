#!/bin/bash
# WALKER v5 tests-only autopilot (2026-10-02 13:50) — takes over the test loop of walker_v5_pipeline.sh for the
# run that is already training. It never starts or stops a training.
# Why: at 13:43:49 the dashboard's Render button (kbot-watch service) rendered iteration 400; training 8.3 GB +
# that render ~6 GB left no GPU memory for the six test runs, and all of them failed ("TEST FAILED"). This loop
#   * waits for free GPU memory before each test and retries a test that produced no result (up to 3 times),
#   * takes a lock shared with eval_watch/walker_v5_render.sh, so tests and those renders never overlap.
# Usage: bash eval_watch/walker_v5_tests.sh <best score so far> <pid to wait for, or 0> N1 N2 ...
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
ST=$IL/eval_watch/WALKER_V5_STATUS.md
LOGD=$IL/logs/autopilot
LOCK=$LOGD/gpu_side_job.lock
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
eval "$(sed -n '/^WALK_ENV=/p' eval_watch/walker_v5_pipeline.sh)"
eval "$(sed -n '/^score_one() /,/^}/p' eval_watch/walker_v5_pipeline.sh)"   # the pipeline's own scorer
c="train_am"; d="p.py"; PAT_AMP="$c$d"
BEST=${1:?best score so far}; WAITPID=${2:?pid to wait for or 0}; shift 2
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_walker_v5/ 2>/dev/null | head -1)
[ -n "$RUN" ] || { echo "no walker_v5 run folder"; exit 1; }

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() {  # $1 = MiB needed; gives up after 15 minutes
    local n=0
    while [ "$(free_mb)" -lt "$1" ]; do n=$((n + 1)); [ "$n" -ge 90 ] && return 1; sleep 10; done
    return 0
}
test_ck() {  # $1 checkpoint file, $2 label -> logs a line, sets SC (mean over the six tests of survival x share of the commanded speed x step-height factor)
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag try
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=walker_v5_${lab}_$name
        rm -f "eval_watch/$tag.json"
        for try in 1 2 3; do
            (
                flock -w 1800 9 || exit 1
                wait_room 3500
                env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$ck" --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
            ) 9> "$LOCK"
            [ -f "eval_watch/$tag.json" ] && break
            sleep 30
        done
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    SC=$(python3 -c "print(round($tot / $n, 3))")
    log "$lab: score $SC | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3; reference lift 10.7 / 6.2 side / 4.1 pivot cm) $parts"
}

log "tests restarted with a wait-for-GPU-memory + retry loop (the first model_400 test collided with a dashboard render and found no free GPU memory)"
if [ "$WAITPID" != "0" ]; then while [ -d "/proc/$WAITPID" ]; do sleep 10; done; fi
for N in "$@"; do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; [ -f "$LOGD/walker_v5_$N.pt" ] || cp "$CK" "$LOGD/walker_v5_$N.pt"
    test_ck "$LOGD/walker_v5_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" archive_anchors/walker_v5_best.pt; log "  -> new best (score $SC): archive_anchors/walker_v5_best.pt = model_$N"
    fi
done
log "DONE: best score $BEST. Pushes are still frozen; hardening is the next decision."
