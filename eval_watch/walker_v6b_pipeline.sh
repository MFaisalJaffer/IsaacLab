#!/bin/bash
# WALKER v6b pipeline — v6 restarted 2026-10-03 15:30 on the user's decision with ONE addition: the rig's watchdog as a
# training rule (while standing, joint speed > 5 rad/s or tilt > 12 deg ends the episode like a fall). v6 itself ran
# 600 iterations: walking kept, engage not improved (6/15), a loaded quiet stand drifted and fell (45-60% in 10 s).
# Why: first hardware engage of walker_v5_3200 (eval_watch/RIG_HW_ENGAGE_FINDINGS.md) — violent motion, power cut
# at 0.88 s. The real robot carries a standing load at the zero pose and its ankles do not answer small commands
# at once; the policy integrated on its own last_action. Reproduced in Isaac (eval_watch/amp_engage_test.py).
# v6 = fine-tune of v5's best checkpoint with the rig's asks in the data, everything else as v5:
#   standing load +-3 Nm pitch/roll per episode; dead band U(0,1) Nm on hips/knees/yaw; ankle rotor stiction
#   U(0,1.5) Nm; weak start on half the episodes (gains x U(0.3,1) ramping over U(0,1) s); ankle spring 52
#   (band 30-120); stands drawn at spawn are kept (the robot is engaged standing at the zero pose).
# Tests per checkpoint: the six-direction battery, the engage test under the rig's watchdog limits, and the
# rig's own toy-plant test. Score = walking score x share of engage conditions passed.
# It needs an idle GPU and never stops a training. Progress: eval_watch/WALKER_V6B_STATUS.md.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
ST=$IL/eval_watch/WALKER_V6B_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
LOCK=$LOGD/gpu_side_job.lock
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
WARM=archive_anchors/walker_v5_best.pt
HW=eval_watch/rig_hw_engage/hw_engage_ep_20261003_195937.npz
ITERS=${WALKER_ITERS:-2000}
WALK_ENV="KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5 KBOT_AMP_MOTION_FILES=$DS_TURN,$DS_MULTI"
V6_ENV="KBOT_AMP_STAND_MOMENT=3.0 KBOT_AMP_DEADBAND=1.0 KBOT_AMP_ROTOR_FC=1.5 KBOT_AMP_ENGAGE_P=0.5 KBOT_AMP_SERIES_K=52:30:120 KBOT_AMP_SPAWN_STAND_P=1.0 KBOT_AMP_WATCHDOG=5:12"
TRAIN_ENV="$WALK_ENV KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0 KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_INIT_STD=0.15 KBOT_AMP_LIFT_W=5 $V6_ENV"
c="train_am"; d="p.py"; PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() { local n=0; while [ "$(free_mb)" -lt "$1" ]; do n=$((n + 1)); [ "$n" -ge 90 ] && return 1; sleep 10; done; return 0; }
eval "$(sed -n '/^score_one() /,/^}/p' eval_watch/walker_v5_pipeline.sh)"   # the v5 walking scorer (survival x speed share x step height)

echo "# WALKER v6b pipeline $(date '+%Y-%m-%d %H:%M') (v5 best + standing load, unanswered commands, weak start, spawn stands, ankle spring 52, stand watchdog 5 rad/s / 12 deg)" >> "$ST"
[ -f "$WARM" ] && [ -f "$DS_MULTI" ] && [ -f "$DS_TURN" ] && [ -f "$HW" ] || { log "missing a checkpoint, dataset or the rig's episode — stopping"; exit 1; }

# ---------------------------------------------------------------- 1. idle GPU, launch
n=0
while [ "$(gpu_n)" != "0" ]; do
    n=$((n + 1)); [ "$n" -ge 20 ] && { log "the GPU is still busy after 10 minutes — not launching"; exit 1; }
    sleep 30
done
env $TRAIN_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$WARM" --run_name walker_v6b > "$LOGD/walker_v6b.log" 2>&1 < /dev/null &
sleep 240
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_walker_v6b/ 2>/dev/null | head -1)
echo "$LOGD/walker_v6b.log" > logs/amp_pilot/CURRENT; echo "$LOGD/walker_v6b.log" > logs/track_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
    log "WALKER v6b launched: $ITERS iterations from $WARM; on TensorBoard (port 6007) as kbot_legs_amp/$(basename "$RUN") — $(grep -a -m1 '\[amp-env\] v6' "$LOGD/walker_v6b.log" | cut -c1-330)"
else
    log "walker launch FAILED — see $LOGD/walker_v6b.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/walker_v6b.log" | cut -c1-200)"; exit 1
fi

# ---------------------------------------------------------------- 2. tests
gpu_job() {  # $1 MiB needed, rest = command; one GPU side job at a time, only when it fits
    local need="$1"; shift
    ( flock -w 1800 9 || exit 1; wait_room "$need"; "$@" ) 9> "$LOCK"
}
engage_summary() {  # $1 engage json, $2 toy-test output -> prints "<factor>|<text>"
python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, os, re, sys
if not os.path.isfile(sys.argv[1]):
    print("0|engage TEST FAILED (no result file)"); raise SystemExit
g = json.load(open(sys.argv[1]))["groups"]
ok = {k: (v["fell"] == 0 and v["peak_joint_speed_2s"] < 5.0 and v["peak_tilt_deg_2s"] < 12.0) for k, v in g.items()}   # the rig's watchdog limits
worst = max(g.items(), key=lambda kv: kv[1]["peak_joint_speed_2s"])
hw = next((v for k, v in g.items() if "load y-" in k and "1 s ramp" in k), None)
toy = ""
try:
    t = open(sys.argv[2]).read()
    a0 = float(re.search(r"reaches\s+0% of target:.*?= \S+ \S+ \S+ (\S+)", t).group(1)); a1 = float(re.search(r"reaches 100% of target:.*?= \S+ \S+ \S+ (\S+)", t).group(1))
    toy = f"; rig toy test (ankle action at tick 20, no response / full response): {a0:+.2f} / {a1:+.2f}"
except Exception:
    toy = "; rig toy test n/a"
txt = (f"engage: {sum(ok.values())}/{len(ok)} conditions within the rig's watchdog limits; hardware condition (load + weak start): ankle action at 0.4 s "
       f"{hw['ankle_action_t20'][0]:+.2f}/{hw['ankle_action_t20'][1]:+.2f}, peak joint speed {hw['peak_joint_speed_2s']:.1f} rad/s, tilt {hw['peak_tilt_deg_2s']:.1f} deg, fell {hw['fell']:.2f} "
       f"(v5: -0.58/+1.33, 21.6 rad/s); worst condition '{worst[0]}' {worst[1]['peak_joint_speed_2s']:.1f} rad/s{toy}")
print(f"{sum(ok.values()) / len(ok):.3f}|{txt}")
PY
}
test_ck() {  # $1 checkpoint file, $2 label -> logs two lines, sets SC = walking score x engage factor
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag try
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=walker_v6b_${lab}_$name
        rm -f "eval_watch/$tag.json"
        for try in 1 2 3; do
            gpu_job 3500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$ck" --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
            [ -f "eval_watch/$tag.json" ] && break
            sleep 30
        done
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    WALK=$(python3 -c "print(round($tot / $n, 3))")
    tag=walker_v6b_${lab}_engage; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 4500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_engage_test.py --checkpoint "$ck" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    python eval_watch/rig_hw_engage/cf_tracking.py "$HW" "$ck" > "$LOGD/walker_v6b_${lab}_toy.txt" 2>&1
    out=$(engage_summary "eval_watch/$tag.json" "$LOGD/walker_v6b_${lab}_toy.txt"); [ -n "$out" ] || out="0|engage TEST FAILED"
    ENG=${out%%|*}
    SC=$(python3 -c "print(round($WALK * $ENG, 3))")
    log "$lab: score $SC (walking $WALK x engage $ENG) | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3; reference lift 10.7 / 6.2 side / 4.1 pivot cm) $parts"
    log "  $lab ${out#*|}"
}
BEST=0
for N in 400 800 1200 1600 $((ITERS - 1)); do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/walker_v6b_$N.pt"
    test_ck "$LOGD/walker_v6b_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" archive_anchors/walker_v6b_best.pt; log "  -> new best (score $SC): archive_anchors/walker_v6b_best.pt = model_$N"
    fi
done
log "DONE: best score $BEST."
