#!/bin/bash
# WALKER v7 pipeline — launched 2026-10-03 on the user's go ("go ahead and launch it").
# Why: hardware session 2 (eval_watch/RIG_HW_STAND_FINDINGS.md, RIG_HW_STAND_ADDENDUM_SENSING.md). With the rig's
# crossfade removed walker_v5_3200 engaged cleanly, stood 1.9 s, then rocked at ~2 Hz until the watchdog cut it.
# Cause (rig, confirmed here with eval_watch/amp_stand_sensing_test.py): on the robot the IMU gave 20 new samples
# a second, 20-70 ms old, the joints were ~14 ms late and the policy server stalled ~116 ms once per episode; this
# env delivered every observation fresh. v5 keeps its balance at one tick of IMU delay and loses it at two.
# v7 = fine-tune of v5's best checkpoint with ONE change, the sensing path:
#   joints 0-1 policy steps late; IMU terms 1-3 steps late and a new sample only every 1-3 steps (per episode);
#   one 5-6 step hold of the joint targets per episode.
# None of the v6 ingredients (standing load, dead band, stiction, weak start, spawn stands, watchdog rule): the
# two v6 fine-tunes came out worse than v5 and were never taken apart.
# Tests per checkpoint: six-direction walking battery; stand test under 12 sensing cases (score factor = share of
# cases in which no robot loses balance, before or after the kick); engage test and the rig's toy test (reported).
# It needs an idle GPU and never stops a training. Progress: eval_watch/WALKER_V7_STATUS.md.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
ST=$IL/eval_watch/WALKER_V7_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
LOCK=$LOGD/gpu_side_job.lock
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
WARM=archive_anchors/walker_v5_best.pt
HW=eval_watch/rig_hw_engage/hw_engage_ep_20261003_195937.npz
ITERS=${WALKER_ITERS:-2000}
WALK_ENV="KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5 KBOT_AMP_MOTION_FILES=$DS_TURN,$DS_MULTI"
SENSE_ENV="KBOT_AMP_OBS_DELAY=0:1:1:3:3 KBOT_AMP_ACT_HOLD=5:6"
TRAIN_ENV="$WALK_ENV KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0 KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_INIT_STD=0.15 KBOT_AMP_LIFT_W=5 $SENSE_ENV"
c="train_am"; d="p.py"; PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() { local n=0; while [ "$(free_mb)" -lt "$1" ]; do n=$((n + 1)); [ "$n" -ge 90 ] && return 1; sleep 10; done; return 0; }
eval "$(sed -n '/^score_one() /,/^}/p' eval_watch/walker_v5_pipeline.sh)"   # the v5 walking scorer (survival x speed share x step height)

echo "# WALKER v7 pipeline $(date '+%Y-%m-%d %H:%M') (v5 best + the robot's sensing path: delayed joints, delayed and repeated IMU, one frozen command per episode)" >> "$ST"
[ -f "$WARM" ] && [ -f "$DS_MULTI" ] && [ -f "$DS_TURN" ] && [ -f "$HW" ] || { log "missing a checkpoint, dataset or the rig's episode — stopping"; exit 1; }

# ---------------------------------------------------------------- 1. idle GPU, launch
n=0
while [ "$(gpu_n)" != "0" ]; do
    n=$((n + 1)); [ "$n" -ge 20 ] && { log "the GPU is still busy after 10 minutes — not launching"; exit 1; }
    sleep 30
done
env $TRAIN_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$WARM" --run_name walker_v7 > "$LOGD/walker_v7.log" 2>&1 < /dev/null &
sleep 240
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_walker_v7/ 2>/dev/null | head -1)
echo "$LOGD/walker_v7.log" > logs/amp_pilot/CURRENT; echo "$LOGD/walker_v7.log" > logs/track_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
    log "WALKER v7 launched: $ITERS iterations from $WARM; on TensorBoard (port 6007) as kbot_legs_amp/$(basename "$RUN") — $(grep -a -m1 '\[amp-env\] v6' "$LOGD/walker_v7.log" | cut -c1-260)"
else
    log "walker launch FAILED — see $LOGD/walker_v7.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/walker_v7.log" | cut -c1-200)"; exit 1
fi

# ---------------------------------------------------------------- 2. tests
gpu_job() {  # $1 MiB needed, rest = command; one GPU side job at a time, only when it fits
    local need="$1"; shift
    ( flock -w 1800 9 || exit 1; wait_room "$need"; "$@" ) 9> "$LOCK"
}
sensing_summary() {  # $1 stand-sensing json -> prints "<factor>|<text>"
python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
if not os.path.isfile(sys.argv[1]):
    print("0|stand/sensing TEST FAILED (no result file)"); raise SystemExit
g = json.load(open(sys.argv[1]))["groups"]
ok = {k: (v["tripped_before_kick"] == 0 and v.get("tripped_after_kick", 1.0) == 0) for k, v in g.items()}
row = lambda key: next(v for k, v in g.items() if key in k)
ctl, meas, cliff, fix = row("control"), row("robot as measured"), row("IMU 20 Hz + 2 ticks"), row("rig after its fix")
f = lambda v: f"{100 * v['tripped_before_kick']:.0f}% lost" + (f", rocking {v['pitch_fast_rms_deg']:.2f} deg" if "pitch_fast_rms_deg" in v else "")
txt = (f"stand under the robot's sensing: {sum(ok.values())}/{len(ok)} cases with every robot up (v5: 5/12) | fresh {f(ctl)} | as measured {f(meas)} (v5 16%, 0.76) | "
       f"held IMU + 2 ticks {f(cliff)} (v5 78%) | rig's fix {f(fix)}")
print(f"{sum(ok.values()) / len(ok):.3f}|{txt}")
PY
}
engage_summary() {  # $1 engage json, $2 toy-test output -> prints one line of text
python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, os, re, sys
if not os.path.isfile(sys.argv[1]):
    print("engage TEST FAILED (no result file)"); raise SystemExit
g = json.load(open(sys.argv[1]))["groups"]
hard = [v for k, v in g.items() if "ramp" not in k]
hw = next(v for k, v in g.items() if "load y-" in k and "1 s ramp" in k)
toy = "n/a"
try:
    t = open(sys.argv[2]).read()
    toy = "%+.2f / %+.2f" % (float(re.search(r"reaches\s+0% of target:.*?= \S+ \S+ \S+ (\S+)", t).group(1)), float(re.search(r"reaches 100% of target:.*?= \S+ \S+ \S+ (\S+)", t).group(1)))
except Exception:
    pass
print(f"engage (plant as trained): hard starts fell {sum(v['fell'] for v in hard) / len(hard):.2f} (v5 0.00), tilt {sum(v['peak_tilt_deg_2s'] for v in hard) / len(hard):.1f} deg; load + weak start: "
      f"{hw['peak_joint_speed_2s']:.1f} rad/s, fell {hw['fell']:.2f} (v5 18.2, 0.12); rig toy test (no response / full response) {toy} (v5 -0.36 / -0.04)")
PY
}
test_ck() {  # $1 checkpoint file, $2 label -> logs three lines, sets SC = walking score x sensing factor
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag try
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=walker_v7_${lab}_$name
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
    tag=walker_v7_${lab}_sensing; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 4500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_stand_sensing_test.py --checkpoint "$ck" --max_speed 1000 --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    out=$(sensing_summary "eval_watch/$tag.json"); [ -n "$out" ] || out="0|stand/sensing TEST FAILED"
    SENS=${out%%|*}; STXT=${out#*|}
    tag=walker_v7_${lab}_engage; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 4500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_engage_test.py --checkpoint "$ck" --seconds 10 --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    python eval_watch/rig_hw_engage/cf_tracking.py "$HW" "$ck" > "$LOGD/walker_v7_${lab}_toy.txt" 2>&1
    ETXT=$(engage_summary "eval_watch/$tag.json" "$LOGD/walker_v7_${lab}_toy.txt")
    SC=$(python3 -c "print(round($WALK * $SENS, 3))")
    log "$lab: score $SC (walking $WALK x sensing $SENS) | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3; reference lift 10.7 / 6.2 side / 4.1 pivot cm) $parts"
    log "  $lab $STXT"
    log "  $lab ${ETXT:-engage summary failed}"
}
BEST=0
for N in 400 800 1200 1600 $((ITERS - 1)); do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/walker_v7_$N.pt"
    test_ck "$LOGD/walker_v7_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" archive_anchors/walker_v7_best.pt; log "  -> new best (score $SC): archive_anchors/walker_v7_best.pt = model_$N"
    fi
done
log "DONE: best score $BEST."
