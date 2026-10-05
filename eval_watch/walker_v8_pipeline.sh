#!/bin/bash
# WALKER v8 pipeline — launched 2026-10-04 on the user's go ("go ahead and launch v8").
# Why: the mirror-symmetry loss mirrored the IMU inputs on the wrong axes in every run so far. The policy's gravity
# and gyro are in the imu link frame (x down, y backward, z left; confirmed on the robot by the rig's hand test,
# eval_watch/RIG_REPLY_IMU_FRAME.md), and kbot_legs/symmetry.py flipped them as if in a standard frame — it called
# a forward lean the mirror image of a backward lean and never mirrored a sideways lean. Fixed 2026-10-04
# (eval_watch/mirror_check.py); v7_800 stands lopsided (hip pitch -4.5 / +1.5 deg, both hip yaws -2.8 deg).
# v8 = walker v7 run again with ONE change, the corrected mirror (KBOT_MIRROR_IMU=imu):
#   same start (walker v5's best checkpoint), same recipe (v5's rewards + the robot's sensing path), same length,
#   same tests at the same iterations — so v8 reads against v7's table line by line (WALKER_V7_STATUS.md).
# Tests per checkpoint: as v7 (six-direction walking battery; stand under 12 sensing cases; engage test and the
# rig's toy test) plus the stand pose (left + right of the joint targets: 0 = a mirror-symmetric stand).
# Score = v7's (walking x share of sensing cases with every robot up), so "best" means the same thing in both.
# It needs an idle GPU and never stops a training. Progress: eval_watch/WALKER_V8_STATUS.md.
# Dry run of the whole pipeline (6 iterations, short tests, nothing written outside logs/): V8_DRY=1.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD" "$IL/logs/smoke_runs"
LOCK=$LOGD/gpu_side_job.lock
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
WARM=archive_anchors/walker_v5_best.pt
HW=eval_watch/rig_hw_engage/hw_engage_ep_20261003_195937.npz
DRY=${V8_DRY:-0}
if [ "$DRY" = "1" ]; then
    ITERS=${WALKER_ITERS:-6}; NENV=2048; NAME=dry_v8pipeline; TP=dry_v8; ST=$IL/logs/smoke_runs/WALKER_V8_DRY_STATUS.md
    CKLIST="$((ITERS - 1))"; BESTF=$LOGD/dry_walker_v8_best.pt
    GAIT_ARGS="--num_envs 64 --seconds 6"; SENS_ARGS="--per_group 8 --seconds 5 --kick_at 3"; ENG_ARGS="--per_group 8 --seconds 3"; POSE_ARGS="--num_envs 64 --seconds 6"
else
    ITERS=${WALKER_ITERS:-2000}; NENV=8192; NAME=walker_v8; TP=walker_v8; ST=$IL/eval_watch/WALKER_V8_STATUS.md
    CKLIST="400 800 1200 1600 $((ITERS - 1))"; BESTF=archive_anchors/walker_v8_best.pt
    GAIT_ARGS="--num_envs 256 --seconds 20"; SENS_ARGS=""; ENG_ARGS="--seconds 10"; POSE_ARGS="--num_envs 256 --seconds 20"
fi
WALK_ENV="KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5 KBOT_AMP_MOTION_FILES=$DS_TURN,$DS_MULTI"
SENSE_ENV="KBOT_AMP_OBS_DELAY=0:1:1:3:3 KBOT_AMP_ACT_HOLD=5:6"
TRAIN_ENV="$WALK_ENV KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0 KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_INIT_STD=0.15 KBOT_AMP_LIFT_W=5 $SENSE_ENV KBOT_MIRROR_IMU=imu"
c="train_am"; d="p.py"; PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_room() { local n=0; while [ "$(free_mb)" -lt "$1" ]; do n=$((n + 1)); [ "$n" -ge 90 ] && return 1; sleep 10; done; return 0; }
eval "$(sed -n '/^score_one() /,/^}/p' eval_watch/walker_v5_pipeline.sh)"   # the v5 walking scorer (survival x speed share x step height)

echo "# WALKER v8 pipeline $(date '+%Y-%m-%d %H:%M') (walker v7 again with one change: the mirror-symmetry loss now mirrors the IMU on the right axes)$([ "$DRY" = "1" ] && echo ' — DRY RUN')" >> "$ST"
[ -f "$WARM" ] && [ -f "$DS_MULTI" ] && [ -f "$DS_TURN" ] && [ -f "$HW" ] || { log "missing a checkpoint, dataset or the rig's episode — stopping"; exit 1; }

# ---------------------------------------------------------------- 1. idle GPU, launch
n=0
while [ "$(gpu_n)" != "0" ]; do
    n=$((n + 1)); [ "$n" -ge 20 ] && { log "the GPU is still busy after 10 minutes — not launching"; exit 1; }
    sleep 30
done
env $TRAIN_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs "$NENV" --max_iterations "$ITERS" --checkpoint "$WARM" --run_name "$NAME" > "$LOGD/$TP.log" 2>&1 < /dev/null &
n=0; RUN=""
while [ -z "$RUN" ] || [ -z "$(ls "$RUN"events.out.tfevents.* 2>/dev/null)" ]; do      # the run folder and its first TensorBoard file
    n=$((n + 1)); [ "$n" -ge 40 ] && break
    sleep 10
    RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_"$NAME"/ 2>/dev/null | head -1)
done
[ "$DRY" = "1" ] || { sleep 120; echo "$LOGD/$TP.log" > logs/amp_pilot/CURRENT; echo "$LOGD/$TP.log" > logs/track_pilot/CURRENT; }
MIR=$(grep -a -m1 '^\[symmetry\]' "$LOGD/$TP.log" | cut -c1-120)
if [ -n "$RUN" ] && { pgrep -f "$PAT_AMP" >/dev/null || [ -f "$RUN/model_$((ITERS - 1)).pt" ]; } && echo "$MIR" | grep -q "imu mirror 'imu'"; then
    log "WALKER v8 launched: $ITERS iterations from $WARM; on TensorBoard (port 6007) as kbot_legs_amp/$(basename "$RUN") — $MIR — $(grep -a -m1 '\[amp-env\] v6' "$LOGD/$TP.log" | cut -c1-260)"
else
    log "walker launch FAILED (run folder '$RUN', mirror line '$MIR') — see $LOGD/$TP.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/$TP.log" | cut -c1-200)"; exit 1
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
pose_summary() {  # $1 stand-pose json -> prints one line of text
python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
if not os.path.isfile(sys.argv[1]):
    print("stand pose TEST FAILED (no result file)"); raise SystemExit
r = json.load(open(sys.argv[1])); p = r["pairs"]; n = r["robots"]
f = lambda k: f"{k} {p[k]['target_sum']:+.1f} ({p[k]['robots_with_target_sum_within_1deg']}/{n} robots within 1 deg)"
print("stand pose, left + right of the joint targets in deg (0 = mirror-symmetric): " + ", ".join(f(k) for k in ("hip pitch", "hip roll", "hip yaw", "ankle")) + f"; {r['fell']} fell"
      + " | v7_800: hip pitch -3.1 (0/256), hip roll +0.3 (246), hip yaw -5.7 (0), ankle -2.6 (23) | v5_3200: +0.6 (163), -3.9 (0), +0.8 (86), -0.4 (193)")
PY
}
test_ck() {  # $1 checkpoint file, $2 label -> logs four lines, sets SC = walking score x sensing factor
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag try
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=${TP}_${lab}_$name
        rm -f "eval_watch/$tag.json"
        for try in 1 2 3; do
            gpu_job 3500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$ck" $GAIT_ARGS --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
            [ -f "eval_watch/$tag.json" ] && break
            sleep 30
        done
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    WALK=$(python3 -c "print(round($tot / $n, 3))")
    tag=${TP}_${lab}_sensing; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 4500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_stand_sensing_test.py --checkpoint "$ck" --max_speed 1000 $SENS_ARGS --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    out=$(sensing_summary "eval_watch/$tag.json"); [ -n "$out" ] || out="0|stand/sensing TEST FAILED"
    SENS=${out%%|*}; STXT=${out#*|}
    tag=${TP}_${lab}_engage; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 4500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_engage_test.py --checkpoint "$ck" $ENG_ARGS --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    python eval_watch/rig_hw_engage/cf_tracking.py "$HW" "$ck" > "$LOGD/${TP}_${lab}_toy.txt" 2>&1
    ETXT=$(engage_summary "eval_watch/$tag.json" "$LOGD/${TP}_${lab}_toy.txt")
    tag=${TP}_${lab}_pose; rm -f "eval_watch/$tag.json"
    for try in 1 2 3; do
        gpu_job 3500 env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_stand_pose.py --checkpoint "$ck" $POSE_ARGS --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        [ -f "eval_watch/$tag.json" ] && break
        sleep 30
    done
    PTXT=$(pose_summary "eval_watch/$tag.json")
    SC=$(python3 -c "print(round($WALK * $SENS, 3))")
    log "$lab: score $SC (walking $WALK x sensing $SENS) | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3; reference lift 10.7 / 6.2 side / 4.1 pivot cm) $parts"
    log "  $lab $STXT"
    log "  $lab ${ETXT:-engage summary failed}"
    log "  $lab ${PTXT:-stand pose summary failed}"
}
BEST=0
for N in $CKLIST; do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/${TP}_$N.pt"
    test_ck "$LOGD/${TP}_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" "$BESTF"; log "  -> new best (score $SC): $BESTF = model_$N"
    fi
done
log "DONE: best score $BEST. (v7, same start and recipe with the old mirror: 0.128 / 0.81 / 0.715 / 0.756 / 0.535 at 400 / 800 / 1200 / 1600 / 1999.)"
if [ "$DRY" = "1" ]; then      # leave nothing of the dry run where a real run or a reader would find it
    n=0; while pgrep -f "$PAT_AMP" >/dev/null && [ "$n" -lt 30 ]; do n=$((n + 1)); sleep 10; done
    mv "$RUN" "$IL/logs/smoke_runs/" 2>/dev/null
    mv "$IL"/eval_watch/${TP}_*.json "$IL/logs/smoke_runs/" 2>/dev/null
    log "dry run: run folder and test results moved to logs/smoke_runs/"
fi
