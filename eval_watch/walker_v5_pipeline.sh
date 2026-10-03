#!/bin/bash
# WALKER v5 pipeline — PREPARED 2026-10-02 13:15, to be started only on the user's go.
# Walker v4 (same settings) kept every direction and learned to stand, but slid its feet instead of stepping:
# lift 2 cm forward/backward (reference 10.7), under 1 cm for side-steps and pivots. v5 = v4 + ONE change:
# the stepping anchor (mdp_amp.ref_foot_lift, weight $WALKER_LIFT_W): each foot's height follows the reference
# cycle's lift at the gait clock's phase. Dry-run 512 envs x 30 iterations: OK.
#   1. needs an idle GPU (it never stops a training itself; walker v4 must be stopped first)
#   2. launches from the tracker's weights again (clean stepping in all six directions), same dataset as v4
#   3. tests checkpoints per direction; the score now also counts step height (v4's score was blind to it)
#   4. best checkpoint -> archive_anchors/walker_v5_best.pt
# Progress in eval_watch/WALKER_V5_STATUS.md; TensorBoard (port 6007) run kbot_legs_amp/<time>_walker_v5.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0   # the lineage environment — always
ST=$IL/eval_watch/WALKER_V5_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz
DS_MULTI=$A/multitrack_v2_kbot.npz
WARM=archive_anchors/trackmulti_v2_best_final.pt
ITERS=${WALKER_ITERS:-4000}
LIFT_W=${WALKER_LIFT_W:-5}
WALK_ENV="KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5 KBOT_AMP_MOTION_FILES=$DS_TURN,$DS_MULTI"
TRAIN_ENV="$WALK_ENV KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0 KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_INIT_STD=0.25 KBOT_AMP_LIFT_W=$LIFT_W"
c="train_am"; d="p.py"; PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }

echo "# WALKER v5 pipeline $(date '+%Y-%m-%d %H:%M') (v4 + stepping anchor, weight $LIFT_W)" >> "$ST"
[ -f "$WARM" ] && [ -f "$DS_MULTI" ] && [ -f "$DS_TURN" ] || { log "missing the tracker checkpoint or a dataset — stopping"; exit 1; }

# ---------------------------------------------------------------- 1. idle GPU
n=0
while [ "$(gpu_n)" != "0" ]; do
    n=$((n + 1)); [ "$n" -ge 20 ] && { log "the GPU is still busy after 10 minutes — not launching (stop the other training first)"; exit 1; }
    sleep 30
done

# ---------------------------------------------------------------- 2. launch the walker
env $TRAIN_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations "$ITERS" --checkpoint "$WARM" --run_name walker_v5 > "$LOGD/walker_v5.log" 2>&1 < /dev/null &
sleep 240
RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*_walker_v5/ 2>/dev/null | head -1)
echo "$LOGD/walker_v5.log" > logs/amp_pilot/CURRENT; echo "$LOGD/walker_v5.log" > logs/track_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null && [ -n "$RUN" ]; then
    log "WALKER v5 launched: $ITERS iterations; on TensorBoard (port 6007) as kbot_legs_amp/$(basename "$RUN") — $(grep -a -m1 '\[amp-env\] v5' "$LOGD/walker_v5.log" | cut -c1-160); $(grep -a -m1 '\[amp-env\] v4' "$LOGD/walker_v5.log" | cut -c1-260)"
else
    log "walker launch FAILED — see $LOGD/walker_v5.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/walker_v5.log" | cut -c1-200)"; exit 1
fi

# ---------------------------------------------------------------- 3. per-direction tests, best checkpoint
score_one() {  # $1 name, $2 command "vx:vy:wz", $3 result file -> prints "<score 0..1>|<text>"
python3 - "$1" "$2" "$3" <<'PY' 2>/dev/null
import json, os, sys
name, cmd, path = sys.argv[1], [float(x) for x in sys.argv[2].split(":")], sys.argv[3]
# highest foot lift of the reference gait in this same test (tracker weights, 2026-10-02), cm
REF = {"fwd": 10.7, "back": 10.7, "turn": 10.7, "side": 6.2, "pivot": 4.1}
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
cl = float(r.get("clearance_cm") or 0.0)
lift = min(1.0, cl / (0.6 * REF[name]))  # full marks from 60% of the reference's lift; sliding feet score ~0
frac = sv * min(1.0, max(0.0, (fw, lt, yr)[k] / cmd[k])) * lift
print(f"{frac:.3f}|{name} {fw:+.2f} {lt:+.2f} {yr:+.2f} surv {sv:.2f} lift {cl:.1f} cm;")
PY
}
test_ck() {  # $1 checkpoint file, $2 label -> logs a line, sets SC (mean over the six tests of survival x share of the commanded speed x step-height factor)
    local ck="$1" lab="$2" tot=0 parts="" n=0 out name cmd tag
    for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
        name=${pair%%:*}; cmd=${pair#*:}; tag=walker_v5_${lab}_$name
        rm -f "eval_watch/$tag.json"
        env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$ck" --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        out=$(score_one "$name" "$cmd" "eval_watch/$tag.json"); [ -n "$out" ] || out="0|$name TEST FAILED;"
        tot=$(python3 -c "print($tot + ${out%%|*})"); parts="$parts ${out#*|}"; n=$((n + 1))
    done
    SC=$(python3 -c "print(round($tot / $n, 3))")
    log "$lab: score $SC | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3; reference lift 10.7 / 6.2 side / 4.1 pivot cm) $parts"
}
BEST=0
for N in 400 800 1200 1600 2000 2600 3200 $((ITERS - 1)); do
    CK="$RUN/model_$N.pt"
    while [ ! -f "$CK" ]; do
        pgrep -f "$PAT_AMP" >/dev/null || break
        sleep 30
    done
    [ -f "$CK" ] || { log "walker process gone before model_$N; newest $(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)"; break; }
    sleep 20; cp "$CK" "$LOGD/walker_v5_$N.pt"
    test_ck "$LOGD/walker_v5_$N.pt" "model_$N"
    if python3 -c "import sys; sys.exit(0 if float('$SC') > float('$BEST') else 1)"; then
        BEST=$SC; cp "$CK" archive_anchors/walker_v5_best.pt; log "  -> new best (score $SC): archive_anchors/walker_v5_best.pt = model_$N"
    fi
done
log "DONE: best score $BEST. Pushes are still frozen; hardening is the next decision."
