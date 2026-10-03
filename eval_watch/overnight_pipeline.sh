#!/usr/bin/env bash
# OVERNIGHT PIPELINE 2026-09-30 -> 10-01 (user away ~8 h). Progress: eval_watch/OVERNIGHT_STATUS.md
# Chain: tracker v2c finishes -> turn test (retry once) -> record dataset v2 -> AMP pilot 2 (warm from the
# tracker, datasets v1+v2, pushes frozen) -> eval + gif -> hardening (push ramp on) if it passes.
# Rules: one training at a time; anchor before stopping; nothing deleted; lineage 11 stays paused.
set -u
IL=/home/faisal/IsaacLab
cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0
ST=$IL/eval_watch/OVERNIGHT_STATUS.md
LOGD=$IL/logs/overnight; mkdir -p "$LOGD"
TRK_RUN=$IL/logs/rsl_rl/kbot_legs_track/2026-09-30_23-24-08
TRK_ENV="KBOT_TRACK_YAW=0.3 KBOT_TRACK_GATE=0.1 KBOT_TRACK_GATE_ANG=0.05 KBOT_TRACK_TASK_W=2"
AMP_ENV="KBOT_AMP_VX=0.30:0.50 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_PUSH_FREEZE=1"
DS1=eval_watch/amp_refs/asimov_tracked_kbot.npz
DS2=eval_watch/amp_refs/asimov_tracked_v2_kbot.npz
a="Isaac-Tr"; b="ack-Kbot"; PAT_TRK="$a$b"          # pattern built from parts: never matches this script
c="train_am"; d="p.py";      PAT_AMP="$c$d"

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 60; done; }
newest() { ls -t "$1"/model_*.pt 2>/dev/null | head -1; }
jget() {  # jget file.json key1.key2.key3  (dotted path; no quotes needed)
    python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    r = r[k]
print(r)
PY
}

# ---------------------------------------------------------------- 1. tracker v2c
log "pipeline started; waiting for tracker v2c ($TRK_RUN) to finish"
wait_gone "$PAT_TRK"
CK=$(newest "$TRK_RUN"); cp "$CK" archive_anchors/track_v2c_final.pt
log "tracker v2c finished: $(basename "$CK") anchored as archive_anchors/track_v2c_final.pt"

turn_test() {  # $1 = checkpoint, $2 = tag -> prints "yaw rmse surv"
    env $TRK_ENV ./isaaclab.sh -p eval_watch/amp_track_eval.py --checkpoint "$1" --num_envs 128 --seconds 20 --cmd 0.3:0:0.3 --tag "$2" --headless > "$LOGD/$2.log" 2>&1
    echo "$(jget eval_watch/$2.json "yaw.achieved_mean_signed") $(jget eval_watch/$2.json "rmse_all_deg") $(jget eval_watch/$2.json "survival_frac")"
}
read -r YAW RMSE SURV <<< "$(turn_test archive_anchors/track_v2c_final.pt track_v2c_final_turn)"
log "turn test v2c: yaw ${YAW:-?} rad/s (cmd 0.30), stride rmse ${RMSE:-?} deg, survival ${SURV:-?}"
PASS=$(python3 -c "print(int(float('${YAW:-0}') >= 0.2 and float('${RMSE:-99}') <= 4.0))")
if [ "$PASS" != "1" ]; then
    log "turn test below bar (yaw >= 0.2, rmse <= 4): ONE retry — 1200 more iterations"
    env $TRK_ENV setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Track-KbotLegs-v0 --headless --num_envs 8192 --max_iterations 1200 --resume --load_run "$(basename "$TRK_RUN")" --checkpoint "$(basename "$CK")" > "$LOGD/tracker_v2c_retry.log" 2>&1 < /dev/null &
    sleep 120; wait_gone "$PAT_TRK"
    RUN2=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_track/*/ | head -1); CK=$(newest "$RUN2"); cp "$CK" archive_anchors/track_v2c_final.pt
    read -r YAW RMSE SURV <<< "$(turn_test archive_anchors/track_v2c_final.pt track_v2c_retry_turn)"
    log "turn test after retry: yaw ${YAW:-?}, rmse ${RMSE:-?}, survival ${SURV:-?} (proceeding regardless)"
fi

# ---------------------------------------------------------------- 2. record dataset v2 (turn commands sampled by the task)
env $TRK_ENV ./isaaclab.sh -p eval_watch/amp_track_eval.py --checkpoint archive_anchors/track_v2c_final.pt --num_envs 256 --seconds 20 --record "$DS2" --tag track_v2c_record --headless > "$LOGD/record_v2.log" 2>&1
if [ -f "$DS2" ]; then log "dataset v2 recorded: $DS2 (survivors only, turn commands +-0.3), merged with v1 for the judge"; else log "RECORD FAILED — see $LOGD/record_v2.log; using dataset v1 only"; DS2=""; fi
FILES="$DS1"; [ -n "$DS2" ] && FILES="$DS1,$DS2"

# ---------------------------------------------------------------- 3. AMP pilot 2 (warm start from the turning tracker)
env $AMP_ENV KBOT_AMP_INIT_STD=0.08 KBOT_AMP_MOTION_FILES="$FILES" setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations 3000 --checkpoint archive_anchors/track_v2c_final.pt > "$LOGD/pilot2.log" 2>&1 < /dev/null &
sleep 120
AMP_RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*/ | head -1)
log "AMP pilot 2 launched: $AMP_RUN (3000 iters, datasets: $FILES, pushes frozen, style 2.0, std 0.08)"
wait_gone "$PAT_AMP"
CK2=$(newest "$AMP_RUN"); cp "$CK2" archive_anchors/amp_pilot2_final.pt
log "pilot 2 finished: $(basename "$CK2") anchored as archive_anchors/amp_pilot2_final.pt"

# ---------------------------------------------------------------- 4. evaluate + render
for cmd in 0:0:0 0.3:0:0.3 0.4:0:0; do
    tag=amp_pilot2_final_cmd_$(echo "$cmd" | tr ':' '_')
    env $AMP_ENV KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_pilot2_final.pt --num_envs 256 --seconds 20 --cmd "$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
    if [ "$cmd" = "0:0:0" ]; then
        STAND=$(jget eval_watch/$tag.json "standing.survival"); STILT=$(jget eval_watch/$tag.json "standing.tilt_deg.mean"); SW=$(jget eval_watch/$tag.json "standing.width_cm")
        log "pilot 2 STAND: survival ${STAND:-?}, tilt ${STILT:-?} deg, stance ${SW:-?} cm"
    else
        log "pilot 2 cmd $cmd: survival $(jget eval_watch/$tag.json "survival_walking"), fwd $(jget eval_watch/$tag.json "speed.achieved_fwd_mean") m/s, yaw $(jget eval_watch/$tag.json "yaw.achieved_mean_signed") (cmd $(jget eval_watch/$tag.json "yaw.cmd_mean")), width $(jget eval_watch/$tag.json "stance_width_cm.median") cm, clearance $(jget eval_watch/$tag.json "clearance_cm") cm, stride $(jget eval_watch/$tag.json "stride_period_s.median") s, knee p5/p95 $(jget eval_watch/$tag.json "joint_ranges_deg.policy.knee.p5")/$(jget eval_watch/$tag.json "joint_ranges_deg.policy.knee.p95"), judge $(jget eval_watch/$tag.json "style_score_walking")"
        [ "$cmd" = "0.3:0:0.3" ] && YAW2=$(jget eval_watch/$tag.json "yaw.achieved_mean_signed")
    fi
done
env $AMP_ENV KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_pilot2_final.pt --num_envs 16 --seconds 12 --no_standers --render_seconds 10 --tag amp_pilot2_final --headless --rendering_mode preview > "$LOGD/pilot2_render.log" 2>&1
[ -f eval_watch/amp_pilot2_final.gif ] && log "render: eval_watch/amp_pilot2_final.gif + amp_pilot2_final_filmstrip.png (also on :8800)"

# ---------------------------------------------------------------- 5. hardening if it passes
HARD=$(python3 -c "print(int(float('${STAND:-0}') >= 0.85 and float('${YAW2:-0}') >= 0.2))")
if [ "$HARD" = "1" ]; then
    env KBOT_AMP_VX=0.30:0.50 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_MOTION_FILES="$FILES" setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations 2000 --resume --load_run "$(basename "$AMP_RUN")" --checkpoint "$(basename "$CK2")" > "$LOGD/pilot2_hardening.log" 2>&1 < /dev/null &
    sleep 120
    log "PASSED the bar (stand >= 0.85, yaw >= 0.2): HARDENING launched (push ramp ON) from $(basename "$CK2"), 2000 iters: $(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*/ | head -1)"
    wait_gone "$PAT_AMP"
    CK3=$(newest "$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*/ | head -1)"); cp "$CK3" archive_anchors/amp_pilot2_hardened.pt
    for cmd in 0:0:0 0.4:0:0; do
        tag=amp_pilot2_hard_cmd_$(echo "$cmd" | tr ':' '_')
        env KBOT_AMP_VX=0.30:0.50 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_pilot2_hardened.pt --num_envs 256 --seconds 20 --cmd "$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
    done
    log "hardened: $(basename "$CK3") anchored; STAND survival $(jget eval_watch/amp_pilot2_hard_cmd_0_0_0.json "standing.survival") (pushes ramping in training; eval at the start band), WALK survival $(jget eval_watch/amp_pilot2_hard_cmd_0.4_0_0.json "survival_walking") fwd $(jget eval_watch/amp_pilot2_hard_cmd_0.4_0_0.json "speed.achieved_fwd_mean")"
else
    log "did NOT pass the bar (stand ${STAND:-?} >= 0.85 and yaw ${YAW2:-?} >= 0.2) — hardening skipped, left for review"
fi
log "pipeline done. GPU idle. Lineage 11 still paused (resume: systemctl --user start kbot-train, then kbot-watchdog)."
