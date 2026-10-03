#!/bin/bash
# AUTOPILOT 2026-10-01 17:40 -> ~22:00 (user away 3-4 h). Progress: eval_watch/AUTOPILOT_STATUS.md
# Chain: clip tracker v7 (library v4) evaluated per label at 1000/1500/2000 (+2500/final) -> best checkpoint by a
#        body-motion score -> early stop if 2000 is not better than 1500 -> record dataset v3 (fall windows
#        masked) -> AMP stage (warm = best walker amp_pilot2_hard_mirroroff, judge on v1+v2+v3, commands widened
#        to backward/lateral/yaw, pushes frozen) -> evals + gifs -> hardening (push ramp on) if it passes.
# Rules: one training at a time; anchor before stopping; nothing deleted; kill by PID only; lineage 11 paused.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1
ST=$IL/eval_watch/AUTOPILOT_STATUS.md
LOGD=$IL/logs/autopilot; mkdir -p "$LOGD"
V7_RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_trackclip/2026-10-01_17-0*/ | head -1)
CLIP_ENV="KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 KBOT_CLIP_GATE=0 KBOT_CLIP_GATE_ANG=0 KBOT_CLIP_DIR=$IL/eval_watch/amp_refs/lafan1/clips_v4"
AMP_ENV="KBOT_AMP_VX=-0.30:0.50 KBOT_AMP_VY=-0.15:0.15 KBOT_AMP_WZ=-0.50:0.50 KBOT_AMP_STYLE_W=2.0 KBOT_AMP_PUSH_FREEZE=1 KBOT_AMP_STANCE_W=5 KBOT_AMP_MIRROR=0"
DS1=eval_watch/amp_refs/asimov_tracked_kbot.npz
DS2=eval_watch/amp_refs/asimov_tracked_v2_kbot.npz
DS3=eval_watch/amp_refs/lafan1_tracked_v4_kbot.npz
DS4=eval_watch/amp_refs/lafan1_v4_kinematic.npz   # library v4 itself, label-balanced (user go 18:10): v7 copies the joint
FILES="$DS1,$DS2,$DS3,$DS4,$DS4"                   # shapes but travels at ~1/3 speed, so the judge also sees the real clips (x2 weight)
c="train_am"; d="p.py"; PAT_AMP="$c$d"   # pattern from parts: never matches this script's own command line

log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
newest() { ls -t "$1"/model_*.pt 2>/dev/null | head -1; }
v7_pids() { for p in $(pgrep -f python 2>/dev/null); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q 'KBOT_CLIP_DIR=.*clips_v4' && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'rsl_rl/train.py' && echo "$p"; done; }
v7_alive() { [ -n "$(v7_pids)" ]; }
stop_v7() { for p in $(v7_pids); do pp=$(ps -o ppid= -p "$p" | tr -d ' '); kill "$p" "$pp" 2>/dev/null; done; sleep 20; for p in $(v7_pids); do kill -9 "$p" 2>/dev/null; done; sleep 5; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 60; done; }
jget() { python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    r = r[k]
print(r)
PY
}
score() {  # body-motion score of a per-label clip eval: mean over labels of survival x primary-speed achievement
python3 - "$1" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))
P = {"backward": 0, "forward_straight": 0, "forward_turn": 0, "start_stop": 0, "stop_start": 0, "side_left": 1, "side_right": 1}
tot = n = 0
for lab, v in r["labels"].items():
    surv = 1.0 - v["fall_frac_of_ends"]
    if lab == "turn_in_place":
        ach = max(0.0, 1.0 - v["yaw_err"] / 0.6)
    else:
        k = P[lab]; c = v["cmd_mean"][k]; a = v["achieved_mean"][k]
        ach = min(1.0, max(0.0, a / c)) if abs(c) > 0.03 else 1.0
    tot += surv * ach; n += 1
print(f"{tot / max(n, 1):.3f}")
PY
}
summ() {  # one-line human summary of a per-label clip eval
python3 - "$1" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1])); L = r["labels"]
def g(lab, k): return L[lab]["achieved_mean"][k], L[lab]["cmd_mean"][k]
b = g("backward", 0); s = g("side_left", 1); f = g("forward_straight", 0)
surv = sum(1 - v["fall_frac_of_ends"] for v in L.values()) / len(L)
rm = sum(v["rmse_deg"] for v in L.values()) / len(L)
print(f"survival {surv:.2f}, rmse {rm:.1f} deg, backward {b[0]:.2f}/{b[1]:.2f}, side {s[0]:.2f}/{s[1]:.2f}, fwd {f[0]:.2f}/{f[1]:.2f} m/s, pivot yaw err {L['turn_in_place']['yaw_err']:.2f}")
PY
}
clip_eval() {  # $1 ckpt  $2 tag  $3 seconds  [$4 record path]  [$5 num_envs]
    rec=""; [ -n "${4:-}" ] && rec="--record $4"
    env $CLIP_ENV ./isaaclab.sh -p eval_watch/amp_clip_eval.py --checkpoint "$1" --num_envs "${5:-256}" --seconds "$3" $rec --tag "$2" --headless > "$LOGD/$2.log" 2>&1
}

echo "# AUTOPILOT v7b $(date '+%Y-%m-%d %H:%M') — v7 run $(basename "$V7_RUN"); AMP judge = v1 + v2 + v3 (v7 recording, small) + kinematic library v4 (x2)" >> "$ST"
log "started; v7 alive: $(v7_alive && echo yes || echo no); GPU procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"

# ---------------------------------------------------------------- 1. v7 evals + best checkpoint
TABLE="$LOGD/v7_scores.txt"; : > "$TABLE"
BEST=""; BEST_S=0; S1500=0; S2000=0; EARLY=0
for N in 1000 1500 2000 2500; do
    while [ ! -f "$V7_RUN/model_$N.pt" ] && v7_alive; do sleep 30; done
    [ -f "$V7_RUN/model_$N.pt" ] || { log "v7 ended before model_$N"; break; }
    sleep 20
    REUSED=""; if [ -f eval_watch/amp_clip_eval_v7_$N.json ]; then REUSED=" (eval reused)"; else clip_eval "$V7_RUN/model_$N.pt" "amp_clip_eval_v7_$N" 30; fi
    S=$(score eval_watch/amp_clip_eval_v7_$N.json); [ -n "$S" ] || S=0
    echo "$N $S" >> "$TABLE"
    log "v7 model_$N$REUSED: score $S — $(summ eval_watch/amp_clip_eval_v7_$N.json)"
    if python3 -c "import sys; sys.exit(0 if float('$S') > float('$BEST_S') else 1)"; then BEST="$V7_RUN/model_$N.pt"; BEST_S=$S; fi
    [ "$N" = 1500 ] && S1500=$S
    if [ "$N" = 2000 ]; then
        S2000=$S
        if python3 -c "import sys; sys.exit(0 if float('$S2000') < float('$S1500') * 1.05 else 1)"; then
            log "2000 not >5% better than 1500 (${S2000} vs ${S1500}): stopping v7 early (anchor kept)"
            EARLY=1; stop_v7; break
        fi
    fi
done
# let the run finish on its own unless it was stopped early, then evaluate its final checkpoint
[ "$EARLY" = 1 ] || while v7_alive; do sleep 30; done
if ! v7_alive; then
    CK=$(newest "$V7_RUN"); N=$(basename "$CK" .pt | sed 's/model_//')
    if ! grep -q "^$N " "$TABLE"; then
        clip_eval "$CK" "amp_clip_eval_v7_$N" 30
        S=$(score eval_watch/amp_clip_eval_v7_$N.json); [ -n "$S" ] || S=0
        echo "$N $S" >> "$TABLE"; log "v7 final model_$N: score $S — $(summ eval_watch/amp_clip_eval_v7_$N.json)"
        if python3 -c "import sys; sys.exit(0 if float('$S') > float('$BEST_S') else 1)"; then BEST="$CK"; BEST_S=$S; fi
    fi
fi
[ -n "$BEST" ] || { log "no evaluated checkpoint — pipeline stopped"; exit 1; }
while v7_alive; do stop_v7; done
BN=$(basename "$BEST" .pt); cp "$BEST" "archive_anchors/trackclip_v7_best_${BN}.pt"
log "v7 BEST = $BN (score $BEST_S) anchored as archive_anchors/trackclip_v7_best_${BN}.pt; v7 stopped"

# ---------------------------------------------------------------- 2. record dataset v3
clip_eval "$BEST" "record_v7_${BN}" 30 "$DS3" 64   # small on purpose: the full-amplitude clips must outweigh the shuffle
python3 - "$DS3" <<'PY' >> "$LOGD/ds3_check.txt" 2>&1
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("md", "/home/faisal/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/kbot_legs/amp/motion_dataset.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
ds = md.MotionDataset([sys.argv[1]], device="cpu", fps_expected=50.0, mirror=True)
d = np.load(sys.argv[1], allow_pickle=True)
print(ds.describe(), "| falls masked:", int(d["done"].sum() - d["done_raw"].sum()), "frames")
PY
log "dataset v3 recorded: $(tail -n 1 "$LOGD/ds3_check.txt")"

# ---------------------------------------------------------------- 3. AMP stage
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { sleep 60; }
env $AMP_ENV KBOT_AMP_INIT_STD=0.1 KBOT_AMP_MOTION_FILES="$FILES" setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations 2000 --checkpoint archive_anchors/amp_pilot2_hard_mirroroff.pt > "$LOGD/amp_v3.log" 2>&1 < /dev/null &
sleep 180
AMP_RUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*/ | head -1)
echo "$LOGD/amp_v3.log" > logs/amp_pilot/CURRENT; echo "$LOGD/amp_v3.log" > logs/track_pilot/CURRENT
if pgrep -f "$PAT_AMP" >/dev/null; then
    log "AMP stage launched: $(basename "$AMP_RUN") — warm amp_pilot2_hard_mirroroff, judge v1+v2+v3+kinematic library v4 (x2), vx -0.3..0.5 vy ±0.15 wz ±0.5, pushes frozen, style 2.0, stance x5, mirror loss off, std 0.1, 2000 iters"
else
    log "AMP launch FAILED (no process after 3 min) — see $LOGD/amp_v3.log: $(grep -a -m1 -E 'Error|Traceback' "$LOGD/amp_v3.log" | cut -c1-160)"; exit 1
fi
wait_gone "$PAT_AMP"
CK2=$(newest "$AMP_RUN"); cp "$CK2" archive_anchors/amp_v3_final.pt
log "AMP stage finished: $(basename "$CK2") anchored as archive_anchors/amp_v3_final.pt"

# ---------------------------------------------------------------- 4. evaluate + render
WALK=0; BACK=0
for cmd in 0.4:0:0 -0.3:0:0 0:0.15:0 0:0:0.5 0:0:0; do
    tag=amp_v3_cmd_$(echo "$cmd" | tr ':' '_' | tr -d '-' )
    [ "${cmd:0:1}" = "-" ] && tag=amp_v3_cmd_back
    env $AMP_ENV KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_v3_final.pt --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
    if [ "$cmd" = "0:0:0" ]; then
        log "AMP v3 standing: survival $(jget eval_watch/$tag.json standing.survival), tilt $(jget eval_watch/$tag.json standing.tilt_deg.mean) deg, width $(jget eval_watch/$tag.json standing.width_cm) cm"
    else
        log "AMP v3 cmd $cmd: walking survival $(jget eval_watch/$tag.json survival_walking), speed $(jget eval_watch/$tag.json speed), yaw $(jget eval_watch/$tag.json yaw)"
        [ "$cmd" = "0.4:0:0" ] && WALK=$(jget eval_watch/$tag.json survival_walking)
        [ "$cmd" = "-0.3:0:0" ] && BACK=$(jget eval_watch/$tag.json survival_walking)
    fi
done
for cmd in -0.3:0:0 0:0.15:0; do
    tag=amp_v3_gif_$(echo "$cmd" | tr ':' '_' | tr -d '-'); [ "${cmd:0:1}" = "-" ] && tag=amp_v3_gif_back
    env $AMP_ENV KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_v3_final.pt --num_envs 16 --seconds 12 --no_standers --cmd="$cmd" --render_seconds 10 --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
    log "gif for cmd $cmd: $(ls -t eval_watch/${tag}*.gif 2>/dev/null | head -1)"
done

# ---------------------------------------------------------------- 5. hardening if it passes and time allows
HARD=$(python3 -c "print(int(float('${WALK:-0}') >= 0.8 and float('${BACK:-0}') >= 0.6))")
if [ "$HARD" = "1" ] && [ "$(date +%H%M)" -lt 2230 ]; then
    env $AMP_ENV KBOT_AMP_PUSH_FREEZE=0 KBOT_AMP_MOTION_FILES="$FILES" setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0 --headless --num_envs 8192 --max_iterations 1500 --resume --load_run "$(basename "$AMP_RUN")" --checkpoint "$(basename "$CK2")" > "$LOGD/amp_v3_hard.log" 2>&1 < /dev/null &
    sleep 180
    HRUN=$(ls -dt "$IL"/logs/rsl_rl/kbot_legs_amp/*/ | head -1); echo "$LOGD/amp_v3_hard.log" > logs/amp_pilot/CURRENT
    log "PASSED (walk $WALK >= 0.8, backward $BACK >= 0.6): HARDENING launched (push ramp on) $(basename "$HRUN"), 1500 iters"
    wait_gone "$PAT_AMP"
    CK3=$(newest "$HRUN"); cp "$CK3" archive_anchors/amp_v3_hardened.pt
    for cmd in 0:0:0 0.4:0:0 -0.3:0:0; do
        tag=amp_v3_hard_cmd_$(echo "$cmd" | tr ':' '_' | tr -d '-'); [ "${cmd:0:1}" = "-" ] && tag=amp_v3_hard_cmd_back
        env $AMP_ENV KBOT_AMP_PUSH_FREEZE=0 KBOT_AMP_MOTION_FILES="$FILES" ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint archive_anchors/amp_v3_hardened.pt --num_envs 256 --seconds 20 --cmd="$cmd" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
        log "hardened cmd $cmd: $( [ "$cmd" = "0:0:0" ] && echo "stand survival $(jget eval_watch/$tag.json standing.survival)" || echo "walking survival $(jget eval_watch/$tag.json survival_walking), speed $(jget eval_watch/$tag.json speed)")"
    done
    log "hardened anchored as archive_anchors/amp_v3_hardened.pt"
else
    log "hardening skipped (walk ${WALK:-?} >= 0.8 and backward ${BACK:-?} >= 0.6 needed, or past 22:30) — left for review"
fi
log "DONE. GPU idle. Lineage 11 still paused."
