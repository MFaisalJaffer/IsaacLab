#!/usr/bin/env bash
# ON-DEMAND render watcher for the :8800 K-Bot dashboard.
#   IDLE BY DEFAULT — zero GPU cost during training. Renders ONE checkpoint only
#   when the dashboard's "Render" button writes render_request.flag, then returns
#   to idle until the next click. (Was a 10-min periodic loop costing ~3% training
#   throughput; made on-demand 2026-07-24 at user request.)
#
# WHICH EXPERIMENT (2026-09-29): the flag's content picks it —
#   auto   whatever is training right now: train_amp.py -> amp, a Track task -> track,
#          else the lineage (kbot_legs_rough); nothing running -> lineage's newest ckpt
#   rough | amp | track | clip   explicit choice from the dashboard dropdown (clip = LAFAN1 clip tracker)
# The render uses the SAME KBOT_* env vars as the running trainer (read from
# /proc/<pid>/environ) so it sees the config the checkpoint was trained on
# (clock, command band, DR); with no trainer running it falls back to the service
# defaults. AMP checkpoints go through play_amp.py (stock runner rejects AMPPPO).
# Outputs: dr_latest.gif / dr_filmstrip.png / dr_status.txt (current), and the
# previous render is kept as prev_* so two checkpoints can be compared on the page.
set -u
IL=/home/faisal/IsaacLab
OUT=$IL/eval_watch
PY=$IL/kbot_env/bin/python
FF=$IL/kbot_env/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
POLL=10                 # seconds between flag checks (click -> render within ~10 s)
BOOT_WAIT=300
FLAG=$OUT/render_request.flag
STATE=$OUT/render_state.txt
DEFAULT_ENV="KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0"

mkdir -p "$OUT"
log() { echo "[$(date '+%H:%M:%S')] $*" >> "$OUT/watcher_isaac.log"; }

# resolve_exp <choice>  -> sets EXP TASK SCRIPT LABEL and RENDER_ENV (space-separated VAR=val)
resolve_exp() {
    local choice="$1" pid=""
    if [ "$choice" = "auto" ] || [ -z "$choice" ]; then
        if pgrep -f "train_amp.py" >/dev/null; then choice=amp
        elif pgrep -f "TrackMulti-Kbot" >/dev/null; then choice=multi
        elif pgrep -f "TrackClip-Kbot" >/dev/null; then choice=clip
        elif pgrep -f "Isaac-Track-Kbot" >/dev/null; then choice=track
        else choice=rough; fi
    fi
    case "$choice" in
        amp)   EXP=kbot_legs_amp;   TASK=Isaac-Velocity-Rough-KbotLegs-AMP-v0; SCRIPT=play_amp.py; LABEL="AMP pilot";  pid=$(pgrep -f "train_amp.py" | head -1) ;;
        track) EXP=kbot_legs_track; TASK=Isaac-Track-KbotLegs-v0;              SCRIPT=play.py;     LABEL="tracking";   pid=$(pgrep -f "Isaac-Track-Kbot" | head -1) ;;
        multi) EXP=kbot_legs_trackmulti; TASK=Isaac-TrackMulti-KbotLegs-v0;    SCRIPT=play.py;     LABEL="multi-cycle tracker"; pid=$(pgrep -f "TrackMulti-Kbot" | head -1) ;;
        clip)  EXP=kbot_legs_trackclip; TASK=Isaac-TrackClip-KbotLegs-v0;      SCRIPT=play.py;     LABEL="clip tracking"; pid=$(pgrep -f "TrackClip-Kbot" | head -1) ;;
        *)     EXP=kbot_legs_rough; TASK=Isaac-Velocity-Rough-KbotLegs-v0;     SCRIPT=play.py;     LABEL="lineage";    pid=$(pgrep -f "rsl_rl/train.py --task=Isaac-Velocity-Rough-KbotLegs-v0" | head -1) ;;
    esac
    RENDER_ENV=""
    if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
        RENDER_ENV=$(tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^KBOT_' | grep -v '^KBOT_NUM_ENVS=' | tr '\n' ' ')
    fi
    [ -z "$RENDER_ENV" ] && RENDER_ENV="$DEFAULT_ENV"
    CHOICE=$choice
}

# render <prefix> <tb_tag>   (uses EXP TASK SCRIPT RENDER_ENV newest iter RUN)
render() {
    local pre="$1"; shift
    local tag="$1"; shift
    local vglob="$IL/logs/rsl_rl/$EXP/*/videos/play"
    rm -f $vglob/*.mp4 2>/dev/null
    cd "$IL"
    env PYTHONUNBUFFERED=1 $RENDER_ENV "$PY" scripts/reinforcement_learning/rsl_rl/$SCRIPT \
        --task $TASK --num_envs 2 \
        --checkpoint "$newest" --video --video_length 200 --headless \
        --rendering_mode preview > "$OUT/play_${pre}.log" 2>&1 &
    local play_pid=$! waited=0 mp4=""
    while [ $waited -lt $BOOT_WAIT ]; do
        mp4=$(ls -t $vglob/*.mp4 2>/dev/null | head -1)
        [ -n "$mp4" ] && break
        sleep 5; waited=$((waited+5))
    done
    [ -n "$mp4" ] && sleep 8
    kill -9 "$play_pid" 2>/dev/null
    pkill -9 -f "$SCRIPT --task $TASK" 2>/dev/null
    sleep 3
    if [ -z "$mp4" ]; then log "  [$pre] no mp4 (boot/hang) — see play_${pre}.log"; return 1; fi
    # keep the previous render for side-by-side comparison
    for f in latest.gif filmstrip.png status.txt; do
        [ -f "$OUT/${pre}_$f" ] && cp "$OUT/${pre}_$f" "$OUT/prev_$f"
    done
    "$FF" -y -i "$mp4" -vf "select='not(mod(n\,28))',scale=360:-1,tile=7x1" \
        -frames:v 1 "$OUT/${pre}_filmstrip.png" >/dev/null 2>&1
    "$FF" -y -i "$mp4" -vf "fps=12,scale=620:-1" "$OUT/${pre}_latest.gif" >/dev/null 2>&1
    echo "[$LABEL] $EXP · $(basename "$RUN") · iter $iter · $TASK · rendered $(date '+%Y-%m-%d %H:%M:%S') · env: $RENDER_ENV" > "$OUT/${pre}_status.txt"
    "$PY" "$OUT/video_to_tb.py" "$OUT/${pre}_latest.gif" "$RUN" "$iter" "$tag" \
        >> "$OUT/watcher_isaac.log" 2>&1
    log "  [$pre] ok ($LABEL $EXP iter $iter)"
    return 0
}

log "isaac watcher started (ON-DEMAND: idle until Render clicked; auto-detects the running experiment)"
rm -f "$FLAG"
echo "idle — no renders yet (click Render on the dashboard)" > "$STATE"
while true; do
    if [ ! -f "$FLAG" ]; then sleep "$POLL"; continue; fi
    choice=$(head -c 32 "$FLAG" 2>/dev/null | tr -cd 'a-z')
    rm -f "$FLAG"
    resolve_exp "$choice"
    newest=$(ls -t $IL/logs/rsl_rl/$EXP/*/model_*.pt 2>/dev/null | head -1)
    if [ -z "$newest" ]; then echo "idle — no checkpoint yet for $EXP" > "$STATE"; log "no checkpoint for $EXP"; continue; fi
    RUN=$(dirname "$newest")/
    iter=$(basename "$newest" .pt | sed 's/model_//')
    echo "rendering $LABEL ($EXP) iter $iter (started $(date '+%H:%M:%S')) — ~3-5 min…" > "$STATE"
    log "ON-DEMAND render choice=$choice -> $EXP iter=$iter (run $(basename "$RUN")) env: $RENDER_ENV"

    # cache guard: each terrain boot cooks a multi-100MB DerivedDataCache blob,
    # never evicted (69 GB in a day). Purge entries >90 min old before rendering.
    find "$IL/kbot_env/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache" \
        -mindepth 1 -mmin +90 -delete 2>/dev/null

    # RENDER REGIME = TRAINING REGIME (2026-08-10): same env vars as the trainer.
    render dr "policy/rollout"

    echo "idle — last rendered $LABEL ($EXP) iter $iter at $(date '+%H:%M:%S') (click Render to refresh)" > "$STATE"
    log "done $EXP iter=$iter"
done
