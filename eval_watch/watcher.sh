#!/usr/bin/env bash
# Dual checkpoint-watcher: each cycle renders BOTH flat runs (zero-pose and
# bent-pose) into per-run filmstrip + GIF, served on 8800 (index.html shows
# both). Renders are sequential, so GPU only holds one render at a time.
set -u
IL=/home/faisal/IsaacLab
OUT=$IL/eval_watch
PY=$IL/kbot_env/bin/python
FF=$IL/kbot_env/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
SLEEP_SECS=600          # 10 min between cycles
BOOT_WAIT=300           # max seconds to wait for an mp4 before giving up

mkdir -p "$OUT"
log() { echo "[$(date '+%H:%M:%S')] $*" >> "$OUT/watcher.log"; }

# render_one TAG CKPT_GLOB VIDEO_GLOB ENV_PREFIX OUT_PREFIX
render_one() {
    local tag="$1" ckpt_glob="$2" video_glob="$3" envp="$4" pre="$5"
    local newest last iter mp4 waited play_pid expdir
    newest=$(ls -t $ckpt_glob 2>/dev/null | head -1)
    if [ -z "$newest" ]; then log "$tag: no checkpoint yet"; return; fi
    last=$(cat "$OUT/${pre}last_ckpt.txt" 2>/dev/null)
    if [ "$newest" = "$last" ]; then return; fi      # nothing new since last render
    iter=$(basename "$newest" .pt | sed 's/model_//')
    log "$tag: rendering iter=$iter"
    rm -f $video_glob/*.mp4 2>/dev/null
    cd "$IL"
    # preview RTX mode (parts visible, low VRAM); env vars match each run's training
    env $envp "$PY" scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Velocity-Rough-KbotLegs-v0 --num_envs 2 \
        --checkpoint "$newest" --video --video_length 150 --headless \
        --rendering_mode preview > "$OUT/play_${tag}.log" 2>&1 &
    play_pid=$!
    waited=0; mp4=""
    while [ $waited -lt $BOOT_WAIT ]; do
        mp4=$(ls -t $video_glob/*.mp4 2>/dev/null | head -1)
        [ -n "$mp4" ] && break
        sleep 5; waited=$((waited+5))
    done
    [ -n "$mp4" ] && sleep 8
    kill -9 "$play_pid" 2>/dev/null
    pkill -9 -f "play.py --task Isaac-Velocity" 2>/dev/null
    sleep 3
    if [ -z "$mp4" ]; then log "$tag: no mp4 (boot/hang); retry next cycle"; return; fi
    "$FF" -y -i "$mp4" -vf "select='not(mod(n\,22))',scale=340:-1,tile=7x1" \
        -frames:v 1 "$OUT/${pre}filmstrip.png" >/dev/null 2>&1
    "$FF" -y -i "$mp4" -vf "fps=12,scale=600:-1" "$OUT/${pre}latest.gif" >/dev/null 2>&1
    expdir="$(dirname "$newest")/exported"
    cp -f "$expdir/policy.onnx" "$OUT/${pre}policy.onnx" 2>/dev/null
    echo "iter $iter   rendered $(date '+%Y-%m-%d %H:%M:%S')" > "$OUT/${pre}status.txt"
    echo "$newest" > "$OUT/${pre}last_ckpt.txt"
    log "$tag: done iter=$iter"
}

log "dual watcher started"
echo 0 > "$OUT/zero_last_ckpt.txt"; echo 0 > "$OUT/bent_last_ckpt.txt"
while true; do
    render_one zero "$IL/logs/rsl_rl/kbot_legs_rough/*/model_*.pt" \
        "$IL/logs/rsl_rl/kbot_legs_rough/*/videos/play" "KBOT_FLAT=1" "zero_"
    render_one curric "$IL/logs/rsl_rl/kbot_legs_curric/*/model_*.pt" \
        "$IL/logs/rsl_rl/kbot_legs_curric/*/videos/play" "KBOT_MAXINIT=0" "curric_"
    sleep "$SLEEP_SECS"
done
