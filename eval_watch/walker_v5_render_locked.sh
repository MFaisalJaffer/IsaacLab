#!/bin/bash
# Render a walker v5 checkpoint in six directions (mesh render, side + rear views) -> eval_watch/walker_v5_<N>_gif_<dir>.gif
# Usage: bash eval_watch/walker_v5_render.sh <N> <next test iteration>
# Waits until the autopilot has finished testing checkpoint N (training + test + render do not fit on the GPU
# together); each render takes the lock shared with eval_watch/walker_v5_tests.sh and waits for free GPU memory.
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0
N=${1:?checkpoint iteration}; NEXT=${2:?next test iteration}
LOGD=$IL/logs/autopilot
A=eval_watch/amp_refs
DS_TURN=$A/asimov_tracked_v2_kbot.npz; DS_MULTI=$A/multitrack_v2_kbot.npz
eval "$(sed -n '/^WALK_ENV=/p' eval_watch/walker_v5_pipeline.sh)"
CK=$LOGD/walker_v5_$N.pt
RUN=$(ls -dt logs/rsl_rl/kbot_legs_amp/*_walker_v5/ | head -1)
until grep -q -E "model_$N: score" eval_watch/WALKER_V5_STATUS.md; do sleep 15; done
sleep 10
for pair in fwd:0.4:0:0 back:-0.4:0:0 side:0:0.13:0 pivot:0:0:0.56 turn:0.3:0:0.3 stand:0:0:0; do
    name=${pair%%:*}; cmd=${pair#*:}
    it=$(grep -a -o "Learning iteration [0-9]*" "$LOGD/walker_v5.log" | tail -n 1 | grep -o "[0-9]*$")
    if [ "${it:-99999}" -ge $((NEXT - 45)) ] || [ -f "${RUN}model_$NEXT.pt" ]; then echo "skip $name: the autopilot's next test is close (iteration ${it:-?})"; continue; fi
    extra="--no_standers"; [ "$name" = "stand" ] && extra=""
    (   # one GPU side job at a time (lock shared with walker_v5_tests.sh), and only when the renderer fits
        flock -w 1800 9 || exit 1
        n=0; while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6800 ]; do n=$((n + 1)); [ "$n" -ge 90 ] && break; sleep 10; done
        env $WALK_ENV ./isaaclab.sh -p eval_watch/amp_gait_eval.py --checkpoint "$CK" --num_envs 16 --seconds 12 $extra --cmd="$cmd" --render_seconds 10 --tag "walker_v5_${N}_gif_$name" --headless > "$LOGD/walker_v5_${N}_gif_$name.log" 2>&1
    ) 9> "$LOGD/gpu_side_job.lock"
    echo "$(date '+%H:%M:%S') $name exit=$? at iteration $it: $(grep -a '\[render\]' "$LOGD/walker_v5_${N}_gif_$name.log" | tail -n 1 | cut -c1-140)"
done
echo "RENDER DONE"
