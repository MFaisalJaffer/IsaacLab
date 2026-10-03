#!/bin/bash
# Render an obstacle-course policy meeting obstacles head-on (mesh render, side + rear view).
#   bash eval_watch/obstacle_render.sh <checkpoint> <prefix> <kind>:<level> [<kind>:<level> ...]
#   e.g. ... logs/autopilot/obst_s1_map_800.pt obst_s1_map_800_gif platform:2 platform:4 beam:4
# -> eval_watch/<prefix>_<kind><level>.gif (+ filmstrip). Level k = (2 + 2k) cm. KBOT_OBST_BLIND=1 in the
# environment renders the blind policy. One GPU side job at a time (lock shared with the autopilot's tests).
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
source eval_watch/obstacle_env.sh
CK=${1:?checkpoint}; PRE=${2:?prefix}; shift 2
LOGD=$IL/logs/autopilot
for pair in "$@"; do
    kind=${pair%%:*}; level=${pair#*:}; tag=${PRE}_${kind}${level}
    (
        flock -w 1800 9 || exit 1
        n=0; while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6800 ]; do n=$((n + 1)); [ "$n" -ge 90 ] && break; sleep 10; done
        ./isaaclab.sh -p eval_watch/obstacle_eval.py --checkpoint "$CK" --num_envs 200 --seconds 14 --yaw0 --render_seconds 13 --render_kind "$kind" --render_level "$level" --tag "$tag" --headless > "$LOGD/$tag.log" 2>&1
    ) 9> "$LOGD/gpu_side_job.lock"
    echo "$(date '+%H:%M:%S') $pair: $(grep -a '\[render\]' "$LOGD/$tag.log" | tail -n 1 | cut -c1-200)"
done
echo "RENDER DONE"
