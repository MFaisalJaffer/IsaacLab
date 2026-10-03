#!/bin/bash
# Switch from walker v4 to walker v5 — RUN ONLY ON THE USER'S GO (it stops the walker v4 training).
#   1. keep + verify walker v4's newest checkpoint (archive_anchors/walker_v4_last_model_<N>.pt)
#   2. stop the v4 autopilot (process group of eval_watch/walker_v4_pipeline.sh) and then the v4 training
#   3. start eval_watch/walker_v5_pipeline.sh detached (it waits for the GPU to be idle, then launches)
# Nothing is deleted: the v4 run folder, its checkpoints and walker_v4_best.pt stay where they are.
# Usage: bash eval_watch/switch_to_walker_v5.sh <v4 autopilot pid> <v4 training python pid>
set -u
IL=/home/faisal/IsaacLab; cd "$IL" || exit 1
source kbot_env/bin/activate
AP=${1:?v4 autopilot pid}; TR=${2:?v4 training pid}
ST=$IL/eval_watch/WALKER_V4_STATUS.md
log() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; echo "$*"; }
cmdline() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

a="walker_v4_pipe"; b="line.sh"; c="train_am"; d="p.py"
case "$(cmdline "$TR")" in *"$c$d"*) ;; *) echo "pid $TR is not the walker training — nothing done"; exit 1 ;; esac
RUN=$(ls -dt logs/rsl_rl/kbot_legs_amp/*_walker_v4/ | head -1)

# 1. newest checkpoint: not being written right now, copied, and loadable
NEW=$(ls -t "$RUN"model_*.pt | head -1)
while [ $(( $(date +%s) - $(stat -c %Y "$NEW") )) -lt 20 ]; do sleep 5; done
KEEP=archive_anchors/walker_v4_last_$(basename "$NEW")
cp "$NEW" "$KEEP"
python - "$KEEP" <<'PY' || { echo "the checkpoint copy does not load — nothing stopped"; exit 1; }
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert "model_state_dict" in d and d["model_state_dict"]["actor.0.weight"].shape[1] == 430
print("checkpoint copy verified:", sys.argv[1], "iteration", d.get("iter"))
PY

# 2. stop the autopilot (it only waits and tests), then the training
case "$(cmdline "$AP")" in
    *"$a$b"*) kill -- "-$AP" 2>/dev/null; echo "v4 autopilot stopped" ;;
    *) echo "pid $AP is not the v4 autopilot (already gone?) — continuing" ;;
esac
kill "$TR"
for i in $(seq 1 30); do [ -d "/proc/$TR" ] || break; sleep 2; done
[ -d "/proc/$TR" ] && { kill -9 "$TR"; sleep 3; }
log "walker v4 STOPPED on the user's decision at $(basename "$NEW") (kept as $KEEP); switching to walker v5 (v4 + stepping anchor)"

# 3. walker v5 autopilot, detached
setsid nohup bash eval_watch/walker_v5_pipeline.sh > logs/autopilot/walker_v5_pipeline.out 2>&1 < /dev/null &
echo "walker v5 autopilot started; status in eval_watch/WALKER_V5_STATUS.md"
