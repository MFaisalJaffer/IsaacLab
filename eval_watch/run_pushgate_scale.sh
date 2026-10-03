#!/usr/bin/env bash
set -u
IL=/home/faisal/IsaacLab
cd "$IL"
CK=$(ls $IL/logs/rsl_rl/kbot_legs_rough/*/model_*.pt 2>/dev/null | awk -F'model_' '{n=$2+0; print n"\t"$0}' | sort -n | tail -1 | cut -f2-)
OUT=$IL/eval_watch/pushgate_scale.txt
: > "$OUT"
echo "=== PUSH GATE at SCALE: same 20N x 1.5s lean, env-count sweep (was 32) ===" >> "$OUT"
echo "=== ckpt=$(basename $CK) ===" >> "$OUT"
for N in 32 256 1024; do
  for t in $(seq 1 90); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -gt 4000 ] && break; sleep 10
  done
  KBOT_ADAPT=1 KBOT_FLAT=1 "$IL/kbot_env/bin/python" "$IL/eval_watch/push_response_probe.py" \
    --checkpoint "$CK" --vx 0.0 --axis y --force_n 20 --force_steps 75 --pin_gains \
    --num_envs "$N" --label "pushgate_n${N}" --out "$OUT" \
    > "$IL/eval_watch/pg_n${N}.log" 2>&1
  echo "[n=$N exit $?]" >> "$OUT"
done
