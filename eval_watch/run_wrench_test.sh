#!/usr/bin/env bash
set -u
IL=/home/faisal/IsaacLab; cd "$IL"
CK=$(ls $IL/logs/rsl_rl/kbot_legs_rough/*/model_*.pt 2>/dev/null | awk -F'model_' '{n=$2+0; print n"\t"$0}' | sort -n | tail -1 | cut -f2-)
OUT=$IL/eval_watch/wrench_flag_test.txt; : > "$OUT"
for N in 64 4096; do
  echo "=== num_envs=$N ===" >> "$OUT"
  for t in $(seq 1 90); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -gt 4000 ] && break; sleep 10
  done
  KBOT_ADAPT=1 KBOT_FLAT=1 "$IL/kbot_env/bin/python" "$IL/eval_watch/burst_audit.py" \
    --checkpoint "$CK" --num_envs $N --steps 600 --out "$OUT" > "$IL/eval_watch/wf_$N.log" 2>&1
done
