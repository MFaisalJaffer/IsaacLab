#!/usr/bin/env bash
set -u
IL=/home/faisal/IsaacLab; cd "$IL"
CK=$(ls $IL/logs/rsl_rl/kbot_legs_rough/*/model_*.pt 2>/dev/null | awk -F'model_' '{n=$2+0; print n"\t"$0}' | sort -n | tail -1 | cut -f2-)
OUT=$IL/eval_watch/fix_verify.txt; : > "$OUT"
echo "=== POST-FIX: base_external_force_torque removed; flag should stay ~83% ===" >> "$OUT"
for N in 4096; do
  for t in $(seq 1 90); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -gt 4000 ] && break; sleep 10
  done
  KBOT_ADAPT=1 KBOT_FLAT=1 "$IL/kbot_env/bin/python" "$IL/eval_watch/burst_audit.py" \
    --checkpoint "$CK" --num_envs $N --steps 600 --out "$OUT" > "$IL/eval_watch/fv_$N.log" 2>&1
done
