#!/usr/bin/env bash
# FREQ-MAP STUDY part A — sleep-proof runner (2026-08-04).
# Runs the frequency-transfer grid on the deploy policy. IDEMPOTENT: skips any
# cell whose label already appears in the output file, so it can be re-run /
# resumed safely. Launched via systemd-run --user so it survives the Claude
# session, SSH drops, and the user's Mac sleeping.
set -u
IL=/home/faisal/IsaacLab
cd "$IL"
DEPLOY=$IL/logs/rsl_rl/kbot_legs_rough_rigcand_calmref198600/model_198600.pt
OUT=$IL/eval_watch/freq_map_study.txt
LOG=$IL/eval_watch/freq_map_study_runner.log
ts() { date '+%H:%M:%S'; }
echo "[$(ts)] runner start" >> "$LOG"
[ -f "$OUT" ] || echo "=== FREQ-MAP STUDY part A: deploy policy (trained@1.4) driven at off-nominal clocks ===" > "$OUT"

run_cell() {
    local F="$1" V="$2"
    local LABEL="deploy_f${F}_v${V}"
    if grep -q "^${LABEL}:" "$OUT"; then
        echo "[$(ts)] skip $LABEL (done)" >> "$LOG"
        return 0
    fi
    # GPU guard: wait for >3500 MiB free (training shares the card)
    for t in $(seq 1 60); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -gt 3500 ] && break
        sleep 10
    done
    echo "[$(ts)] run $LABEL (gpu free ${free}MiB)" >> "$LOG"
    KBOT_FLAT=1 "$IL/kbot_env/bin/python" "$IL/eval_watch/posture_probe.py" \
        --checkpoint "$DEPLOY" --vx "$V" --gait_freq "$F" --pin_gains \
        --num_envs 32 --steps 400 --label "$LABEL" --out "$OUT" \
        > "$IL/eval_watch/fs_${F}_${V}.log" 2>&1
    echo "[$(ts)]   $LABEL exit $?" >> "$LOG"
}

for F in 1.0 1.1 1.2 1.3 1.4; do
    for V in 0.15 0.30; do
        run_cell "$F" "$V"
    done
done
echo "[$(ts)] ALL CELLS DONE" >> "$LOG"

# ---- EXTENSION (2026-08-04): sub-1.0 posture cells + TORQUE surface ----
# posture at the pendulum-preferred band
for F in 0.8 0.9; do
    for V in 0.15 0.30; do
        run_cell "$F" "$V"
    done
done

# torque surface: which frequency is thermally cheapest at each speed
run_tq() {
    local F="$1" V="$2"
    local LABEL="tq_f${F}_v${V}"
    if grep -q "^${LABEL} " "$OUT" || grep -q "^${LABEL}:" "$OUT"; then
        echo "[$(ts)] skip $LABEL (done)" >> "$LOG"; return 0
    fi
    for t in $(seq 1 60); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -gt 3500 ] && break
        sleep 10
    done
    echo "[$(ts)] run $LABEL (gpu free ${free}MiB)" >> "$LOG"
    KBOT_FLAT=1 "$IL/kbot_env/bin/python" "$IL/eval_watch/walk_gap_probe.py" \
        --checkpoint "$DEPLOY" --vx "$V" --gait_freq "$F" --pin_gains \
        --num_envs 32 --steps 500 --label "$LABEL" --out "$OUT" \
        > "$IL/eval_watch/tq_${F}_${V}.log" 2>&1
    echo "[$(ts)]   $LABEL exit $?" >> "$LOG"
}
for F in 0.8 0.9 1.0 1.1 1.2 1.4; do
    for V in 0.15 0.30; do
        run_tq "$F" "$V"
    done
done
echo "[$(ts)] EXTENSION DONE" >> "$LOG"
