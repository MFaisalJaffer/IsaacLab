#!/usr/bin/env bash
# Parallel bent-knee bisect experiment: legs on FLAT terrain with the original
# bent-knee default pose, logged to a SEPARATE experiment (kbot_legs_bent) so it
# doesn't collide with the main flat/zero-pose run. Run via kbot-train-bent.service
# (which sets KBOT_FLAT=1, KBOT_BENTKNEE=1, KBOT_EXP=kbot_legs_bent, KBOT_NUM_ENVS).
set -u
cd /home/faisal/IsaacLab
source kbot_env/bin/activate

NUM_ENVS="${KBOT_NUM_ENVS:-4096}"
EXP="${KBOT_EXP:-kbot_legs_bent}"
latest=$(ls logs/rsl_rl/$EXP/*/model_*.pt 2>/dev/null \
         | awk -F'model_' '{n=$2+0; print n"\t"$0}' | sort -n | tail -1 | cut -f2-)

{
  echo "==== [$(date '+%F %T')] $EXP start (envs=$NUM_ENVS flat=${KBOT_FLAT:-0} bent=${KBOT_BENTKNEE:-0}) ===="
  if [ -n "$latest" ]; then
      run=$(basename "$(dirname "$latest")"); ckpt=$(basename "$latest")
      echo "==== resuming $EXP from $run / $ckpt ===="
      exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
          --task=Isaac-Velocity-Rough-KbotLegs-v0 --headless --num_envs "$NUM_ENVS" \
          --resume --load_run "$run" --checkpoint "$ckpt"
  else
      echo "==== fresh $EXP run ===="
      exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
          --task=Isaac-Velocity-Rough-KbotLegs-v0 --headless --num_envs "$NUM_ENVS"
  fi
} >> /home/faisal/IsaacLab/train_kbot_bent.log 2>&1
