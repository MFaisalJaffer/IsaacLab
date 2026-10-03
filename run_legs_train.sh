#!/usr/bin/env bash
# Resume-from-latest launcher for the legs training, run under a systemd --user
# service (Restart=always). On each (re)start it finds the highest-iteration
# checkpoint across all kbot_legs_rough runs and resumes from it, so an Isaac
# crash or a reap just costs the last checkpoint interval.
set -u
cd /home/faisal/IsaacLab
source kbot_env/bin/activate

NUM_ENVS="${KBOT_NUM_ENVS:-16384}"

latest=$(ls logs/rsl_rl/kbot_legs_rough/*/model_*.pt 2>/dev/null \
         | awk -F'model_' '{n=$2+0; print n"\t"$0}' | sort -n | tail -1 | cut -f2-)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
{
  echo "==== [$(ts)] supervisor start (num_envs=$NUM_ENVS) ===="
  if [ -n "$latest" ]; then
      run=$(basename "$(dirname "$latest")")
      ckpt=$(basename "$latest")
      # fast-forward step-based curricula (push ramp) past the resume point:
      # num_steps_per_env = 24, so global env-steps ~= iter * 24
      iter=$(echo "$ckpt" | sed 's/model_\([0-9]*\).pt/\1/')
      export KBOT_RESUME_STEP_OFFSET=$(( iter * 24 ))
      echo "==== resuming from $run / $ckpt (curriculum offset $KBOT_RESUME_STEP_OFFSET) ===="
      exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
          --task=Isaac-Velocity-Rough-KbotLegs-v0 --headless --num_envs "$NUM_ENVS" \
          --resume --load_run "$run" --checkpoint "$ckpt"
  else
      echo "==== no checkpoint found -> fresh run ===="
      exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
          --task=Isaac-Velocity-Rough-KbotLegs-v0 --headless --num_envs "$NUM_ENVS"
  fi
} >> /home/faisal/IsaacLab/train_kbot_legs.log 2>&1
