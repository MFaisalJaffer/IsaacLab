# Environment of the obstacle course, stage 1b (2026-10-03): stage 1's (obstacle_env.sh) plus
#   mix      40% flat (was 25%), stairs added: 20% platform, 15% beam, 25% stairs
#   noise    action noise capped at 0.2 and started there (stage 1 rose 0.22 -> 0.36)
#   levels   falling behind the command no longer demotes a robot that got onto its obstacle; 20% of episodes
#            replay a lower height
#   episode  25 s (a staircase is 3.5 m to get past)
# `source` it.
source /home/faisal/IsaacLab/eval_watch/obstacle_env.sh
export KBOT_OBST_MIX=flat:0.40,platform:0.20,beam:0.15,stairs:0.25
export KBOT_OBST_BEHIND_FAILS=0 KBOT_OBST_REPLAY=0.2 KBOT_OBST_EPISODE_S=25
export KBOT_AMP_STD_MAX=0.2 KBOT_AMP_INIT_STD=0.2
