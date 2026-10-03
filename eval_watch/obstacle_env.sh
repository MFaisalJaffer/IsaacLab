# Environment of the obstacle course (stage 1): exactly walker v5's training environment
# (eval_watch/walker_v5_pipeline.sh TRAIN_ENV, minus the start-up noise override). `source` it.
# KBOT_FLAT=1 stays: it selects the lineage's plant and events; obstacle_env_cfg replaces the plane it installs.
export PYTHONUNBUFFERED=1 KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0
export KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 KBOT_AMP_PUSH_FREEZE=1
export KBOT_AMP_VX=-0.40:0.45 KBOT_AMP_VY=-0.13:0.13 KBOT_AMP_WZ=-0.56:0.56
export KBOT_AMP_STYLE_W=2.0 KBOT_AMP_STANCE_W=5
export KBOT_AMP_MOTION_FILES=eval_watch/amp_refs/asimov_tracked_v2_kbot.npz,eval_watch/amp_refs/multitrack_v2_kbot.npz
export KBOT_AMP_AXIS_P=0.6 KBOT_AMP_VEL_STD=0.25 KBOT_AMP_VEL_W=4.0 KBOT_AMP_YAW_STD=0.35 KBOT_AMP_YAW_W=3.0
export KBOT_AMP_DRIFT=1.5 KBOT_AMP_DRIFT_YAW=1.5 KBOT_AMP_ENTROPY=0.005 KBOT_AMP_LIFT_W=5
