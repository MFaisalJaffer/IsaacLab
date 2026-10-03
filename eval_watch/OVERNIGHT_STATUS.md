# Overnight pipeline — 2026-09-30 → 10-01

Plan: tracker v2c (turning) finishes → turn test (retry once if yaw < 0.2) → record dataset v2 → AMP pilot 2
(warm start from the tracker, datasets v1+v2, pushes frozen, style 2.0) → eval + gif → hardening (push ramp on)
if stand ≥ 85% and yaw ≥ 0.2. One training at a time, checkpoints anchored in archive_anchors/, nothing deleted,
lineage 11 stays paused. Logs: logs/overnight/. Dashboard: :8800 (auto = whatever is training).

## Log
- 09-30 23:29 pipeline started; waiting for tracker v2c (/home/faisal/IsaacLab/logs/rsl_rl/kbot_legs_track/2026-09-30_23-24-08) to finish
- 10-01 00:28 tracker v2c finished: model_1199.pt anchored as archive_anchors/track_v2c_final.pt
- 10-01 00:29 turn test v2c: yaw ? rad/s (cmd 0.30), stride rmse ? deg, survival ?
- 10-01 00:29 turn test below bar (yaw >= 0.2, rmse <= 4): ONE retry — 1200 more iterations
- 10-01 01:34 turn test after retry: yaw ?, rmse ?, survival ? (proceeding regardless)
- 10-01 01:35 dataset v2 recorded: eval_watch/amp_refs/asimov_tracked_v2_kbot.npz (survivors only, turn commands +-0.3), merged with v1 for the judge
- 10-01 01:37 AMP pilot 2 launched: /home/faisal/IsaacLab/logs/rsl_rl/kbot_legs_amp/2026-10-01_01-35-40/ (3000 iters, datasets: eval_watch/amp_refs/asimov_tracked_kbot.npz,eval_watch/amp_refs/asimov_tracked_v2_kbot.npz, pushes frozen, style 2.0, std 0.08)
- 10-01 04:20 pilot 2 finished: model_2999.pt anchored as archive_anchors/amp_pilot2_final.pt
- 10-01 04:21 pilot 2 STAND: survival ?, tilt ? deg, stance ? cm
- 10-01 04:22 pilot 2 cmd 0.3:0:0.3: survival , fwd  m/s, yaw  (cmd ), width  cm, clearance  cm, stride  s, knee p5/p95 /, judge 
- 10-01 04:23 pilot 2 cmd 0.4:0:0: survival , fwd  m/s, yaw  (cmd ), width  cm, clearance  cm, stride  s, knee p5/p95 /, judge 
- 10-01 04:24 render: eval_watch/amp_pilot2_final.gif + amp_pilot2_final_filmstrip.png (also on :8800)
- 10-01 04:24 did NOT pass the bar (stand ? >= 0.85 and yaw ? >= 0.2) — hardening skipped, left for review
- 10-01 04:24 pipeline done. GPU idle. Lineage 11 still paused (resume: systemctl --user start kbot-train, then kbot-watchdog).
