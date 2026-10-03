# WALKER v4 pipeline 2026-10-02 08:53
- 10-02 08:53 waiting for the tracker run and its watcher to finish
- 10-02 09:33 autopilot restarted after a dry run of the launch (it was only waiting; no training touched): the run is now named ..._walker_v4 for TensorBoard, and single-direction commands are always real movement commands (>= 0.1) and no longer disturb the stand transition
# WALKER v4 pipeline 2026-10-02 09:33
- 10-02 09:33 waiting for the tracker run and its watcher to finish
- 10-02 09:35 autopilot restarted once more (still only waiting): clearer test lines. For reference, the STARTING point (tracker weights in the walker env, before any walker training): fwd +0.41 surv 0.71; back -0.40 surv 0.85; side +0.11 surv 0.93; pivot +0.51 surv 0.92; standing not learned yet
# WALKER v4 pipeline 2026-10-02 09:35
- 10-02 09:35 waiting for the tracker run and its watcher to finish
- 10-02 10:10 tracker finished; best checkpoint frozen as archive_anchors/trackmulti_v2_best_final.pt (new best (score 0.938): archive_anchors/trackmulti_v2_best.pt = model_9999)
- 10-02 10:12 dataset recorded: MotionDataset: 1 file(s), 511057 real transitions + mirror = 1022114 rows x 40 features, joints=10 | share per direction: {'forward': 0.2, 'backward': 0.21, 'side_left': 0.14, 'side_right': 0.14, 'pivot_left': 0.15, 'pivot_right': 0.16} | pre-fall frames masked: 175
- 10-02 10:12 recording per direction: forward surv 0.99 achieved 0.40/0.00/0.03; backward surv 1.00 achieved -0.41/-0.01/-0.00; side_left surv 0.99 achieved 0.00/0.13/-0.00; side_right surv 1.00 achieved -0.01/-0.12/0.01; pivot_left surv 0.96 achieved 0.01/-0.04/0.50; pivot_right surv 0.98 achieved -0.03/0.03/-0.51; 
- 10-02 10:16 WALKER v4 launched: 4000 iterations; on TensorBoard (port 6007) as kbot_legs_amp/2026-10-02_10-12-44_walker_v4 — [amp-env] v4: history 10, signed clock True, lin vel w 4.0 std 0.25, yaw w 3.0 std 0.35, axis bias 0.6 (single-direction commands >= 0.1), drift 1.5 m, standing envs 0.3
- 10-02 10:48 model_400: score 0.691 | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3)  fwd +0.35 -0.01 -0.01 surv 0.90; back -0.41 -0.01 -0.09 surv 0.96; side +0.00 +0.02 -0.01 surv 0.96; pivot +0.02 +0.01 +0.31 surv 0.94; turn +0.24 -0.00 +0.23 surv 0.93; stand surv 0.95 tilt 1.0 deg;
- 10-02 10:48   -> new best (score 0.691): archive_anchors/walker_v4_best.pt = model_400
- 10-02 11:17 model_800: score 0.79 | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3)  fwd +0.33 -0.02 +0.06 surv 0.86; back -0.35 -0.01 -0.10 surv 0.92; side -0.00 +0.12 -0.01 surv 0.93; pivot +0.02 +0.00 +0.47 surv 0.99; turn +0.17 +0.02 +0.32 surv 0.98; stand surv 0.98 tilt 1.5 deg;
- 10-02 11:17   -> new best (score 0.79): archive_anchors/walker_v4_best.pt = model_800
- 10-02 11:46 model_1200: score 0.839 | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3)  fwd +0.30 -0.00 +0.06 surv 0.91; back -0.32 -0.02 -0.08 surv 0.95; side -0.01 +0.14 -0.03 surv 0.94; pivot +0.02 +0.01 +0.57 surv 1.00; turn +0.20 +0.01 +0.35 surv 0.99; stand surv 0.99 tilt 1.1 deg;
- 10-02 11:46   -> new best (score 0.839): archive_anchors/walker_v4_best.pt = model_1200
- 10-02 12:14 model_1600: score 0.867 | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3)  fwd +0.33 -0.01 +0.04 surv 0.95; back -0.30 -0.02 -0.02 surv 0.96; side -0.01 +0.16 +0.00 surv 0.96; pivot +0.02 +0.01 +0.62 surv 0.96; turn +0.25 +0.00 +0.35 surv 0.99; stand surv 0.97 tilt 1.4 deg;
- 10-02 12:14   -> new best (score 0.867): archive_anchors/walker_v4_best.pt = model_1600
- 10-02 12:43 model_2000: score 0.876 | (achieved fwd lat yaw; cmd fwd 0.4, back -0.4, side 0.13, pivot 0.56, turn 0.3+0.3)  fwd +0.33 -0.00 +0.05 surv 0.97; back -0.31 -0.01 +0.05 surv 0.96; side -0.00 +0.14 -0.01 surv 0.98; pivot +0.02 +0.00 +0.59 surv 0.96; turn +0.23 +0.01 +0.36 surv 1.00; stand surv 0.99 tilt 1.4 deg;
- 10-02 12:43   -> new best (score 0.876): archive_anchors/walker_v4_best.pt = model_2000
- 10-02 13:15 QUALITY CHECK (by hand, not the autopilot) — the scores above do not measure step height, and the gait has turned into a SLIDE:
  highest foot lift in the tests, cm (reference gait = tracker weights in this same test: fwd 10.7, back 13.2, side 6.2, pivot 4.1):
    fwd   400: 10.1 | 800: 8.7 | 1200: 3.1 | 1600: 4.1 | 2000: 4.2
    back  400:  6.1 | 800: 3.6 | 1200: 2.3 | 1600: 2.4 | 2000: 2.7
    side  400:  0.6 | 800: 0.8 | 1200: 0.7 | 1600: 0.7 | 2000: 0.6   (no strides detected at all from 1600)
    pivot 400:  1.0 | 800: 0.9 | 1200: 0.9 | 1600: 0.9 | 2000: 0.9   (no strides detected from 800)
  typical lift over the whole walk (95th percentile) at 2000: fwd/back about 2 cm, side/pivot about 0.5 cm. The judge's style score fell from ~0.75 to 0.5 (fwd/back) and 0.4 (side/pivot).
  Same checkpoints re-scored with step height counted (survival x speed x lift): start 0.54 (no stand yet) | 400: 0.61 | 800: 0.46 | 1200: 0.39 | 1600: 0.43 | 2000: 0.45.
  So: standing was learned (99%) and every direction moves and stays up, but by sliding the feet. "walker_v4_best.pt = model_2000" is best only by the blind score; by gait quality model_400 is the best of this run (it still steps forward/backward/turning, but had lost the side-step and most of the pivot).
  Gifs of model_2000: eval_watch/walker_v4_2000_gif_{fwd,back,side,pivot,turn}.gif.
  Prepared, NOT started: walker v5 = v4 + one change, a stepping anchor (each foot's height follows the reference cycle at the clock's phase; eval_watch/walker_v5_pipeline.sh). Waiting for the user's decision. The v4 run keeps going meanwhile.
- 10-02 13:18 walker v4 STOPPED on the user's decision at model_2400.pt (kept as archive_anchors/walker_v4_last_model_2400.pt); switching to walker v5 (v4 + stepping anchor)
