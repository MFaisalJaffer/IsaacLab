# Obstacle course, stage 1 — status

Question: can the walker climb what it sees? Two runs from walker v5, same course and settings:
**map** (sees the height map) and **blind** (map zeroed). Details: OBSTACLE_PERCEPTION_PROPOSAL.md, "Stage 1 as built".

How to read a test line: crossing shares are listed by obstacle height 2, 4, ..., 20 cm (1200 robots, forward
0.35 m/s, random heading, 20 s; crossed = got 0.5 m past the obstacle without falling). "climbs N cm" = the
tallest height that at least 80% cross, with every lower height passing too. Score = mean crossing share.

What to look for:
- map run: how high it climbs (my expectation before the run: 10-15 cm on the platform);
- the same checkpoint with the map zeroed: if that is as good, the policy is not using the map;
- blind run vs map run: what the map buys over simply training on the course;
- flat walking (the walker's own six-direction test, walker v5 scored 0.876): must not collapse.

# OBSTACLE COURSE stage 1 — 2026-10-03 00:18 (runs: map blind; 3000 iterations each; crossing shares listed by height 2,4,...,20 cm)
- 10-03 00:18 start point (walker v5, no map yet): flat 1.00; platform 0.83 0.09 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -> climbs 2 cm; beam 0.89 0.17 0.00 0.06 0.00 0.00 0.00 0.00 0.00 0.00 -> climbs 2 cm
- 10-03 00:22 RUN 'map' launched: 3000 iterations; TensorBoard (port 6007) run kbot_legs_obstacle/2026-10-03_00-18-43_obst_s1_map — [obstacle-env] 10 heights 2-20 cm, start level 0, obstacle-tile vx (0.3, 0.45), policy history 10 + height map (on, z_nominal 1.0), lift anchor terrain-aware, contact penalties -5.0
- 10-03 00:50 map model_400: score 0.384 | flat 1.00; platform 1.00 0.83 0.65 0.24 0.20 0.07 0.06 0.02 0.02 0.00 -> climbs 4 cm; beam 1.00 0.97 0.75 0.50 0.47 0.33 0.25 0.14 0.14 0.03 -> climbs 4 cm; with the map zeroed: score 0.151, climbs 2/2 cm | training heights now (platform, beam): 3.8723 cm 5.7166 cm 
- 10-03 00:50   -> new best (score 0.384): archive_anchors/obstacle_s1_map_best.pt = model_400
- 10-03 01:21 map model_800: score 0.435 | flat 1.00; platform 0.96 0.91 0.81 0.67 0.35 0.06 0.07 0.00 0.00 0.00 -> climbs 6 cm; beam 1.00 1.00 0.78 0.67 0.53 0.31 0.22 0.28 0.06 0.03 -> climbs 4 cm; with the map zeroed: score 0.101, climbs 0/0 cm | training heights now (platform, beam): 4.7622 cm 7.6360 cm 
- 10-03 01:21   -> new best (score 0.435): archive_anchors/obstacle_s1_map_best.pt = model_800
- 10-03 01:49 map model_1200: score 0.614 | flat 1.00; platform 0.96 0.96 0.94 0.83 0.65 0.56 0.37 0.20 0.09 0.04 -> climbs 8 cm; beam 1.00 0.97 0.94 0.81 0.67 0.69 0.61 0.42 0.36 0.19 -> climbs 8 cm; with the map zeroed: score 0.101, climbs 0/0 cm | training heights now (platform, beam): 5.3773 cm 9.2865 cm 
- 10-03 01:49   -> new best (score 0.614): archive_anchors/obstacle_s1_map_best.pt = model_1200
- 10-03 02:02   flat walking (walker's own test; walker v5 scored 0.876): score 0.865 | fwd +0.38 -0.01 +0.03 surv 1.00 lift 13.6 cm; back -0.30 -0.01 -0.06 surv 0.99 lift 13.1 cm; side +0.03 +0.09 +0.05 surv 1.00 lift 10.0 cm; pivot +0.06 +0.01 +0.64 surv 0.96 lift 8.5 cm; turn +0.26 +0.01 +0.34 surv 1.00 lift 11.5 cm; stand surv 0.98 tilt 2.0 deg;
- 10-03 02:23 map model_1600: score 0.666 | flat 1.00; platform 0.94 0.80 0.93 0.74 0.72 0.59 0.56 0.33 0.22 0.09 -> climbs 2 cm; beam 0.94 0.92 0.89 0.92 0.81 0.72 0.78 0.56 0.42 0.44 -> climbs 10 cm; with the map zeroed: score 0.104, climbs 0/0 cm | training heights now (platform, beam): 6.8041 cm 12.2981 cm 
- 10-03 02:23   -> new best (score 0.666): archive_anchors/obstacle_s1_map_best.pt = model_1600
- 10-03 02:50 map model_2000: score 0.869 | flat 1.00; platform 0.94 0.93 0.85 0.96 0.94 0.87 0.81 0.74 0.63 0.57 -> climbs 14 cm; beam 0.94 0.94 0.94 0.92 0.94 0.92 0.97 0.86 0.86 0.81 -> climbs 20 cm; with the map zeroed: score 0.094, climbs 0/0 cm | training heights now (platform, beam): 8.4452 cm 13.7251 cm 
- 10-03 02:50   -> new best (score 0.869): archive_anchors/obstacle_s1_map_best.pt = model_2000
- 10-03 03:18 map model_2400: score 0.863 | flat 0.99; platform 0.93 0.65 0.81 0.87 0.93 0.94 0.83 0.85 0.80 0.65 -> climbs 2 cm; beam 0.92 0.81 0.81 1.00 0.92 1.00 0.83 0.94 0.89 0.89 -> climbs 20 cm; with the map zeroed: score 0.074, climbs 0/0 cm | training heights now (platform, beam): 8.6743 cm 13.4604 cm 
- 10-03 03:55 map model_2999: score 0.921 | flat 1.00; platform 0.96 0.76 0.87 0.98 0.96 1.00 0.98 0.93 0.85 0.76 -> climbs 2 cm; beam 0.97 0.86 0.94 0.97 0.92 0.97 0.94 0.89 1.00 0.89 -> climbs 20 cm; with the map zeroed: score 0.072, climbs 0/0 cm | training heights now (platform, beam): 8.7908 cm 13.0114 cm 
- 10-03 03:55   -> new best (score 0.921): archive_anchors/obstacle_s1_map_best.pt = model_2999
- 10-03 04:01   flat walking (walker's own test; walker v5 scored 0.876): score 0.656 | fwd +0.40 -0.00 +0.04 surv 1.00 lift 13.4 cm; back -0.30 +0.00 -0.13 surv 0.92 lift 13.8 cm; side -0.01 +0.00 +0.01 surv 0.96 lift 0.9 cm; pivot +0.03 -0.00 +0.21 surv 0.94 lift 5.4 cm; turn +0.30 -0.02 +0.26 surv 0.99 lift 12.6 cm; stand surv 0.90 tilt 1.7 deg;
- 10-03 04:01 run 'map' DONE: best score 0.921
- 10-03 04:05 RUN 'blind' launched: 3000 iterations; TensorBoard (port 6007) run kbot_legs_obstacle/2026-10-03_04-01-31_obst_s1_blind — [obstacle-env] 10 heights 2-20 cm, start level 0, obstacle-tile vx (0.3, 0.45), policy history 10 + height map (BLIND: zeroed, z_nominal 1.0), lift anchor terrain-aware, contact penalties -5.0
- 10-03 04:30 blind model_400: score 0.224 | flat 1.00; platform 0.98 0.78 0.26 0.06 0.02 0.00 0.00 0.00 0.00 0.00 -> climbs 2 cm; beam 1.00 0.69 0.50 0.06 0.06 0.00 0.00 0.08 0.00 0.00 -> climbs 2 cm | training heights now (platform, beam): 3.1356 cm 3.8825 cm 
- 10-03 04:30   -> new best (score 0.224): archive_anchors/obstacle_s1_blind_best.pt = model_400
- 10-03 04:57 blind model_800: score 0.308 | flat 1.00; platform 1.00 0.72 0.50 0.20 0.13 0.06 0.02 0.00 0.00 0.00 -> climbs 2 cm; beam 1.00 0.86 0.58 0.31 0.33 0.17 0.06 0.03 0.06 0.14 -> climbs 4 cm | training heights now (platform, beam): 3.4660 cm 4.7120 cm 
- 10-03 04:57   -> new best (score 0.308): archive_anchors/obstacle_s1_blind_best.pt = model_800
- 10-03 05:23 blind model_1200: score 0.394 | flat 1.00; platform 0.98 0.80 0.63 0.44 0.28 0.11 0.04 0.04 0.00 0.00 -> climbs 2 cm; beam 1.00 0.92 0.58 0.56 0.42 0.33 0.33 0.11 0.22 0.08 -> climbs 4 cm | training heights now (platform, beam): 4.0069 cm 6.1359 cm 
- 10-03 05:23   -> new best (score 0.394): archive_anchors/obstacle_s1_blind_best.pt = model_1200
- 10-03 05:36   flat walking (walker's own test; walker v5 scored 0.876): score 0.811 | fwd +0.35 -0.00 -0.00 surv 1.00 lift 14.9 cm; back -0.28 -0.01 -0.07 surv 0.99 lift 12.6 cm; side +0.02 +0.08 -0.03 surv 1.00 lift 9.5 cm; pivot +0.04 +0.01 +0.51 surv 0.99 lift 7.7 cm; turn +0.25 +0.00 +0.26 surv 1.00 lift 13.0 cm; stand surv 0.95 tilt 1.7 deg;
- 10-03 05:56 blind model_1600: score 0.589 | flat 1.00; platform 1.00 0.83 0.74 0.61 0.56 0.48 0.46 0.22 0.09 0.00 -> climbs 4 cm; beam 1.00 0.89 0.83 0.78 0.75 0.58 0.69 0.61 0.44 0.19 -> climbs 6 cm | training heights now (platform, beam): 4.2116 cm 8.7857 cm 
- 10-03 05:56   -> new best (score 0.589): archive_anchors/obstacle_s1_blind_best.pt = model_1600
- 10-03 06:22 blind model_2000: score 0.627 | flat 1.00; platform 0.98 0.74 0.70 0.61 0.67 0.72 0.50 0.33 0.22 0.06 -> climbs 2 cm; beam 0.97 0.81 0.83 0.50 0.69 0.72 0.78 0.72 0.69 0.28 -> climbs 6 cm | training heights now (platform, beam): 4.3845 cm 9.5344 cm 
- 10-03 06:22   -> new best (score 0.627): archive_anchors/obstacle_s1_blind_best.pt = model_2000
- 10-03 06:49 blind model_2400: score 0.674 | flat 0.99; platform 0.96 0.81 0.83 0.78 0.80 0.67 0.56 0.50 0.35 0.06 -> climbs 6 cm; beam 0.94 0.83 0.83 0.75 0.69 0.78 0.72 0.58 0.61 0.42 -> climbs 6 cm | training heights now (platform, beam): 5.1767 cm 10.7302 cm 
- 10-03 06:49   -> new best (score 0.674): archive_anchors/obstacle_s1_blind_best.pt = model_2400
- 10-03 07:26 blind model_2999: score 0.784 | flat 1.00; platform 1.00 0.96 0.85 0.91 0.76 0.74 0.80 0.56 0.30 0.17 -> climbs 8 cm; beam 1.00 0.97 0.92 0.86 0.94 0.94 0.69 0.75 0.81 0.75 -> climbs 12 cm | training heights now (platform, beam): 5.4560 cm 11.0214 cm 
- 10-03 07:26   -> new best (score 0.784): archive_anchors/obstacle_s1_blind_best.pt = model_2999
- 10-03 07:32   flat walking (walker's own test; walker v5 scored 0.876): score 0.658 | fwd +0.36 -0.03 +0.00 surv 1.00 lift 13.8 cm; back -0.28 -0.00 -0.07 surv 0.95 lift 13.4 cm; side +0.00 +0.00 +0.01 surv 0.96 lift 0.9 cm; pivot +0.05 +0.00 +0.32 surv 0.95 lift 7.0 cm; turn +0.27 -0.02 +0.26 surv 1.00 lift 12.9 cm; stand surv 0.94 tilt 2.7 deg;
- 10-03 07:32 run 'blind' DONE: best score 0.784
- 10-03 07:32 ALL DONE (map blind).

## Result of stage 1 (2026-10-03 09:10)

Share of robots that cross, by height 2, 4, ..., 20 cm (1200 robots, forward 0.35 m/s, random heading, 20 s):

| | platform | mean | beam | mean |
|---|---|---|---|---|
| start (walker v5) | 83 9 0 0 0 0 0 0 0 0 | 9% | 89 17 0 6 0 0 0 0 0 0 | 11% |
| blind run, iteration 2999 | 100 96 85 91 76 74 80 56 30 17 | 70% | 100 97 92 86 94 94 69 75 81 75 | 86% |
| **map run, iteration 2999** | 96 76 87 98 96 100 98 93 85 76 | **91%** | 97 86 94 97 92 97 94 89 100 89 | **94%** |
| map run with the map zeroed | 28 11 17 6 2 0 0 0 0 0 | 6% | 28 17 22 8 0 3 0 0 3 0 | 8% |

- The walker can climb what it sees: 85% or more cross every platform from 6 to 18 cm, 76% the 20 cm one; beams 86% or more at every height.
  (Expectation before the run was 10-15 cm.) Toe/leg hits before crossing: 3-4 steps at every height (blind: 1 -> 23 steps as the obstacle grows).
- What the map buys over blind practice: tall platforms — 16 / 18 / 20 cm: 93 / 85 / 76% vs 56 / 30 / 17%. On beams the blind policy is nearly as good.
- Both runs were still improving at the end (map 0.863 -> 0.921, blind 0.674 -> 0.784 over the last 600 iterations).
- Weak spot of the map policy: low obstacles — 4 cm platform 76%, 4 cm beam 86%.

PROBLEM — flat skills decayed in BOTH runs (walker's own six-direction test, walker v5 = 0.876):
- map run: 0.865 at iteration 1200 -> 0.656 at the end. Side-step gone (0.00 of 0.13 m/s, feet not lifting), turn in place 0.21 of 0.56 rad/s, standing 90%.
  Located: side-step still 0.08 m/s at 1600, gone at 2000; at 2400 half of the robots fall when asked to side-step.
- blind run: 0.811 at 1200 -> 0.658 at the end (same pattern).
- Forward, backward, mixed turn are intact (forward 0.40 of 0.40 m/s — better than walker v5).
- Likely causes: action noise rose all run (0.22 -> 0.36; walker v5 sat at 0.15-0.21) and only ~3.5% of the robots practise side-stepping
  (25% flat tiles x the walker's command mix), against 14% in walker v5.

Checkpoints: best climber `archive_anchors/obstacle_s1_map_best.pt` (= iteration 2999, flat skills damaged);
all skills intact but lower climbing: `logs/autopilot/obst_s1_map_1600.pt` (platform 94 80 93 74 72 59 56 33 22 9, side 0.08, turn 0.45).
Gifs: eval_watch/obst_s1_map_2999_gif_{platform4,platform7,platform9,beam9}.gif, obst_s1_blind_2999_gif_platform7.gif.
