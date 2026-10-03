# Obstacle course, stage 1b — status

One run from stage 1's iteration-1600 checkpoint. New: stairs; 40% flat tiles; action noise capped at 0.2;
falling behind the command no longer demotes a climbing robot; lower heights are replayed; 25 s episodes.
Details: OBSTACLE_PERCEPTION_PROPOSAL.md, "Stage 1 result and stage 1b".

How to read a test line: crossing shares by height 2, 4, ..., 20 cm (for stairs: the riser; 3 risers up, 3 down).
1200 robots, forward 0.35 m/s, random heading, 26 s. "climbs N cm" = tallest height at least 80% cross, lower
ones passing too. A checkpoint is ELIGIBLE only if every flat direction (walker's own test) scores >= 0.5.

# OBSTACLE COURSE stage 1b — 2026-10-03 09:38 (3000 iterations from archive_anchors/obstacle_s1_map_1600_allskills.pt; crossing shares listed by height 2,4,...,20 cm; for stairs that is the riser height)
- 10-03 09:38 start point on the new course: flat 1.00; platform 0.97 0.75 1.00 0.86 0.78 0.53 0.61 0.53 0.17 0.03 -> climbs 2 cm; beam 0.97 0.92 0.86 0.86 0.86 0.78 0.83 0.69 0.56 0.31 -> climbs 10 cm; stairs 0.89 0.64 0.42 0.25 0.11 0.08 0.00 0.00 0.00 0.00 -> climbs 2 cm
- 10-03 09:38 for reference, stage 1's best climber on the new course: flat 0.99; platform 0.94 0.89 0.83 0.92 0.94 0.97 0.92 0.97 0.86 0.89 -> climbs 20 cm; beam 0.92 0.97 0.94 0.94 0.92 0.94 0.94 0.94 0.94 0.89 -> climbs 20 cm; stairs 0.86 0.69 0.53 0.39 0.36 0.19 0.14 0.06 0.00 0.00 -> climbs 2 cm
- 10-03 09:42 RUN launched: 3000 iterations; TensorBoard (port 6007) run kbot_legs_obstacle/2026-10-03_09-38-24_obst_s1b — [obstacle-env] 10 heights 2-20 cm, columns per kind {'flat': 8, 'platform': 4, 'beam': 3, 'stairs': 5}, start level 2, falling behind demotes False, replay of lower heights 0.2, episode 25 s, obstacle-tile vx (0.3, 0.45), policy history 10 + height map (on, z_nominal 1.0), lift anchor terrain-aware, contact penalties -5.0; [AMP] action noise capped at std 0.2 (KBOT_AMP_STD_MAX)
- 10-03 10:11 model_400: crossing 0.569 | flat 1.00; platform 1.00 0.97 0.86 0.78 0.75 0.64 0.56 0.28 0.25 0.19 -> climbs 6 cm; beam 1.00 1.00 0.89 0.94 0.86 0.83 0.89 0.64 0.47 0.39 -> climbs 14 cm; stairs 0.94 0.89 0.61 0.19 0.17 0.06 0.00 0.00 0.00 0.00 -> climbs 4 cm | training heights (platform beam stairs): 9.9 13.5 4.3  cm, noise 0.15
- 10-03 10:38   flat walking: score 0.9 (walker v5 0.876), weakest side 0.773 -> ELIGIBLE | fwd +0.37 -0.00 +0.03 surv 1.00 lift 13.5 cm; back -0.33 -0.01 -0.08 surv 0.95 lift 14.2 cm; side +0.01 +0.10 +0.03 surv 0.99 lift 10.4 cm; pivot +0.04 +0.00 +0.56 surv 1.00 lift 8.6 cm; turn +0.28 -0.00 +0.32 surv 1.00 lift 12.3 cm; stand surv 0.98 tilt 0.9 deg;
- 10-03 10:38   -> new best (crossing 0.569, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_400
- 10-03 10:50 model_800: crossing 0.681 | flat 1.00; platform 0.97 0.97 0.92 0.89 0.81 0.78 0.78 0.81 0.61 0.42 -> climbs 10 cm; beam 0.97 0.97 0.94 0.97 0.89 0.89 0.83 0.72 0.72 0.86 -> climbs 14 cm; stairs 0.94 0.97 0.64 0.53 0.33 0.28 0.03 0.00 0.00 0.00 -> climbs 4 cm | training heights (platform beam stairs): 13.6 14.9 5.5  cm, noise 0.14
- 10-03 11:02   flat walking: score 0.859 (walker v5 0.876), weakest side 0.651 -> ELIGIBLE | fwd +0.37 -0.01 +0.02 surv 1.00 lift 13.6 cm; back -0.31 -0.01 -0.08 surv 0.92 lift 14.5 cm; side +0.01 +0.08 +0.02 surv 1.00 lift 10.1 cm; pivot +0.03 +0.01 +0.55 surv 1.00 lift 8.4 cm; turn +0.27 +0.00 +0.29 surv 1.00 lift 11.9 cm; stand surv 1.00 tilt 1.0 deg;
- 10-03 11:02   -> new best (crossing 0.681, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_800
- 10-03 11:23 model_1200: crossing 0.741 | flat 1.00; platform 1.00 1.00 0.97 0.92 0.89 0.94 0.94 0.78 0.81 0.61 -> climbs 14 cm; beam 1.00 0.92 0.97 0.94 0.94 0.92 0.89 0.89 0.86 0.83 -> climbs 20 cm; stairs 0.97 0.86 0.64 0.53 0.44 0.50 0.19 0.06 0.00 0.00 -> climbs 4 cm | training heights (platform beam stairs): 15.2 14.7 6.2  cm, noise 0.14
- 10-03 11:36   flat walking: score 0.886 (walker v5 0.876), weakest side 0.740 -> ELIGIBLE | fwd +0.37 +0.00 +0.00 surv 1.00 lift 13.2 cm; back -0.34 -0.01 -0.14 surv 0.93 lift 12.9 cm; side -0.00 +0.10 -0.00 surv 1.00 lift 9.4 cm; pivot +0.04 +0.00 +0.59 surv 1.00 lift 7.9 cm; turn +0.26 +0.00 +0.30 surv 1.00 lift 11.2 cm; stand surv 0.99 tilt 1.3 deg;
- 10-03 11:36   -> new best (crossing 0.741, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_1200
- 10-03 11:58 model_1600: crossing 0.746 | flat 1.00; platform 1.00 0.97 0.94 0.89 0.89 0.92 0.94 0.86 0.83 0.86 -> climbs 20 cm; beam 1.00 0.97 0.94 0.92 0.89 0.92 0.89 0.81 0.86 0.89 -> climbs 20 cm; stairs 0.97 0.86 0.69 0.67 0.44 0.39 0.08 0.08 0.00 0.00 -> climbs 4 cm | training heights (platform beam stairs): 15.4 14.3 7.4  cm, noise 0.14
- 10-03 12:11   flat walking: score 0.864 (walker v5 0.876), weakest side 0.655 -> ELIGIBLE | fwd +0.37 -0.00 +0.02 surv 1.00 lift 13.3 cm; back -0.31 -0.01 -0.08 surv 0.94 lift 12.9 cm; side -0.00 +0.09 +0.00 surv 1.00 lift 9.0 cm; pivot +0.02 +0.02 +0.53 surv 1.00 lift 7.9 cm; turn +0.27 -0.00 +0.30 surv 1.00 lift 10.5 cm; stand surv 1.00 tilt 1.0 deg;
- 10-03 12:11   -> new best (crossing 0.746, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_1600
- 10-03 12:31 model_2000: crossing 0.775 | flat 1.00; platform 1.00 0.97 0.89 0.86 0.92 0.94 0.86 0.75 0.89 0.75 -> climbs 14 cm; beam 1.00 0.94 0.94 0.94 0.94 0.97 0.89 0.94 0.89 0.75 -> climbs 18 cm; stairs 1.00 0.89 0.69 0.75 0.56 0.50 0.53 0.22 0.06 0.00 -> climbs 4 cm | training heights (platform beam stairs): 15.1 14.2 8.3  cm, noise 0.14
- 10-03 12:44   flat walking: score 0.868 (walker v5 0.876), weakest side 0.734 -> ELIGIBLE | fwd +0.37 -0.01 +0.00 surv 1.00 lift 12.8 cm; back -0.34 -0.01 -0.13 surv 0.93 lift 13.5 cm; side +0.02 +0.10 -0.01 surv 1.00 lift 9.4 cm; pivot +0.04 +0.00 +0.51 surv 1.00 lift 8.3 cm; turn +0.26 -0.01 +0.26 surv 1.00 lift 10.4 cm; stand surv 0.98 tilt 1.0 deg;
- 10-03 12:44   -> new best (crossing 0.775, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_2000
- 10-03 13:05 model_2400: crossing 0.781 | flat 1.00; platform 1.00 0.92 0.89 0.92 0.94 0.94 0.97 0.86 0.89 0.81 -> climbs 20 cm; beam 1.00 1.00 0.86 0.97 0.94 0.92 0.86 0.94 0.94 0.81 -> climbs 20 cm; stairs 0.94 0.78 0.53 0.75 0.58 0.64 0.47 0.28 0.03 0.03 -> climbs 2 cm | training heights (platform beam stairs): 14.9 14.0 9.5  cm, noise 0.14
- 10-03 13:18   flat walking: score 0.855 (walker v5 0.876), weakest side 0.735 -> ELIGIBLE | fwd +0.36 -0.01 +0.01 surv 1.00 lift 12.0 cm; back -0.35 -0.01 -0.10 surv 0.96 lift 12.2 cm; side +0.01 +0.10 +0.02 surv 1.00 lift 8.6 cm; pivot +0.02 +0.01 +0.48 surv 1.00 lift 7.8 cm; turn +0.25 -0.01 +0.29 surv 1.00 lift 9.5 cm; stand surv 0.98 tilt 1.1 deg;
- 10-03 13:18   -> new best (crossing 0.781, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_2400
- 10-03 13:51 model_2999: crossing 0.842 | flat 1.00; platform 0.97 0.97 0.94 0.94 0.97 0.89 0.94 0.92 0.97 0.94 -> climbs 20 cm; beam 1.00 0.89 1.00 0.92 0.94 0.94 0.89 0.92 0.94 0.94 -> climbs 20 cm; stairs 0.97 0.89 0.78 0.83 0.86 0.67 0.67 0.42 0.25 0.06 -> climbs 4 cm | training heights (platform beam stairs): 14.4 14.0 10.9  cm, noise 0.14
- 10-03 13:57   flat walking: score 0.855 (walker v5 0.876), weakest side 0.639 -> ELIGIBLE | fwd +0.36 +0.00 -0.00 surv 1.00 lift 11.3 cm; back -0.36 -0.01 -0.07 surv 0.96 lift 11.7 cm; side -0.00 +0.08 +0.02 surv 1.00 lift 8.3 cm; pivot +0.01 +0.00 +0.50 surv 1.00 lift 7.3 cm; turn +0.25 +0.01 +0.27 surv 1.00 lift 9.7 cm; stand surv 0.98 tilt 1.2 deg;
- 10-03 13:57   -> new best (crossing 0.842, every flat skill kept): archive_anchors/obstacle_s1b_best.pt = model_2999
- 10-03 13:59 best model_2999 with the map zeroed: crossing 0.161 | flat 0.98; platform 0.86 0.28 0.36 0.19 0.00 0.00 0.00 0.00 0.00 0.00 -> climbs 2 cm; beam 0.81 0.31 0.53 0.47 0.11 0.17 0.17 0.11 0.17 0.06 -> climbs 2 cm; stairs 0.19 0.06 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -> climbs 0 cm
- 10-03 13:59 DONE: best eligible checkpoint model_2999 (crossing 0.842); highest crossing of any checkpoint: model_2999 (0.842).

## Result of stage 1b (2026-10-03 14:00)

Share of robots that cross, by height 2, 4, ..., 20 cm (1200 robots, forward 0.35 m/s, random heading, 26 s; stairs: riser height):

| | platform | mean | beam | mean | stairs | mean |
|---|---|---|---|---|---|---|
| start (stage 1, iteration 1600) | 97 75 100 86 78 53 61 53 17 3 | 62% | 97 92 86 86 86 78 83 69 56 31 | 76% | 89 64 42 25 11 8 0 0 0 0 | 24% |
| stage 1's best climber (flat skills damaged) | 94 89 83 92 94 97 92 97 86 89 | 91% | 92 97 94 94 92 94 94 94 94 89 | 94% | 86 69 53 39 36 19 14 6 0 0 | 32% |
| **stage 1b, iteration 2999** | 97 97 94 94 97 89 94 92 97 94 | **95%** | 100 89 100 92 94 94 89 92 94 94 | **94%** | 97 89 78 83 86 67 67 42 25 6 | **64%** |
| stage 1b with the map zeroed | 86 28 36 19 0 0 0 0 0 0 | 17% | 81 31 53 47 11 17 17 11 17 6 | 29% | 19 6 0 0 0 0 0 0 0 0 | 3% |

- Platform and beam: 89% or more at every height up to 20 cm — as good as stage 1's best climber, WITHOUT losing a flat skill.
- Flat walking (walker's six-direction test, walker v5 = 0.876): 0.900 / 0.859 / 0.886 / 0.864 / 0.868 / 0.855 / 0.855 at the seven
  checkpoints; every checkpoint eligible. At the end: forward 0.36 and backward 0.36 of 0.40 m/s, side 0.08 of 0.13, turn in place 0.50 of
  0.56 rad/s, standing 98%.
- Stairs (new): from 24% to 64% overall. Reliable (80%+) at 2, 4, 8 and 10 cm risers (6 cm: 78%); 12-14 cm 67%, 16 cm 42%, 18 cm 25%, 20 cm 6%.
  Still improving when the run ended (score 0.781 -> 0.842 over the last 600 iterations; practice height 9.6 -> 10.9 cm).
- The fixes did what they were for: noise settled at 0.14 by itself (stage 1: 0.22 -> 0.36); practice heights kept up with ability
  (platform 15 cm, beam 14 cm by iteration 1200; stage 1 had 5 and 9 cm there); the 4 cm weak spot is gone (97%).

Best checkpoint: `archive_anchors/obstacle_s1b_best.pt` (= iteration 2999).
