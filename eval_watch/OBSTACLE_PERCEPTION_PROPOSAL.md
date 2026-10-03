# Proposal — obstacle-aware walking with a height map ("climb it or stop")

**2026-10-03 · status: proposal, nothing built or trained yet.**

## 1. The idea in one paragraph

Today the walker is blind: it feels the ground through its joints and IMU only. The proposal is to give it a
small **height map** of the ground around it — a grid of numbers, one per 10 cm cell, saying how high the
ground is there — so it can see a step coming, lift its foot onto it if the step is low, and stop in front of
it if it is too tall. In research this is called *perceptive locomotion with a robot-centric elevation map*
(or "height scan"). It is the most common way legged robots are given sight, and it has been shown on bipeds
and humanoids.

## 2. What we already have

- **The map exists in our simulator.** Every training env already carries a height scanner: a 17 × 11 grid,
  1.6 m long × 1.0 m wide, 10 cm cells (187 values), centred on the robot. So far it is only fed to the
  critic (the training-time "coach"); the policy never sees it.
- **Obstacle terrain exists too.** Isaac Lab's terrain generator has boxes, steps and stairs with a difficulty
  curriculum. Our lineage deliberately removed them with the note "this policy has no height scan, so stairs
  (5–23 cm) and boxes are untraversable".
- **A walker to start from:** walker v5 (all directions, stands, real steps of 8–14 cm lift).
- **What we do not have:** any sensor on the real robot that could produce this map, and any state estimate
  to keep a map steady while the robot moves. That is the real cost of this project (§6).

## 3. What others have done

**Height map into the policy (the basic recipe)**
- Rudin et al., *Learning to Walk in Minutes…* (CoRL 2021): height scan + terrain curriculum (stairs, boxes)
  in massively parallel simulation. Our simulator's scanner and terrain generator come from this line.
- Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild* (Science Robotics
  2022): the policy learns **when to trust the map** — a "belief" module falls back to feel when the map is
  wrong (reflections, soft ground, drift). The standard answer to imperfect maps.

**Two-legged robots with a height map**
- Duan et al., *Learning Vision-Based Bipedal Locomotion for Challenging Terrain* (ICRA 2024, Cassie): train
  the walking policy on a **local height map** in simulation; separately train a small network that turns a
  single depth camera's images into that height map. Crossed single high blocks, stairs and random blocks on
  the real robot with no real-world fine-tuning.
- Gadde et al., *No More Blind Spots* (2025): the follow-up for **omnidirectional** walking (backward and
  sideways, where a forward camera cannot see): a robust blind controller first, then a teacher supervises a
  vision student.
- Long et al., *Learning Humanoid Locomotion with Perceptive Internal Model* (PIM, 2024): the policy reads
  heights sampled from an **onboard elevation map** built from LiDAR; trained with ground-truth heights in
  simulation; about 3 hours on one RTX 4090; continuous stair climbing on several humanoids.
- Wang et al., *BeamDojo* (RSS 2025, Unitree G1): LiDAR elevation map; **two-stage training** — first walk on
  flat ground while already *seeing* the obstacle map, then fine-tune on the real terrain.
- Sun et al., *Learning Perceptive Humanoid Locomotion over Challenging Terrain* (2025): teacher with a clean
  map, student with a denoising model for noisy maps; about 2 km outdoors without intervention.
- 2026 work in the same direction: RPL, PRIOR (height map reconstructed from depth, combined with
  **reference gait priors** — close to our setup), DPL, TACT-ful.

**Deciding to climb, go around or stop**
- Rudin et al., *Advanced Skills by Learning Locomotion and Local Navigation End-to-End* (2022): give the
  policy a **target position and a time limit** instead of a speed command; it is then free to choose how (and
  whether) to cross what is in the way.
- Lee et al. (Science Robotics 2024) and GuideWalk (2026): a separate navigation layer decides where to go and
  hands the walking policy a feasible speed command.
- He et al., *Agile But Safe* (RSS 2024): a learned safety value watches the agile policy and switches to a
  recovery policy before a collision.
- Xue et al., *Collision-Free Humanoid Traversal in Cluttered Indoor Scenes* (2026): a learned obstacle
  representation for stepping over, crouching under and squeezing past obstacles.

**What the literature says about our two goals**
- *Climbing with a map* is well established, including on bipeds. The standard ingredients: height map in the
  observation, an obstacle curriculum from easy to hard, and training against map errors.
- *Stopping for obstacles that are too large* is usually **not** left to the walking policy. Most systems put
  that decision in a layer above it (a rule or a planner). It can be learned end to end, but only if the reward
  makes stopping the best choice — a plain "follow the speed command" reward teaches the robot to push against
  the obstacle.

## 4. Proposed approach

Simulation first, in stages; each stage has one question and a pass test. One change at a time.

**Stage 0 — decisions before any training** (§6): the sensor we would put on the robot, the obstacle family,
and the first target heights. The map we train on must be one the real sensor can deliver.

**Stage 1 — can it climb with a perfect map?**
- Add the height map to the policy's input (current frame only, next to the 10-frame history of the other
  signals). Start with the grid we have; shrink or reshape it once we know which cells matter.
- Terrain: flat floor with single steps, curbs and boxes, 2–20 cm high, rising with a curriculum; a share of
  plain flat tiles stays in so flat walking does not decay (the lineage's lesson).
- Start from walker v5. Forward walking first; sideways and backward onto obstacles later.
- Three known conflicts to fix in the walker's rewards:
  1. the foot-lift reward measures height above a flat floor — it must become height above the local ground,
     and a minimum rather than an exact target, so a higher step onto a curb is not punished;
  2. the reference steps are flat-ground steps, so the style judge will mark a step-up as "wrong" — its weight
     has to come down on obstacle tiles;
  3. add a penalty for hitting the obstacle with the shin or toe.
- Run a **blind** copy alongside (same terrain, no map). The gap between the two is what the map buys.
- Result: a curve of success against obstacle height. That curve *is* the "small enough to climb" limit — we
  measure it rather than guess it. My expectation is 10–15 cm: today's steps lift 8–14 cm, and the ankles
  only pitch.

**Stage 2 — stop for what is too tall**
- Add obstacles above the measured limit (tall boxes, walls).
- Two ways, and I would build both:
  - **A. Rule (safety net):** outside the policy — if anything taller than the limit lies within about 0.5 m
    in the direction of travel, the command sent to the policy is zero. A few lines of code, no training,
    fully predictable.
  - **B. Learned:** during training, when a too-tall obstacle is ahead, the speed the robot is *paid for*
    becomes zero, touching the obstacle is penalised, and standing in front of it counts as success. The
    policy then learns to stop from the map itself.
- Tests: distance at which it stops, contacts with the obstacle, and false stops on flat ground or at
  climbable obstacles.

**Stage 3 — imperfect maps**
- Train against what a real map does wrong: height noise, the whole map shifted by a few centimetres, missing
  cells, delay, and the blind area under and behind the body.
- If plain noise training is not enough, use the teacher–student recipe (our critic already sees the perfect
  map), or a belief module in the style of Miki et al.

**Stage 4 — out of our simulator**
- The RIG's MuJoCo simulator has to produce the same map (ray casts against its obstacles): a new interface
  note, like the one for the 10-frame input.
- On the robot: sensor + mapping pipeline + a check of the map against a known obstacle before any policy
  runs on it.

## 5. Risks particular to our robot

- **No sensor and no odometry on the robot today.** A height map needs both a depth sensor (camera or LiDAR)
  and an estimate of how the robot has moved, to keep the map steady between frames. This is hardware and
  software we do not have yet, and it sets the timeline more than the training does.
- **Ankles only pitch.** Stepping onto an edge, or sideways onto an obstacle, loads the direction the ankle
  cannot correct. Expect a lower climb limit than humanoids with two-axis ankles.
- **A forward-looking sensor cannot see behind.** Our walker also walks backward and sideways; without a map
  kept in memory, those directions stay blind. First version: obstacles handled when walking forward only.
- **Another interface change.** The policy input grows from 430 values to 430 + the map, and the RIG and the
  robot both need the map source.
- **The walker still does about 80% of the commanded speed**, and it has not been hardened against pushes.
  Obstacles add a third open front.

## 6. Decisions needed from you

1. **Sensor direction:** depth camera on the pelvis, or a LiDAR — or "simulation first, decide later". My
   recommendation is simulation first with a deliberately modest map (coarse, short range, forward-biased), so
   that either sensor could supply it.
2. **Obstacle family and first targets:** I suggest single curbs/steps and boxes on a flat floor, up to 20 cm
   in training.
3. **How the stop is done in the first version:** rule only, learned only, or both (my recommendation).

## 7. Cost

Stage 1 is about a day of tooling (terrain, map input, per-height tests and renders) plus two training runs
of roughly 4 hours each (with and without the map). Stages 2 and 3 are one or two runs each. Stage 4 depends
on the hardware decision.

## Sources

- Long et al., PIM — https://arxiv.org/abs/2411.14386
- Duan et al., vision-based bipedal locomotion (Cassie) — https://arxiv.org/abs/2309.14594
- Gadde et al., No More Blind Spots — https://arxiv.org/abs/2508.11929
- Wang et al., BeamDojo — https://arxiv.org/abs/2502.10363
- Sun et al., perceptive humanoid locomotion — https://arxiv.org/abs/2503.00692
- Rudin et al., locomotion and local navigation end-to-end — https://arxiv.org/abs/2209.12827
- He et al., Agile But Safe — https://arxiv.org/abs/2401.17583
- Lee et al., navigation and locomotion for wheeled-legged robots — https://arxiv.org/abs/2405.01792
- Han et al., GuideWalk — https://arxiv.org/abs/2606.10449
- Xue et al., collision-free humanoid traversal — https://arxiv.org/abs/2601.16035
- RPL — https://arxiv.org/abs/2602.03002 · PRIOR — https://arxiv.org/abs/2603.18979 · DPL — https://arxiv.org/abs/2510.07152 · TACT-ful — https://arxiv.org/abs/2606.20645
- From memory, not re-checked today: Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning* (CoRL 2021); Miki et al., *Learning robust perceptive locomotion for quadrupedal
  robots in the wild* (Science Robotics 2022).
