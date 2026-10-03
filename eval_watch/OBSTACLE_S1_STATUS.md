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

