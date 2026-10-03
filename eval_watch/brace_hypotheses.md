# Hip-roll A-frame brace — hypothesis tracker

## ★ SOLVED @51k (2026-07-23) by H3+H4 TOGETHER ★
Hip-roll stand torque **20 -> 4.7 Nm**, 0/32 falls, tilt 2.8/4.9 (BETTER than
braced). stand_pose reward reclaimed 0.02->1.03 (policy dropped torque to earn
it back); H3 EMA penalty -> 0 (command < deadband). **THE BRACE WAS NOT
LOAD-BEARING** — it was a reward-shaped local optimum (rigid = high stand_pose,
torque free in sim), not morphological necessity. Every prior build braced only
because no single lever could break it; H4 (withhold the reward) + H3 (charge the
SUSTAINED command, exempt transient catches) together dislodged it -> a genuinely
stable LOW-torque stance. Confirm-holds at the 58k gate + verify walk/knee wins.
The morphological/hardware fallback (H6/H7) is NOT needed.

---
# (history) hypothesis tracker

**Problem:** at commanded stand, the policy holds both hip rolls at ~18-20 Nm
CONTINUOUSLY (thermal hazard on hardware). Trace: joints AT default (q~0) but
COMMANDED ~0.7 rad apart (action ~1.4) — an isometric outward preload against
friction-pinned feet. Feet don't slide, joints don't move; only the torque is high.
**Morphology note:** robot has NO ankle-roll DOF, so lateral balance = hip-roll or
stepping only; ~9 Nm continuous hip-roll is the unavoidable geometric gravity load
(hardware-fine per user). Only the ~9 Nm BRACE EXCESS (9->18) must go.

**Unifying law (5 failures + analysis):** price the CAUSE (the sustained COMMAND /
action offset), never the OUTCOME (foot slide / displacement / instantaneous
torque — all SHARED between brace and balance, so penalizing them kills balance
first). And distinguish SUSTAINED (brace) from TRANSIENT (balance catch) — the
missing discrimination behind every failure.

**Meta-hypothesis:** EVERY build ever measured braces (~19 Nm: 210k/289k/326k/
l2-40k, across wildly different reward configs) -> the brace may be a MORPHOLOGICAL
attractor (no ankle-roll + narrow stance need lateral stiffness), in which case NO
reward form wins and the answer is structural (H6/H7).

---
## DEPLOYED — under test at the ~50k gate
- **H1 · calm-gate stand_pose + stand_feet_planted** (vel_release 0.15) — free the
  lateral CATCH-STEP during a genuine push so bracing isn't the only push-defense.
  Deployed @40.8k. Risk: quiet-stand fidget if threshold too loose.
- **H2 · stand_hip_roll_brace -0.5 -> -1.5** — flip the arithmetic: at -0.5 the brace
  (+0.8 stand_pose rigidity bonus) beat the 0.55 penalty; -1.5 makes relaxed win.
  Deployed @44k. **VERDICT @48.2k: SAFE BUT INEFFECTIVE.** Brace torque UNCHANGED
  (19-20 Nm, target 9-12); command trimmed |action| 1.4->1.1 (~20%, real but tiny)
  yet still deep in saturation so ZERO torque relief. NO harm: upright 61/59 of 64,
  gyro 0.13, lifts 39/min symmetric — NOT the falls-spike branch. Signature: policy
  PAYS the penalty and KEEPS the brace, trimming command only to the stability floor
  -> the brace is worth more than any payable penalty = LOAD-BEARING for lateral
  stability. Rules out "harder instantaneous penalty" (H3 alone would just convert
  pay->fall). -> H4 (withhold reward, different axis).

## DEPLOYED TOGETHER @48.4k (2026-07-23, user: "do both H3 and H4")
- **H3 · TIME-WINDOWED brace penalty** [user idea, converged]. Class term
  stand_hip_roll_brace_ema, -2.5: EMA (alpha 0.04 ~0.5s) of |commanded hip-roll
  action| (idx 2,3); penalize SUSTAINED excess over deadband 0.3, hip-roll-scoped,
  calm-gated. Sustained brace -> EMA high -> pays ~2.2/step; transient catch ->
  EMA barely moves -> ~free. Replaces the zeroed instantaneous H2. SAFE to crank
  (no correction-suppression).
- **H4 · torque-GATED stand_pose** [user idea]. stand_pose *= exp(-(hip_roll_torque
  - 12).clamp(min=0)/4): braced 20 Nm -> keeps 13% of the +15; <=12 Nm -> keeps
  100%. Withholds the brace's REWARD (graded, gives a lower-torque gradient) vs
  adding a payable penalty. Hip roll = joint idx 2,3, from applied_torque.
- Combined arithmetic (braced vs relaxed / step): stand_pose 0.37 vs 2.8; H3 -2.2
  vs 0; NET -1.8 vs +2.8 = 4.6 gap toward relaxed. Restart healthy (ep_len 641, VF
  0.66; stand_pose reading collapsed 0.88->0.02 = H4 working). Anchor:
  eval_watch/pre_h3h4_ckpt.txt. **GATE @~58k: brace torque 20->9-12 + |action|
  1.4-><0.3 (EMA down) WITHOUT falls (upright holds ~57-61/64) or fidget. If the
  brace holds at 20 despite THIS max-safe two-angle pressure -> MORPHOLOGICAL
  confirmed, go H6/H7.**

## PENDING — if H3+H4 fail
- **H5 · reduce stand_pose weight/kernel** — blunt: shrinks the brace's rigidity
  advantage but risks fidget's return (stand_pose +15 is the anti-fidget solution).
  Only if H2-H4 all fail and we're sure fidget won't regress.

## STRUCTURAL / HARDWARE — if the reward lane is exhausted (H2-H5 all plateau)
- **H6 · stance-width change** — feet directly under hips -> ~0 lateral lever ->
  ~0 gravity hip-roll torque. BUT no ankle-roll DOF means a hip-roll offset TILTS
  the soles; needs hardware's sole-tilt tolerance. Also narrower base = tippier
  (more dynamic hip-roll). Reward-only via stand_pose target offset (warm-startable).
- **H7 · add ankle-roll DOF** — the morphological root fix; gives the natural
  low-torque lateral "ankle strategy" humans use. Hardware redesign. Last resort /
  long-term.

## GRAVEYARD — TRIED, FAILED, do NOT repeat
- stand_foot_slide (-3): Goodhart, slip-speed down / net-drift up. Outcome channel.
- stand_foot_anchor v1/v2/v3: tap-ratchet / correction-suppression (falls) / both.
- stand_torque (-0.3, |torque|>12 Nm deadband, all joints): INSTANTANEOUS -> hit
  transient catches; brace unchanged (19 Nm), falls TRIPLED 2->15/32. (H4 differs:
  WITHHOLDS reward vs ADDS penalty; H3 differs: TIME-WINDOWED vs instantaneous.)
- tv_headroom: motoring-gated; a stalled joint reads as braking -> structurally blind.
- stand_action_magnitude (-1.0, deadband 1.0, all joints): too weak / diluted 10x.

## Decision at the 50k gate
- H2 unwinds brace (torque->9-12, |action|->0.3) + no fall spike -> WORKING, done.
- H2 unwinds but falls spike -> correction-suppression -> deploy H3 (time-window).
- H2 does nothing -> if H3/H4 also plateau, brace is MORPHOLOGICAL -> H6/H7, accept 9 Nm.

---
# W-series — GENTLE WALKING (peak-torque reduction), designed 2026-07-24
**Goal:** carry the stand lessons into the gait: lower PEAK torques / softer motion
while walking, without repeating the two known failure modes (stand_torque falls
3x; tv_headroom -5.0 fleet collapse — terrain vigor is LOAD-BEARING).
**Measured waste signature (walk probes @71.8k-93k):** (a) VIGOR FLOOR — ankle
sat barely scales with speed (36-39% @cmd 0.15, 43-46% @0.3, 54-56% @0.6): the
gait spends near-constant push-off effort regardless of commanded speed = the
walking analog of the brace (sustained effort the task doesn't need). (b) SPIKE
GAP — rms >> mean on knee (7.8 mean / 11 rms) & hip pitch (9.0/10.7): foot-strike
impacts + push-off transients carry the peaks.

- **W1 · gentleness-GATED walk rewards (H4-analog).** track_lin_vel_xy_exp (and/or
  feet_phase) *= exp(-(max motoring T-V frac - 0.85).clamp(min=0)/tau). Full pay
  only for under-limit walking; "fast by slamming torque" earns nothing extra.
  WITHHOLDS reward (unGoodhartable into falling) vs pricing. Pairs naturally with
  the deferred lin_vel_error_l1 tracking fix.
- **W2 · EMA headroom (H3-analog) — RECOMMENDED FIRST.** EMA (~1.5 s, ~2 gait
  cycles) of per-joint motoring T-V fraction; penalize sustained excess over ~0.65.
  Targets the VIGOR FLOOR exactly; single-step spikes (terrain catch, one hard
  push-off) barely move the EMA = exempt — the discrimination the -5.0
  instantaneous form lacked. Refines the proven tv_headroom (-2.5 @0.93 stays).
- **W3 · impact pricing.** Raise foot_impact_penalty (currently inherited, tiny):
  prices hard heel-strikes -> softer landings -> lower structural peaks. Walking-
  specific; doesn't touch continuous effort at all. Safest single lever.
- **ANTI-PATTERN (do NOT):** raise dof_torques_l2 (-1e-5, always on, never stopped
  rail-riding) — instantaneous, all-joint, OUTCOME-channel = the stand_torque
  mistake; would tax catches and terrain vigor first.
**Deploy law:** one lever at a time, graded, gate on: walk tracking (>=78% floor),
falls, terrain level (>=2.2), ankle/knee sat%, per-joint |torque| p95. If W2 alone
plateaus, add W1 as the second orthogonal pressure (the H3+H4 lesson).
