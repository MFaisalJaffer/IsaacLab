# Gentle-walking campaign — 2026-07-24

## ✅ RIG-CANDIDATE BUNDLE SHIPPED (2026-07-30):
## logs/rsl_rl/kbot_legs_rough_rigcand_calmref198600/
model_198600 (zip-verified) + policy.pt/.onnx (nominal gains) +
legs_policy_meta.json (NOW carries the per-joint action_clip table + semantics —
the long-flagged must-ship) + probe references (torque/clearance/stride) +
traces @cmd 0.3 AND 0.5 + BUILD_NOTES.md + HIL_ASKS.md (headline ask: thermal
duty-cycle on ankles + hip_pitch; walk gate at both cmds; watch-items listed).
Training deliberately PAUSED (plain-resume explosion law — see stability memory);
resume only with a genuine objective change.

## ★★★ LEVER #3 (speed-scaled stride) ALSO GATE-FAILED — CAMPAIGN CONCLUSION ★★★
(2026-07-29, stride_gate.txt; run archived gatefail_stride_13-53-52.) STEP-0 had
revised the mechanism honestly (no visible overreach; realized stride ~= natural
v/f; the fixed 0.30 target is unsatisfiable = isometric pull) and the lever DID
align demand with reality (realized 7.8 ~= new target) — but hip_pitch got HOTTER
(rms 10.5->12.0 @0.15, sat 9-12->20%), ankle sat 37-42->54%, tilt +3 deg. The
unsatisfiable splay demand was a POSTURE PRIOR (fore-aft pressure = wider
effective base = passive pitch stability); relaxing it shifted work onto ACTIVE
hip/ankle balancing. Clean falsification (gait stayed healthy, tracking 101-105%,
0 falls). One-line revert executed; fixed 0.30 stays.
**CAMPAIGN CONCLUSION after 5 falsified levers** (knee-cut, lift-cut, cadence,
headroom escalation, stride-scaling — every one caught by a pre-registered gate):
the reference build's torque profile IS this gait family's demonstrated floor.
Every reward-side push either degenerates the gait (shuffle/limp/dither) or
relocates heat to a hotter joint. Residual thermal state (ankle ~1.5x, hip_pitch
~1.3-1.4x continuous, all others <=1.1x) is a MORPHOLOGY+DUTY-CYCLE question:
no ankle-roll DOF + stance-dominated torque. NEXT: consolidate the reference
build, HIL thermal duty-cycle measurement (minutes-of-walking per thermal limit),
hardware answers (bigger ankle motor / ankle-roll) if duty-cycle insufficient.

# (history) CALM-WALK PACKAGE: BOTH LEVERS FAILED THEIR GATES — FULLY REVERTED
## (2026-07-29; runs archived: fallen_calmwalk_fulldose + gatefail_cadence115)
**Lever 2 (headroom -4.0/0.88) failed @208.8k**: pre-registered Goodhart trio —
gate-flip dithering (tick-flips +40%), hip_pitch shunt, shuffle + asymmetric LIMP
(L knee rms 4.4 vs R 7.5), tracking 31%. Escalation lane CLOSED on flat too.
**Lever 1 (cadence 1.15) failed @216.6k**: f^2-swing premise FALSIFIED — missed the
stride coupling (stride = v/f): hip_pitch rms 9.8 -> 15.1 Nm (2.0x cont), sat 7 ->
57%, tilt 10.5, tracking 79%, apex 1.7-2.0 cm. Walk torque is STANCE-dominated;
slower clock = longer strides + longer support = hotter hips. NEVER lower the
clock at fixed commanded speed.
**Net state**: reverted to the restored reference build (1.4 Hz / 0.55 / 0.12 /
-2.5/0.93, flat), resumed @198.6k healthy anchor. The calm_walk_reference.txt
numbers (ankle sat 43-47%, all-joint rms 7.7-9.8, apex 2.8-3.0 cm, tracking 106%)
stand as the gait family's demonstrated floor. Remaining calm-walk lanes, all
requiring their own STEP-0 first: (a) speed-scaled step_separation (SHORTENS
strides at low cmd — the exact inverse of the coupling that killed cadence; the
wildcard's probe regression showed hip_pitch -0.7 Nm), (b) reserved withhold-gate
after its frame bug fix. The ankle's ~45% sat / 1.5x thermal may simply be this
morphology's walking cost (no ankle-roll, stance-dominated) — a hardware
duty-cycle question as much as a policy one.

# (history) CALM-WALK PACKAGE DEPLOYED @198.6k (2026-07-28, run 2026-07-28_15-05-51)
User-approved two-lever deploy (lever-hunt synthesis: calm_walk_lever_synthesis.json;
15-agent analysis + adversarial verify; thermal-EMA & impact levers REJECTED by
arithmetic, posture a no-op, withhold/stride reserved):
- **Lever 1 · cadence**: _GAIT_FREQ 1.4 -> 1.15 Hz (swing torque + strike velocity
  ~f^2). Exporter GAIT["gait_freq"] updated in lockstep (rig phase-obs coupling).
- **Lever 2 · tv_headroom FULL dose**: -2.5/0.93 -> -4.0/0.88. STEP-0 measured the
  ankle's at-clamp time is ~50-60% MOTORING-classified (visible to the term) ->
  full-dose band. Flat-only branch, so the old terrain-vigor collapse is moot.
**Reference anchor** (@198.4k, calm_walk_reference.txt + REF-restored clearance):
cmd0.3 ankle sat 47/43, knee 21/20, rms hip_pitch 9.7/9.8 knee 9.5/9.2 ankle 7.9/7.7,
tilt 6.3, apex 2.8-3.0 cm, off-ground 11%, tracking 106%, flicker 8.1, duty 77%,
tick-flips ankle 26/25 knee 28/28. Anchor ckpt: safe_ckpts/pre_calmwalk_model_198400.pt.
**GATE @~208.6k** (32env, flat, pinned, cmd 0.15/0.3/0.5 vs reference):
HARD: 0 falls, ep_len>=900, apex>=2.5cm, off-ground>=7%, duty<=85%, tilt<=ref+2,
hip_roll rms<=ref+0.5, tracking>=85%. CADENCE: knee rms<=ref-0.5 AND hip_pitch<=
ref-0.3, OR apex>=3.0 with both <=ref+0.2 ("dividend lands on clearance" = pass).
HEADROOM: ankle sat<=39 (ref-6), tick-flips NOT above baseline, motoring-share
falls in the 3-way split, hip_pitch rms<=10.4, stand-command sat not regressed.
ROLLBACK LADDER: (1) balance fail -> full revert to anchor; (2) shuffle signature
-> revert headroom only, keep cadence, re-gate +5k; (3) lateral regression ->
cadence 1.25 half-dose; (4) ankle sat unmoved -> headroom exhausted, revert it,
next-cycle = speed-scaled stride (reserved).
Early health @+150 iters: ep_len 983, falls 0.08, VF 0.76, headroom bite -0.08. OK.

## STATUS: Phase 2 FLAT-ONLY gentle branch LIVE @142k (run 2026-07-26_17-51-48).
Terrain build FROZEN as robust fallback (eval_watch/terrain_fallback.txt,
safe_ckpts/terrain_robust_fallback_model_142000.pt). Flat branch: warm-start from
142k, foot-lift 0.10->0.06 (clearance not load-bearing on a plane), knee stays 0.35,
KBOT_FLAT=1, 8192 envs. Flat CONFIRMED (terrain_type=plane, no terrain curriculum).
GATE PASSED @146k (gentle_walk_flat.txt, probe vs baseline/p1b, ×real continuous):
  cmd0.15: ankle 1.52->1.33->1.26 | hip_pitch 1.41->1.30->1.21 | knee 1.23->1.05->0.95
           | hip_roll 0.99->0.91->0.81 | hip_yaw 0.24
  cmd0.30: ankle 1.60->1.48->1.37 | hip_pitch 1.31->1.23->1.17 | knee ->0.99 | hip_roll ->0.95
0 falls, ep_len 998, tracking 93-98%. WIN: flat 6cm cooled the ANKLE (worst,
balance mech) 1.62->~1.3x; only ANKLE (~1.3x) & HIP_PITCH (~1.2x) remain over
continuous, everything else UNDER. TRADEOFF: torso tilt 6.5->9.2 deg mean (max
13.8->18.2) — gentler = wobblier; dial foot-lift back toward 0.08 if posture matters.
Options now: (a) consolidate this as the gentle FLAT deploy build; (b) push lift
6->5cm for a bit more ankle/hip-pitch (watch tilt); (c) back to 0.08 if tilt too high.

## (prior) Phase 1b HELD + WON @141k. Knee solved; ankle was the worst offender.

### Phase 1b RESULT (probe @141.8k vs baseline @100.6k, flat/pinned, gentle_walk_phase1b.txt)
Recovery complete + gentleness confirmed. terrain 2.19, 0 falls, ep_len 850,
tracking 98-100% (IMPROVED). cmd 0.30 rms Nm / overload (÷ real continuous):
  knee     9.35 -> 7.5  (1.25x -> **1.00x** SOLVED, at rating)
  hip_pitch 9.85 -> 9.25 (1.31x -> 1.23x, down)
  ankle    8.0  -> 7.4  (1.60x -> 1.48x, down — now the WORST offender)
  hip_roll 8.15 -> 8.3  (~flat)
Pattern holds at 0.15/0.50. **The feared knee->hip-pitch torque shift did NOT
happen** — hip-pitch & ankle both dropped; the whole leg got flatter/gentler.
CAVEAT: +40k more train iters than baseline, so some is refinement; the knee-
specific 20% drop is clearly the 0.55->0.35 config change. Remaining thermal
offenders: **ankle 1.48x (balance mechanism, hard to touch), hip_pitch 1.23x.**
Next options: (a) gradual foot-lift 0.10->0.09 for more; (b) leave gait as-is and
address ankle/hip-pitch via a corrected Phase-2 thermal finisher; (c) consolidate.



### Real motor ratings (user-confirmed 2026-07-24)
`_04` hip_pitch/hip_roll/knee = 7.5 Nm cont / 22 stall. `_02` ankle + `_03` yaw
= 5.0 cont / 11 stall. (Supersede the 9.0/4.5 guesses.)

### Baseline probe (model_100600, flat, pinned gains) — gentle_walk_baseline.txt
Leg torque is a FLAT 8–10 Nm rms across 0.14–0.48 m/s (3.5× speed) → heat is a
gait-STYLE cost, NOT propulsion; "walk slower to run cooler" DOESN'T work.
Overload (rms/cont): ankle 8.1/5=1.62×, hip_pitch 10.5/7.5=1.40×, knee &
hip_roll 9.4/7.5=1.25×, hip_yaw 0.36× (cold). Gait healthy: tilt 5–7°, 0 falls,
stance-duty 76–79%.

### Phase 1a — ONE-SHOT lift+knee drop: COLLAPSED (do not repeat)
Warm-start from 101000; foot-lift 0.12→0.08 AND knee 0.55→0.35 together, @8192 envs.
By 110k: terrain 2.2→**0.32** (stuck, not climbing), ep_len 810→**~250**,
bad_orientation falls **~22/window** (vs time_out ~2.7), reward stuck ~7. Flat for
9k iters = not a transient, a basin. **LESSON: the 12 cm foot-lift is largely
LOAD-BEARING for terrain clearance** (same class as the tv_headroom -5.0 vigor
collapse) — cutting it to 8 cm broke terrain-walking. Archived to
logs/rsl_rl/kbot_legs_rough_archive/fallen_phase1a_15-03-18 (preserved, not wiped).

### Phase 1b DEPLOYED — recover + isolate (run 2026-07-25_00-02-15, @8192)
Reverted to the healthy 101000 anchor. **feet_phase rewards foot-BODY height
directly, so clearance is set by max_foot_height, NOT the knee angle** (user insight
2026-07-24): the knee-flex cut is a THERMAL lever, ~clearance-independent. So:
- feet_phase max_foot_height **0.12 → 0.10** (mild -2 cm; fixes clearance)
- knee_swing flex **KEPT at 0.35** (~20°; keep the knee thermal win)
Early recovery (first ~few iters): bad_orientation **22 → 0.5**, terrain climbing
0.32→0.9. **GATE @+8-10k:** terrain level HOLDS **≥2.0** + falls low; then re-probe
torque — check **knee AND hip-pitch** rms together (a straighter knee may shift
swing torque onto hip-pitch, itself 1.40× over rating). If terrain still demotes →
0.10 lift insufficient OR knee couples to clearance → revert knee to 0.55 to
disambiguate. If it holds → clearance hypothesis confirmed + knee gentleness free.
**Then:** step foot-lift lower gradually (0.10→0.09…) with a gate between each — NO
more one-shot drops. **Phase 2 (later):** corrected torque_thermal_ema (real 5.0/7.5
ratings, threshold above the NEW floor, ankle spared/capped) for residual heat.

### Open strategy question (surfaced by 1a)
If foot-lift is genuinely load-bearing on THIS terrain, gentleness trades against
terrain robustness. Deploy surface (HIL rig) is FLAT — a lower-clearance gentle gait
may be fine for deploy while costing rough-terrain margin. Option if 1b plateaus:
develop the gentle gait on flat-only, keep terrain robustness as a separate build.

---
# (history) W2 thermal term — pre-deploy review that caught the blocker

**Verdict: NOT DEPLOYED. Blocker found + verified inline. Live run untouched.**

## What was built
`torque_thermal_ema` (mdp_gait.py, kept): per-joint EMA(tau^2) (~1.5 s) priced
above continuous rating^2; sum of max(0, EMA/rating^2 - 1). weight -0.2,
ratings {_04:9.0, _03:4.5, _02:4.5}. Intent: winding-heat model, EMA exempts
transient catches (H3 analog), no cmd gate (one law stand+walk).

## The blocker (2 independent lenses converged; I re-derived it myself)
I set the **ankle threshold BELOW the honest operating point** — the inverse of
what made the stand fix work.
- Stand fix WORKED: 12 Nm deadband > 9 Nm geometric floor -> only brace EXCESS paid.
- This term: ankle rating 4.5 < honest walk ankle ~7.8 rms -> EMA(tau^2)~60.8 vs
  4.5^2=20.25 = **3.0x over threshold during NORMAL walking**. The term prices
  the FLOOR, not the excess.
- Bill: ankle 2.0 excess x2 = 4.0 of a 5.82 total (**69% is ankle**); knee 0.49x2,
  hip-pitch 0.41x2. x0.2 weight = 1.16/step vs track_lin 3.0x0.8 = 2.4 -> **~48%
  the size of the entire tracking reward**, deployed as a step change on a 95k policy.
- The ankle is the **sole sagittal balance actuator** (no ankle-roll DOF). Its load
  is **speed-independent** (probes), so the sanctioned "walk slower" escape can't
  reduce it. The gait clock (feet_phase 12cm lift +3.5, knee_swing 32deg +4.0,
  feet_alternation 30cm +2.0) freezes the "gentler gait" DOFs. => the ONLY payable
  direction is cutting ankle balance/toe-off torque = the stand_torque failure
  (falls 2->15/32) with a 1.5 s delay line in front of it.

## Compounding problems
1. **Rating 4.5 is a GUESS** (48%-of-peak of the 9.35 Nm class). The whole
   calibration rests on a made-up number. Need the real _02/_03 continuous rating.
2. **Fuzzy operating point**: "ankle rms 7.8" and "ankle T-V sat 19-84%" are
   different metrics from different probes/builds. No clean current-build torque
   number exists to anchor a threshold to.

## Rejected finding (verified false, for the record)
"EMA reads only last-of-4 substeps -> aliasing." Real code fact, but the failure
model is wrong: 14-15 Hz dither is sub-Nyquist and the 77-sample EMA sweeps phase
(within +/-1.2% of true mean of tau^2); it's the established known-good read idiom
(tv_headroom, stand_torque, Isaac's own applied_torque all use it). Not blocking.

## The reframe (why walk != stand for this)
At stand the brace was **clean excess above a calculable floor** (9 Nm geometric),
so a threshold-above-floor penalty carved it off cleanly. Walking has **no
demonstrated gentle-gait floor** — the current torque IS set by the gait-shaping
clock (12cm foot lift, 32deg knee swing, springy anti-hop push-off). Taxing torque
on top of a clock that MANDATES vigor can only be paid by breaking balance.

**=> Gentleness comes from relaxing the gait clock FIRST (making a gentler gait
reachable), with a properly-calibrated thermal/impact term as the finisher — not
from a torque tax as the driver.**

## Corrected-design requirements (if a thermal term is used later)
- Real _02/_03 continuous rating (from user/datasheet), not a guess.
- Threshold ABOVE a *demonstrated gentle-gait* floor (price excess, not the floor).
- Spare or CAP the ankle (balance mechanism); concentrate on headroom joints
  (hip-pitch sat 0-13% has the most; knee some).
- Per-joint cap so no single joint dominates the gradient.
- Ramp weight in + tv_headroom-style rollback guards (ep_len<300, VF>10).
