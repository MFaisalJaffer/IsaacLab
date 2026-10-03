# CALM-FROM-SCRATCH experiment plan (drafted 2026-07-30)

## ★★ PROBE-THRESHOLD BUG — "ALL FELL" WAS AN ARTIFACT (2026-08-03) ★★
The alive mask in walk_gap_probe.py was STICKY total-tilt > 25 deg. Attempt 3
WALKS with a steady **+36 deg forward lean**, so every env was masked out on step
1 and the probe reported "ALL FELL (alive 0/32)" — I reported "cannot walk" TWICE.
**The user caught it from the 8800 render** ("it looks like it walks with its
torso bent more than 25 degrees"). Ground truth (posture_probe.py): pitch +36.1 ->
+36.3 (steady, not growing), roll ~0, base height 0.95 m constant, **32/32 upright
at the end**, translating at 112% of cmd 0.15. FIXED: the mask now uses the ENV'S
OWN fall criteria (base height > 0.50 m AND tilt <= 57 deg = the bad_orientation
termination limit) — posture-agnostic.
**LESSONS:** (a) a probe threshold tuned on one gait family silently invalidates
another — key metrics must key off the environment's own termination criteria,
not hand-picked constants; (b) DECOMPOSE tilt into pitch vs roll (a forward lean
is a posture; roll is a fall); (c) base height is the robust fall detector;
(d) when the render and the metrics disagree, the render wins until proven
otherwise. New tool: eval_watch/posture_probe.py.

## ANTI-LEAN ESCALATION @21.2k (2026-08-03, user-approved, warm-continued)
With the mask fixed, the REAL numbers: the leaning walk is thermally ~2x WORSE
than the deploy build — hip_pitch 18.2 rms (2.4x continuous), hip_roll 18.8
(2.5x), knee 12.7 (1.7x), ankle 9.4 (1.9x); T-V sat hip_roll 95%, ankle 87-91%,
hip_pitch 88% (deploy: 8-47%). Holding a 36-deg torso is a continuous
gravitational hip load = the single cause. flat_orientation_l2 **-3.0 -> -10.0**
(‑3.0 is the value that historically cut a 29-deg lean to 1.4, and costs ~1.04/
step here — the policy pays it, so the lean is buying propulsion).
**GATE @+10k:** pitch < ~15 deg AND still translating (>= 60% of cmd) -> re-probe
torque. If the lean persists or it stops moving -> 1.0 Hz forces falling-forward
propulsion => cadence is the blocker; stop the calm experiment.

## ★ ATTEMPT 3 (2026-08-03, run 2026-08-03_14-05-49, fresh @0): THERMAL-PRICE ISOLATION
**Why:** attempts 1-2 at 1.0 Hz + tau^2 -1.5e-4 NEVER LEARNED TO WALK. Alive-masked
probes at BOTH 29.8k and 56.4k: all 32 envs fall under a sustained walk command at
every calm speed; clearance shows tumbling (apex 33-42 cm, max 82 cm, off-ground
51-67%, stance-duty 42-51% vs healthy ~78%). The fast fresh control was walking well
before that age (ep_len 493@20k -> 615@40k, climbing). **TB metrics LIED** — ep_len
634 and rising gait rewards while the policy could not actually sustain walking;
only the forced-command probe exposed it. LESSON: gate a from-scratch gait on an
ALIVE-MASKED FORCED-COMMAND PROBE, never on ep_len/reward curves.
**Confound:** 5 things changed at once. Two prime suspects — the 1.0 Hz clock and
the 1000x tau^2 price. User call: isolate the PRICE first.
**This run:** dof_torques_l2 -1.5e-4 -> **-1.5e-7** (base). EVERYTHING else in the
calm regime unchanged: 1.0 Hz, cmds (0.12,0.30), stride 0.22, track 2.0,
tv_headroom -2.5/0.93, stand_pose sens 0.5, rel_standing 0.2, flat.
**GATE @~30-40k (alive-masked probe, cmd 0.15/0.30):** envs must STAY UP
(alive > 0/32), stance-duty toward ~75%, apex ~2-3 cm.
  - WALKS -> the effort price was suppressing the vigor needed to learn balance;
    re-introduce pricing gradually AFTER the gait forms.
  - FALLS -> 1.0 Hz cadence is the blocker (4th independent low-cadence failure:
    warm 1.15 blew up hip_pitch; cold 1.0 failed at 29.8k, 56.4k, and here)
    -> conclude this morphology needs ~1.4 Hz and the deploy build's profile is
    the floor.
**Also reverted (failed their own gate):** stand_pose sensitivity 2.0 -> 0.5 and
rel_standing 0.35 -> 0.2 — see the stand deadlock section below.

## STAND-BOOTSTRAP DEADLOCK (found @30k, 2026-08-02; fix warm-deployed)
**Symptom:** Episode_Reward/stand_pose EXACTLY 0.00 for 30k iters. Control: the
fast fresh lineage paid 0.67 from iter 0 -> 1.39 @40k. User spotted it visually
first (the stand "doesn't look right").
**Diagnosis (3 measurements, eval_watch/stand_gate_diag.py + stand_sampling_test.py):**
1. Sampling is FINE — forced _resample_command gives 20/14/20/31/16% standing.
2. But standing envs COLLAPSE in ~0.5 s while walkers survive ~645 steps ->
   env-step share **standing 0.9% / walking 99.1%** (survivorship, not sampling).
3. When standing does occur, pose-err ~2 rad^2 where exp(-err/0.5)=0.02 ->
   ~no reward AND ~no gradient (mean term value 0.0002).
=> closed loop: can't stand -> standing envs die instantly -> ~1% experience +
no gradient -> never learns to stand. Probe confirmed 0/32 upright.
**Contributing cause (honest):** the dead-zone fix (walk cmds >= 0.12) removed an
ACCIDENTAL bootstrap — in the fast lineage, near-zero-command walkers fell into
the standing gate while still near default pose and paid stand_pose from iter 0.
Fixing the lunge also cut that unintended stand signal.
**Fix (user-approved, warm-resumed @29.8k — a genuine objective change, so the
plain-resume explosion law is satisfied):** stand_pose sensitivity 0.5 -> **2.0**
(gradient at the visited pose-err; mirrors the knee_swing 0.1->0.3 precedent) +
rel_standing_envs 0.2 -> **0.35**. KBOT_CALM-only; deploy config untouched.
**GATE @+5k:** stand_pose > 0 and RISING (it moved 0.000 -> 0.001 immediately).
If it stalls near 0, the deadlock is survivorship-dominated, not gradient-
dominated -> next lever is stand-specific (e.g. shorter stand episodes, or
warm-start the stand skill from the deploy build). Watch: a wider kernel pays
for sloppier poses — tighten toward 1.0 once it can hold itself up.
**LESSON:** an exp-kernel reward that reads exactly 0.00 for thousands of iters
is a GATE/GRADIENT bug, not slow learning — check the control lineage's same-age
value before attributing to lateness.

## ATTEMPT 2 (2026-08-01, run 2026-08-01_16-33-09, fresh @0) — DEAD-ZONE BUG FIXED
Attempt 1 (2026-07-31_14-28-03, archived calm_deadzone_lunge_*) failed by design
bug, caught @32k via render + stand probe (0/32 could stand; creeping wide-lunge
attractor): the (-0.15, 0.30) command range made the stand/walk DEAD ZONE
(|cmd|<0.1: gait rewards gate OFF, stand rewards gate ON) ~44% of walker envs
(vs ~10% at the fast build's ±1.0) — half the fleet trained on contradictory
orders (track 0.05 m/s while reward-classified as standing) and the lunge-creep
was the optimal response. LESSON: when shrinking a command range, RESCALE it
against every cmd-norm threshold in the reward stack — gate fractions, not
absolute ranges, are the invariant.
Fix: lin_vel_x (0.12, 0.30) — walkers start ABOVE the 0.1 stand threshold;
vy ±0.05; wz ±0.30; 20% rel_standing envs (pinned 0) own the stand skill.
Reverse walking dropped (not a calm-gait goal). User call: FRESH restart
(nothing worth keeping from the contaminated distribution). All other design
elements unchanged. Probe schedule restarts from 0: forming ~15k, verdict ~40k.

**Goal:** grow a walking gait FROM SCRATCH in a slow, soft regime and measure
whether a thermally sustainable walk (every joint ≤1.0× continuous rating)
exists for this morphology. Speed explicitly does NOT matter. This is a science
run, not a deploy candidate — the deploy build (rigcand_calmref198600) is
frozen and untouched.

**Why from scratch + slow-speed fixes what killed the warm cadence attempt:**
stride = v/f (lowering f at fixed v forced LONGER strides → hip_pitch 2.0×);
and warm-starts refine, never reshape (campaign law). Lower BOTH f and v and
let the gait FORM there: stride at 0.25 m/s / 1.0 Hz = 0.25 m; at 0.15 → 0.15 m.

## Design (all gated behind KBOT_CALM=1 — deploy config untouched)
REGIME (the primary lever):
- gait_freq 1.4 → **1.0 Hz** (swing torque + strike velocity ~f²: ~half)
- commands: lin_vel_x (-1.0,1.0) → **(-0.15, +0.30)**; lin_vel_y ±0.5 → ±0.10;
  ang_vel_z ±1.0 → ±0.30; rel_standing 0.20 kept (stand stack still trains)
GEOMETRY (matched to the regime):
- feet_alternation step_separation 0.30 → **0.22** (≈ natural stride at the
  0.22-0.25 m/s design point — kills the isometric-pull mismatch at its root
  instead of speed-scaling it; the posture-prior effect is preserved, just sized
  for the slow regime)
- feet_phase max_foot_height **0.12 kept**, knee_swing flex **0.55 kept**
  (knee=clearance law; at 1.0 Hz the same lift costs ~half the torque)
THERMAL SHAPING FROM ITER 0 (fresh-run = prevention, the proven dose class):
- dof_torques_l2 −1.5e-7 → **−1.5e-4** (the smooth rms/thermal price; ≈0.10/step
  at the OLD hot gait's Στ² — real pressure, not dominant)
- tv_headroom −2.5 / 0.93 from iter 0 (documented safe fresh dose)
- track_lin_vel weight 3.0 → **2.0** (speed devalued vs effort; the 3.0 bump
  existed to fight the lurch optimum — lurch is a pre-registered watch item)
- ratings-EMA term NOT at start (exploration-noise tax during formation);
  held as escalation if the emerged gait still exceeds ratings
UNCHANGED (the proven stack): mirror loss, log-std, LR 3e-4, containment guards,
stand stack (H3/H4/calm gates), per-joint action clips, bent-start resets,
flat-only (KBOT_FLAT=1), DR set, 8192 envs, save_interval 200.

## Ops
- Legacy kbot_legs_rough run dirs → archive (bundle + safe_ckpts hold every
  deploy artifact); launcher finds no checkpoint → fresh run at iter 0.
- Watchdog RE-ENABLED (fresh-run collapse @32k precedent — it caught one live);
  VF-onset trigger armed; no KBOT_RESUME_STEP_OFFSET → push curriculum ramps
  naturally from 0.
- 8800 on-demand renders work unchanged.

## Measurement plan (pre-registered)
Probes at ~15k (forming?), ~40k (walk verdict), 70k+ (converged):
walk_gap + foot_clearance + stride at cmd **0.10 / 0.20 / 0.30**, flat, pinned.
- **SUCCESS:** stable walk (0/32 falls, apex ≥2 cm, real alternation) with ALL
  joints ≤**1.0× continuous** at cmd 0.20 (stretch ≤1.1× at 0.30). Any
  tracking%/speed accepted.
- **INFORMATIVE FAILURE:** walks but ankle still >1.3× → near-conclusive that
  ankle heat is morphological support cost (no ankle-roll) → hardware answer.
- **Watch items:** lurch optimum (track 2.0 may be too weak — the 3.0 bump's
  original reason), shuffle (apex <1.5 cm), stand-brace regression (H3/H4
  should hold), VF explosions (fresh-run class, watchdog armed).
Timeline: walking historically emerges ~30-50k fresh iters ≈ 1.5-2 days to the
40k verdict at 3.3 s/iter.
