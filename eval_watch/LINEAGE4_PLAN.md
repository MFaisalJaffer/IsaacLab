# LINEAGE 4 PLAN (drafted 2026-08-16)

Lineage 3 status at boundary: iter ~55.5k, ep_len 940–989, reward ~160–167, VF ~1.0,
level 1.0 (25 N cap), noise σ 0.052. Gate battery: 20N@1024 96.3%, 25N 87%, impulse 98–99%,
thermals hips/knees under continuous / ankles 1.18–1.24×, quiet stand p95 roll ~0.6°,
free-base 8° roll step recovers <1° in ~0.5 s. Known gaps → lineage 4 targets.

## 0. Close out lineage 3 FIRST (before any stop)
1. Checkpoint-safety protocol: copy newest checkpoint out of the run dir, `torch.load`-verify
   the copy (never stop mid-save; warm-start/eval only from the verified copy).
2. Final full gate battery on that copy: stations 1–6 + 7 (tilt_decay, roll) + NEW 8 (pose_rehome).
3. Export the rig deploy bundle from it (policy + deploy contract + probe results) — the rig
   thread continues on lineage 3 while lineage 4 cooks.
4. Archive run dir untouched (preserve-training-runs rule). Renders/watcher keep pointing at it
   until the new run exists.

## 1. Carried unchanged (proven stack)
- De-chatter gains (kp60/kd3 on 02/03 joints; kp150/kd2 on 04s), T-V actuator, delays.
- walk_at_spawn (+velocity REDRAW), stand corridor 1.5 s @0.12, spawn z (0.0,0.1),
  bad_orientation stand grace, base_height grace 30 + stand_grace 1.5 s,
  stand exemption from base_height + stand_height_slope (0.85 m, −3.0).
- (π,π) stand pin with 1 s annealed shortest-arc entry.
- stand_tilt_wall (3° deadband, −5.0, never released) + tilt-servo holds inside
  sustained_push_bursts — ONE change: `hold_roll_bias 0.7 → 0.5` (pitch is the weak axis
  once ankles are loose; rebalance practice).
- Sustained-push thermostat: completed-episode capture, min_samples 200, end_force (10, 25),
  ep_len>620/+, <480/−. Push/hold release split (`_push_release` vs `_push_or_hold_release`)
  exactly as wired.
- push_step_shaping (0.41/0.65, 18°), push_brace_shaping v2, walk_upright_wall (2°/3°, −4.0).
- PPO: entropy 0.003, LR 3e-4 adaptive at birth → RE-PIN 1e-5/fixed at convergence (law),
  [kbot guard] straight-through std ceiling (verify present after any pip event).
- 12288 envs, KBOT_FLAT=1 KBOT_ADAPT=1, watchdog OFF until ep_len>400 then armed,
  save_interval 200, renders on 8800 watcher.

## 2. NEW package A — pose re-homing (the headline gap)
Evidence: after 20 N pushes, joints partially re-home (8–12°→3–6°) but foot geometry
ratchets and parks: width 36→46→45→51 cm, stagger 1.8→5–6.5 cm, flat for 10 s; falls
climb 0→3→14% across pushes. Mechanism: stand_pose (joint kernel) is nearly blind to
width; the corrective step is taxed; recurring bursts make braced-wide optimal.

A1. `stand_stance_geometry` reward (new, mdp_gait.py):
    - body-frame foot geometry: width = |y_L − y_R|, stagger = |x_L − x_R|
    - penalty = w_w·relu(|width − nominal| − 5 cm) + w_s·relu(stagger − 3 cm), standing envs only
    - nominal from default-pose FK measured at startup (≈34–36 cm; constant param after measuring)
    - released by `_push_or_hold_release` (protective stepping stays free)
    - weight ≈ −2.0 (start; must beat the ~1° joint-kernel blindness but not fight push response)
A2. Post-push re-home grace: for ~2.5 s after a burst/hold ends on an env,
    soften stand_feet_planted + feet_slide (standing branch) + stand_still_joint_motion
    (e.g., ×0.25) so the step home is cheap. Publish `env._rehome_until` from
    sustained_push_bursts at burst end; taxes read it. stance_geometry stays LIVE during
    grace (it is the gradient home).
A3. stand_pose unchanged (it works for joints); no new always-on width term beyond A1's
    deadband (A1 is always-on-at-stand already, deadbanded — covers the entry case).
Acceptance (new gate station 8, pose_rehome_probe): after 20 N ×1.5 s from settled stand:
    width within +4 cm of baseline by +5 s; stagger < 3.5 cm; pose_err < 3.5°; and across
    3 sequential pushes no ratchet (final width within +5 cm of original baseline); falls
    on ±Y ≤ 2%.

## 3. NEW package B — ankle free-play DR + motor-side encoder obs
Hardware truth: ankles ±7.5° (15° total) free band; other joints ~0.
B1. Wire `randomize_joint_play` (already written) into events: startup+reset draw,
    ankles from a CURRICULUM range, others U(0°, 0.5°).
B2. Play thermostat (reuse sustained-push thermostat pattern, completed-episode ep_len):
    ankle band cap starts 2° → +1°/step when ep_len>620 → cap 16°; floor stays 60% of cap
    (draw U(0.6·cap, cap)); never decreases (ratchet, like push level; −1° only if ep_len<480).
B3. Motor-side encoder obs (the essential half at 15°): TVCurveActuator keeps a virtual
    motor angle m per env-joint: m steps toward the commanded target with the servo, and is
    clamped each step to within ±play/2 of the true link angle (linkage constraint).
    Ankle joint_pos obs (and joint_vel for ankles) report m, ṁ instead of link truth —
    ACTIVE FROM BIRTH so the policy never learns to rely on information the robot lacks.
    Rigid joints (play=0): m ≡ link angle, exactly legacy obs.
B4. Gate addition: station 7 decay test runs BOTH roll and PITCH offsets (pitch is the
    play-vulnerable axis); station: pitch 8° step → upright <2° within 1.5 s at band 15°.
    Plus stand battery re-run at fixed 15° band (wandering budget: net drift <15 cm/12 s
    target by end of lineage; lineage-3 measured 52–58 cm).

## 4. Phasing
- Phase 0 (birth): everything above live: re-home package (native learning), play DR at 2°
  cap, motor-side obs on, holds 50/50 roll/pitch. Watchdog off.
- Phase 1 (ep_len > 400): arm watchdog (VF-ONSET rollback to oldest anchor).
- Phase 2 (ep_len > 620 sustained): thermostats climb — push force toward 25 N AND ankle
  band toward 16° (both gated on the same completed-episode ep_len; they will alternate
  naturally as headroom allows).
- Phase 3 (plateau at full difficulty): LR re-pin 1e-5/fixed; wean ladder only as
  PROPOSED rungs at natural boundaries (no unnecessary restarts).
- Gate batteries at ~2k, ~10k, then per boundary; stations 1–8 + pitch decay.

## 5. Execution sequence at switch time
1. Close-out (section 0) — battery ~1–1.5 h while implementing.
2. Implement A1–A3, B1–B4 (+ hold_roll_bias 0.5) with compile + 200-iter smoke test at
   1024 envs in a throwaway dir (verify: new reward channels logged, play draw visible in
   actuator, motor-obs path exact-legacy when play=0, no NaN).
3. Launch lineage 4 in a FRESH run dir (from scratch), kbot-train.service env updated;
   watcher render lines already KBOT_FLAT=1 KBOT_ADAPT=1.
4. Overnight: expect gait formation; morning check: ep_len trajectory vs lineage-3's
   from-scratch curve (crossed 600 within ~day 1), stance_geometry channel engaged,
   play thermostat still at 2° until 620.

## 6. Risks / falsifiables
- A1 weight too strong → fights push-response stepping (watch push gates at station 1–2;
  if 20N% drops >3 pts vs lineage 3 at comparable maturity, halve weight).
- Re-home grace exploited as free fidget window → watch feet_slide/lifts-per-min in stand
  battery; if fidget grows, shorten grace to 1.5 s.
- Motor-side obs slows gait formation (less ankle info) → expected mild; judge only vs
  from-scratch curve, not intuition; do NOT revert before 2 days unless NaN/collapse.
- Play thermostat + push thermostat both gated on ep_len could oscillate → both use
  ratchets with hysteresis (620/480), alternation is acceptable; cap climb rate 1 step
  per 200 iters each.
