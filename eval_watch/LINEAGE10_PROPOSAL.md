# Lineage 10 proposal — reward audit, what to change, and why from scratch

**2026-09-22** · Lineage 9 passed the standing gate at 177k (4.3° at the hardware stiffness)
and lost it again by 382k (7.1°) while total reward went UP. This is a reward-economy failure,
not a plant or capacity failure. The audit below is measured from TensorBoard on both
checkpoints, not inferred.

---

## 1. What the optimizer actually did (177k → 382k)

Per-term change, Episode_Reward units (≈ mean_reward / 20):

| bought | | paid | |
|---|---|---|---|
| feet_phase | **+0.424** | joint_deviation_hip (wider stance) | −0.066 |
| knee_swing | +0.187 | stand_upright | −0.017 |
| track_ang_vel_z | +0.121 | stand_stance_geometry | −0.015 |
| feet_alternation | +0.093 | stand_tilt_excursion | −0.012 |
| track_lin_vel_xy | +0.037 | | |
| **total gait/tracking** | **≈ +0.90** | **total standing** | **≈ −0.10** |

A 9:1 trade. The policy sold standing quality for gait quality at that price and the total went
+17.85. **Nothing about this is a bug — the economy priced it and the optimizer took it.**

## 2. Why standing is worth so little

**(a) The pot is tiny.** Every standing penalty combined at 382k = **0.257 units**. The five
gait/tracking carrots = **8.86 units**. Standing penalties are 2.9% of the carrots. Doubling
every standing penalty costs less than the feet_phase gain alone.

**(b) There is nothing left to *earn* by standing better.** The only positive standing term,
`stand_pose` (+15), is saturated at 0.92 and has not moved in 200k iters. Standing can only
*lose* small amounts; walking has five carrots still growing. The gradient points one way.

**(c) The tilt penalties are switched off for most of a standing episode.** Measured duty from
the event parameters:

| | duty | effect on standing terms |
|---|---|---|
| sustained bursts (dur U(0.5,2.5), rest U(3,8)) | ~21% | `stand_upright`, `excursion` ZEROED (`_push_release`) |
| tilt holds (dur U(1,3), rest U(4,10)) | ~22% | motion taxes zeroed; tilt terms live |
| re-home grace (2.5 s after each burst) | ~36% of cycle | motion taxes ×0.25 |
| velocity shove every 10–15 s (×0.35) | impulse | recovery transient |

Roughly **40% of standing time is released** and most of the remainder is recovery from the
last disturbance. A quiet stand — the thing the probe measures and the thing the hardware
needs — is a minority regime that is never isolated. **This is the mechanism behind the L8
puzzle** (excursion reward improved 2.8× while quiet-stand tilt stayed flat): the reward was
earned during push-recovery, the probe measures quiet standing. They were never looking at the
same robot.

**(d) The stance ratchet is losing on the arithmetic.** Stance 57 cm vs nominal 26 cm. The
penalty is (0.57−0.26−0.03)×4 = 1.1/step when live; with ~40% release and 30% standing envs the
aggregate is 0.09 — matches the measured −0.087 exactly. The policy pays it because wide makes
push survival cheap, and in training a push is always coming. **On hardware no push comes.**
Braced-wide is genuinely optimal in the training regime and wrong on the robot.

## 3. What is fine and stays untouched

- All walking terms: `feet_phase`, `feet_alternation`, `knee_swing`, `flight_phase`,
  `hip_yaw_deviation`, tracking. Walking is the good news of lineage 9.
- `termination_penalty` −150, the plant DR (K_s 20–120, play 0–3°), the push curricula.
- `stand_still` −0.03 and `vel_coeff` 0 — the graveyard markers from the freeze-the-joints
  failures. Do not raise standing motion taxes; that is where bracing comes from.

## 4. Proposed changes

### Change 1 — split standing into QUIET and DISTURBED, and train the quiet half ★ the structural fix
At reset, each standing-command env draws `quiet_stand` with p = 0.5. Quiet envs get **no**
sustained bursts, **no** tilt holds, **no** velocity shoves; every standing term is live at
full weight for the whole episode. Disturbed envs keep the entire current program unchanged.

- The flag is **not observed** by the policy. With 50/50 mixing it cannot tell which regime it
  is in until a push arrives, so it must stand calmly AND stay recoverable. Mixing, not
  separating, is what prevents a brittle statue.
- Push robustness is not reduced: the disturbed half is identical to today.
- This is the one change that makes the training reward and the probe measure the same robot.
- ~30 lines: a per-env bool set in a reset event; `sustained_push_bursts`, the tilt-hold
  state machine and `push_by_setting_velocity_cmd_scaled` skip flagged envs.
- Log `stand_*` terms separately for quiet vs disturbed envs (two extra TB scalars) so the
  drift of §1 is visible in training, not only in a probe.

### Change 2 — give standing a carrot the optimizer can keep earning
New positive term `stand_calm` = exp(−(tilt/3°)²) · exp(−(|ω_xy|/0.3 rad/s)²), standing envs,
weight **+4.0** — the same magnitude as `feet_phase` (3.5) and `knee_swing` (4.0). Softened to
×0.25 during bursts/holds (`_push_or_hold_soften`), never fully released, so it never pays for
a slam. A bounded kernel cannot Goodhart into an explosion the way an unbounded penalty can.
Positive kernels are what worked here before (`stand_pose` was "the target-based fix";
`feet_phase` is the gait's entire backbone).

### Change 3 — modest re-weights, not 20×
Once Changes 1–2 land, the effective weight of the tilt terms already rises (no release in
quiet envs). Then: `stand_upright` −80 → **−120**; `stand_tilt_excursion` −0.2 → **−0.5**
(keep as the tail guard); `stand_stance_geometry` −4 → **−6**, nominal 0.26 unchanged. Not
more. The graveyard says freezing and bracing appear when standing taxes get large.

### Change 5 — price the splayed WALK (added 2026-09-22 after the user's observation)
Measured with `walk_width_probe.py` at 0.3 m/s, K_s = 52, body-frame lateral foot separation:

| | anatomical (default pose) | stand | **walk 0.3 m/s** | hip roll while walking |
|---|---|---|---|---|
| now (381800) | 28.2 cm | 44.9 cm | **71.1 cm (2.5×)** | **19.9°** |
| anchor (176800) | 28.2 cm | 32.0 cm | **63.8 cm (2.3×)** | **16.4°** |

The walk is two and a half hip-widths wide with 16–20° of hip abduction, and it is **not the
recent drift** — the gate-passing anchor walks nearly as wide. It has never been priced:
`feet_phase` scores foot HEIGHT, `feet_alternation` fore-aft swap, `knee_swing` knee angle;
`stand_stance_geometry` is stand-gated. The only thing opposing abduction is
`joint_deviation_hip` (L1, −0.5 on yaw+roll): 20° of roll costs 0.17/hip/step against ~9 units
of gait carrots. Wide is free and wide is stable under pushes, so wide it is. Changes 1–4 do
nothing for this.

**Lever (cause-priced, per the campaign law):** `walk_hip_abduction` = Σ_hips
max(|hip_roll| − 3°, 0), walking envs only, released during bursts/holds like the other
motion terms, weight **−2.0** → ~1.2/step at today's 20°, ~0.36/step at 8°, zero inside 3°.
One lever only, so attribution stays clean; if hip roll narrows but width does not (policy
re-widens via yaw/knee), add the outcome term (body-frame width, nominal 0.28 ± 0.06) as a
second step. **Ramped**, not switched on: weight 0 → −2 over the first ~10k iters on the
completed-episode thermostat, because a narrow walk on a compliant ankle has less lateral
margin and may not be learnable cold.

**Walk gate (pre-registered):** width at 0.3 m/s ≤ 40 cm (≈1.4× anatomical) AND survival and
`feet_phase` not below lineage 9. If width comes down and falls go up, the plant needs the
width and the lever is falsified — stop there rather than push the weight.

### Change 4 — remove dead terms (zero behavioural effect, clarity only)
`stand_hip_roll_brace` (weight 0.0), `stand_height_slope` (−0.0001 measured),
`dof_torques_l2` (−1.5e-7, an "isolation test" leftover), `lin_vel_z_l2` (0.0000).

## 5. From scratch, or warm from the 176800 anchor?

**Recommendation: from scratch (lineage 10).** Change 1 alters what "standing" means during
training, and the two entrenched habits — the 57 cm stance and recover-instead-of-stand — are
exactly its targets. The lineage-8 law: a term that fights an entrenched habit belongs at
birth. `l9_gatepass_model_176800` is archived and stays the deploy candidate regardless, so
nothing is at risk.

Cost: ~150k iters (~7 days) to lineage-9 quality. **Requirement:** a newborn topples on a soft
ankle, so the K_s draw must start stiff and widen. Re-introduce the stability ramp as a
*band* ramp: draw log-uniform (60, 120) at birth, lower bound anneals 60 → 20 on the same
completed-episode thermostat every other ramp uses. ~15 lines reusing
`series_stiffness_curriculum`'s structure. Probes still null the draw, so evaluation is
unaffected.

**Option B (faster, less clean):** warm-start `l9_gatepass_model_176800` with the same reward
package. The stability law says resume + genuine objective change is the stable case (3/3).
Reads out in ~20–30k iters. Downside: starts with the 57 cm stance already learned, so it tests
the economy against the habit rather than without it.

## 6. Pre-registered gate for lineage 10

Nominal probe (`rig_asks_3b.py --nominal --series_k {52,120}`, 128 envs, 45 s):
1. worst-tilt **median < 5° at both 52 and 120**;
2. **and it holds for ≥ 50k iterations after first crossing** — lineage 9 passed once and
   drifted, so a single pass no longer counts;
3. stance width ≤ 32 cm; 20 N push survival ≥ 97%; walking not below L9 (`feet_phase` ≥ 1.5).

If the median converges above 5° again with `stand_calm` saturated, the carrot is falsified
and the remaining lever is the regime mix (raise the quiet fraction), not the weights.

## 7. What this does NOT address

- Walking speed-tracking plateau (vel error ~0.37, flat all lineage). Separate problem;
  untouched here so the standing result stays attributable.
- The rig's load-dependent play band (opens in swing, closes in stance). Still queued.
- Left/right asymmetry watch item.
