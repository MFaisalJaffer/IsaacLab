# AMP for K-Bot legs — plan (lineage 12 candidate)

**2026-09-29** · Written while lineage 11 trains (do not disturb it). Two questions:
(1) how to put Adversarial Motion Priors into *our* stack, (2) how to decide which
gait motion to hand it. Everything below is grounded in the live config dumped from
the L11 service environment and in Menlo's `isaac_asimov` implementation (BSD-3).

---

## 0. Recommendation in five lines

1. **Replace the gait-shape carrots with one learned style reward; keep task, safety and
   the whole standing package.** Today the four hand-made gait carrots (`knee_swing`,
   `feet_phase`, `feet_alternation`, `feet_air_time`) pay **2.6 units vs 1.5 for velocity
   tracking — 63 % of the positive return is us telling the robot what a step looks like.**
   That is the part AMP replaces; nothing else in the L10/L11 package needs to move.
2. **Wire AMP the way rsl_rl already wires RND** (extra observation group → per-step bonus in
   `process_env_step` → auxiliary net trained in `update()`), porting Menlo's three files.
   One snag: our rsl_rl 2.3.3 runner hard-checks `class_name == "PPO"`, so it needs a small
   runner subclass and its own launch script, all in-repo (survives pip reinstalls).
3. **First reference motion = our own best historical walker, recorded and lightly edited**
   (Candidate A). It is feasible by construction, covers our exact command range and gait
   clock, needs no retargeting, and it is literally the question you asked: *can the good
   pre-ankle-flex gait be carried to the new plant?* A designed gait (Candidate C) is the
   second entry; raw human clips are a shape donor, not a reference (§4.3 — Froude).
4. **Decide the gait by a battery, not by taste alone:** kinematic checks (seconds) →
   pinned-base tracking under our actuator model with delays and K_s (minutes) → a short
   AMP pilot per surviving candidate (hours), judged on the existing gates plus filmstrips.
5. **GPU timing:** everything up to the pilots is offline or sub-minute sim. Pilots start
   at L11's first gate probe (~45 k iterations, ≈ 2 days at the current 4.1 s/iter), so
   L11 is never slowed or restarted.

---

## 1. What AMP replaces — the reward economy as it stands

Live term table (KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0), per-term episode contribution at
L11 iteration ~2.2 k, 10-iteration mean, TB units (episode-sum / 20 s):

| bucket | terms | share of positive return |
|---|---|---|
| **gait shape (hand-made)** | knee_swing +1.88, feet_phase +0.39, feet_alternation +0.33, feet_air_time +0.01 | **2.6 → 63 %** |
| task | track_lin_vel_xy +0.85, track_ang_vel_z +0.67 | 1.5 → 32 % |
| standing carrots | stand_pose +0.59, stand_calm +0.09 | (other regime) |
| push shaping | push_brace, push_step | ~0 |

Three of the five walking curricula/thermostats exist to keep those carrots honest
(walk-splay ramp, gait clock freq-map, knee=clearance). Every one of them was a Goodhart fix
for the previous one (see `isaac-legs-gait-rewards` and `isaac-legs-skating` in memory).
AMP's premise is that a discriminator trained on "what a good step looks like" gives the
same guidance in one term, without us encoding each feature of the step by hand.

What AMP does **not** replace: velocity tracking (task), termination, safety walls
(`walk_upright_wall`, `dof_pos_limits`, `tv_headroom`, `foot_impact`), skating (`feet_slide`),
regularisers (`action_rate`, `dof_acc`), and the **entire standing package + push curricula**,
which live in a separate command regime that the style reward never sees (command-gated).

### 1.1 Term-by-term decision for the AMP variant

| term | weight | decision | why |
|---|---|---|---|
| knee_swing | +4.0 | **retire** | shape; the reference has knee flexion in it |
| feet_phase | +3.5 | **retire** (keep as *diagnostic* log only) | shape; phase-locked to our clock |
| feet_alternation | +2.0 | retire in variant P, keep at +1.0 in variant H | alternation is in the reference; H keeps it as a metronome anchor |
| feet_air_time | +1.0 | retire | shape |
| flight_phase | −2.5 | retire | protects feet_phase from hopping; moot without it |
| walk_hip_abduction + its ramp | 0 → −2 | **retire** | splay is a *style* property — the reference has hip roll within ±5°; this is the cleanest test that AMP does the job |
| hip_yaw_deviation | −2.0 | keep at −0.5 | yaw wander is a heading/safety issue, not only style |
| joint_deviation_hip / ankles / hip_pitch_knee | −0.5/−0.5/−0.02 | retire while walking (they already release on push) | posture priors conflict with the reference posture |
| track_lin_vel_xy_exp / track_ang_vel_z_exp / ang_vel_error_l1 | +2/+2/−1 | **keep** | task |
| walk_upright_wall, flat_orientation_l2, ang_vel_xy_l2 | −4/−3/−0.05 | keep | safety |
| tv_headroom, foot_impact_penalty, dof_pos_limits | −2.5/−0.0015/−1 | keep | hardware envelope |
| feet_slide | −0.1 | keep (consider −0.3) | the disc sees joint kinematics only; skating is invisible to it |
| action_rate_l2, dof_acc_l2 | −0.05/−1.25e-7 | keep | smoothness (jerk probe gates) |
| termination_penalty | −150 | keep | |
| every `stand_*`, `push_*`, `stand_calm` | as is | **keep untouched** | other regime; command-gated off from style |
| `gait_phase` observation | — | keep for pilot 0/1 | deployment interface unchanged; the reference cadence is generated from the same freq-map so the clock stays truthful. Dropping it is a later, rig-coordinated change |

Variant **P** (pure) = table as written. Variant **H** (hybrid) = P + `feet_alternation` +1.0 +
`knee_swing` at +1.0. P runs first; H is the fallback if P's gait degenerates (§7).

---

## 2. Architecture — how it plugs into our stack

### 2.1 What Menlo did (confirmed from `menloresearch/isaac_asimov`, BSD-3)

- `AMPPPO(PPO)` subclass; observation group **`amp` = `joint_pos_rel` + `joint_vel_rel`**,
  no noise, concatenated. Discriminator input = **[s_t, s_t+1]** (2 × obs dim).
- Discriminator MLP **[256, 256] ReLU**, running-mean/var feature normaliser (clip ±10),
  **LS-GAN**: `mse(D(expert), +1) + mse(D(policy), −1)`, **gradient penalty λ = 10** on expert
  samples, **replay buffer 100 k** policy transitions, disc updated every PPO update.
- Style reward **`r_s = clamp(1 − 0.25·(D − 1)², 0)`**, mixed as
  `total = 0.3·(0.3·r_s) + 0.7·r_task` (their `amp_reward_coef=0.3`, `lerp=0.7`), **command
  gate**: style zeroed when ‖twist cmd‖ < 0.1.
- PPO otherwise ordinary (24 steps/env, 5 epochs, 4 minibatches; their LR 1e-3 — we keep our
  3e-4 and the guard patches).
- Motion file `policy_delay_walk_slow.npz` (joint_pos, joint_vel, body pos/quat/vel in world,
  fps). **The name suggests a recorded policy rollout rather than mocap; the README does not
  say.** The npz layout is the one our recorder should write.

### 2.2 Our wiring (all in-repo; nothing in site-packages)

```
source/isaaclab_tasks/.../config/kbot_legs/amp/
    motion_dataset.py     npz loader; consecutive-frame transition sampler; per-clip weights
    discriminator.py      MLP + normaliser + grad-pen + reward map      (port, BSD attribution)
    replay_buffer.py      policy-transition ring buffer                  (port)
    amp_ppo.py            AMPPPO(PPO): style reward in process_env_step, disc step in update()
    runner.py             KbotAmpRunner(OnPolicyRunner): accepts class_name "AMPPPO",
                          sizes the disc from extras["observations"]["amp"], saves/loads
                          disc + normaliser state, logs Train/mean_amp_reward,
                          Train/mean_task_reward, Train/disc_acc_expert, Train/disc_acc_policy
scripts/reinforcement_learning/rsl_rl/train_amp.py     train.py with the runner swapped
rough_env_cfg.py   `if os.environ.get("KBOT_AMP") == "1":` block: observations.amp group,
                   the §1.1 retirements, amp_* params
agents/rsl_rl_ppo_cfg.py   KbotLegsAmpRunnerCfg (algorithm.class_name = "AMPPPO")
eval_watch/amp_record_reference.py   rollouts of a checkpoint -> npz (Candidate A)
eval_watch/amp_gen_gait.py           IK gait generator -> npz (Candidate C)
eval_watch/amp_ref_battery.py        §5 stages 1-2
```

Design choices that differ from Menlo, and why:

| choice | Menlo | ours | reason |
|---|---|---|---|
| mixing | lerp (scales task by 0.7) | **additive**: `r = r_task + w_s·gate·r_s` | keeps the L10 economy (stand/push weights) untouched; one knob |
| `w_s` | 0.09 effective | set so style ≈ 40–50 % of the *walking* positive return at pilot start, then hold | that is the budget the four retired carrots had (63 %); logged as `Train/style_share` |
| gate | ‖cmd‖ < 0.1 | same, **and** `~env._quiet_stand` | our quiet/disturbed standers must never see style |
| disc features | joint pos + vel | pilot 0: same (joint-side, 20 + 20); pilot 1 option: + base ang-vel + projected gravity | start minimal; add torso posture only if walk tilt needs it |
| guards | — | style added *after* the runner's reward clamp, bounded in [0, w_s]; disc loss goes through the same non-finite mini-batch skip | keep the containment that stopped the value-loss blow-ups |
| reset handling | — | transition (s_t, s_t+1) masked when `dones` (post-reset obs is a new episode) | avoids teaching the disc that resets are policy style |

Deployment interface: **unchanged.** Same observation vector, same actions, same
`legs_policy_meta.json`. AMP is train-time only; nothing goes to the rig.

### 2.3 Unit tests before any pilot (no GPU contention)

1. Discriminator alone: train on reference vs time-scrambled reference → accuracy → ~1.0;
   `r_s(reference transitions)` → ~1.0; `r_s(scrambled)` → ~0. Gradient-penalty finite.
2. 20-iteration smoke on 64 envs with `KBOT_AMP=1`: losses finite, `Train/mean_amp_reward`
   in (0, 1), disc accuracies logged, checkpoint saves and reloads the disc.
3. Gate check: style reward is exactly 0 on every quiet-stand env and every ‖cmd‖ < 0.1 env
   over 500 steps.
4. Rollout-storage check: the masked fraction of transitions equals the done rate.

---

## 3. Reference motion — the candidates

The discriminator only ever sees **joint positions and velocities of the ten leg joints
across one 20 ms policy step**. So a "reference motion" is a 50 Hz table of 10 angles and 10
rates, nothing more; foot paths, width and clearance are implied through the kinematic chain.
That makes three sources practical:

### A · Our own best walker, recorded (self-reference) — *first*
Play a historical build on flat ground with DR nulled, sweep the current command distribution
(0.12–0.5 m/s, ±0.1 lateral, ±0.3 rad/s yaw, 30 % standers excluded), keep only clean
episodes (no fall, tracking error < 20 %, no skating), write npz.
Which build: the one you consider "stand and walk were good" — `model_210000` (build of record,
pre-ankle-flex, de-chattered) is the default; `l10_candidate_model_100000` (current plant,
splay-taxed, 35 cm width) is the *control* reference for §5 pilot 0.
- **Pros:** feasible by construction on this robot (torque, T-V, limits, our foot); covers our
  command range *and* lateral/yaw so the disc never punishes a commanded turn; matches the
  gait clock; zero retargeting; also the honest answer to "carry the old gait to the new plant".
- **Cons:** inherits that build's flaws. Mitigation that costs nothing: **edit the clip.**
  Because the disc only sees joint kinematics, a recorded clip can be post-processed —
  subtract a hip-roll offset to bring width from 35 → 28 cm, scale knee flexion ×1.2, shorten
  the double-support fraction — and re-run the §5 battery on the edited clip. Editing the
  prior is a far cheaper style lever than editing a reward.

### C · Designed gait from foot trajectories (IK generator) — *second*
Choose base-frame foot paths (step length = v / f from the freq-map, apex clearance 3–4 cm,
flat-foot landing, lateral separation 28 cm, hip roll for weight shift ≤ 5°, hip yaw 0) and
solve the sagittal 3-joint chain (hip pitch, knee, ankle pitch with foot-flat constraint —
closed form) per leg; left = right shifted by half a cycle. Generate 5 clips across 0.12–0.5
m/s. ~150 lines.
- **Pros:** every style property is an explicit, inspectable parameter; cadence is the clock's
  by construction; no historical baggage.
- **Cons:** it is a *kinematic* proposal — nothing guarantees the robot can balance it. That is
  what §5 stage 2b (tracking pilot) is for. No lateral/turning clips (accept: ranges are small).

### B · Human gait profiles (clinical datasets / mocap) — *shape donor only*
Public per-speed joint-angle curves (hip flex/abd/rot, knee, ankle; e.g. Fukuchi 2018,
Bovi 2011) map 1:1 onto our five joints per leg, so retargeting is a sign/offset table, not IK.
But **Froude scaling is against us.** Hip height: robot ≈ 0.72 m (URDF: hip-pitch axis 0.28 m
under the base, thigh 0.36, shank 0.29), human ≈ 0.92 m. Fr = v²/(g·L):

| our command | Fr | human-equivalent speed |
|---|---|---|
| 0.12 m/s | 0.002 | 0.13 m/s |
| 0.30 m/s | 0.013 | 0.34 m/s |
| 0.50 m/s | 0.035 | 0.56 m/s |
| (comfortable human 1.3 m/s, Fr 0.19) | | → **1.16 m/s on our robot** |

Our whole command range sits below the speeds gait datasets normally sample, and slowing a
comfortable-speed clip 2–3× changes the regime (long single support at low speed — the
thing a 20 ms, compliant-ankle robot is worst at). So: use human curves to *inform* C's
parameters (knee-flexion timing, ankle profile, hip-roll amplitude) and as a visual
comparator; do not feed them to the discriminator directly. Real mocap retargeting (GMR /
AMASS) only becomes worth it if we later want turning and side-stepping style from data.

### Cadence — the one place the reference and the policy input must agree
The policy observes a gait clock (`_FREQ_MAP = 0.9 Hz @ 0.15 m/s → 1.4 Hz @ 0.45 m/s`,
i.e. step length 8 → 16 cm). A Froude-scaled human at 0.5 m/s would take ~0.7 Hz strides of
~29 cm — half our cadence, twice our step. Under hand-shaped rewards, lowering the clock at
fixed speed always failed (memory: falsified lever). Under AMP it might not, but that is a
*second* experiment: **pilot references use the clock's cadence.** A slow-cadence clip
("C-slow": f × 0.7, stride × 1.4) is queued as experiment 3, with the clock obs adjusted to
match — never a clip that disagrees with the clock the policy sees.

---

## 4. Deciding the gait — what "best" means here, in order

1. **Feasible on this plant under our DR** — delays 10–25 ms, K_s 20–120 with 3° play,
   masses ±20 %, our T-V envelope. Non-negotiable; §5 stages 1–2b test it.
2. **Inside the hardware envelope** — RMS torque per joint against the continuous ratings
   (7.5 Nm `_04`, 5.0 Nm `_02/_03`; walking already runs 1.25–1.62× over), ankle series
   deflection at K_s 52, flat-foot landing (no toe joint), stance width near the anatomical
   28 cm (gate ≤ 32 standing, ≤ 40 walking).
3. **Consistent with the command range and the clock** — implied speed from FK stance-foot
   velocity must match the commanded speed the clip is labelled with (±15 %).
4. **The style we want** — knee flexion in swing (30–60°), hip roll ≤ ±5°, apex clearance
   3–5 cm, L/R symmetric, no double-support shuffle, and it *looks* right on the :8800
   filmstrip. Only this last item is taste, and it only ranks candidates that passed 1–3.

---

## 5. The battery — how a candidate earns a pilot

### Stage 1 · kinematic (no sim, seconds) — `amp_ref_battery.py --stage 1`
FK of the clip through the URDF chain. Pass thresholds:

| check | threshold |
|---|---|
| joint-limit margin, every joint | ≥ 5° (note the ankle has only 13° one way — heel-strike-style dorsiflexion can hit it) |
| peak joint speed | ≤ 60 % of the motor's no-load speed |
| implied speed (stance-foot FK velocity) vs label | ±15 %; both feet stationary in double support |
| swing apex clearance | 2.5–6 cm; sole never below ground |
| foot pitch at touchdown | ≤ 5° |
| lateral foot separation | 26–32 cm |
| hip roll / hip yaw range | ≤ ±6° / ≤ ±8° |
| L/R mirror symmetry (half-cycle shift) | RMS ≤ 2° |
| cadence vs clock | ±15 % (or the declared alt cadence) |

### Stage 2 · pinned-base tracking under our actuator model (16 envs, ~1 min) — `--stage 2`
`fix_root_link=True`, raised spawn (the pinning method that survived sim2sim). Track the clip
with the live PD, delays sampled 2–5 steps, K_s ∈ {52, 120}, play 3°, Coulomb/viscous on.
Measures what a *pinned* robot can tell us: tracking RMSE ≤ 4° on hips/knee, phase lag
≤ 25 ms, swing-leg RMS torque ≤ 60 % continuous, ankle series deflection ≤ 3° at K_s 52.
(Stance torques need body weight — that is stage 2b/3.)

### Stage 2b · tracking pilot (GPU, ~2 h) — only for C and edited-A clips
A DeepMimic-style run rewarded purely for tracking the clip + staying upright + the velocity
term, 2 k iterations. If PPO cannot learn to walk the clip while balancing, the clip is
infeasible on this plant — reject before spending AMP time. If it can, **the tracked rollouts
become the AMP dataset** (a physically consistent version of the clip). This is very likely
what Menlo's `policy_*` file is.

### Stage 3 · AMP pilot bake-off (GPU, ~7 h per candidate at 4096 envs)
Identical config and seed, 6 k iterations, variant P. Scored on:
- the existing gates at matched iteration vs L11: walk width ≤ 40 cm, world-frame drift
  (`skate_diag`), stance sway (must be unchanged — standing is untouched), 20 N push;
- AMP health: disc accuracy on expert/policy sits 0.6–0.85 (saturation at 1.0 = no
  gradient), `Train/mean_amp_reward` rising then plateauing, `style_share` near its budget;
- hardware envelope: RMS torque / continuous per joint, `tv_headroom`, jerk probe;
- filmstrip review (yours).
Eligibility is the gates; the choice between eligible candidates is the filmstrip.

### Pilot 0 — machinery validation, before any style argument
Reference = rollouts of `l10_candidate_model_100000` (a policy whose gait metrics we know).
AMP-trained policy must reproduce that policy's own width, cadence and feet_phase within
tolerance, on the *new* plant. Pass = the port works and the style signal transfers; fail =
fix the machinery before arguing about gaits. 3 k iterations is enough.

Decision tree after pilot 0:
```
A (historical best) passes stages 1-2 and pilot P? --yes--> like it on film? --yes--> reference = A (+ edits)
        |                                                          |
        no                                                         no --> edit A (width/knee) -> re-battery -> re-pilot
        v
C (designed) passes stage 2b tracking pilot? --yes--> AMP pilot on tracked rollouts
        | no
        v
C parameters wrong for this plant: shorten stride / raise cadence / lower clearance, retry;
B curves only as a shape reference for those parameters.
```

---

## 6. Schedule and GPU

| phase | work | GPU | when |
|---|---|---|---|
| 0 | port AMP modules, runner, `train_amp.py`, `KBOT_AMP` cfg block, unit tests §2.3 | none / 2-min smoke | now |
| 1 | recorder (A), generator (C), battery stages 1–2; run them on A, edited-A, C | seconds–minutes | now |
| 2 | pilot 0 (machinery), then P on A | ~3 h + ~7 h | **after L11's 45 k gate probe** (≈ 2 days) — L11 stays untouched; if you'd rather not wait, a 2048-env pilot can share the card at roughly −35 % L11 speed |
| 3 | stage 2b + AMP pilot on C; C-slow (cadence experiment) | ~2 h + ~7 h each | after phase 2 verdicts |
| 4 | lineage 12 from scratch on the chosen reference, full DR, full gates, rig export | full | after phase 3 |

Phase 0–1 is 2–3 working days of implementation and lands before L11 reaches its probe, so
the pilots can start at that natural boundary. Lineage 12 is a **from-scratch** birth: adding
an adversarial reward to a converged policy is the same category of mistake as the
plain-resume explosion (memory: resume-at-optimum law).

---

## 7. Risks and what we do about each

| risk | signal | mitigation |
|---|---|---|
| disc wins outright → style ≈ 0, no gradient | expert/policy accuracy → 1.0 within a few hundred iterations | LS-GAN already bounded; lower disc LR (1e-4), GP λ 10 → 20, update disc every 2nd iteration, replay buffer keeps old policy samples |
| policy fools the disc with a frozen mean pose / walking in place | style high, tracking low, drift high | command gate + task weight unchanged + `feet_slide`; variant H if it persists |
| style vs task at lateral/yaw commands (clip has none) | tracking drops only when ‖cmd_y‖ or yaw large | Candidate A covers them by recording the full command distribution; for C, halve lateral/yaw ranges in the pilot |
| reference infeasible on this plant | stage 2/2b failure | battery before any pilot; edit clip |
| standing regressions | quiet/disturbed sway, push survival move at all | they must not — style is gated off there; the regime_report and push probes are the tripwire |
| reward scale drift → value blow-up | guard prints skipped batches | style bounded in [0, w_s]; same guards |
| gait clock obs disagrees with the reference cadence | policy ignores clock, feet_phase diagnostic near 0 | pilot references are generated *from* the clock's freq-map; C-slow changes both together |
| thermal — style may prefer long strides | RMS torque / continuous rises above L11's | gate; edit clip (shorter stride) |

---

## 8. Open decisions for you

1. **Which build is "the good one" for Candidate A?** Default `model_210000` (build of
   record, pre-ankle-flex). If a different checkpoint is the one you remember as good, name it.
2. **GPU:** wait for L11's 45 k probe (recommended) or share the card now at ~−35 % L11 speed.
3. **Gait clock:** keep it in the observation for the pilots (recommended, interface-safe) or
   drop it now and tell the rig.

Everything else has a default in this document and can start today without touching L11.

*— training, 2026-09-29*

---

## Status log

**2026-09-29 (evening) — Phase 0/1 done; Candidate D (Menlo clip) chosen and proven; machinery built and tested.**

- Reference: Menlo's `policy_delay_walk_slow.npz` sign-mapped onto our joints from geometry
  (`amp_replay_clip.py`), cut into a symmetric 1.06 s / 0.41 m/s stride (`amp_build_ref.py`),
  passed the pinned-base actuator test (`amp_track_pinned.py`), then **walked by a tracking
  policy** on our plant under full DR (`Isaac-Track-KbotLegs-v0`, RSI + self-paced kernel +
  command-tracking gate): 99.2% survival / 20 s, 2.1° rms, torques ≤ 1.08× continuous.
  Its rollouts are the dataset: `eval_watch/amp_refs/asimov_tracked_kbot.npz` (127 × 20 s).
- Machinery (`config/kbot_legs/amp/`, `mdp_amp.py`, `amp_env_cfg.py`,
  `agents: KBotLegsAmpPPORunnerCfg`, `scripts/.../train_amp.py`, task
  `Isaac-Velocity-Rough-KbotLegs-AMP-v0`): 24/24 unit tests (`eval_watch/amp_unit_tests.py`),
  Isaac smoke on 64/256 envs incl. layout guard, discriminator training and checkpoint
  round-trip with the discriminator. Style reward additive, gated on commanded motion,
  masked on done steps; expert data mirror-augmented; TensorBoard section `AMP/`.
- Launch (pilot 0 = machinery validation on the single-speed dataset, command band narrowed):
  `KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 KBOT_AMP_VX=0.30:0.50 ./isaaclab.sh -p
  scripts/reinforcement_learning/rsl_rl/train_amp.py --task=Isaac-Velocity-Rough-KbotLegs-AMP-v0
  --headless --num_envs 4096 --max_iterations 6000`
  (`KBOT_AMP_STYLE_W`, `KBOT_AMP_MOTION_FILES`, `KBOT_AMP_CLOCK` override the defaults).
  Pass = walking envs reproduce the dataset's gait (width ~30 cm, stride 1.06 s, clearance
  ~11 cm, knee 6–70°) while standing metrics stay at L11 levels; disc accuracy 0.6–0.85.
- Known limits before a full lineage: single-speed dataset (0.41 m/s) — record the tracking
  policy at more speeds or add cycles; the gait clock is fixed at 0.943 Hz.

### 2026-10-01 13:25 — clip tracker v3: give the policy the reference (dataset expansion)
- v1 (tight gate 0.1/0.05, 12:46) and v2 (loose gate 0.3/0.25, 13:02) both sat at rmse ≈ 19° on
  every label for 150+ iterations: the policy's only clip-specific input was the 4-vector progress
  phase, so it could not know WHICH of the 98 clips (backward / side / pivot / stop…) it was in — the
  pose carrot was unlearnable, not just gated. Both runs set aside in kbot_legs_trackclip_ABANDONED/.
- Fix (one change): `clip_ref` observation = upcoming reference joints at +1 and +10 frames minus the
  current joints (20-D), appended to BOTH the policy and the critic groups (mdp_clip.clip_ref_obs,
  clip_env_cfg). Recordings stay physics-only (joint_pos/joint_vel), so the AMP stage is untouched.
- Warm start: archive_anchors/amp_track_asimov_model_2598_clipref.pt = the straight tracker with
  the first actor/critic layers zero-padded 43→63 / 288→308 (behaves exactly like the old policy until
  it learns the new inputs) + a FRESH Adam state (eval_watch/amp_clip_prep_warm.py). Mirror loss and
  mirror data-aug are OFF for this runner (symmetry.py does not know the new block; the library already
  holds mirrored clips).
- Launch: 8192 envs, 3000 iters, gate 0.3/0.25, KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0; log
  logs/track_pilot/trackclip_v3_refobs_*.log. Dashboard: watcher/server/page now know a `clip` mode
  (auto-detects the TrackClip task). Lineage 11 stays paused.
- Read-out: Curriculum/clip_report/rmse_deg (was flat 19.2) must fall; per-label rmse_<label>_deg;
  Episode_Reward/track_clip_pose (was 0.04). If STILL flat by ~300: next suspect is the RSI start
  (arbitrary frames of unfamiliar motion) → bias t0 toward clip starts.

### 2026-10-01 13:35 — v3 also flat → the real blocker is the policy's action noise (std 0.035)
- v3 (clip_ref obs, 160 iters): rmse 22.4 flat, base_height falls up; value loss fell faster than v2
  (the critic uses the reference) but the actor could not move. Set aside (refobs_std035_*).
- TensorBoard calibration (Policy/mean_noise_std, Loss/learning_rate):
  * stride tracker 09-29_20-53 (learned, rmse 18.7→2.0): std 1.0 → 0.53 (50) → 0.12 (500) → 0.04; lr up to 6.7e-4.
  * AMP pilots (all learned): std 0.998 at step 0, lr climbing to 1e-3.
  * every warm-started tracker v2/v2b/v2c (rmse flat 2.9) and clip v1/v2/v3: std 0.035→0.016, lr pinned at
    the 3e-5 floor (kbot-patched adaptive schedule: lr ∈ [3e-5, 1e-3]; KL ∝ Δμ²/σ² so a tiny σ makes every
    step exceed desired_kl and the policy is frozen).
- LAW: a warm start onto a NEW task must reset log_std to 0 (std 1.0). Weights carry the skill; the noise
  must be re-inflated or the KL schedule locks the policy. Checkpoint: amp_track_asimov_model_2598_clipref_std1.pt.
- v4 = v3 + std reset (same gate 0.3/0.25, 8192 envs, 3000 iters).

### 2026-10-01 13:40 — v4 (std reset) survives like the stride tracker but still does not track → the GATE
- v4 vs the stride tracker (09-29_20-53, trained FROM SCRATCH at std 1.0, no checkpoint) at the same iteration:
  eplen 47 vs 46 (same survival learning), but rmse 22-23° flat vs 11° falling, pose reward 0.007 vs 0.105.
- The stride tracker learned its pose tracking UNGATED (the command gate was only added in the 22:29
  continuation at iter 2000+ to enforce obedience). Every clip run (v1-v4) had the gate on from step 0:
  with a noisy, not-yet-tracking policy the velocity-error gate multiplies the pose carrot by ~0.1-0.3, so
  the policy learns survival only and never gets the tracking signal (chicken-and-egg).
- In the clip env the command IS the clip's own body motion, so obedience is built into the reference; the
  gate is unnecessary. v5 = v4 with KBOT_CLIP_GATE=0 (pose carrot ungated; velocity-tracking task terms x2 remain).
- Clip library check: |qd| max 3.6-7.4 rad/s, qd rms 0.4-1.1 (stride cycle 6.4 / 1.5) → RSI velocities are
  not the problem; one stop_start clip dips to base_z 0.54 (< 0.55 fall threshold) — negligible.

### 2026-10-01 14:30 — v5 tracks the joints (7°) but the clip COMMANDS do not match our robot
- model_600 per-label eval (amp_clip_eval_v5_600.json): survival 78-99% to clip end, rmse 5-8.6°, but
  achieved body speed = 30-40% of the command (backward -0.08 of -0.27, side 0.04 of 0.21, yaw err 0.5-0.8).
- Kinematic probe (amp_clips_kinematic_speed.py: stance-foot motion with the base fixed = the speed a
  perfect tracker would have): forward/backward labels are consistent (ratio 0.85-1.1), but LATERAL is
  0.2 of the label and YAW ~0.5. Cause: amp_lafan_edit.py scales hip roll x0.3 for every clip (sway
  removal) — for side-steps hip roll IS the motion. The tracker's ±0.04 m/s side-step equals the kinematic
  value: it is faithful; the labels are wrong. Backward clips also carry weaving yaw (rms 0.25-0.9).
- Fix (library v2 → clips_v2): lateral-aware roll scale (0.3 at |vy|≈0 → 1.0 when |vy|≥0.15), commands
  relabelled by each clip's kinematic ratio, glitch clips (ratio<0.15, e.g. turn_in_place_6 with 5 rad/s
  spikes) dropped. v6 = v5 recipe on clips_v2. v5 anchor: archive_anchors/trackclip_v5_model_800.pt.

### 2026-10-01 14:48 — clip library v2 built (clips_v2, 96 clips) → v6
- Edit v2 (KBOT_EDIT_SIDE_ROLL=1.0): hip-roll scale blends 0.3 → 1.0 with |vy| (side-steps keep their
  abduction; forward walking keeps the sway removal). Kinematic lateral speed of side clips 0.035 → 0.106 m/s.
- Commands relabelled by per-clip kinematic ratios (amp_clips_relabel.py; cmd_orig kept): backward x0.82,
  forward x1.0, side vy x0.61, turn_in_place wz x0.66, backward yaw weave x0.60. turn_in_place_6(+m) dropped
  (yaw ratio 0.04, 5 rad/s spikes).
- v5 stopped at its newest checkpoint (archive_anchors/trackclip_v5_model_*.pt; run dir kept in place).
  v6 = v5 recipe (std-1 stride warm start, clip_ref obs, no gate) with KBOT_CLIP_DIR=.../clips_v2.
- AMP-stage prep for dataset v3 (no launch yet): KBOT_AMP_VY="lo:hi" / KBOT_AMP_WZ="lo:hi" command bands in
  amp_env_cfg; walk_gate speed_fraction now blends a yaw fraction (achieved/commanded yaw rate) in by command
  dominance (0.3 m lever), so turn-in-place commands earn style when the robot actually turns (before: divided by
  a ~0 planar command). Synthetic check: full fwd 1.0 | pivot achieved 1.0 | pivot standing 0.0 | half/half 0.5.

### 2026-10-01 15:30 — route B launched: derived single cycles on the proven stride tracker
- v6 (LAFAN1 library v2) at 600: joints 7-9° but body speed still 20-50% of the (kinematic) commands:
  backward -0.10 of -0.23, side 0.03 of 0.14; rendered = wide-stance shuffle (clip_v6_600_*.gif). Same as
  v5 → the multi-clip tracker's fidelity, not the labels, is the bottleneck (noisy, infeasible human
  references; 98 clips). v6 stopped at 900 (anchor trackclip_v6_lib2_model_800.pt, rmse ~6.9°).
- Derived cycles (amp_design_cycles.py + amp_cycle_ground.py): asimov_walk_cycle_rev.npz = the Asimov
  stride reversed in time (kinematic vx -0.403 m/s); sidestep_left/right_cycle.npz = designed side-step on the
  cycle's double-support posture (roll ±8°, knee 25°, period 1.2 s; kinematic vy 0.075 m/s — pitch-only
  ankles limit lateral stepping by construction). Track env now reads vel_b (vx, vy) from the cycle file.
- Launched in parallel (4096 envs each, 2000 iters, KBOT_TRACK_GATE=0, warm = straight tracker with std reset
  amp_track_asimov_model_2598_std1.pt): logs/track_pilot/track_back_rev_*.log, track_side_left_*.log.
  The right side-step comes from mirror augmentation of the left recording. Turn-in-place cycle: TODO.

### 2026-10-01 15:45 — user-spotted posture bug in the LAFAN1 references: hips behind the feet
- Puppet probe (amp_posture_probe.py, base upright): proven stride keeps the stance foot 0.03 m BEHIND the
  hip; LAFAN1 clips put it 0.05-0.16 m IN FRONT (backward_0 +0.05, side_left_0 +0.06, forward_straight_0
  +0.08, turn_in_place_0 +0.16; stop_start_0 fine). A reference that sits back forces the tracker to
  over-step/shuffle — consistent with the v5/v6 renders.
- Causes (amp_lafan_to_kbot.py): (1) G1's thigh is tilted 9.1° back at its zero pose (knee origin 0.053 m
  behind the hip axis, URDF) while ours hangs vertical → same angles = extra flexion on us; (2) the mocap
  pelvis pitch/roll were dropped (yaw only) → a tilted pelvis became a crouch on our upright base.
- Fix (KBOT_LAFAN_POSTURE=1): hip+knee flexion −9.1° (zero-pose), hip pitch −= pelvis pitch(t), hip roll
  += pelvis roll(t). Old raw conversions kept in lafan1/_raw_v1/. Library v3 to be rebuilt after the probe
  confirms the feet are back under the hips. Route-B cycles (reversed stride, designed side-step) were
  already upright (stance foot −0.03 / −0.07 m).
- Route B at iter 100 (15:39): backward rmse 21.6→10.6, side-left 16.9→11.8 — on the stride tracker's pace.
- 16:06 — library v3 verified: stance foot now +0.01/−0.00/+0.04/+0.05 m (backward/side/forward/pivot) vs
  +0.05..+0.16 before (stride −0.03). Kinematic side-step 0.15 m/s (was 0.035/0.106), backward −0.23. 95 clips
  (5 pivots dropped for yaw ratio < 0.15). Stick-figure before/after gifs: eval_watch/posture_<clip>.gif
  (amp_stick_gif.py — physics puppet only, runs next to training; the viewport renderer needs ~5 GB and OOMs
  while two trackers hold 10 GB → mesh renders after the B runs stop). Lesson: never leave a replay/render
  process behind (pid-kill only; pgrep patterns inside a chain match the chain itself and deadlock it).

### 2026-10-01 16:36 — balance pass → library v4 (user: "at stop the legs are behind the hips")
- Standing phase of stop_start_0: stance foot −0.11 m (old) / −0.13 m (posture fix) behind the hip — the
  source G1 stands hips-extended 7° with the pelvis tilted back; no constant angle fits standing AND walking.
- amp_clips_balance.py: per frame, pivot both legs about the hips so the (1 s-smoothed) stance-foot x hits
  the balance target (0 at rest, −0.085 m per m/s forward: the stride's −0.034 at 0.4), ankles re-levelled by
  the same angle (sign found by a sole-pitch test: +1 → 0.16° change). Pivots 2-5° mean, max 20°.
  Result: standing stop_start_0 −0.13 → −0.04 m; all labels within ±0.03 m in both phases.
- clips_v4 = v3 + balance, re-grounded, kinematic-relabelled (95 clips; turn_in_place_5m dropped). Gifs:
  posture4_<clip>.gif. This is the LAFAN1 library for the next clip tracker (v7) once the B runs free the GPU.

### 2026-10-01 16:40 — route B at checkpoint 600 (amp_track_eval, 128 envs, 20 s, DR on)
- BACKWARD (reversed stride): speed −0.32 m/s of −0.40 commanded (80%), survival 97%, rmse 4.2°, stride
  period 1.06 s = reference, yaw tracking err 0.08, torque rms/continuous hip 1.75× knee 1.4× (like the
  forward walk). → the first real backward gait on this robot. Record at ~1000 → backward_tracked_kbot.npz.
- SIDE-STEP (designed cycle): lateral 0.064 m/s abs, survival 99%, but rmse 9.0° with right knee 22.5° off,
  clearance 1.4 cm, stride detector sees a 5 s period → it SHUFFLES sideways instead of stepping: the
  8° roll / 25° knee-lift design is not balanceable with pitch-only ankles. Options: accept the shuffle as
  the lateral reference (feet_slide will fight it in AMP), or redesign gentler (roll 5°, knee 15°, 1.6 s).
- Next GPU slot (after B hits 1000, ~17:05): clip tracker v7 on clips_v4 (posture + balance fixed LAFAN1).
- 16:55 — side-step tracker stopped at its newest checkpoint (archive_anchors/track_side_left_model_700.pt;
  rmse flat 8.9° since 400, shuffle) to free GPU memory for mesh renders; backward run continues to 1000
  (auto eval + record). Mesh renders queued: library v4 side_left_3/0 (lafan4_*), backward + side-step
  trackers in action (track_<tag>_model_N.gif).

### 2026-10-01 17:00 — USER DECISION: drop the single-cycle trackers, go all-in on library v4
- Route B stopped: backward tracker at model_900 (archive_anchors/track_back_rev_model_900.pt, rmse 3.9°,
  −0.32 m/s at 600), side-step at model_700 (shuffle). Run dirs kept in kbot_legs_track/. No recordings made;
  either can be recorded later in ~3 min with amp_track_eval.py --record if wanted.
- Next: clip tracker v7 on clips_v4 (8192 envs, std-1 clipref warm start, gate 0) — launch only after the
  user has reviewed the v4 mesh renders (lafan4_<clip>_{side,rear34}.gif, one clip per label).
- 17:03 — v7 launched on clips_v4 after the user reviewed the mesh renders ("these look great"): 8192 envs,
  3000 iters, std-1 clipref warm start, gate 0, log logs/track_pilot/trackclip_v7_lib4_*.log. Per-label eval
  auto-runs at model_600 (compare with v6: side 0.03 m/s, backward −0.10, joints 7-9°).

### 2026-10-01 17:35 — AUTOPILOT armed (user away 3-4 h): eval_watch/autopilot_v7.sh → AUTOPILOT_STATUS.md
- v7 per-label evals at 1000/1500/2000(/2500/final); body-motion score = mean over labels of survival ×
  primary-speed achievement; early stop if 2000 < 1.05 × 1500; best → archive_anchors/trackclip_v7_best_*.pt.
- Dataset v3 = 256 envs × 60 s of the best v7 (fall windows masked) → amp_refs/lafan1_tracked_v4_kbot.npz.
- AMP stage: warm amp_pilot2_hard_mirroroff, judge v1+v2+v3, vx −0.3..0.5 / vy ±0.15 / wz ±0.5, pushes frozen,
  style 2.0, stance ×5, mirror loss off, std 0.1, 2000 iters → archive_anchors/amp_v3_final.pt; evals at
  0.4:0:0, −0.3:0:0, 0:0.15:0, 0:0:0.5, 0:0:0 + gifs (backward, lateral).
- Hardening (push ramp on, 1500 iters) only if walk survival ≥ 0.8 and backward ≥ 0.6 and before 22:30 →
  amp_v3_hardened.pt. Logs in logs/autopilot/. Lineage 11 stays paused.

### 2026-10-01 18:12 — v7 at 1000: survives, copies joints, barely travels → judge also gets the real clips
- amp_clip_eval_v7_1000: survival 98%, rmse 7.7° (plateau since ~800, std 0.06), but body speed ≈ 1/3 of the
  clip: backward −0.07/−0.25, side 0.02/0.16, forward 0.08/0.20, pivot yaw err 0.42. Posture/balance fixes
  improved survival (stop_start falls 24% → 1%), not travel: a 7-9° error is 1/3-1/2 of the swing amplitude,
  i.e. short, partly sliding steps. Same ceiling as v5/v6 → it is the 94-clip tracker's fidelity, not the data.
- USER GO (18:10): feed library v4 itself to the AMP judge. eval_watch/amp_refs/lafan1_v4_kinematic.npz =
  clips_v4 concatenated as (T,1,J) with done at clip starts, label-balanced by tiling (each label ≈ 300 s,
  120k transitions). Judge mix: v1 127k + v2 239k + v3 (v7 recording, 64 envs × 30 s ≈ 96k) + v4kin ×2 (240k)
  → full-amplitude clips 34%, shuffle 14%. Risk to watch: kinematic data is not physics-filtered, so the judge
  may separate policy from reference too easily (style reward saturating low) → retune style weight / logit_reg.
- Autopilot relaunched as eval_watch/autopilot_v7b.sh (reuses the 1000 eval; otherwise identical chain).

### 2026-10-01 21:35 — AMP v3 interim test (model_1400) vs the starting policy: mostly worse
- straight 0.4: survival 0.55 (was 0.94), 0.34 m/s | backward −0.3: 0.62 (was 0.00), −0.09 m/s, clearance 2.4 cm
  | sideways 0.15: 0.75 (was ~0.93), lateral 0.00 | pivot 0.5: 0.68, yaw 0.06 (was 0.23 of 0.3).
- v7 full run: rmse 9.1 (400) → 7.6 (1000) → 7.1 (2999); noise 0.19 → 0.04, lr at the 3e-5 floor from ~1500:
  the last 2000 iters bought 0.5°. (Autopilot eval points 1500/2500 never fired: save_interval is 200.)
- Read: (1) no exploration (std 0.1 → 0.06) so no new gait is discovered, only bracing; (2) the speed_fraction
  gate pays style only once the robot already moves along the command, so sideways/pivot get no clip signal;
  (3) wide commands + new judge mix loosened the forward gait. Options given to the user: rerun with noise +
  gate floor + forward-heavy commands; or clean-cycle multi-tracker → warm start (recommended); or backward only.
- Best walker archive_anchors/amp_pilot2_hard_mirroroff.pt untouched. Hardening gate (straight ≥ 0.8) will fail.

### 2026-10-01 21:48 — USER: option 2 → multi-cycle tracker (command-selected clean cycles)
- Clean set, all at the stride's period (1.06 s, 50 samples) so ONE gait clock serves every direction:
  forward (Asimov stride 0.406 m/s), backward (stride reversed, −0.402), side-step L/R (designed: roll 6°,
  knee 45°/hip 22° lift = 6.8 cm clearance, kinematic ±0.066 m/s), pivot L/R (designed: hip yaw ±6° sawtooth
  + tiny fore/aft pitch, kinematic ±0.270 rad/s). Zero-speed cycles stand over the feet (stride posture
  pivoted 4.3°, stance foot −0.02 m). Mirrors stored half a period apart (symmetry.py's phase swap).
  amp_design_cycles.py (fixed a hip-pitch sign bug in the first side-step: it extended instead of flexed),
  amp_cycle_ground.py, library eval_watch/amp_refs/multicycle_v1.npz (weights 0.2/0.2/0.15×4).
- Task Isaac-TrackMulti-KbotLegs-v0 (mdp_trackmulti.py, trackmulti_env_cfg.py; experiment kbot_legs_trackmulti):
  the stride tracker with per-env cycle + command written every step; obs = lineage 43-D (command + clock), so
  the weights warm-start the AMP walker directly. Ungated pose carrot, task weights ×2, symmetry kept.
- Smoke test with the straight tracker's weights: forward 2.7° / 0.42 m/s / 98% (machinery consistent); other
  cycles fail as expected (policy ignores commands). Eval: amp_trackmulti_eval.py (per-cycle survival, rmse,
  achieved vx/vy/wz, --record, --render_seconds, KBOT_TM_ONLY). Dashboard: `multi` mode.
- Launch queued behind the autopilot's official AMP v3 test (auto, when the GPU is free).
- 21:59 user approved the revised side-step (roll 12 deg) and pivot (yaw +-12 deg) cycles: 'these gifs look good'

### 2026-10-01 22:25 — AMP v3 official result + multi-cycle tracker v1a → v1b
- AMP v3 final (amp_v3_final.pt, official): straight 0.52 survive / 0.36 m/s; backward 0.65 / −0.11; sideways 0.78 /
  0.00; pivot 0.69 / 0.05 rad/s; standing 0.74. Hardening gate failed (as intended). Not a base for anything.
- Multi-cycle tracker v1a (22:06, global self-paced sigma) at 200: eplen 436, lin_r 1.03, falls 18/window —
  but per-cycle rmse forward 12.7, side 11.2, pivot 11.5, BACKWARD 17.8 (18.3 at 100): one global sigma
  (12.5°) gives the lagging cycle a carrot of e^-2 that shrinks as the others improve. Fix: per-cycle
  sigma/vel-sigma buffers set by tm_report and read by tm_pose/tm_vel. v1a set aside
  (kbot_legs_trackmulti_ABANDONED/v1a_globalsigma_*, anchor trackmulti_v1a_model_200.pt); v1b launched 22:24.
- Cycle library revised per user review (21:55): side-step roll 12° (kin ±0.129 m/s), pivot yaw ±12°
  (kin ±0.563 rad/s); user: "these gifs look good".

### 2026-10-01 22:51 — multi-cycle tracker v1b → v1c (reward balance)
- v1b (per-cycle sigma) at 200/300/400: per-cycle sigma changed nothing early (backward 17.8/17.6/17.4°, others
  11→9.2°). Checkpoint-400 eval: survival 0.92-1.00 (backward 0.73) but travel forward 0.12/0.41, backward
  −0.14/−0.40, side 0.00/0.13, pivot 0.09 & −0.14 / ±0.56. Same shuffle symptom as v5-v7; the warm-start's own
  forward gait (0.42 m/s at iter 0) was lost.
- Difference from the single-cycle runs that travelled: velocity-tracking weights 4/4 here (KBOT_TM_TASK_W=2,
  copied from the clip tracker) vs 2/2 there, pose weight 6 in both. Episode sums at 440: lin 2.87 + ang 1.12
  vs pose 2.13 — the wide velocity kernel pays ~70% for creeping at 30% speed, and it outweighs the stride
  carrot, so the policy creeps (backward by posture, not by the reversed stride).
- v1c = v1b with KBOT_TM_TASK_W=1.0 (weights 2/2, the proven balance), launched 22:51. v1b set aside
  (kbot_legs_trackmulti_ABANDONED/v1b_taskw2_*, anchor trackmulti_v1b_model_400.pt).

### 2026-10-01 23:30 — multi-cycle tracker v1c result → v1d (signed clock + adaptive draw)
- v1c (velocity weights 2/2) at checkpoint 600: forward 0.83 surv / 8.2° / 0.19 m/s; pivot 0.97 / 8.3° / +0.27 & −0.25
  rad/s (46% of 0.56) — real turning; side 0.91-1.00 / 8.3° / 0.00 m/s; BACKWARD 0.98 / 17.5° / −0.01: it steps in
  place. Backward rmse was 17.4-18.6° in v1a/v1b/v1c at every checkpoint = never learned.
- Root cause: with a forward-running clock the same clock value demands OPPOSITE leg motions for forward and
  backward (q(phi) vs q(−phi)); the command sign had to flip the whole mapping and PPO never found it (also, the
  failing cycle's short episodes starved it of steps under a fixed draw).
- Fixes in v1d (launched 23:29): (1) SIGNED CLOCK — backward = the forward stride table with clock_dir −1
  (library `clock_dir`, mdp_trackmulti._tm_phase / tm_phase_obs, qd sign); smoke test with the old weights:
  backward rmse 18.1 → 12.1° at iter 0, forward unchanged 2.5°. (2) adaptive cycle draw
  p_k ∝ base_k / EMA(episode length_k) × rmse tilt, clamped [0.5 base, 0.6] (KBOT_TM_ADAPT=1), logged as
  draw_p_* / share_*. (3) per-cycle sigma, velocity weights 2/2 kept.
- INTERFACE NOTE for the AMP stage: the main walker's gait clock must also run backward when vx_cmd < 0 (stateful
  signed phase) for these weights to warm-start it; same rule on the robot. Tell the user before that stage.
- Anchors: trackmulti_v1c_model_600.pt (best pivot so far). v1c dir → kbot_legs_trackmulti_ABANDONED/v1c_fwdclock_*.

### 2026-10-02 00:14 — tracker v2: history + task-space rewards + drift termination (user: "apply all 3")
- v1d (signed clock, adaptive draw) at 400 / 600: forward 0.15 → 0.20 m/s, backward −0.12 → −0.17, pivot ±0.20 → ±0.27,
  side 0.00; survival 92-100%; rmse 7.2 / 10.0 (backward) / 6.7; noise 0.10 → stalling. Anchor trackmulti_v1d_model_600.pt.
- Architecture comparison (BeyondMimic, GMT, KungfuBot, ExBody2, OmniH2O, Humanoid-Gym, Unitree G1): our MLP
  512-256-128 is the same; the differences are (1) no memory — everyone uses 5-25 frames of history or an LSTM,
  ours saw one frame and has no body-velocity sensor; (2) joint-angle-only reward — others pay for foot/body
  positions and root velocity; (3) no termination on drifting from the reference; (4) 10-30k iterations with
  entropy 0.005-0.01 vs our 600-3000 at 0.003; (5) one motion per policy unless MoE + teacher-student.
- v2 changes (all env-var controlled, defaults = v2): KBOT_TM_HIST=10 (policy obs 430, term-contiguous, oldest
  first; symmetry.py mirrors per frame — checked equal to the per-frame mirror); track_ref_feet weight 3,
  sigma 0.06 m (feet [R, L] relative to the base in the yaw frame vs `feet` tables from the puppet);
  velocity terms tight: lin weight 4 std 0.2, yaw weight 3 std 0.3 (was 2/2 at std 0.5); termination
  root_drift: > 1.0 m or > 1.0 rad behind the integrated command; entropy 0.005; 10 000 iterations.
- Warm start archive_anchors/trackmulti_v2_warm_from_v1d_600.pt: first layer padded 43 → 430 onto the newest
  frame's columns (action diff vs v1d 3.6e-7), std 0.35, fresh optimizer (eval_watch/amp_tm_prep_warm.py).
- Eval: amp_trackmulti_eval.py now separates falls from drift-ends. v1d evals need KBOT_TM_HIST=1
  KBOT_TM_FEET_W=0 KBOT_TM_DRIFT=0 KBOT_TM_VEL_STD=0 to rebuild the old env.
- INTERFACE: the main walker and the robot must now also (a) stack 10 frames and (b) run the clock backward for
  vx < 0, for these weights to warm-start it.

### 2026-10-02 00:42 — tracker v2 at checkpoint 400: forward/backward/pivot travel (the three changes worked)
- amp_trackmulti_eval (256 envs, 20 s): forward +0.38/+0.41 m/s surv 0.99 rmse 6.8; backward −0.36/−0.40 surv 1.00
  rmse 9.2; pivot +0.44 & −0.46 / ±0.56 rad/s surv 1.00 rmse 7.1; side +0.02 & −0.03 / ±0.13 surv 1.00 rmse 7.2.
  Score 0.630 (v1d@600 ≈ 0.35). Training: noise holds 0.22 (entropy 0.005), drift ends 41 → 16 per window.
- Remaining gap: side-step travel (15-25% of command) — pitch-only ankles; consider a larger share / longer run /
  redesign once the others converge.
- Detached: training (10k iters) + eval_watch/tracker_v2_watch.sh → eval_watch/TRACKER_V2_STATUS.md, best checkpoint
  archive_anchors/trackmulti_v2_best.pt. Gifs eval_watch/tmv2_400_<direction>.gif.

### 2026-10-02 08:50 — tracker v2 overnight result; CORRECTION on AMP v3; walker v4 build (user: "go ahead")
- Tracker v2 ran to ~8900/10000 overnight, no errors. Best = model_8000, score 0.915: forward +0.41/+0.41 (surv 0.99,
  rmse 4.2°), backward −0.40/−0.40 (1.00, 3.2°), side +0.11 & −0.12 / ±0.13 (1.00, 3.8°), pivot +0.50 & −0.49 / ±0.56
  (1.00, 5.4°). Score per checkpoint: 0.63 (400) → 0.76 (800) → 0.82 (1600) → 0.87 (3200) → 0.90 (6000) → 0.915 (8000).
  Noise held at 0.14 (entropy 0.005) — no stall. Gifs eval_watch/tmv2_8000_<dir>.gif.
- CORRECTION: the 2026-10-01 "AMP v3" main-walker stage (autopilot_v7b.sh) was launched WITHOUT
  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 (the script exported them only for the clip evals). Its event list shows
  base_external_force_torque and no randomize_joint_play / draw_quiet_stand / sustained_push — a different plant
  and observation model than its starting policy. Its numbers (straight 94% → 52% etc.) are NOT a valid test of
  "kinematic clips in the judge". amp_env_cfg now raises unless those vars are set (KBOT_AMP_ALLOW_MISMATCH=1
  overrides). The vars are only active when == "1"; every launcher must export them.
- Walker v4 code (all opt-in env vars on the AMP env): KBOT_AMP_HIST=10 (policy obs 430), KBOT_AMP_SIGNED_CLOCK=1
  (mdp_gait._phase_signed: integrated, runs backward while cmd_vx < −0.05; gait_phase_obs(signed=True)),
  KBOT_AMP_AXIS_P=0.6 (mdp_amp.axis_bias: 60% of resampled commands keep one component), KBOT_AMP_VEL_STD /
  _YAW_STD / _VEL_W / _YAW_W (tight velocity kernels), KBOT_AMP_DRIFT (mdp_amp.root_drift termination),
  KBOT_AMP_ENTROPY. Pipeline eval_watch/walker_v4_pipeline.sh → WALKER_V4_STATUS.md: wait for tracker → freeze
  best → record 256 envs × 40 s (multitrack_v2_kbot.npz) → train_amp from the tracker weights (judge = turning
  dataset v2 + the new recording; vx −0.40..0.45, vy ±0.13, wz ±0.56; pushes frozen; 4000 iters) → per-direction
  tests at 400/800/…, best → archive_anchors/walker_v4_best.pt.
- 10-02 09:01 transfer check (tracker v2 @8000 in the walker env, no training):   survival_walking 0.70703125 | fwd 0.409 lat -0.001 | yaw 0.024 | note None;  survival_walking 0.8515625 | fwd -0.402 lat -0.002 | yaw 0.006 | note None;  survival_walking 0.9296875 | fwd 0.006 lat 0.113 | yaw 0.011 | note None;  survival_walking 0.921875 | fwd 0.028 lat -0.039 | yaw 0.512 | note None;
- 10-02 09:30 WALKER v4 DRY RUN before the unattended launch (same settings and code paths as walker_v4_pipeline.sh at small scale: record 64 envs x 12 s -> dataset check -> train_amp.py 512 envs x 30 iterations from tm_v2_9000 -> the pipeline's own test_ck). Launch path OK (430-D input, counter reset to 0, noise 0.25, fresh judge, 2 datasets). Found and fixed two faults in mdp_amp.axis_bias with a new probe (eval_watch/amp_cmd_probe.py, real env, zero actions, forced resample):
  1. dead zone: single-direction commands were uniform in [0, max]; the lateral band is 0.13 wide, so 77% of lateral-only draws were under the 0.1 stand threshold (side-step practice ~4% of envs). Now the kept component is rescaled to [0.1, max] (KBOT_AMP_AXIS_MIN, default 0.1): fwd/back, lateral and yaw each ~20% of the moving envs, none under 0.1.
  2. the mask zeroed the lineage's stand-entry corridor command (0.12, 0, 0) whenever the env's mask was lateral- or yaw-only: 40% of stand entries were abrupt stops. Standing and corridor envs are now never masked (probe: 200/200 corridor envs carry (0.12, 0, 0)), and a re-draw by walk_at_spawn is detected (command differs from the one written last step) and treated as a fresh draw.
  Start-of-training picture from the dry run (512 envs, 20 iterations): episode ~4.5 s, return about -30, dominated by standing terms (stand_gyro, ang_vel_error_l1): the tracker has never seen a zero command or the pinned clock. Expected; it is what this stage has to learn.
  Pipeline relaunched with --run_name walker_v4 (TensorBoard :6007 run kbot_legs_amp/<time>_walker_v4). Dry-run folders are in logs/smoke_runs/ (outside the TensorBoard tree). Note: train_amp.py ignores --experiment_name (cli_args.update_rsl_rl_cfg never applies it).
- 10-02 13:15 WALKER v4 RESULT SO FAR (launched 10:12 from tracker model_9999, at iteration ~2500/4000): survival x speed score 0.69 -> 0.88, standing 99%, all directions move — but the GAIT COLLAPSED INTO SLIDING. Max foot lift (cm) fwd 10.1@400 -> 4.2@2000 (p95 over the walk ~2), back 6.1 -> 2.7, side 0.6 from the first test, pivot 0.9; no strides detected for side/pivot; judge style 0.75 -> 0.5/0.4. The checkpoint score (survival x speed) was blind to it — Goodhart again. Why: (1) the tracker's pose/feet rewards are gone and the judge alone (weight 2, fresh at the start, 97% sure the policy is fake from ~1200) does not hold stepping; (2) the tight INSTANTANEOUS velocity kernels pay smooth sliding more than stepping (a stepping pivot's yaw rate swings 0-1 rad/s: ~1.3/s less than a smooth slide); (3) low steps survive pushes/noise better. Also: action noise creeping up 0.14@400-800 -> 0.21@2300 (entropy 0.005), training reward flat/declining since 2000.
  PREPARED (not launched; user's decision): walker v5 = v4 + mdp_amp.ref_foot_lift (KBOT_AMP_LIFT_W=5, sigma = 0.5 x each cycle's peak lift, no tracking gate): foot height above ground follows the reference cycle's lift at the signed clock's phase; cycle picked by the command's dominant direction. Probe eval_watch/amp_lift_probe.py: tracker weights score 0.91 fwd / 0.92 back / 0.92 side / 0.80 pivot (its pivot only lifts one foot), walker v4@2000 scores 0.50 / 0.50 / 0.60 / 0.66. The lineage's feet_phase could not be reused as is: the cycles' swing peaks at clock phase ~0.17, feet_phase assumes 0. Dry run 512 envs x 30 its OK. eval_watch/walker_v5_pipeline.sh also scores checkpoints with a step-height factor (re-scoring v4: 400: 0.61, 2000: 0.45).
- 10-02 13:22 WALKER v5 LAUNCHED on the user's go ("go ahead and start v5"): v4 stopped at model_2400 (kept: archive_anchors/walker_v4_last_model_2400.pt), run kbot_legs_amp/2026-10-02_13-18-12_walker_v5, 8192 envs x 4000 its from trackmulti_v2_best_final.pt, v4 settings + KBOT_AMP_LIFT_W=5. Autopilot eval_watch/walker_v5_pipeline.sh (status eval_watch/WALKER_V5_STATUS.md, best -> archive_anchors/walker_v5_best.pt, score = survival x speed share x step-height factor).
- 10-02 18:20 WALKER v5 DONE (4000 its, 13:18-18:04). Steps held in every direction for the whole run (the lift anchor works): best model_3200, score 0.876 — fwd 0.35/0.40 surv 0.96 lift 13.7 cm; back -0.34/-0.40 surv 0.99 lift 14.3; side 0.10/0.13 surv 1.00 lift 9.0; pivot 0.52/0.56 surv 1.00 lift 8.2; turn 0.26+0.32 surv 1.00 lift 11.2; stand 0.98. Scores: 400 0.824, 800 0.84, 1200 0.852, 1600 0.845, 2000 0.841, 2600 0.85, 3200 0.876, 3999 0.863. Remaining gap: speed 75-90% of the command. User decision: NO hardening run; the RIG side decides hardware-readiness. Bundles exported (exporter now handles AMP runner + 10-frame layout + signed clock + command envelope; eval_watch/export_io_vectors.py writes io_test_vectors.npz): deploy_candidates/walker_v5_1200 and walker_v5_3200 (final, sent). Note: eval_watch/RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md. Lesson: the dashboard Render button + an autopilot test + training do not fit on the GPU; tests now wait for free memory and retry (walker_v5_tests.sh), renders share a lock (walker_v5_render_locked.sh).
- 10-03 14:40 FIRST HARDWARE ENGAGE of walker_v5_3200 (rig, eval_watch/RIG_HW_ENGAGE_FINDINGS.md + rig_hw_engage/): gates 0/1 passed (golden vectors exact; 3x60 s stands + walk battery in their sim), live engage = violent motion, operator cut power at 0.88 s, no fall. Rig's cause: the policy integrates on its own last_action when the plant does not answer; real robot carries a standing load at the zero pose (ankles 0.94/1.59 Nm, L hip roll 2.32 Nm), ankle spring 52 Nm/rad, and their 1 s engage crossfade starved the first commands. VERIFIED HERE: their toy test on our weights = -0.36/-0.23/-0.12/-0.04 (identical); Isaac reproduction eval_watch/amp_engage_test.py: -2.5 Nm pitch + 1 s ramp -> ankle action @0.4 s -0.58/+1.33, 21.6 rad/s, tilt 9.8 deg (HW -1.42/+1.25, 15-22 rad/s); the load alone (+2.5, hard start) drops 16% in 6 s. Gaps in our training: standing load (torso CoM DR only ~0.9 Nm), no stiction at rest (ankle rotor frictionless), constant gains, and NO EPISODE EVER BEGAN STANDING (walk_at_spawn converts every spawn stand). PREPARED (all off by default; KBOT_AMP_STAND_MOMENT / _DEADBAND / _ROTOR_FC / _ENGAGE_P / _SERIES_K / _SPAWN_STAND_P): standing moment in sustained_push_bursts, actuator _gain_scale/_deadband/_rotor_fc, mdp_amp.randomize_unanswered + engage_gain_ramp, walk_at_spawn keep_stand_p. Walker v6 = fine-tune of v5 best with all of it, 2000 its, eval_watch/walker_v6_pipeline.sh (score = walking x share of 15 engage conditions inside the rig's watchdog limits; v5 = 7/15). Dry run 512 x 30 OK; v5 forward test unchanged with the switches off. Reply to the rig drafted: eval_watch/REPLY_HW_ENGAGE.md. Awaiting the user's go.
- 10-03 15:25 WALKER v6 @400 (launched 14:35 from v5 best, 2000 its): walking 0.867 (unchanged from v5), engage 6/15 (v5 7/15), hardware condition (load y-2.5 + 1 s ramp) 26.1 rad/s, fell 25%; rig toy test -0.66 at 0% response (v5 -0.36). Diagnostic eval_watch/amp_engage_diag.py (96 robots at rest, 10 s, +2.5 Nm pitch, hard start, no ramp): NOT an engage transient — a slow drift. The robot settles into a 1.6 deg lean, holds ~2.5 s, then creeps in the direction of the load at ~0.2 m/s under a stand command and falls by bad_orientation at 4-10 s: v5 45% in 10 s, v6@400 60% (52% with the training reset), -2.5 Nm direction 14% with a steady -0.2 m/s backward drift. So the standing-load gap is "cannot hold a quiet stand against a constant moment" (the August tilt-servo problem again), on top of the weak-start wind-up. Both are in v6's data (load every episode) but diluted: ~5% of episodes are heavily loaded stands. Hypothesis on the toy test: stiction training rewards pushing harder when a joint does not move; nothing in training makes pushing harder useless (the crossfade case), so the toy number can get worse. Decision: let v6 run to the 800/1200 tests before changing anything.
- 10-03 15:35 WALKER v6b LAUNCHED (user: "restart now with the watchdog rule"): v6 stopped at model_600 (archive_anchors/walker_v6_last_model_600.pt). v6b = v6 recipe + mdp_amp.stand_watchdog as a termination (KBOT_AMP_WATCHDOG=5:12: while standing, joint speed > 5 rad/s or tilt > 12 deg ends the episode with the fall penalty; armed 0.3 s into a spawn stand, 1.5 s after a corridor stand, not during pushes), from v5 best, 2000 its, run kbot_legs_amp/2026-10-03_15-30-25_walker_v6b, autopilot eval_watch/walker_v6b_pipeline.sh, status eval_watch/WALKER_V6B_STATUS.md. Dry run (1024 x 40): the watchdog is the leading end-of-episode cause at the start (~1 robot/step vs 0.1-0.2 falls) without collapsing episode length. Why the rule: the standing penalties were only nudges — averaged over 20 s, bad_orientation at 57 deg with a 1.5 s stand grace.
- 10-03 18:10 CORRECTION + v6b RESULT. My engage test/diag used a STRIPPED plant ("nominal": training DR off, sensor noise off — which also removes randomize_joint_friction_ankles, the PhysX ankle joint friction every policy trained with). On that never-seen plant even an unloaded v5 stand drifts and falls (34% in 20 s); the "v5 falls 16-45% under the standing load / slow drift" finding of 15:20 was THAT ARTIFACT. Correct numbers (amp_engage_test.py --plant trained, now the default): v5 holds -3..+3 Nm pitch and roll for 20 s with 0 falls at a hard start (tilt 2-3.5 deg, joint speed peaks 4.4-6.7 rad/s — ~3 without sensor noise); the weak-start wind-up DOES reproduce on the trained plant (-2.5 Nm + 1 s ramp: ankle -0.77/+1.39 at 0.4 s, 18.2 rad/s, tilt 15.5 deg, 12% fell). v6 and v6b re-tested on the trained plant are WORSE than v5: within the 5 rad/s limits v5 7/15, v6@400 4/15, v6b@400 0/15, @800 2/15, @1600 5/15; hard-start falls v5 0%, v6b 3-15%; hardware condition unchanged ~19 rad/s. v6b finished 18:06 (its pipeline scores used the stripped plant: ignore them). No bundle from v6/v6b; v5_3200 remains best. Correction sent to the rig: eval_watch/REPLY2_HW_ENGAGE.md ("do not re-engage" withdrawn; 5 rad/s watchdog limit is at the level of a normal start). LESSON: a new test must first reproduce a known-good result (control row at the same duration, same plant as training) before any of its failures is believed.
