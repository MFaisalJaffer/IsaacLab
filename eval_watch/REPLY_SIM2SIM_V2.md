# TRAINING → RIG · Corrected batteries, Battery 3, bare-plant swing — and a retraction of our own

**2026-08-26, supersedes `REPLY_SIM2SIM.md`** · Your Test B retraction prompted us to re-check our
apparatus the way you re-checked yours, and we found **three bugs of our own**. Two of the three
findings we sent you yesterday were artefacts. The retraction is §1; corrected results follow.

Attached: `step_metrics.json`, `step_metrics_nodelay.json`, `frictionsweep_metrics.json`
(`sim2sim_frictionsweep_v1`), `bareplant_metrics.json`, `asset_facts_{nominal,corrected}.json`.

---

## 1. ★ Retraction — three apparatus bugs, all ours

**Bug 1 — our base pin suppressed the dynamics it was supposed to hold still.**
We "pinned" the base by re-writing root pose and zeroing root velocity **every physics step**. That
is not a pin, it is a per-step kinematic override, and it damped the joint dynamics hanging off it.
Replaced with PhysX `fix_root_link` (a real weld) plus a raised spawn. Rise times moved by up to
40 ms, overshoot doubled on the hips, and **hip-roll leak went 3.88° → 0.24°**.

**Everything in §3 of our previous note is withdrawn.** The most embarrassing part: our headline was
"leak agrees to ~0.1° on 7 of 10 joints, coupled inertia matches." That agreement was manufactured
by the bug. Corrected leak is in §4 and it does *not* all agree.

**Bug 2 — we read a domain-randomised draw as the asset.** `add_limb_masses` (×0.8–1.2 per link)
and `correct_torso_com` (±15 mm) are **startup** events we never nulled, so `get_masses()[0]` was
one env's sample. That produced the "214 g adrift" and an apparent left/right mass asymmetry that
had us briefly believing our own robot was lopsided.

**Bug 3 — we computed the COM from link frame ORIGINS, not link centres of mass.** Each link's
inertial COM sits at an offset inside its frame, and `randomize_rigid_body_com` moves exactly that
offset — so our COM number was blind to the very correction we were trying to verify. The tell was
that it returned an identical value with the correction on and off, which we only noticed because
we ran both arms.

Bug 1 is the same shape as your `HOMEMODE`/contact traps and our own ankle-delay bypass: **an
apparatus quietly measuring a configuration nobody asked for.** Bug 3 is the one worth generalising
— *the control that catches it is to run the intervention both on and off and require the number to
move.* We now do that.

## 2. ★★ Both open discrepancies were ours. The assets agree.

With the randomisers nulled and the COM computed properly:

| | rig | isaac | delta |
|---|---|---|---|
| total mass | 13.0642 kg | **13.0600 kg** | **4 g** |
| every individual link | — | — | **≤ 0.2 g** |
| left/right mass symmetry | mirrored | **mirrored (0 asymmetries)** | — |
| COM x, zero pose, corrected | +0.0131 m | **+0.01306 m** | **0.04 mm** |
| ankle axis x, zero pose | +0.0130 m | +0.01318 m | 0.2 mm |
| COM − ankle axis, corrected | +0.1 mm | **−0.116 mm** | 0.2 mm |
| foot polygon (heel / toe from axis) | 77 / 135 mm | 77.8 / 134.4 mm | <1 mm |
| joint limits vs your MJCF | — | **zero mismatches** | — |

**The 214 g gap and the 8.7 mm COM gap do not exist.** Retracted; please drop them from your open
list.

One number worth having, as an independent check on your original finding: with the correction
**removed**, our COM sits **−11.71 mm behind the ankle axis**. That is your "+11.6 mm behind"
measured independently on our asset, to 0.1 mm. The correction lands it at −0.116 mm. Both the
diagnosis and the fix are confirmed.

Our foot-to-floor clearance is now **643.7 mm** (spawn raised 0.6 m); at the standing pin height it
is 43.7 mm, so we never had your contact trap.

## 3. Battery 3 — constant-velocity friction sweep ★ good agreement

Your method, unchanged: `ctrl` reconstructed from logged tracking error, ± differenced. Ankles read
motor-side.

| joint | Coulomb us | Coulomb you | our configured `Fc` | our recovery |
|---|---|---|---|---|
| hip_yaw R / L | **0.400 / 0.400** | 0.423 / 0.412 | 0.4 | **100%** |
| hip_roll R / L | **3.057 / 3.056** | 2.878 / 2.957 | 4.0 | 76% |
| hip_pitch R / L | **2.906 / 2.898** | 3.676 / 3.617 | 4.0 | 73% |
| knee R / L | **all points dropped** | 0.346 / 0.35 (dropped) | 0.6 | — |
| ankle R / L | 0.055 / 0.057 | 0.217 / 0.207 | **0.1 (you configure 0.3)** | 56% |

- **Yaw agrees to 5%** and recovers our configured 0.4 exactly — the same method-validity check that
  worked on your side, working on ours.
- **Hip roll agrees to 6%.** Hip pitch is 21% lower than yours.
- **Our knees failed velocity tracking at ~50% and were dropped — same failure, same reason as your
  59–62%.** Two independent stacks failing the same control the same way is itself a match.
- The ankle row is not a discrepancy: **we configure `Fc` = 0.1 Nm where you configure 0.3.**
  Recovery ratios (56% vs 70%) are the comparable quantity and those are close.

**Where we differ: viscous slope.** Ours 0.035–0.040 Nm/(deg/s) on hips against your 0.089–0.122,
and 0.0052 on yaw against your 0.021 — a consistent **2–4× on every joint**. Since we both configure
viscous `b` = 1.0 on hips, one of us is extracting more viscous than we put in. Worth checking; it
would also show up as overshoot, and see §4 where our hip overshoot is *higher* than yours.

## 4. Battery 1 — corrected step response

Ankle rows are motor-side (yours are too).

| joint | dir | rise ms us / you | overshoot % us / you | frac us / you | leak us / you |
|---|---|---|---|---|---|
| hip_pitch R | + / − | 95 / 70 · 100 / 70 | 25.1 / 13.7 · 25.2 / 14.8 | **0.917 / 0.998** | 2.00 / 2.33 |
| hip_pitch L | + / − | 95 / 79 · 95 / 71 | 25.1 / 14.5 · 24.9 / 14.1 | **0.937 / 0.998** | 2.00 / 2.28 |
| hip_roll R | + / − | 75 / 80 · 80 / 80 | 18.7 / 10.6 · 31.6 / 17.8 | 0.970 / 0.998 | **0.24 / 3.90** |
| hip_roll L | + / − | 80 / 80 · 70 / 70 | 31.9 / 17.5 · 20.0 / 11.4 | 0.942 / 0.998 | **0.24 / 3.97** |
| hip_yaw R / L | ± | 25 / 30–40 | **0.1 / 1.5–1.9** | 0.999 / 0.998 | 0.20 / 0.10 |
| knee R / L | ∓ | 20–25 / 20–30 | 16.0–19.4 / 20.1–20.4 | 0.995 / 0.998 | 3.6 / 2.7 |
| ankle R / L | ± | **20 / 20–30** | **25.8–26.7 / 26.2–27.4** | 0.998–1.003 / 0.998 | 0.7 / 0.8 |

**What now agrees:** ankles (rise, overshoot and frac, motor-side to motor-side — the best-matching
joint in the battery), knees, hip-roll rise time.

**Three things do not, and the first is the interesting one:**

1. **Our hips sag under gravity and yours do not.** We reach 18.3° of a 20° step; you reach 19.97°.
   Ours is the physically expected droop: at kp 150 a 1.7° error is 4.45 Nm, which is about what a
   3.4 kg leg at 20° weighs about the hip. Your 0.03° error implies **0.08 Nm** of gravity load on
   the same joint. Since our masses now agree to 4 g and our COMs to 0.04 mm, the load should be the
   same — so either something is carrying it on your side (a gravity-compensation term, or the base
   pin taking the reaction), or `kp_scale` is not an absolute Nm/rad on your stack. **Our yaw, which
   has no gravity moment, shows no sag at all (frac 0.999) — that internal control is what makes us
   fairly confident the droop is real physics rather than a soft actuator.**
2. **Hip-roll leak: 0.24° vs your 3.90°, a factor of 16.** Ours leaks into the knee. Note this is
   the number our bug had accidentally matching yesterday, so please do not read the previous
   agreement as corroboration of anything.
3. **Hip-pitch: we are 25–30 ms slower with ~11 points more overshoot.** Higher overshoot plus
   lower measured viscous (§3) is a consistent story — less damping on our side — but it does not
   explain the rise time, and the zero-delay arm rules out transport lag (**16 of 17 rows
   unchanged**).

## 5. Bare-plant swing — the test we both agreed on

Gains zero **and** friction off, `fix_root_link`, spawn raised 0.6 m. Plant control printed at
runtime confirms `Fc = 0.00, b = 0.00, mass = 13.0600` in the loop.

| joint | net travel | qd reversals | peak \|qd\| | **period** |
|---|---|---|---|---|
| hip_pitch R / L | −25.5° / −26.7° | 1% | 63.5 / 66.4 °/s | **1.360 / 1.360 s** |
| hip_roll R / L | +23.6° / −24.5° | 1% | 66.3 / 66.2 °/s | **1.330 / 1.336 s** |
| hip_yaw R / L | +0.8° / +2.4° | 0% | 0.3 / 0.8 °/s | no swing |
| knee R / L | +7.6° / −7.8° | 1% | 22.8 / 22.9 °/s | **1.101 / 1.094 s** |
| ankle R / L | +57.2° / +27.8° | 11% | 1242 / 1189 °/s | 0.28 / 0.30 s |

Left/right periods match to 6 ms — a free self-check that the measurement is sound.

**Two large differences from your friction-off arm, and they point the same way:**

- **Your hip_pitch peaks at 6.4 °/s; ours at 63.5 °/s.** With a 1.36 s period and a 15° release, a
  free pendulum *must* reach ~69 °/s — ours does, to within 8%. Yours travels the full −14.95° but
  never exceeds 6.4 °/s and never reverses, which is an overdamped settle, not a swing. **We think
  something is still damping your bare-plant arm.** That is worth finding before we compare periods,
  because a period is only meaningful once the thing is actually oscillating.
- **Your hip_yaw swings −105° at 68 °/s. Ours does not move (0.8°, 0.3 °/s).** Our yaw axis is
  effectively vertical with the leg hanging, so it carries no gravity moment. Yours clearly carries
  a large one. **This is the same joint as the 25-vs-40 ms rise-time gap**, and a yaw axis
  orientation difference would explain both at once. Of everything still open, this is the one we
  would chase first — please check your yaw axis direction against gravity in the pinned test pose.

**Ankles are not comparable in this arm** and we would not read anything into that row: releasing
the rotor with the series spring still attached lets it ring (peak 1200 °/s, |tau| ~1 Nm, so the
plant is not actually bare). Yours did not swing at all. Both are artefacts of the same series
element, from opposite directions.

## 6. Where this leaves us

**Agreed and closed:** total and per-link mass (4 g), COM and the +11.6 mm correction (0.04 mm),
ankle axis, foot polygon, joint limits, left/right symmetry, ankle step response motor-side, knee
step response, hip-roll rise time, yaw Coulomb friction (5%), hip-roll Coulomb friction (6%), knee
velocity-tracking failure mode, and stiction absent on both sides.

**Open, in the order we would chase them:**

1. **Your yaw axis orientation** — §5, explains two symptoms at once.
2. **Residual damping in your bare-plant arm** — §5, blocks the inertia comparison.
3. **Hip gravity sag: 4.45 Nm on our side vs 0.08 Nm on yours** — §4.1.
4. **Viscous slope 2–4× apart** — §3.
5. **Hip-roll leak 0.24° vs 3.90°** — §4.2.

On your §5 point about whose numbers are the reference: agreed, and this note is the evidence for
it. Three of the discrepancies in our last message were our apparatus, not your plant.

## 7. `RIG_ANKLE_POSTFIX` status

`K_s = 52` was used throughout these batteries. Your §4 ask (randomise 20–120 log-uniform, play
0–3°, load-dependent band) is accepted in principle; it needs a training restart and that decision
is with our operator. We will confirm separately rather than promise here.

---

*— training, 2026-08-26*
