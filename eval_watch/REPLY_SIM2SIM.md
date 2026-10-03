# TRAINING → RIG · Both sim2sim batteries, Isaac side

**2026-08-26** · Reimplemented both protocols per `00_README_SIM2SIM.md`. Two `metrics.json` in your
schemas are attached, plus the six answers. Scripts are `eval_watch/sim2sim_step.py` and
`eval_watch/sim2sim_friction.py`; raw decay traces are in `decay_traces.tsv`.

---

## 0. Headline

**The two plants agree.** Every joint reaches target (frac 0.992–1.023), all ten pass `reach`, and
on your two gates:

- **Overshoot within 5 points: PASSES on all eight non-ankle joints** (largest gap 1.6 pts).
- **Leak: agrees to ~0.1° on 7 of 10 joints** — 2.33 vs 2.33, 3.88 vs 3.90, 2.68 vs 2.77, 2.35 vs
  2.28, 3.95 vs 3.97. This is the number you said would expose an inertia difference, and it does
  not. Coupled inertia matches.
- **Rise time: ours is consistently FASTER** — hips 60–65 vs your 70–80 ms, yaw 25 vs 30–40 ms.
  Hips land inside your ±20% band; **yaw does not** (25 vs 40 ms, 37% faster). Details in §3.

**And the ankle agrees too, once compared like with like.** We logged motor-side as well as
joint-side. Our motor-side ankle overshoot is **22–30% against your 26–27%**, and motor-side rise
20 ms against your 20–30 ms. The joint-side rows (36–46%) are the ones that manufacture a
discrepancy — exactly as your readout warning predicted. Your warning was right and it saved a
false alarm.

---

## 1. The six questions

1. **Foot-to-floor clearance at the test pose: 43.7 mm.** We do not have your trap. We lifted 0.6 m
   anyway and report the number as asked. Feet are clear by an order of magnitude more than yours.
2. **Position quantum: float, none.** No encoder quantiser anywhere in our chain — joint state is
   float32 straight from PhysX. Our ladder therefore resolves to ~0.0002°, ~100× finer than your
   0.0219° LSB. See §4 — this matters more than expected.
3. **Ankle rows: we report BOTH.** Joint-side is primary; every ankle entry carries a `motor_side`
   sub-dict taken from the rotor state inside our actuator. Compare against `motor_side`.
4. **Stiction: NONE, and it is the same form as yours** — literally `-fc*tanh(qd/0.02)`, zero at
   qd = 0. **Your hypothesis is confirmed: neither sim has stiction, and no policy either of us has
   trained has ever met a joint that refuses to close the last fraction of a degree.** §4 measures
   this at your resolution and then some.
5. **USD limits vs your MJCF table: ZERO mismatches.** All ten agree within 1°, knees one-sided as
   you have them ([−155, 0] R / [0, 155] L).
6. **COM +11.6 mm: YES, it landed** — 2026-08-24. It took two attempts and the first was wrong in a
   way worth passing on: `randomize_rigid_body_com` applies **body-frame** offsets, and our torso
   link carries a +90° quaternion about z, so our "+11.6 mm x" was a pure *lateral* shift. It was
   also undersized — the torso is 6.24 kg of 13.06, so moving the whole robot +11.6 mm needs
   −24.3 mm on the torso, which is your MJCF number exactly. Now applied as body-frame y −0.0243 m
   and verified. **But see §5 — our zero-pose COM still does not land where yours does.**

---

## 2. Asset facts (your §6)

| | rig | isaac | |
|---|---|---|---|
| total mass | 13.06 kg | **12.85 kg** | −0.21 kg (−1.6%) |
| foot polygon, heel side of ankle axis | 77 mm | **77.8 mm** | agrees |
| foot polygon, toe side of ankle axis | 135 mm | **134.4 mm** | agrees |
| COM x at zero pose | +13.1 mm (= on the ankle axis) | **+5.1 mm** | |
| ankle axis x at zero pose | (implied +13.1 mm) | **+13.7 mm** | agrees |
| COM − ankle axis at zero pose | 0 | **−8.7 mm (behind)** | **see §5** |

Foot geometry and ankle axis location match to under a millimetre. Mass and zero-pose COM do not.

## 3. Battery 1 — step response

Full table in `step_metrics.json` (`sim2sim_step_v2`). Rise/overshoot/leak vs yours:

| joint | dir | rise ms (us / you) | overshoot % (us / you) | leak (us / you) | frac |
|---|---|---|---|---|---|
| right_hip_pitch | + / − | 60 / 70 · 60 / 70 | 13.2 / 13.7 · 13.8 / 14.8 | 2.33 / 2.33 | 1.003 / 1.001 |
| right_hip_roll | + / − | 65 / 80 · 65 / 80 | 12.2 / 10.6 · 18.9 / 17.8 | 3.88 / 3.90 | 0.997 / 0.999 |
| right_hip_yaw | + / − | 25 / 40 · 25 / 30 | 0.4 / 1.5 · 0.4 / 1.9 | 0.23 / 0.10 | 0.999 |
| right_knee | − | 25 / 30 | 18.6 / 20.4 | 2.68 / 2.77 | 0.998 |
| right_ankle | + / − | *motor* 20 / 29 · 20 / 20 | *motor* 29.7 / 26.3 · 29.3 / 27.4 | 0.79 / 0.84 | 0.992 / 1.023 |
| left_hip_pitch | + / − | 60 / 79 · 60 / 71 | 13.6 / 14.5 · 13.2 / 14.1 | 2.35 / 2.28 | 1.001 / 0.999 |
| left_hip_roll | + / − | 65 / 80 · 60 / 70 | 18.0 / 17.5 · 10.7 / 11.4 | 3.95 / 3.97 | 1.001 / 1.002 |
| left_hip_yaw | + / − | 25 / 40 · 25 / 40 | 0.2 / 1.5 · 0.2 / 1.7 | 0.23 / 0.14 | 0.999 |
| left_knee | + | 25 / 20 | 20.3 / 20.1 | 2.94 / 2.70 | 1.002 |
| left_ankle | + / − | *motor* 20 / 20 · 20 / 30 | *motor* 22.0 / 26.2 · 24.8 / 27.2 | 0.57 / 0.78 | 1.023 / 0.992 |

**Latency is not the explanation for the rise-time gap.** We ran a second arm with all command
delay removed (`step_metrics_nodelay.json`). **16 of 17 rows did not move at all**; one moved by a
single 5 ms sample. So our 0–25 ms per-family latency is not what makes us faster — it is plant or
actuator. Our sampling is 5 ms, so all rise times are quantised to that; a 25 vs 40 ms yaw
difference is 3 samples and real, a 60 vs 70 ms hip difference is 2.

**Where we would look first:** yaw. It is the joint where we differ most in relative terms (37%),
it has the lowest inertia so it is the most sensitive to an armature or rotor-inertia difference,
and our overshoot is also lower (0.2–0.4% vs 1.5–1.9%) — both consistent with our yaw carrying
*less* effective inertia than yours. Our yaw armature is 0.005 kg·m². What is yours?

## 4. Battery 2A — amplitude ladder

`friction_metrics.json` (`sim2sim_friction_v1`).

**We independently reproduce your central finding, at 100× your resolution.** Our eight non-ankle
intercepts:

| joint | our intercept | implied Coulomb | nominal `Fc` in our plant |
|---|---|---|---|
| hip_pitch R / L | −0.0177 / +0.0158° | 0.046 / 0.041 Nm | **4.0** |
| hip_roll R / L | +0.0002 / +0.0006° | 0.0005 / 0.0015 Nm | **4.0** |
| hip_yaw R / L | −0.0005 / −0.0005° | 0.0005 Nm | **0.4** |
| knee R / L | −0.0002 / −0.0001° | 0.0005 Nm | **0.6** |

You reported bounds because your intercepts sat inside your quantiser. **We have no quantiser and we
still measure essentially zero** — 0.0005 Nm against a nominal 4.0. This is no longer an upper
bound: it is a measurement, and it confirms `-fc*tanh(qd/0.02)` contributes *nothing* statically.
The ~0.017° on hip pitch (both sides, opposite signs) is the only non-zero row and is gravity sag,
not friction.

**The ankle ladder is a clean confirmation of your readout warning, with a number attached:**

| | joint-side intercept | motor-side intercept |
|---|---|---|
| right_ankle | **+0.1602°** | +0.0070° |
| left_ankle | **+0.1659°** | +0.0074° |

We ran play = 0.3° to match your stack. **Half the play band is 0.15°, and the joint-side intercept
is 0.160/0.166°.** The motor closes to target; the joint sits short by half the deadband, and the
motor-side encoder cannot see it. That is your ~54%-invisible geometry showing up as a static
offset — and it is a useful independent check that both play implementations are doing the same
thing. Your ankle rows (0.0255 / −0.0401°) match our *motor-side* rows, as they should.

## 5. Battery 2B — free decay: WE CANNOT RUN IT, and the reason is worth having

**We are not reporting half-lives. Our joints do not swing when the PD is switched off, and the
control says why.**

With kp = kd = 0, per joint, over 6 s at 200 Hz:

| joint | qd sign reversals | \|tau\| still applied | verdict |
|---|---|---|---|
| hip_pitch R / L | **100% of steps** | 4.20 / 4.23 Nm | dither, not swing |
| hip_roll R / L | **100%** | 4.13 / 4.14 Nm | dither |
| hip_yaw R / L | **100%** | 0.399 Nm | dither |
| knee R / L | **100%** | 0.639 / 0.650 Nm | dither |
| ankle R / L | 49% | 0.000 Nm | did not move at all |

**The applied torque with the gains at zero is exactly our configured `Fc`** (4.0 / 0.4 / 0.6).
Our friction lives in the actuator in Nm and does not care that the PD gains went to zero — so
"actuator off" on our side still leaves the full friction term in the loop. And at 5 ms it is
numerically stiff: one step of 4 Nm on a ~0.19 kg·m² leg changes velocity by 0.105 rad/s, five
times the 0.02 rad/s tanh width, so the term overshoots zero velocity **every single step**. The
joint neither sticks nor swings — it limit-cycles and creeps at ~0.5°/s.

Confirmed by a three-way release trial on one joint (`decay_diag.py`):

| release condition | motion in 3 s | \|tau\| | qd reversals |
|---|---|---|---|
| gains = 0 only | +1.54° (creep) | 4.21 Nm | 599/600 |
| gains = 0, Coulomb off | +0.19° | 0.06 Nm | 598/600 |
| gains = 0, Coulomb **and** viscous off | **−11.98° (real swing)** | 0.000 Nm | **6/600** |

Only with both friction terms removed does the pendulum behave like one — 6 reversals in 3 s ≈ 1 Hz,
which is the free-swing mode.

**Three things follow, and the second is the one that matters:**

1. Our Test B and yours are not measuring the same thing. Your `kp_scale=kd_scale=0` evidently drops
   your friction with the gains; ours does not. **Please check what your emulator does with `Fc`
   when the gains are zero** — if it also stays live, your clean 1.5–2.5 s half-lives are hard to
   reconcile with a 4 Nm hip.
2. **Back out the friction your Test B actually saw.** A Coulomb-damped pendulum loses
   `4·tau_c/k` of amplitude per cycle. From your hip_pitch row (15° start, 2.01 s half-life,
   ~6.5 Hz) that is on the order of **0.02 Nm, not 4.0 Nm.** If that is right, your Test B
   characterised a residual, and the "linear envelope → Coulomb" verdict is about your MuJoCo
   `frictionloss`, not about the 4 Nm the policy actually feels. Worth a look before either of us
   treats that half-life as a plant constant.
3. Our ankles do not free-swing either (0.000 Nm, no motion) — with kp = kd = 0 the rotor floats and
   the joint hangs off the series spring. **Same result as yours, same reason.** One row where the
   two sims agree by agreeing not to work.

To get you a comparable number we would need a bare-plant arm on both sides (friction off, gains
off). Say the word and we will run ours; it is one flag (`--actuator_off plant_only`).

## 6. Two discrepancies we would like you to check on your side

1. **Total mass: 12.85 kg vs your 13.06 kg.** 210 g, 1.6%. Small, but it is 1.6% off every torque
   and every static margin, and it is the kind of thing that explains a 15% rise-time gap if it
   sits in the wrong link. Can you send per-link masses? We will diff them; ours are in
   `step_metrics.json → asset_facts.link_mass_kg`.
2. **Zero-pose COM sits 8.7 mm BEHIND our ankle axis; yours is on it.** Our ankle axis is at
   +13.7 mm from root and yours is at +13.1 mm — those agree. Our COM is at +5.1 mm. So after
   applying your correction we still land ~9 mm short of where you land. Since your whole
   passive-stability argument is built on COM-vs-axis, this is worth closing. Note our standing-pose
   check reads ≈0 mm, so the gap is configuration-dependent and we may be comparing different poses
   — tell us whether your +13.1 mm is at all-joints-zero or at the standing pose.

## 7. On `RIG_ANKLE_POSTFIX` (K_s 23 → 52, randomise 20–120)

Received and understood; the `K_s` used for this whole battery is **52**, yours, not our training
value. Your §4 ask — randomise rather than converge — is the right call and we are not going to
argue with a moving target. Implementing it needs a training restart, which is a decision on our
side that is being made now; we will confirm in the next note rather than promise here. The other
two parts (play band 0–3°, load-dependent play that opens in swing and closes in stance) we agree
are the more valuable half, and the load-dependent band is the one we had not modelled at all.

---

*— training, 2026-08-26*
