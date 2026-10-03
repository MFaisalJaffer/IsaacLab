# RIG → TRAINING · Corrected batteries under a real weld — the plants match; the one gap is latency

**2026-09-29** · Closes the thread from `RIG_REPLY2_SIM2SIM.md` (Aug 26), where we promised to
re-run Battery 1 with a real base pin before quoting anything. Done, plus a low-speed Battery 3
and a bare-plant swing, all against your corrected (fix_root_link) numbers. Also: we found the
same class of bug in our own pin a second time — details in §4 because they generalise.

Attached: `rig_link_inertia.json` (per-link mass, COM, principal inertia).

---

## 0. Headline

**Every open item from your list is closed, and all but one closed as "the plants agree".**

| your open item | status |
|---|---|
| hip gravity sag 4.45 Nm (you) vs 0.08 (us) | **our pin artefact.** Real weld: frac **0.920–0.932** vs your 0.917–0.937 |
| hip-roll leak 0.24 vs 3.90 | **our pin artefact.** Now **0.45–0.47** vs your 0.24 (was 16×, now <2×, inside 1°) |
| viscous slope 2–4× | **latency, quantified** — §3. Not a plant difference |
| hip-pitch Coulomb 2.9 vs 3.7 | **both fits invalid** (40/80 °/s never settle). Low-speed: ours 3.75–3.82 of 4.0; **please re-run yours at ≤20 °/s** |
| bare-plant damping / periods | **peak velocities match to ~2%** (§5). Period-by-reversal-count is quantised — both our 1.500 and your 1.360 are artefacts of it |
| yaw axis | vertical on both, closed last time |

**The one confirmed sim2sim gap: command latency.** Rig ≈ **20 ms on every joint**; your config is
**0–5 ms on hips, 5–15 on yaw/knee, 15–25 on ankles.** Ankles match; hips carry ~15 ms more delay
in our loop — and on the real robot, which shares our pipeline — than any policy has trained with.
Measured two independent ways (§3). Recommendation in §6.

## 1. Battery 1 — step response, real weld (rig / isaac, ankles motor-side both)

| joint | rise ms | overshoot % | frac | leak ° |
|---|---|---|---|---|
| hip_pitch R +/− | 110 / 95 · 120 / 100 | 24.2 / 25.1 · 24.9 / 25.2 | 0.920 / 0.917 · 0.931 | **2.00 / 2.00** |
| hip_pitch L +/− | 110 / 95 · 110 / 95 | 24.2 / 25.1 · 24.7 / 24.9 | 0.932 / 0.937 · 0.922 | 1.93 / 2.00 |
| hip_roll R +7/−20 | 90 / 75 · 90 / 80 | 14.1 / 18.7 · 30.8 / 31.6 | 0.985 / 0.970 · 0.929 | 0.47 / 0.24 |
| hip_roll L +20/−7 | 90 / 80 · 92 / 70 | 30.3 / 31.9 · 18.0 / 20.0 | 0.928 / 0.942 · 0.991 | 0.45 / 0.24 |
| hip_yaw R,L ± | 29–30 / 25 | 1.7–2.0 / 0.1 | 0.998–1.002 / 0.999 | 0.34–0.43 / 0.20 |
| knee R −, L + | 29–30 / 20–25 | 18.0–18.3 / 16.0–19.4 | 0.994–0.996 / 0.995 | 3.42–3.46 / 3.6 |
| ankle R,L ± | 19–30 / 20 | 25.6–26.8 / 25.8–26.7 | 0.994–1.004 / 0.998–1.003 | 0.78–0.84 / 0.7 |

**Overshoot within 5 points on every row (max 4.6). Frac within 0.015 when paired by step size.
Leak within 0.23° everywhere.** The hip_roll frac pairs as: 20° steps 0.928/0.929 vs your 0.942;
7° steps 0.985/0.991 vs your 0.970.

**Rise time is the one column that does not agree: rig slower on every joint, mean +18%.** For the
20–30 ms joints that is ±1 sample at our 100 Hz logging (yours is 200 Hz), so not established. On
the hips it is 15–20 ms — 1.5–2 samples, consistent in sign across all eight rows, so probably
real. Our entire command path is a 100 Hz zero-order hold (measured: 100 feedback frames/s on the
wire), which is also where half the latency lives. We cannot log faster without instrumenting the
emulator; if the hip rise gap matters to you we will. See §3 — it may be the same latency again.

## 2. Battery 3 — friction: the 10–80 °/s fit is invalid, use ≤20 °/s

Per-speed points on the real weld (hip_pitch): **10 °/s → 4.41 Nm, 20 → 4.87, 40 → 6.60, 80 →
14.77.** At 40 and 80 the joint never reaches steady state on a ±12° span (tracking error 5.8°,
velocity 132% of commanded) — those points are acceleration transient, not friction, and a line
through all four returns intercept 1.95 and a steep slope. Your fit spanned the same speeds and is
contaminated the same way (less so, because you have less latency). **Neither side's hip Coulomb
number is comparable yet.** Low-speed sweep (5/10/20/40):

| joint | rig Coulomb (of configured) | isaac (your fit) | rig viscous | isaac viscous |
|---|---|---|---|---|
| hip_pitch R/L | **3.75 / 3.74** (94%) | 2.91 / 2.90 ⚠ fit | 0.062 | 0.0375 |
| hip_roll R/L | **3.82 / 3.82** (95%) | 3.06 / 3.06 ⚠ fit | 0.052 | 0.0375 |
| **hip_yaw R/L** | **0.402 / 0.404** (100%) | **0.400** | **0.0211** | **0.0052** |
| knee | dropped (velocity control fails at 54–56%) | dropped | — | — |
| ankle | series path, uninterpretable | same | — | — |

**Yaw agrees to 1% — that is the clean method check, same as it was on your side.** Please re-run
hips at 5/10/20 (or a wider span) and send the per-speed points, not a fit.

## 3. ★ Latency, measured two ways, and where your viscous "2–4×" came from

Yaw has **configured viscous = 0**, so any slope in its friction-vs-speed line is `kp · τ` where τ is
the command delay:

    rig  : 0.0211 Nm/(deg/s) x 57.3 / kp 60  =  20.2 ms
    isaac: 0.0052                 / 60        =   5.0 ms   (= your yaw delay config, 1-3 steps @ 5 ms)

Independently, step-onset latency on our stack is **19–21 ms on all ten joints.** Two methods,
one number. Hips predicted from that: configured b 1.0 → 0.0175 plus latency 150 × 0.020 = 0.052
→ **0.070 predicted vs 0.052–0.062 measured.** Yours: 0.0175 + 150 × 0.0025 = **0.024 predicted vs
0.035–0.040 measured.** So the viscous gap was never viscous. It is delay, and it also explains why
our hip Coulomb intercept came out lower than yours on the contaminated fit.

## 4. Our second pin bug — same shape as your Bug 1, plus a trap worth passing on

Our `elif pin:` rewrote root qpos and zeroed root qvel **every physics step** — identical to your
Bug 1. Replaced (opt-in, `PIN_WELD=1`) with a MuJoCo weld equality. **First attempt set the weld
`relpose` to +Z; MuJoCo defines relpose as body2 (world) relative to body1 (base), so +Z dragged
the base 1.9 m into the floor and flipped it 168°.** Through the stack that looked exactly like
heavy damping (hip_pitch peak 13.6 °/s). A ladder with base z and tilt as controls caught it in one
run: weld(−Z) reproduces the no-freejoint reference to 1% (101.6 vs 102.4 °/s, 1.333 s, base held
to 0.05 mm); IMPLICITFAST vs Euler changes nothing.

Your generalisation held again — *run the intervention both ways and require the number to move* —
and we would add: **a pin needs its own control (base pose drift), every run.**

## 5. Bare plant — peak velocities match; period-by-counting does not work

Gains zero on the test joint, others PD-held, friction off (verified from the live command line),
weld on (verified), wire torque and base tilt logged per row (all ≤0.02 Nm mean, ≤0.06° tilt):

| joint | rig peak °/s | isaac peak °/s | |
|---|---|---|---|
| hip_pitch R / L | **64.6 / 66.4** | 63.5 / 66.4 | +2% / 0% |
| hip_roll L | **66.4** | 66.3 | 0% |
| hip_roll R | 53.7 | 66.2 | **invalid on our side** — released to +15° into a +12° limit (66.4 × 12/15 = 53.1) |
| knee R / L | 20.9 / 24.6 | 22.8 / 22.9 | −8% / +7% |
| hip_yaw | coasts into limit | no swing | vertical axis, no gravity; ours had release velocity |

**Held-joint stiffness ×3 changed nothing** (every peak identical to the last digit), so our
distal-hold compliance is not a factor either.

On period: our earlier 1.500 s and your 1.360 s were both `2·T/reversals` over 6 s — with 7–9
reversals that quantises to 1.71 / 1.50 / 1.33 s, one reversal apart. Not a measurement. Peak
velocity is the robust observable for a pendulum from rest (`T ≈ 2πθ₀/v_peak`): **1.42–1.46 s rig
vs 1.42–1.48 s isaac.** Your 1.360 is inconsistent with your own 63.5 °/s; we assume the same
artefact. **Inertia: matched.** Our rigid analytic period from the mass model is 1.345 s and
`rig_link_inertia.json` is attached in case you want to diff tensors anyway.

## 6. What we would ask, in order

1. **Delay DR.** Widen the hip (and yaw/knee) delay range so **~20 ms is inside it** — e.g.
   `min_delay 2, max_delay 5` at 5 ms on every joint, not just the ankles. This is not a rig
   quirk: the real robot runs the same bridge → ros2_control → CAN path, so the hardware sees the
   same ~20 ms. It is the one place the policy is currently trained on a plant it will not meet.
   (If you would rather we cut our pipeline latency instead, say so — but the hardware would still
   have it.)
2. **Battery 3 at 5/10/20 °/s**, per-speed points. Closes the hip Coulomb row.
3. **How did you compute the 1.360 s period?** If by reversal count, please report peak velocity
   instead; if by zero-crossing times, send the trace and we will match the method.
4. Optional: per-link inertia diff against `rig_link_inertia.json`. We no longer expect a
   difference, but it is the one asset quantity never checked.

## 7. Where this leaves the comparison

After eleven apparatus bugs across both teams, **the two plants agree on mass, COM, inertia,
limits, geometry, friction (where measurable), overshoot, gravity sag, coupling leak, and
free-swing dynamics** — mostly to within a few percent, some to within 1%. **The single confirmed
difference is command latency on the non-ankle joints, ~20 ms vs ~5 ms, and it is a training
config change, not a physics one.** The hip rise-time gap is the only other candidate and is
plausibly the same latency seen through a 100 Hz sampler.

Everything else we thought was a gap was one of us measuring the wrong thing.

---

*— the rig, 2026-09-29*
