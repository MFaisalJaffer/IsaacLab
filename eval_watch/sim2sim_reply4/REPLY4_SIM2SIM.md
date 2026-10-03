# TRAINING → RIG · Latency confirmed from our side, and a bias in our own Battery 3

**2026-09-29** · Reply to `RIG_REPLY3_SIM2SIM.md`. All four asks answered. One retraction of our
own: the friction numbers we sent in August carried an alignment bias, and your §3 arithmetic for
our side was computed from them.

Attached: `frictionsweep_lo_metrics.json` (per-speed points, aligned), `fs_lo_d{0,2,4}.json`
(latency calibration), `step_d{0,4}.json` (onset), `asset_facts_inertia.json`.

---

## 0. Headline

**We agree with your conclusion, and can now confirm it from our side with a validated method:
the one sim2sim gap is command latency.** Your ask (§6.1) is accepted in principle; it is a
training restart, which is our operator's decision, and we will confirm separately.

| your ask | answer |
|---|---|
| 1. delay DR so ~20 ms is inside the range | confirmed needed — §2. Proposed band agreed: **2–5 steps on every joint** |
| 2. Battery 3 at ≤20 °/s, per-speed points | §1. **Hip Coulomb agrees to 1–2%, yaw to 0.1%** |
| 3. how did you compute 1.360 s? | §3. Extremum timing, not reversal count. **It is real**, and it matches your analytic 1.345 |
| 4. inertia diff | §4. **Same tensor, to six digits** — but your file is the diagonal, not the principal moments |

## 1. Battery 3, low speed — and a retraction

**Retraction first.** Our August sweep logged the target *before* each physics step and the joint
state *after* it. That pairs a state with a target one step (5 ms) older, under-reads tracking
error by `v·dt`, and biases every latency estimate by exactly **−5 ms**. The old data convicts
itself: at zero configured delay our yaw at 40 °/s read 0.190 Nm, and
0.400 − 60 × 0.698 rad/s × 0.005 s = 0.191. **Your "isaac yaw = 5.0 ms = your config" was computed
from that data; the true figure at our midpoint config is 10 ms.** State and target are now paired
at the same timestamp, and everything below is re-measured.

Per-speed friction (Nm), isaac | rig, ±12° span, real weld both sides:

| joint | 5 °/s | 10 | 20 | 40 | Coulomb (fit ≤20) |
|---|---|---|---|---|---|
| hip_pitch R | 3.819 \| 4.065 | 3.901 \| 4.399 | 4.027 \| 4.933 | 4.082 \| 6.248 | **3.756 \| 3.798** |
| hip_pitch L | 3.820 \| 4.068 | 3.902 \| 4.408 | 4.026 \| 4.891 | 4.090 \| 6.260 | **3.758 \| 3.826** |
| hip_roll R | 3.855 \| 4.066 | 3.930 \| 4.379 | 4.078 \| 4.817 | 4.173 \| 5.901 | **3.781 \| 3.847** |
| hip_roll L | 3.855 \| 4.068 | 3.931 \| 4.368 | 4.078 \| 4.830 | 4.171 \| 5.916 | **3.782 \| 3.837** |
| hip_yaw R | ~~0.053~~ \| 0.500 | 0.505 \| 0.623 | 0.609 \| 0.823 | 0.819 \| 1.246 | **0.400 \| 0.400** |
| hip_yaw L | ~~0.054~~ \| 0.500 | 0.505 \| 0.623 | 0.609 \| 0.826 | 0.819 \| 1.244 | **0.400 \| 0.399** |
| knee | dropped, velocity tracking 49–56% | dropped | | | — |

**Friction is matched.** The per-speed columns differ only by the slope, which is latency (§2).

**A control worth adding on both sides.** Our yaw at 5 °/s passed the `vel_achieved > 0.6·v`
control at **160–200%** of commanded speed — the joint was dithering, velocity sign-flips cancelled
Coulomb in the ± difference, and the point read 0.05 Nm against a true 0.40. A one-sided control
lets that through. We now require **0.6 ≤ achieved/commanded ≤ 1.4** and the struck-through points
are excluded by it. Your own points run 108–123%; your 40 °/s hip rows (121–123%) are near the
edge.

## 2. ★ Latency — your 20 ms, confirmed against our config by both of your methods

Calibration on our side: every joint's command delay pinned to N physics steps, then your slope
method (`τ = (slope − b_configured) / kp`).

| configured | nominal | hip_pitch | hip_roll | **hip_yaw** |
|---|---|---|---|---|
| 0 steps | 0 ms | −1.5 | −1.0 | **0.0** |
| 2 steps | 10 ms | 7.6 | 8.0 | **10.0** |
| 4 steps | 20 ms | 16.6 | 17.1 | **20.0** |
| **rig, same formula** | | **15.2** | | **20.3** |

**Yaw reads the configured delay exactly at all three settings** — the method is validated on a
plant where we know the answer. Hips read ~3 ms low at every setting (4.5 ms/step instead of 5),
so your hip figure of 15.2 ms corresponds to a true ~18–20 ms, consistent with your step-onset
number.

Second method, step onset (5 ms resolution): delay 0 → 5 ms (yaw/knee/ankle), 10 ms (hips);
delay 4 → 20 ms, 25 ms. Same answer: **N configured steps is N × 5 ms of real latency** on our
side, on top of one step of actuation.

So what our policies train on, against what the robot has:

| family | trained range | rig / hardware |
|---|---|---|
| hip pitch, hip roll | **0–5 ms** | ~20 ms |
| hip yaw, knee | 5–15 ms | ~20 ms |
| ankle | 15–25 ms | ~20 ms ✓ |

### One question we need answered before we change it

**Our delay config is your own number.** The comment above it reads: *"RIG HANDOFF 2026-08-20 §3:
MEASURED per-family command latency — hips 0, knee 10 ms, yaw 10 ms, ankle 20 ms."* Today's note
says 19–21 ms on all ten joints. Both cannot describe the same quantity, so:

- Was the August figure measured **at the actuator** (response after the command arrives), with
  today's 20 ms being the **pipeline** in front of it?
- If so, **do they add on hardware?** That would put the real hips at ~20 ms but the real
  **ankles at ~40 ms** — and "ankles match" would be true of your emulator and false of the robot.
- Or does today's measurement supersede August's outright?

This decides the ankle band. If they add, 2–5 steps is right for the hips and too short for the
ankles. **We would rather know than randomise wide to cover it**: at ankle kd 0.5, our notes record
that 40 ms of delay topples a passive stand in sim.

## 3. The period: extremum timing, and it is consistent with our own peak velocity

Method: a hysteresis extremum detector (an extremum is confirmed only after the signal reverses by
more than 2% of the swing), then `2 × mean(interval between successive extrema)`. **Timing, not
counting** — it is not quantised by the number of reversals.

Three independent estimates from the same trace (`bareplant_off.tsv`, in the August bundle):

| joint | release | equilibrium | amplitude A | v_peak | extrema | zero-crossings | 2πA / v_peak |
|---|---|---|---|---|---|---|---|
| hip_pitch R | 14.10° | +0.34° | 13.76° | 63.5 °/s | **1.360** | **1.360** | **1.361** |
| hip_pitch L | 14.05° | −0.34° | 14.38° | 66.4 | 1.360 | 1.360 | 1.361 |
| hip_roll R | −14.08° | −0.05° | 14.03° | 66.3 | 1.330 | 1.302 | 1.330 |
| knee R | −14.95° | −10.92° | 4.03° | 22.8 | 1.101 | 1.101 | 1.111 |

**The inconsistency you found came from using θ₀ = 15°.** The joint had sagged to 14.10° under
gravity before release, and it swings about +0.34°, not 0 — so the amplitude is 13.76°, and
2π × 13.76 / 63.5 = 1.361 s. The same correction applies to your side: with your frac of
0.920–0.932 your release angle was ~13.9°, which moves your 1.42–1.46 s to ~1.35 s.

That puts all four numbers in one place: **your analytic 1.345 s, your weld reference 1.333 s,
our 1.360 s, and your corrected peak-velocity estimate ~1.35 s.** Inertia is matched, and the
period agrees to ~1–2% rather than being unmeasurable. Note the knee: its equilibrium is at
−10.9°, so its amplitude is 4°, not 15° — the θ₀ assumption fails worst there.

## 4. Inertia — same tensor, but your file holds the diagonal

| | result |
|---|---|
| mass, all 11 links | match (total 13.0600 vs 13.0642) |
| COM, all 11 links | match to **≤ 0.01 mm** |
| inertia **diagonal**, link frame | match to **six digits** |
| inertia **principal moments** | differ up to 9% (yoke −7.8 / +9.2%, femur −8.1%) |

The last row is not a model difference. Your `inertia_principal` values equal our tensor's
**diagonal** exactly (yoke: yours 0.00040574 / 0.00054498 / 0.00058602, ours 0.000406 / 0.000545
/ 0.000586), the traces are identical, and your file reports `iquat = [1,0,0,0]` on all eleven
links — which real CAD parts do not have. Our tensors carry products of inertia: femur
I_yz = 0.000702 against a smallest moment of 0.00185, torso I_yz = 0.0048.

**So the question is whether your MJCF uses them.** If it specifies `fullinertia`, the file is just
an export of the diagonal and nothing differs. If it specifies `diaginertia` with no orientation,
your links have no products of inertia and ours do. Either way the dynamic effect is small — the
bare-plant peaks matched to 2% — so this is for completeness, not a lead.

## 5. Where this leaves us

Agreed and closed: mass, COM, inertia (pending §4's one-line answer), limits, geometry, Coulomb
friction (1–2%), overshoot, gravity sag, leak, free-swing period and peak velocity.

**Open, and it is the only one: latency.** Confirmed on both sides by two methods each. We need
your answer on whether the August per-family figures add to the pipeline figure before we set the
ankle band.

Three of the corrections in this exchange were ours — per-step pinning, a randomised draw read as
the asset, and now an alignment bias. The rule that has caught every one of them, on both sides:
**calibrate the method on a case where the answer is known.** Yaw, with zero configured viscous,
is that case for latency, and it is why we trust the 20 ms.

---

*— training, 2026-09-29*
