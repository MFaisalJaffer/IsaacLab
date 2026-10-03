# TRAINING → RIG · Delay change landed; new lineage training on 10–25 ms, all joints

**2026-09-29** · Confirms the one action left open in `REPLY4_SIM2SIM.md`. Thank you for going
back to the raw traces — that settled it in one message.

---

## 1. What changed

| family | before (from the Aug-20 `lag_ms` column) | **now** |
|---|---|---|
| hip pitch, hip roll | 0–1 steps (0–5 ms) | **2–5 steps (10–25 ms)** |
| hip yaw, knee | 1–3 steps (5–15 ms) | **2–5 steps (10–25 ms)** |
| ankle | 3–5 steps (15–25 ms) | **2–5 steps (10–25 ms)** |

Per-environment, redrawn every episode, uniform over the four values. Ankles were **not**
widened toward 40 ms, per your §2.

## 2. Verified before we stopped the old run

- all ten actuators configured (2, 5); sampled lags observed in [2, 5] steps with four distinct
  values on every joint;
- the ankle series-compliance path draws its delayed command from the same buffer (the path that
  once bypassed the delay entirely — your Handoff 3B caught that one);
- calibration, from our last note: a pinned delay of 0 / 2 / 4 steps reads back as
  0.0 / 10.0 / 20.0 ms by your yaw-slope method, so N steps here is N × 5 ms of real latency.

## 3. How we restarted

**From scratch**, not warm-started: every lineage since the one built on the August table trained
its hips on a plant the robot does not have, so we preferred a clean birth to adapting a policy
around the old timing. Nothing else in the plant or the rewards changed, so the result is
attributable to latency alone.

The previous lineage's best build is archived and available if you want a before/after on the rig:
`l10_candidate_model_100000` — median standing sway 3.4–3.6° at K_s 52/120, 20 N push survival
98.8% both directions, walk width 35 cm. It was trained on the old timing, so treat it as the
"before".

## 4. What we will send next

A candidate once the new lineage clears its gates — all of them now probed **at 20 ms on every
joint** as well as at our nominal: standing sway, stance width, 20 N push in both directions,
walking width and gait quality. We will say which checkpoint, and point at the probe output it
came from.

## 5. Adopted from your note

**"A number labelled *measured* needs to point at the raw trace it came from."** We have taken that
as a rule on our side too: the comment above our delay table now names the traces, the three
cross-checks, and what it replaced and why.

---

*— training, 2026-09-29*
