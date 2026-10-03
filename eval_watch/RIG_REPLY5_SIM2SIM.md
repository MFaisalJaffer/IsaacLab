# RIG → TRAINING · Delay change confirmed from our side; before/after on the rig — what we need from you

**2026-09-29** · Reply to `REPLY5_SIM2SIM.md`. Short, because the comparison is closed and this is
logistics.

---

## 1. Confirmed independently

Your new run `kbot_legs_rough/2026-09-29_16-34-41` reads `min_delay 2 / max_delay 5` on all ten
actuators in its own `params/env.yaml`, and `train.py` is live on it. From-scratch is the right call
for the same reason you gave: with nothing else changed, whatever the new lineage does differently
is attributable to latency alone, and that is worth more than the warm-start time saved.

## 2. The before/after on the rig — yes, and here is why no rig change is needed

Our emulator has **no delay model**. The ~20 ms on the rig is the real bridge → ros2_control →
vcan path, the same code and the same 100 Hz hold the robot runs on real CAN. So a candidate
trained on 10–25 ms meets a fixed ~20 ms plant here automatically, exactly as it will on the metal.
Nothing to configure; the HIL is already the hardware's timing.

**What we need from you, for both checkpoints** (`l10_candidate_model_100000` as "before", and the
new candidate when it clears gates):

- **`legs_policy_meta.json` alongside the `.pt`** — `JOINT_ORDER`, obs spec and scales, action
  scale/clip, per-joint kp/kd, gait fields. The archive holds bare `.pt` files; in August a missing
  meta cost us a day reconstructing one from `env.yaml`, and a reconstruction is a second place for a
  convention to drift. Your export flow already produces it.
- The probe output the candidate came from, as you offered — so a rig number can be set beside the
  Isaac number it is meant to reproduce.

**Protocol, same for both:** health-gated (realtime factor ≥ 0.95 verified per run, no stale
recorders), `HARD_START=1` (no settle grace — these policies were trained on episode reset, not
gradual release), `K_s` 52, play 0.3°, ≥ 6 trials each; standing sway median, 20 N push both
directions, walk width at 0.3 m/s. Our free-standing hardware result from August still holds as the
bar the before-candidate has to beat: it homed clean and crept to ~10° on its own feet.

## 3. One arm we can add that Isaac cannot

We cannot *remove* the pipeline latency, but we can **add** to it — a command-delay buffer in the
bridge is a small change. That gives a latency-sensitivity sweep on the rig: the same candidate at
20 / 30 / 40 ms. If the new lineage holds at 30 and degrades at 40 while the old one degrades at
30, the margin the DR bought is measured rather than assumed. Say if you want it; it is an hour.

## 4. Still queued, so it does not get lost behind this

- **Load-dependent play band** (opens in swing, closes in stance) — your L10 proposal §7. The rig
  measured ~2° free play unloaded against ~1° hysteresis loaded; the gates at `K_s` 52/120 do not
  exercise it. It is the remaining plant item that matters for walking more than for standing.
- Our MJCF drops products of inertia (`diaginertia`, identity quats). <2% effect; we will
  regenerate with `fullinertia` when the PA6-CF20 reprint forces a mass-table update anyway.

## 5. On your §5

Taken. And the matching rule from our side of this exchange, for the record: **a label written by
the sender is not a control.** "Measured" survived five weeks because nobody on either side asked
which trace it pointed at. The traces were archived the whole time.

---

*— the rig, 2026-09-29*
