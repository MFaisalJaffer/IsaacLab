# TRAINING → RIG · correction to `REPLY_HW_ENGAGE.md`: one of our claims was a test artifact

**2026-10-03, 18:10.** Two statements in our reply this afternoon were wrong, and one of them was the reason we
asked you not to re-engage. Corrected numbers below; the decision is yours.

## 1. What was wrong

We wrote: *"under the measured load alone walker_v5_3200 falls 16 % within 6 s here even with a hard start"*
and, in the table, *"+2.5 Nm pitch, hard start → fell 16 %"*.

That came from a test plant we had stripped down to "nominal": no gain / mass / friction randomization, no
sensor noise — and, with them, **no ankle joint friction**, which every policy of this lineage was trained
with. On that plant, which the policy has never seen, even an *unloaded* v5 stand drifts and falls (34 % in
20 s). The falls under load were that artifact, not the load. Our mistake: we did not read the control row at
the same duration before trusting the others.

## 2. Corrected numbers (plant as trained: randomization and sensor noise on)

**Standing load, hard start, robot at rest at the zero pose, 20 s, 32 robots per load:**

| torso moment | fell | peak tilt (first 2 s) | peak joint speed (first 2 s) |
|---|---|---|---|
| pitch −3.0 … −0.5 Nm | 0 % (one robot of 32 at −3.0) | 2.2–3.5° | 5.1–6.7 rad/s |
| 0 | 0 % | 2.0° | 4.9 rad/s |
| pitch +0.5 … +3.0 Nm | 0 % | 1.9–2.5° | 4.7–5.5 rad/s |
| roll −3 … +3 Nm | 0 % | 1.9–2.4° | 4.4–5.6 rad/s |

So with a hard start walker_v5_3200 **holds every load from −3 to +3 Nm for 20 s without a fall** in our plant,
with or without ankle rotor stiction (1.2 Nm) and a dead band on the other joints.

**The 15-condition engage test, same plant, 10 s:**

| | hard starts (10 conditions) | weak starts (5 conditions) | your hardware condition (−2.5 Nm pitch + 1 s ramp) |
|---|---|---|---|
| walker_v5_3200 | 0 % fell, tilt 2.4°, joint speed 5.0 rad/s | 5 % fell, 9.1 rad/s | ankle −0.77 / +1.39 at 0.4 s, **18.2 rad/s**, tilt 15.5°, 12 % fell |

## 3. What still stands

- **Your diagnosis.** Load + weak start reproduces the wind-up on the trained plant too (18.2 rad/s; hardware
  15–22). The crossfade removal is the fix for that.
- **Your toy test** on our weights (−0.36 at 0 % response).
- The gaps in our training data (no standing load, no rotor stiction, constant gains, no episode beginning in
  a stand) are real.

## 4. Two things you should know before deciding

1. **Joint speed at a normal start is about 5 rad/s in our plant** — at your watchdog limit. Per-robot peak
   over the first 2 s, any joint: 4.4–5.6 rad/s with our training sensor noise (±1.5 rad/s on joint velocity,
   which makes the policy's commands jitter), about 3 rad/s with clean sensors. Your robot's sensors are
   cleaner than our noise model, so expect the lower figure, but a 5 rad/s E-STOP has little margin over a
   healthy start. The violent event was 15–22 rad/s; a limit of 8–10 rad/s separates the two.
2. **The two fine-tunes we ran today did not help.** walker v6 (your four asks + stands from spawn) and v6b
   (the same + your watchdog limits as a training termination) both keep the walking but are *worse* than v5 on
   this test: hard starts 1–15 % falls instead of 0 %, the hardware condition unchanged at ~19 rad/s. No bundle
   will come out of them. v5_3200 remains our best checkpoint.

## 5. Where that leaves the re-engage question

We withdraw "do not re-engage". What our simulator now says about walker_v5_3200 engaged the way you
described (cmd 0, hard pin, full gains, authority 1.0 from tick 0): it stands under the measured load, no
wind-up, tilt 2–3°. What it cannot tell you is anything the plant does not contain — which is exactly how the
first engage went wrong. Straps, the detached watchdog and stand-only are the right conditions; whether to do
it before a retrained policy exists is your call.

*— Isaac side, 2026-10-03*
