# TRAINING → RIG · walker v8 is finished: no new bundle, and two things we told you turned out wrong (2026-10-05)

Follow-up to `REPLY7_IMU_FRAME.md` and a correction to `REPLY6_STAND4_REVERSAL.md` §3. Nothing for you to do.

## Result

Walker v8 was walker v7 again (same start, recipe, length and tests) with the mirror-symmetry term corrected.

| | v7_800 (your bundle) | v8, best (iteration 800) | v8, last (1999) |
|---|---|---|---|
| score (walking × sensing) | 0.81 | 0.78 | 0.75 |
| walking | 0.81 | 0.85 | 0.75 |
| stand under your sensing cases, all robots up | 12 / 12 | 11 / 12 (one robot of 32) | 12 / 12 |
| load + weak start: peak joint speed, fell | 16.0 rad/s, 9 % | 13.0 rad/s, 0 % | 13.8 rad/s, 0 % |
| your toy test, no response | −0.62 | −0.33 | −0.11 |
| left–right error of the network, walking (rms of target) | 4.3° | 2.2° | 2.3° |
| stand pose, L + R of the hip-yaw targets | −5.7° | −8.6° | −2.0° |
| stand pose, L + R of the hip-pitch targets | −3.1° | −1.0° | −1.5° |

So: about equal. One run each, and differences of this size are within what two runs of the same recipe can
differ by. **v7_800 stays your bundle; we are not posting a v8 bundle.** The corrected term stays in every
future run.

## What we told you that did not hold

- **"Nothing pushed the policy toward left-right symmetry, which fits the lopsided stand"** (`REPLY6` §3). The
  corrected term does make the network about twice as symmetric (4.3° → 2.2°), but **v8 still stands twisted**: the
  hip-yaw sum was −6.9 / −8.6 / −8.0 / +4.1 / −2.0° at its five checkpoints — every robot of a checkpoint to the
  same side, and the side changes between checkpoints. So the twist is not the old term's doing. Our reading now:
  nothing in the rewards holds the pelvis square over the feet at the stand, and a small bias picks the side. Only
  the hip-pitch part shrank.
- **"Of the IMU it uses the rate, hardly the lean — whether that is the term's doing we do not know yet."** Now we
  know: it is not. v8's ankle targets move 0.3–0.5° per degree of lean (v7_800: 0.3°) and about 40° per rad/s of
  pitch rate (the same). Your two observations stand as a property of these policies, with either term.

Two more checks, for the record: on our stripped plant (no randomization, no sensor noise) and under sideways loads
of 5–14 Nm on each side, v8 is no better and no more even than v7_800, which was already even left to right.

## Next

Unchanged: we wait for your two-ankle loops and stand #5 on v7_800.

*— Isaac side, 2026-10-05*
