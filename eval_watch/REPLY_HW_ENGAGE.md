# TRAINING → RIG · reply to `RIG_HW_ENGAGE_FINDINGS.md` (first hardware engage of walker_v5_3200)

**2026-10-03.** Thank you for the write-up and the data — it was enough to reproduce everything on our side
without asking a question. Short version: your diagnosis holds on our weights and in our simulator, three of
your four asks were real gaps in what the policy had seen, and the next bundle will not ship unless it passes
your toy test and our engage test under your watchdog limits.

## 1. What we verified

**Your toy-plant test, our weights** (`rig_hw_engage/cf_tracking.py` on `walker_v5_3200`): L ankle action at tick 20
for 0 / 20 / 50 / 100 % response = −0.36 / −0.23 / −0.12 / −0.04. Identical to your numbers.

**Isaac reproduction** (`eval_watch/amp_engage_test.py`): robot at rest at the zero pose, stand command and hard
phase pin from tick 0, nominal plant, ankle spring 52 Nm/rad with 1° play, 32 robots per condition, constant moment
on the torso as the only load:

| condition | ankle action @ 0.4 s (L / R) | peak joint speed (2 s) | peak tilt | fell (6 s) |
|---|---|---|---|---|
| no load, hard start | −0.02 / −0.03 | 2.9 rad/s | 2.9° | 0 % |
| no load, 1 s ramp (targets 0→1, gains 0.5→1) | −0.08 / +0.09 | 7.7 rad/s | 5.5° | 0 % |
| −2.5 Nm pitch, hard start (your F) | −0.14 / +0.10 | 11.0 rad/s | 5.6° | 0 % |
| **−2.5 Nm pitch, 1 s ramp (your E′)** | **−0.58 / +1.33** | **21.6 rad/s** | 9.8° | 9 % |
| +2.5 Nm pitch, 1 s ramp (your E) | +0.58 / −0.73 | 4.2 rad/s | 2.5° | 12 % |
| +2.5 Nm pitch, hard start | +0.07 / −0.15 | 3.1 rad/s | 2.7° | **16 %** |
| hardware | −1.42 / +1.25 | 15–22 rad/s | 18° | power cut |

Two things beyond your table: (a) in our plant the hard start is not fully quiet under the load (11 rad/s peak),
and (b) the load in the *other* direction makes 16 % of the robots fall within 6 s even with a hard start. So
removing the crossfade helps, but this checkpoint is not safe against the standing load itself. **Please do not
re-engage walker_v5_3200 on hardware.**

## 2. Your asks

| ask | what training had | what the next run has |
|---|---|---|
| 1. standing load | torso CoM ±1.5 cm fore-aft, ±1 cm lateral (≈ ±0.9 / ±0.6 Nm with our 6.2 kg torso) — far below your 2.5 Nm | constant base moment, pitch U(−3, 3) and roll U(−3, 3) Nm, drawn per episode, held all episode. (We apply the moment in Nm rather than a CoM shift: with a 6.2 kg torso, ±3 cm is only ±1.8 Nm.) |
| 2. ankle series stiffness | per-episode log-uniform 20–120 Nm/rad already; but the meta declared 23 as deploy truth | nominal 52, band 30–120; the meta will say 52 |
| 3a. unanswered commands — stiction | friction was `−Fc·tanh(q̇/0.02)` on the link side only (zero at rest); the ankle **rotor had no friction at all** | ankle rotor Coulomb friction with a true static regime, U(0, 1.5) Nm per episode; output dead band U(0, 1) Nm on hips, knees and yaw |
| 3b. unanswered commands — weak start | gains constant over the episode (±10–20 % per-episode DR) | on half the episodes: all gains × U(0.3, 1), ramping to 1 over U(0, 1) s after reset |
| 4. gravity noise | ±0.05 uniform on projected gravity, plus IMU mount rotation ±0.1 rad per episode | unchanged |

One more gap we found ourselves: **no training episode ever began standing.** Every spawn was converted to a
walking command (a from-scratch workaround from August), so "engaged at the zero pose with cmd = 0 from tick 0"
— exactly what you do — was outside the data. The next run keeps the stands drawn at spawn (30 % of episodes),
under the hard pin, with the load, the stiction and the weak start above.

Unchanged: observation layout (430 = 10 × 43), signed clock at 0.9434 Hz, gains, action scale and clip table,
delay band 2–5 steps, command envelope. `RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md` stays valid as is.

## 3. What gates the next bundle

Every checkpoint is tested, and a bundle ships only if:

1. your `cf_tracking.py`: the 0 %-response ankle action at tick 20 stays near the 100 %-response one;
2. our engage test, 15 conditions (load ±2.5 Nm in pitch and roll × hard start / 1 s ramp, rotor stiction 1.2 Nm,
   dead band 0.7 Nm, combinations): **no fall, joint speed < 5 rad/s, tilt < 12°** in the first 2 s — your
   watchdog limits;
3. the walking battery as before (six directions, step height counted).

The numbers go in the bundle README. walker_v5_3200 scores 7 / 15 on (2).

## 4. Questions

1. The left hip roll carries 2.3 Nm and the right 0.25 Nm at home. Is that asymmetry the robot (mass / cable
   routing) or the stance on that day? We cover it with the ±3 Nm roll moment either way.
2. At engage you send cmd = (0, 0, 0) and the hard pin (π, π) from tick 0, joints at home — correct? That is
   what we now train and test.
3. Knee breakaway when you have it (your next motors-on session) — we used U(0, 1) Nm like the hips.
4. You removed the crossfade for this lineage. We train the weak start anyway; is there any other soft-start
   left in the bridge (gain ramp in the drives, command shaping — your meta says `cmd_shaping: true`) that the
   policy should have seen?

*— Isaac side, 2026-10-03*
