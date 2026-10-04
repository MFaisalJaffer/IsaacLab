# TRAINING → RIG · reply to `RIG_HW_STAND_FINDINGS.md` + `RIG_HW_STAND_ADDENDUM_SENSING.md`

**2026-10-03 21:45 (your 10-04).** Your ask 2 first, because it is the one with an answer today: **yes — Isaac
reproduces it.** walker_v5_3200 stands with fresh sensing, loses balance with the robot's sensing as it was, falls
off a cliff with one more tick of IMU delay, and is back to normal with your fix. Second, independent
confirmation of your diagnosis.

## 1. Isaac, walker_v5_3200, hard-start stand with the robot's sensing

`eval_watch/amp_stand_sensing_test.py`. Robot at rest at the zero pose, cmd 0 and hard pin from tick 0, authority
1.0; our plant as trained (gain / mass / friction randomization and sensor noise on), ankle pinned to your
K_s 52 / play 2°. The policy's 430 input is rebuilt from the env's fresh frames through a sensing model per group
(IMU sample-and-hold with a random phase per robot, IMU delay, joint delay, the tick-37 stall). 32 robots per
row, 10 s, then a 5 Nm × 0.25 s pitch kick. "Lost balance" = torso tilt > 12°.

| sensing | lost balance in 10 s | torso pitch, fast part: rms / p2p | ankle encoder rms L / R | ankle torque p2p L / R |
|---|---|---|---|---|
| stock (fresh, no delay) — **control** | **0 %** | 0.41° / 1.9° | 2.4° / 3.1° | 12.1 / 13.9 Nm |
| IMU 20 Hz only | 0 % | 0.57° / 2.9° | 3.2° / 3.8° | 16.1 / 17.1 |
| **robot as measured** (IMU 20 Hz + 1 tick, joints + 1 tick) | **16 %** (median 2.6 s) | **0.76° / 4.3°** | 4.3° / 5.0° | 19.0 / 18.8 |
| as measured + 0.12 s stall at tick 37 | 12 % (3.2 s) | 0.84° / 5.0° | 4.6° / 5.2° | 19.3 / 19.4 |
| as measured + ankle rotor stiction 1 Nm | 16 % (4.0 s) | 0.95° / 5.8° | 4.6° / 4.8° | 19.1 / 18.7 |
| as measured + stall + stiction | 12 % (2.4 s) | 1.00° / 6.0° | 4.9° / 5.3° | 19.6 / 19.2 |
| **IMU 20 Hz + 2 ticks**, joints + 1 tick | **78 %** (2.9 s) | 0.94° / 5.9° | 4.5° / 5.4° | 19.1 / 19.3 |
| **your fix** (fresh IMU + 1 tick, joints + 1 tick) | **0 %** | 0.45° / 2.1° | 3.0° / 3.7° | 14.1 / 16.1 |
| your fix + ankle rotor stiction 1 Nm | 0 % | 0.45° / 2.2° | 2.8° / 3.6° | 12.9 / 14.8 |
| fresh IMU + 2 ticks, joints + 1 tick | 3 % | 0.59° / 3.3° | 3.7° / 4.2° | 16.7 / 17.5 |
| IMU + 2 ticks, joints + 2 ticks (40 ms constant) | 9 % (3.8 s) | 0.78° / 5.5° | 4.5° / 5.2° | 19.1 / 20.5 |
| stock + ankle rotor stiction 1 Nm | 0 % | 0.39° / 1.9° | 2.3° / 2.9° | 11.2 / 13.2 |

Dominant frequency of the fast part: 1.0–1.6 Hz in every row (yours 1.5 Hz in sim, ~2 Hz on the robot).
After the kick the "as measured" rows stay at 0.8–1.0° rms for the rest of the run and lose another 7–19 % of
the robots; the control and "your fix" rows return to their background.

Reading it against yours:
- **Same ordering, same cliff.** As-measured sensing roughly doubles the rocking and tips some robots; one more
  tick on the held IMU is fatal; a fresh IMU at one tick is indistinguishable from stock. Your "≤ ~26 ms stands,
  ≥ ~36 ms does not" matches our "1 tick fine, 2 ticks degraded (3–9 %), held + 2 ticks 78 %".
- **Our background is noisier than yours** (control 0.41° rms, 12–14 Nm p2p, against your 0.07° and 4–5 Nm):
  that is our training sensor noise (±1.5 rad/s on joint velocity, ±0.05 on gravity) exciting the policy, not a
  plant difference. So compare rows within a table, not ours against yours in absolute terms.
- **Rotor stiction does not tip our as-measured case** the way it tips yours (16 % with or without). Ours is a
  Coulomb friction with a true static regime on the rotor; if yours is `ANKLE_ROTOR_FC` as a tanh, that may be
  the difference — worth a line when you next touch the emulator.
- **Watchdog note.** We could not use joint speed as a trip criterion here: our virtual ankle rotor reaches
  ~10 rad/s at a normal start in this plant (link joints ~5), so a speed limit trips the control. On your
  robot engage #3 did not trip 8 rad/s, so your motor side is slower than our rotor model — we are not asking
  for a change, only saying that our rotor speed is not a number to compare.

## 2. Your asks

| ask | answer |
|---|---|
| observation delay in training: joints 0–1 step, IMU terms 1–3 steps, sample-and-hold | **Agreed, and it is the next run's single change.** You are right that our env has actuator delay only (2–5 × 5 ms); observations are fresh every step. |
| does 3200 rock in Isaac with the robot's sensing | yes — §1 |
| one held action per episode (5–6 steps at a random time) | will be in the same run, as a hold of the joint targets at the actuator (the plant ignores new commands for 5–6 steps). The policy keeps running in training, so its `last_action` shows its own ignored outputs rather than the held one — not identical to a frozen server, the closest we can do per-robot. |
| stand-stability gate (1–4 Hz torso oscillation under 1° p2p, not growing) + kick | the test above becomes the gate. With our sensor noise the control already sits at 1.9° p2p, so we will gate on "no worse than the control row" and "no robot loses balance" rather than on an absolute 1°. |
| hysteretic ankle backlash 1–2° | not yet. Our play is a dead band inside the spring, like yours. We would rather add it after the sensing run so each change can be read on its own. |
| ankle rotor friction with a true static regime | exists (`_rotor_fc`), off in training so far; see the note on yesterday's runs below. |
| zero pose: start anywhere in ±1.5° of torso pitch at zero joint angles | noted for the stand-from-spawn episodes. |
| short foot | dropped, as you asked. |

**Yesterday's fine-tunes, for the record.** v6 (standing load ±3 Nm, dead band, rotor stiction, weak-gain start,
stands from spawn) and v6b (the same + your watchdog limits as a training termination) both came out *worse*
than v5 on the engage test (hard-start falls 3–15 % against 0 %). We never separated which ingredient hurt, so
none of them goes into the next run except what this note is about. Your session-2 result says the engage is
already solved by the hard start, so nothing is lost by that.

## 3. Next run (not started — waiting for the operator's go)

walker_v5_3200 fine-tuned with **one** change: the sensing path.
- joints (position, velocity): delay 0–1 policy step, per episode;
- IMU terms (projected gravity, gyro): delay 1–3 steps, and a new sample only every 1–3 steps, per episode;
- one hold of the joint targets for 5–6 steps at a random time in the episode.

Gates for a bundle: walking battery as before; engage test (plant as trained); this stand test — every row
through "IMU 20 Hz + 2 ticks" must keep every robot up, with rocking no worse than today's control.

## 4. Questions

1. After your driver fix, is the IMU age steady (≈ 30 ms) or does it jitter between 20 and 40 ms from tick to
   tick? We will train a per-episode constant delay unless you tell us it jitters.
2. The 14 ms on the joints: constant?
3. Is anything else on the observation side sampled slower than the 50 Hz loop — the command, the clock?

*— Isaac side*
