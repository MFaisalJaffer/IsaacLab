# RIG → TRAINING · Hardware session 2 (2026-10-04): the engage is solved; the stand is not — and our sim does not reproduce it

Follow-up to `RIG_HW_ENGAGE_FINDINGS.md` and your correction. We took your withdrawn "do not re-engage" as
"your call" and ran walker_v5_3200 again with the crossfade removed, a detached watchdog (12° / 8 rad/s, your
suggested limit), CAN-liveness in the feeder, and the IMU re-zeroed. Three engages. Data in `rig_hw_stand/`.

## 1. What happened

| engage | conditions | result |
|---|---|---|
| #2 (02:09) | no crossfade; IMU zeroed at a stance with 1.3 Nm on the ankles | operator cut power at 0.46 s ("moved, not violent"). Ankles ±0.46 action at 0.2 s then reversed; torso 1.6° → 0.5°; plant answered. **No wind-up.** |
| #3 (02:30) | same; IMU re-zeroed at the *balanced* stance (ankle hold torque 0.1 / 0.04 Nm); shadow first | **stood 1.9 s**, then the watchdog tripped (R ankle 17.4 rad/s, torso 0.8°) and latched limp |

Engage transient in #3 (0–0.6 s): ankle action peaks ±0.45 at 0.18 s, body 1.6°, settled to ~0 by 0.5 s; hips
±0.1, knees on the stop. This matches your corrected table (hard start under load: no wind-up, tilt 2–3°).
**The crossfade diagnosis and its fix hold on the robot.**

Then, 0.6–1.8 s, a rocking grew:

| | hardware #3 |
|---|---|
| axis | **pitch** (3.1° p2p; roll 1.1°) |
| frequency | **1.9–2.1 Hz** (torso peaks at 0.86 / 1.32 / 1.80 s) |
| growth | torso peaks 1.8° → 1.5° → 2.8°; R-ankle action 0.64 → 0.83 → 1.43 |
| ankle motor travel vs body | 14–16° encoder p2p for 3.1° of body pitch (ratio 5.3) |
| ankle torque | R mean +3.2 Nm, 14.8 Nm p2p; L mean −1.3 Nm, 9.0 Nm p2p; pitch moment (L−R) 22.4 Nm p2p, reaching 15 Nm one way |
| series stiffness seen | 46 Nm/rad per ankle (torque vs encoder−body deflection, r = 0.93) — consistent with our 52 |
| asymmetry | R ankle action 2.6× the L; L hip roll holding 2.3 Nm throughout |
| end | R ankle held at −0.09 rad against a −0.23 command with 11.8 Nm for 60 ms; policy bends the R knee (−0.42 rad, 10.9 rad/s); ankle releases at 13–17 rad/s → watchdog |

## 2. Our sim does not reproduce it

Same checkpoint, hard start, no crossfade, 12 s stands on the MuJoCo rig:

| variant | result |
|---|---|
| stock (K_s 52, play 0.3°) | stands; 1.0° slow drift, no oscillation |
| play 2.0° | stands; 1.5° slow drift |
| play 2.0° + K_s 23 | stands; sags to a 4° lean, slowly; no 2 Hz |
| play 2.0° + 2 Nm roll moment (left-heavy) | stands; 1.6° |
| play 2.0° + −2.5 Nm pitch moment | stands; 1.3° |
| play 2.0° + 1 Nm ankle rotor stiction | stands; 1.2° |

In every variant the ankle/body travel ratio is 1.9–3.9 and the ankle torques stay within 5 Nm p2p; nothing
near 2 Hz. So the destabilising mechanism is something neither the rig emulator nor (we assume) your plant
contains. Candidates, in the order we believe them:

1. **Foot support polygon.** Open since August: the model foot is 21.2 cm (6.4 cm heel-side, 14.8 cm toe-side
   of the ankle axis), giving static ankle-moment limits of 8.2 Nm (heel) and 19 Nm (toe) at 13.06 kg. The
   real robot has always tipped far more easily than that foot allows. Tonight's rocking drove the ankle
   moment to 15 Nm one way and 7 Nm the other — at or past those limits even for the model foot. If the real
   foot is shorter, the robot was rocking over its foot edges, where ankle torque stops controlling the body.
   **We are measuring the real foot (length, heel→axis, axis→toe, width) before anything else.**
2. **True backlash vs dead band.** Our emulator's play is a dead band inside the spring. A real gear/linkage
   gap is hysteretic: while the motor crosses it the body coasts, which is phase lag, and a 2 Hz limit cycle of
   about the gap's size (±1–1.5° here; play ~2° unloaded) is the textbook result.
3. **Uneven foot loading.** The robot is left-heavy in every stance tonight (L hip roll 1.3–2.3 Nm, R 0.2–0.4
   over four stances — so, your Q1: it is the robot, not the stance). The right foot is lightly loaded and its
   ankle does 2.6× the work for the same body motion.
4. Drive-side damping: the drives' kd acts through a firmware velocity filter; the real damping of the 2 Hz
   mode may be lower than kd = 0.5 implies.

## 3. What we learned about the zero pose (relevant to your stand-from-spawn episodes)

With all joints at zero on a flat floor the torso can sit anywhere in a ~2.3° band — the ankle's free play plus
spring. Our first IMU zero tonight was taken with the body resting 1.3 Nm forward on the ankle springs; the
second with the ankles unloaded (0.1 / 0.04 Nm). Shadow runs at the two: the policy's open-loop ankle demand at
0.3 s was −0.33 / +0.41 at the first and −0.09 / +0.10 at the second. **"Upright" for this robot must mean
"ankle torque ≈ 0", not "joints at zero"**; in your stand-from-spawn episodes the body should start anywhere in
±1.5° of pitch at zero joint angles.

## 4. Asks

1. Can your plant show a 2 Hz pitch rocking in a hard-start stand under any of: hysteretic ankle backlash of
   1–2°; a short foot (support 4–6 cm heel-side, 8–10 cm toe-side); ankle kd × 0.3–1; a 2 Nm lateral load with
   one foot at 30–40 % of body weight? Whichever does is the thing to randomise.
2. A stand-stability gate beside your engage test: 20 s hard-start stand with those effects on, pass = torso
   oscillation at 1–4 Hz stays under 1° p2p and does not grow.
3. Your §2 item "ankle rotor friction with a true static regime" — keep it; it is not what failed tonight
   (rotor breakaway on the robot is 0.6–1.2 Nm and the sim with 1 Nm stood), but the wind-up test still needs it.

We will send the foot measurement as soon as it is taken, and a true-backlash emulator result after that.

## 5. Rig changes since the last note

Crossfade removed (default); detached watchdog — it did the stopping in #3; feeder pauses on CAN silence (caught
both operator power cuts in 0.3 s); bridge IMU QoS fixed; IMU zeroed on the balanced stance (2.35°); a
one-command CAN recovery for the Jetson driver after a motor power cut.

Files in `rig_hw_stand/`: `hw_stand3_ep_20261004_023018_92t.npz` (the 1.9 s stand, with CAN wire tap),
`hw_engage2_ep_20261004_020950_39t.npz`, `hw_shadow_balanced_ep_20261004_022556.npz`, `rock_analyze.py`,
`plant_id.py`, `stand_matrix.py`.

*— the rig, 2026-10-04*
