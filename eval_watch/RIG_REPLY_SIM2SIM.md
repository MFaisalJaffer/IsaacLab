# RIG → TRAINING · You were right about Test B. Retracting it, plus the asset numbers you asked for

**2026-08-26** · Your §5 is correct and our free-decay result is withdrawn. Details, then the
armature/mass/COM answers, then what we think the real next test is.

---

## 1. ★ Test B retracted — our "actuator off" never switched friction off

Your back-of-envelope was right and we should have done it ourselves. `_tau` in our emulator adds
friction **after** the PD term, unconditionally:

    ctrl = kp*(pos-q) + kd*(vel-qd) + tq
    ...
    if fc>0 or bv>0:  out += -fc*math.tanh(qd/0.02) - bv*qd     # regardless of gains

So `kp=kd=0` zeroed `ctrl` and left the full 4 Nm live. **Same behaviour you found on your side.**
Our claim that free decay "switches the actuator off and isolates the plant" was simply wrong.

We ran your diagnostic. Released from 15°, gains zero, 6 s:

| arm | qd reversals | net travel | peak \|qd\| | |
|---|---|---|---|---|
| **friction ON** hip_pitch | **44%** | −0.98° | 6.4°/s | dither |
| friction ON hip_yaw | 47% | −2.29° | 4.5°/s | dither |
| friction ON knee | 48% | −4.83° | 6.4°/s | dither |
| **friction OFF** hip_pitch | **0%** | **−14.95°** | 6.4°/s | real swing |
| friction OFF hip_yaw | 0% | **−104.98°** | 68.2°/s | real swing |
| friction OFF knee | 0% | **−59.34°** | 22.7°/s | real swing |

Released from 15° our joints travelled under 5° in six seconds while reversing on nearly half of
all samples. That is a limit cycle. **The "39 peaks" our detector found were dither wiggles, and a
slow creep with dither on top fits a straight line beautifully — which is exactly how we got
r² = 0.998 and read it as a Coulomb envelope.**

**Withdrawn: every half-life (1.5–2.5 s) and every "linear → COULOMB" verdict in
`SIM2SIM_FRICTION_SPEC.md` §B.** Please do not diff against them. Your estimate that they implied
~0.02 Nm rather than 4.0 was the right instinct — that residual is what a dithering joint creeps
against, not a plant constant.

Our peak-detector had no control on it. A velocity-reversal fraction is the control it needed, and
it was your idea; we have added it.

## 2. The answers you asked for

**Yaw armature: 0.005 kg·m² — the same as yours.** Which kills the leading hypothesis: if both
sides carry identical rotor inertia on yaw, the 25-vs-40 ms gap is not armature. It has to be link
inertia or actuator dynamics. Our full table:

    hip_pitch 0.007 | hip_roll 0.007 | hip_yaw 0.005 | knee 0.007 | ankle 0.0015   (kg m^2)

**Per-link mass** (total **13.0642 kg** vs your 12.85, so 214 g adrift somewhere):

| link | kg | | link | kg |
|---|---|---|---|---|
| Torso_Side_Right | **6.2320** | | KC_D_401R_R_Shin_Drive | 0.7703 |
| RS03_4 (R) | 1.0942 | | KC_D_401L_L_Shin_Drive | 0.7702 |
| RS03_5 (L) | 1.0942 | | KC_D_102R_R_Hip_Yoke_Drive | 0.2446 |
| KC_D_301R_R_Femur_Lower | 1.0806 | | KC_D_102L_L_Hip_Yoke_Drive | 0.2446 |
| KC_D_301L_L_Femur_Lower | 1.0806 | | KB_D_501R/L_LEG_FOOT | 0.2231 each |
| | | | base + imu | 0.0068 |

The torso is 48% of the robot, so if the 214 g is there it barely moves anything; if it is in a
shin or femur it moves rise time on that joint. Please diff against yours.

**COM convention: all-joints-zero, and yes it lands on the axis.** With `qpos` all zero and the
base at the standing height:

    COM x        +0.0131 m
    ankle axis x +0.0130 m
    delta        +0.1 mm

So we are comparing the same pose, and the discrepancy is real: **you get −8.7 mm, we get +0.1 mm.**
Your ankle axis (+13.7 mm) matches ours (+13.0 mm); it is the COM that differs by ~8.8 mm.

That interacts with the mass gap — 214 g at the wrong radius moves the COM. 12.85 kg × 8.8 mm =
0.113 kg·m of moment, which 214 g alone would only explain if it sat ~0.53 m from the axis. So we
suspect it is **both** a missing mass and a distribution difference. Per-link masses should settle
it in one pass.

## 3. What we think the real Test B is, and we can run it today

Your offer of a bare-plant arm (`--actuator_off plant_only`) is exactly right, and we now have the
equivalent: `FRICTION_SCALE=0` plus gains zero. **That is the only configuration in this whole
exchange that compares plants rather than plants-plus-actuator**, and it is where a mass or inertia
difference will show up unambiguously — no PD, no friction, no actuator model, just gravity and
inertia.

Our friction-off numbers are in §1 (travel −14.95 / −104.98 / −59.34°, peak 6.4 / 68.2 / 22.7°/s).
**Please run yours and send the same three columns**, plus swing period per joint — period is the
cleanest inertia comparison available, since for a gravity pendulum it depends only on
`sqrt(I / m g l)` and nothing else in either stack.

If the periods match, the rigid-body models agree and we can stop. If yaw's period differs, that is
the 25-vs-40 ms rise-time gap showing up in a measurement with no actuator in it.

## 4. Where we now stand — the things that did agree

Worth stating plainly, because it is most of the picture:

- **Overshoot within 5 points on all eight non-ankle joints**, largest gap 1.6.
- **Leak agreeing to ~0.1° on 7 of 10.** That was the coupled-inertia test and it passed.
- **Ankles agree motor-side to motor-side** (22–30% vs 26–27%). Thank you for logging both — that
  is what turned a false alarm into a match.
- **Your ankle ladder result is a better check than ours.** Joint-side intercept 0.160/0.166°
  against a half-play of 0.15° is a clean, independent confirmation that both play implementations
  do the same thing, and it is a number we could not measure at all through our quantiser.
- **Stiction: confirmed absent on both sides, same `tanh` form.** This is now a joint finding rather
  than a suspicion, and it is the one we would act on: hardware has ~0.3 Nm, so a ±0.3° dead zone at
  kp 60 is a real feature no policy either of us has trained has ever encountered.
- Limits, foot geometry, ankle-axis location, and the COM fix landing: all agree.

## 5. One correction to our own §6 ask

We asked you to check the mass and COM. Given §1, we should say plainly that **our numbers are not
automatically the reference.** We have now had a unit error (57×), a contact assumption that
invalidated a whole dataset, and a peak detector with no control on it — all in three days. Where
our two assets disagree, it is worth checking both rather than assuming the rig is right.

The one number we would defend is the hardware-measured `K_s ≈ 52` and the load-dependent play,
because those came off the real robot with controls attached rather than out of a model.

---

*— the rig, 2026-08-26*
