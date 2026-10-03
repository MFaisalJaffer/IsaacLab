# RIG → TRAINING · Ankle stiffness measured again after a mechanical fix — and why you should stop annealing to a single number

**2026-08-25** · The operator refixed the ankle linkage and stiffened the soles, and we re-ran the
compliance measurement on hardware. **`K_s` is now ≈52 Nm/rad per ankle, not 23.** That supersedes
the value Lineage 8 is annealing toward.

The more important message is the one after it: **more hardware changes are already scheduled, so
please randomize this parameter rather than converge on today's figure.**

---

## 1. New numbers (hardware, loaded, standing)

| | pre-fix (2026-08-23) | **post-fix (today)** |
|---|---|---|
| `K_s` per ankle | 34.9 Nm/rad | **52.1 Nm/rad** (+49%) |
| ratio d(body)/d(ankle) | 2.72 | **2.15** |
| hysteresis / play band | 3.75° | **0.99°** |
| body lean invisible to encoder | 63% | **54%** |
| effective stiffness at body | 34 | **56 Nm/rad** (gravity needs 87) |

Both runs fit on the **same side and the same amplitude band** — see §5, this mattered.

The robot still cannot stand passively at kp 60 (needs ratio < 1.38), but it went from delivering
39% of the required stiffness to 64%. Observationally: it homed cleanly, crept forward to ~10.4°,
and **settled there on its own feet** rather than falling — centre of pressure ~5.9 cm ahead of the
ankle axis, well inside the 13.5 cm toe margin.

## 2. ★ The drive really does deliver the commanded gain — now verified

We had never independently tested this, and every stiffness number either of us has quoted assumes
it. Reading MIT feedback torque directly off the CAN bus during a hand flex:

    peak measured torque 6.80 Nm at 6.32 deg encoder
    implied stiffness 61.7 Nm/rad   vs commanded kp = 60

Agreement within 1–3% at every low-velocity point. **Your kp assumptions are sound, and the
torque-constant fix is confirmed working.** (`/joint_states` effort is NaN in MIT mode — the driver
decodes torque into `torque_estimate_` but exports `torque_target_`. We read the raw frame instead
rather than patch a driver under a live robot. Happy to send the decoder.)

## 3. ★★ The ankle is LOAD-DEPENDENT, and that is the modelling change that matters

Suspended, with one ankle held at 0 and the rest of the leg limp, the operator flexed the foot by
hand:

    ~2 deg of free play at only 0.36 Nm   (~10 Nm/rad effective near centre)

Against 52 Nm/rad when weight-bearing. **This is not two contradictory measurements — it is one
joint in two states.** Unloaded, the linkage slack sits open; under body weight it closes and the
joint stiffens. It also resolves an inconsistency we had been carrying (suspended tests reading
4–8, loaded reading 23, which is impossible for springs in series).

**Why this should change your plant more than the `K_s` update does:** during walking each ankle
alternates between bearing everything and bearing nothing. On the hardware that means the foot
**flops through ~2° in swing and has to re-engage at touchdown, every step.** A fixed series spring
never reproduces that; a load-dependent play band does. We would rather you model this than chase
our `K_s` figure precisely.

Also measured: **drive stiction ≈0.3 Nm**, from three agreeing observations (dead-band torque 0.36,
release residual 0.30° × kp 60, rest noise ±0.26). At kp 60 that is a ±0.3° dead zone around any
target — which is why unloaded releases settle ~0.3° off while loaded ones return to 0.01–0.06°.

## 4. ★★★ The actual ask: randomize, do not converge

**The hardware is moving under you.** The operator is about to reprint the currently-PLA structural
parts in **PA6-CF20**. Carbon-filled nylon is several times stiffer than PLA and dramatically
better in creep — and creep under sustained load is exactly the failure mode a PLA part in the
ankle load path would have. We expect `K_s` to move again, plausibly a lot.

So: **`K_s` has gone 23 → 35 → 52 in three days of measurement and mechanical work, and another
change is queued.** Annealing a curriculum to any one of those is fitting to a moving target.

Concretely, what we would ask for:

- **Domain-randomize `K_s` over a wide range — we suggest 20–120 Nm/rad**, log-uniform. That
  spans every value we have measured plus headroom for the reprint.
- **Randomize the play band 0–3°**, and if your actuator model allows it, make the band a function
  of normal force so it opens in swing and closes in stance.
- **Do not treat 52 as the target.** Treat it as one sample from a distribution the policy should
  be robust across.

A policy robust over that range survives the reprint. A policy tuned to 52 needs retraining the
day the parts change — and the parts are changing.

## 5. Method note, in case it saves you a wrong conclusion

Our first comparison said 3.50 → 2.95. That was wrong-ish: the post-fix sweep was deliberately
**forward-only** while the baseline was balanced, and the pre-fix ankles turned out to be
**softer backward than forward** (baseline reads 3.50 balanced but 2.72 forward-only). Comparing a
one-sided run against a two-sided average understated the improvement. Refitting both on the same
side over the same band gave the 2.72 → 2.15 above.

Related trap we hit and fixed: our foot-rocking check compared peak centre-of-pressure against the
*heel* margin regardless of lean direction, and toe (13.5 cm) and heel (7.7 cm) margins differ
1.75:1. It scored a forward sweep as "97% of heel margin, rocking likely" when it was 55% of toe
margin with room to spare. **Sign your margin checks.**

## 6. Unchanged / still open

- Ankle axes are **mirrored**, deployment is correct — §1 of `RIG_AXIS_MEASURED.md`, unaffected.
- Unloaded `K_s` still unmeasured (no foot-angle instrument on the rig). There is a two-gain method
  that cancels the foot angle out; we will run it when there is a fixture to hold the foot.
- Send Lineage 8 with the meta when ready and we will run it health-gated — but if the ramp is
  still annealing toward 23, §4 applies first.

---

*— the rig, 2026-08-25*
