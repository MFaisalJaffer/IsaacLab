# HIL handoff — speed-adaptive gait clock

**Status: IMPLEMENT AND HOLD.** The currently shipped bundle
(`kbot_legs_rough_rigcand_calmref198600`) is a **fixed 1.4 Hz** build and needs
none of this. A new *adaptive* build is in training; when it ships, its
`legs_policy_meta.json` will carry the map below and the rig must already speak
it. Implementing early is safe: the spec is frozen.

---

## 1. Why this exists

The policy has **no internal clock**. The gait phase it walks to is synthesized
*outside* the network and handed in as 4 observation values. Today the rig
synthesizes that phase at a constant 1.4 Hz.

The new build varies cadence with commanded speed — slow command → slow, short
steps; fast command → brisk steps (how humans walk). Because the clock lives
on the rig side, **the rig must implement the speed→frequency law**, or the
policy receives a rhythm it never trained on at that speed.

This is the **only** deploy-side change. Stride length, rewards, and training
events are all training-side machinery.

---

## 2. The spec (three rules)

### Rule 1 — frequency from the COMMANDED speed

```
f(v) = clamp( f_min + (f_max - f_min) * (|v_cmd_x| - v_lo) / (v_hi - v_lo),
              f_min, f_max )

f_min = 0.9 Hz   f_max = 1.4 Hz   v_lo = 0.15 m/s   v_hi = 0.45 m/s
```

Equivalently: `f = clamp(0.9 + 1.6667 * (|v_cmd_x| - 0.15), 0.9, 1.4)`

| commanded vx | f |
|---|---|
| ≤ 0.15 m/s | 0.90 Hz |
| 0.30 m/s | 1.15 Hz |
| ≥ 0.45 m/s | 1.40 Hz |

**Use the COMMAND, not measured velocity.** That is what training used; it also
means no state-estimator dependency. Read the exact constants from
`meta["gait"]["gait_freq_map"]` rather than hard-coding them.

### Rule 2 — INTEGRATE the phase (do not multiply)

The old fixed-clock form `phase = tick_count * 2π * f * dt` **must not be used**:
with f varying, changing speed rewrites history and the phase jumps, producing a
discontinuity mid-stride. Accumulate instead, once per 20 ms control tick:

```
theta += 2*pi * f(v_cmd) * dt          # dt = 0.02 s, theta free-running
wrap(x)  = ((x + pi) mod 2*pi) - pi     # -> [-pi, pi)
phi_L = wrap(theta)
phi_R = wrap(theta + pi)                # anti-phase, always
obs   = [cos(phi_L), sin(phi_L), cos(phi_R), sin(phi_R)]
```

`theta` starts at 0 on policy start/reset (gives φ_L = 0, φ_R = π — matches
`meta["gait"]["start_phase"]`). It free-runs forever; never reset it mid-walk.

### Rule 3 — stand pin (unchanged from today)

```
if norm(cmd[vx, vy, yaw]) < stand_still_threshold:   # 0.1
    phi_L = phi_R = pi        # OBSERVATION ONLY -> obs = [-1, 0, -1, 0]
```
The integrator **keeps running underneath** the pin — only the emitted
observation is overridden. Identical to current behavior.

---

## 3. What does NOT change

- Observation layout and size (43-d) — no new inputs, no retraining of your port
- PD gains, action scale/offset, the per-joint `action_clip` table, control rate
- The stand pin value (π, π) and `stand_still_threshold` = 0.1

---

## 4. Meta contract / failure mode

For adaptive builds the exporter writes:

```json
"gait": {
  "gait_freq": null,                 // legacy fixed-clock field, DELIBERATELY null
  "gait_freq_map": {"f_min":0.9,"f_max":1.4,"v_lo":0.15,"v_hi":0.45},
  "phase_synthesis": "integrate: phi += 2*pi*f(|cmd_vx|)*dt ...",
  "stand_still_threshold": 0.1,
  "start_phase": [0.0, 3.141592653589793],
  "stand_phase": [3.141592653589793, 3.141592653589793]
}
```

`gait_freq: null` is intentional: a rig that reads only the legacy field will
**fail loudly** instead of silently running a wrong constant clock. Please make
that a hard error, not a fallback to 1.4.

---

## 5. Verification on the rig (please report)

1. **Cadence follows command.** Walk at commanded 0.15 / 0.30 / 0.45 m/s and
   measure realized step rate (foot-contact period). Expect ≈ 0.9 / 1.15 / 1.4 Hz.
   *Sim note:* our zero-crossing estimator reads **4–14% low** (e.g. 0.69
   measured at a 0.8 Hz clock), so judge the trend and the ratio, not absolutes.
2. **No phase jump on speed change.** Ramp the command 0.15 → 0.45 → 0.15 while
   walking; the gait must transition smoothly. A visible stutter/skip at the
   moment of change means the multiply form is still in use (Rule 2).
3. **Stand/walk boundary.** Cross the 0.1 threshold in both directions; entering
   stand must pin to both-planted without a foot-tap.

---

## 6. Caveats worth knowing

- **Trained command envelope is forward-only, 0.12–0.50 m/s**, with yaw ±0.3
  rad/s applied *while* walking forward. **Backward walking and pure
  turn-in-place are outside the adaptive build's training distribution** — the
  map returns f_min there, but the gait itself was never trained at those
  commands. Don't rely on them without testing. (The shipped fixed-1.4 build
  does cover ±1.0 m/s and in-place turns.)
- The map keys off `vx` only; a large lateral or yaw command with small `vx`
  still yields f_min.
- Frequency changes are safe at any rate thanks to the integrator, but the
  policy has only seen f change as fast as an operator's command changes.

---

## 7. Provenance (why these numbers)

Measured, not guessed — `eval_watch/freq_map_study.txt`:

- The shipped 1.4 Hz policy was replayed at clocks **0.8–1.4 Hz**; it walks
  upright at **every** one (32/32 envs, 7–9° pitch, ~75% stance duty), proving
  the whole band is dynamically feasible for this morphology.
- Realized cadence **phase-locks to the commanded clock** (0.69 / 0.96 / 1.16 /
  1.36 measured at 0.8 / 1.0 / 1.2 / 1.4), so the legs genuinely follow it.
- Torque is mildly cheaper at slower clocks at low speed; the composition shifts
  (knee/hip-pitch down, hip-roll up — longer single-support with no ankle-roll DOF).
- `f_min = 0.9` picked as the cheapest cell that is clean at both test speeds;
  0.8 Hz was viable at 0.15 m/s but showed the first strain at 0.30 m/s.


## Amendment (2026-08-13, lineage-2 builds only): annealed stand-pin entry

Rule 3 (stand pin) gains an entry ramp for lineage-2 checkpoints: when the
commanded speed drops below 0.1 m/s, do NOT snap the phase obs to (pi, pi).
Instead, for the first ~1.0 s after the stand begins, blend each foot's phase
toward pi along the shortest arc:

    s = clamp(t_since_stand / 1.0, 0, 1)
    phi_obs = wrap(phi_live + s * wrap(pi - phi_live))

At s=1 the obs equals the existing (pi, pi) contract exactly — the end state
is unchanged; only the entry is continuous. Rationale: the hard snap is an
observation discontinuity the from-scratch lineage could not learn through
(measured collapse within 1 s of the snap); lineage 1 learned the snap only
because its era allowed unlimited survivable stand practice. Lineage-1
checkpoints (incl. the shipped deploy bundle) keep the original hard pin.
