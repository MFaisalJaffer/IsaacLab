# RIG handoff — multi-direction walker: 10-frame observation history + signed gait clock

**2026-10-02 · Status: IMPLEMENT, then run the first bundle.** The first bundle of this kind is
`walker_v5_3200` (sent with this note; its `legs_policy_meta.json` carries the fields in §6). It is an
interface and walking-quality candidate on flat ground, not a hardened policy (§8). Bundles you already have (43 inputs, fixed 1.4 Hz or speed-adaptive clock)
are **not affected** — keep running them exactly as today.

---

## 1. What is new

The new policies walk forward, **backward**, **sideways**, **turn in place**, and stand. To learn that, the
policy was given two things that live outside the network — on the rig side, like the clock does today:

1. **A short memory:** the network sees the last 10 observations instead of one (§2).
2. **A clock that runs backward** while the command is "walk backward" (§3).

Plus a wider command envelope (§4). Nothing else changes (§5).

---

## 2. Change 1 — the policy input is 10 stacked frames (430 values)

Every 20 ms tick you already build one 43-value frame. **That frame is unchanged** — same 7 terms, same
order, same scaling and filters:

| # | term | per-frame dim |
|---|---|---|
| 1 | `imu_projected_gravity` | 3 |
| 2 | `velocity_commands` `[vx, vy, wz]` | 3 |
| 3 | `joint_pos_rel` | 10 |
| 4 | `joint_vel_rel` (motor side, 4 Hz low-pass, as today) | 10 |
| 5 | `imu_ang_vel` | 3 |
| 6 | `last_action` (previous raw network output) | 10 |
| 7 | `gait_phase` `[cos φL, sin φL, cos φR, sin φR]` | 4 |

The network input is now those values for the **last 10 ticks, grouped per term, oldest first**:

```
input = [ grav(t-9) … grav(t) | cmd(t-9) … cmd(t) | q(t-9) … q(t) | qd(t-9) … qd(t) |
          gyro(t-9) … gyro(t) | act(t-9) … act(t) | phase(t-9) … phase(t) ]
```

| term | offset | length (10 × dim) |
|---|---|---|
| `imu_projected_gravity` | 0 | 30 |
| `velocity_commands` | 30 | 30 |
| `joint_pos_rel` | 60 | 100 |
| `joint_vel_rel` | 160 | 100 |
| `imu_ang_vel` | 260 | 30 |
| `last_action` | 290 | 100 |
| `gait_phase` | 390 | 40 |
| **total** | | **430** |

Inside each block the newest frame is **last** (e.g. the current joint positions are input[150:160]).

> ⚠️ **Not** frame-by-frame (`[frame(t-9) | frame(t-8) | … | frame(t)]`). That ordering puts every signal on
> the wrong inputs; nothing errors, the robot just falls.

Rules:

1. **Start-up:** at policy start / reset, fill all 10 slots of every term with the **first frame** (not zeros).
   `last_action` in that first frame is zero, as today.
2. **Every tick:** build the frame exactly as today, push it (drop the oldest), run the network on the 430 vector.
3. The history is over **control ticks** (20 ms): current frame + 9 past = 0.18 s back.

```
hist = {term: [first_frame[term]] * 10 for term in TERMS}        # at start, TERMS in the order above
each tick:
    frame = build_frame_as_today()                                # 43 values, incl. the clock of §3
    for term in TERMS:
        hist[term] = hist[term][1:] + [frame[term]]
    x = concat(concat(hist[term]) for term in TERMS)              # 430
    action = policy(x)                                            # 10, same meaning as today
```

---

## 3. Change 2 — the gait clock: one fixed frequency, direction follows the command

```
f      = 0.9434 Hz                                # period 1.06 s — read meta["gait"]["gait_freq"]
dir    = -1 if cmd_vx < -0.05 else +1             # COMMANDED vx (m/s), not measured velocity
theta += dir * 2*pi * f * dt                      # dt = 0.02 s — INTEGRATE, never tick_count * f
wrap(x) = ((x + pi) mod 2*pi) - pi
phi_L  = wrap(theta)
phi_R  = wrap(theta + pi)                         # anti-phase, as today
obs    = [cos(phi_L), sin(phi_L), cos(phi_R), sin(phi_R)]
```

- **One frequency for every speed and direction.** These builds do **not** use the speed→frequency map
  (`gait_freq_map` is null); do not apply it.
- **Backward = the clock runs backward.** While `cmd_vx < -0.05`, `theta` decreases. The policy walks backward
  as its forward stride played in reverse; with a forward-running clock it does not walk backward properly.
- **Only the sign of vx matters.** Sideways and turn-in-place commands run the clock forward (`dir = +1`).
  A mixed command with `vx < -0.05` runs it backward whatever `vy` / `wz` are.
- `theta` starts at 0 at policy start (φL = 0, φR = π — `start_phase`, as today), free-runs, and is never
  reset mid-walk. A direction change only flips the sign of the increment — no jump.
- **Stand pin: unchanged.** `norm(cmd[vx, vy, wz]) < 0.1` ⇒ emit (π, π), i.e. `[-1, 0, -1, 0]`; the integrator
  keeps running underneath; the annealed entry over 1.0 s (the lineage-2 amendment in `HIL_ADAPTIVE_CLOCK.md`)
  applies.

*(For bit-exact comparison with our test vectors only: in training `theta` stays 0 for the first two ticks
after start and accumulates from the third. It makes no practical difference.)*

---

## 4. Change 3 — command envelope

| component | trained range | note |
|---|---|---|
| `vx` | −0.40 … +0.45 m/s | backward is now trained |
| `vy` | −0.13 … +0.13 m/s | side-steps; small on purpose (no ankle roll) |
| `wz` | −0.56 … +0.56 rad/s | including turn in place (`vx = vy = 0`) |

- `norm(cmd) < 0.1` is a stand, as today. Single-direction commands were trained at sizes ≥ 0.1, so a pure
  sideways command under 0.1 m/s is a stand; usable side-step speeds are 0.10–0.13 m/s.
- About 60% of training commands were a single direction (forward/backward, sideways, or turn); the rest
  mixed all three.
- **Stopping:** in training a stand is always entered through ~1.5 s of slow forward walking,
  `cmd = (0.12, 0, 0)`, and only then `(0, 0, 0)`. A direct jump from a fast walk, side-step or turn to zero
  was never practiced. If your command source can do it, send `(0.12, 0, 0)` for 1.5 s before zero.

---

## 5. What does NOT change

- The 43 per-frame values: terms, order, scaling, IMU frame, joint-velocity filter, `last_action` meaning.
- Joint order, PD gains, action scale / offset, the per-joint `action_clip` table, 50 Hz control rate —
  read them from the meta as today.
- Stand pin value (π, π), `stand_still_threshold` = 0.1, annealed pin entry.
- Latency: trained with the 2–5 step command-delay band on all ten joints, as agreed in
  `RIG_REPLY4/5_SIM2SIM.md`. No rig change needed.

---

## 6. Meta contract / failure modes

New or changed fields in `legs_policy_meta.json` for these builds:

```json
"task": "Isaac-Velocity-Rough-KbotLegs-AMP-v0",
"obs_dim": 430,
"frame_dim": 43,
"obs_history": {
  "length": 10,
  "layout": "per_term_contiguous_oldest_first",
  "startup": "fill every slot with the first frame",
  "term_slices": [
    {"name": "imu_projected_gravity", "offset": 0,   "length": 30},
    {"name": "velocity_commands",     "offset": 30,  "length": 30},
    {"name": "joint_pos_rel",         "offset": 60,  "length": 100},
    {"name": "joint_vel_rel",         "offset": 160, "length": 100},
    {"name": "imu_ang_vel",           "offset": 260, "length": 30},
    {"name": "last_action",           "offset": 290, "length": 100},
    {"name": "gait_phase",            "offset": 390, "length": 40}
  ]
},
"gait": {
  "gait_freq": 0.9433962264150942,
  "gait_freq_map": null,
  "clock_direction": {"rule": "dir = -1 if cmd_vx < -back_threshold else +1", "back_threshold": 0.05},
  "phase_synthesis": "integrate: theta += dir * 2*pi*gait_freq*dt each tick; pin obs to stand_phase when |cmd| < stand_still_threshold",
  "stand_still_threshold": 0.1,
  "start_phase": [0.0, 3.141592653589793],
  "stand_phase": [3.141592653589793, 3.141592653589793],
  "stand_pin_anneal_s": 1.0
},
"command_envelope": {"vx": [-0.40, 0.45], "vy": [-0.13, 0.13], "wz": [-0.56, 0.56], "stop_via": [0.12, 0.0, 0.0], "stop_via_s": 1.5}
```

`obs_terms` keeps describing **one frame** (43), as today.

- **`obs_dim` 430:** a rig that feeds 43 values gets an ONNX shape error. Loud — good.
- **`gait.clock_direction`:** a rig that ignores it runs a forward clock under a backward command. That is
  silent and wrong. **Please make "clock_direction present but not implemented" a hard error.**
- **`gait.gait_freq` is a number and `gait_freq_map` is null:** fixed clock; do not fall back to the
  adaptive map or to 1.4 Hz.

---

## 7. Verification on the rig (please report)

1. **Golden vectors.** Each bundle includes `io_test_vectors.npz`: 16.5 s of ticks (forward 0.3, backward −0.3,
   side 0.13, turn 0.5, mixed 0.3 + 0.3, then a stop through the corridor and the annealed pin) with, per tick:
   `frame` (43), `input` (430), `action` (10), `cmd` (3), `theta` (the clock before wrapping), `phase_obs` (4),
   `standing`, `corridor`, `t_s`, and `onnx_action` (what `policy.onnx` gives on `input`; it matches the live
   policy to 5e-6). Three checks, none of which needs physics:
   - stack our `frame` rows with your stacker (first frame fills all 10 slots) → must equal `input` exactly;
   - run `policy.onnx` on `input` → must equal `action` to 1e-4;
   - synthesize the clock from `cmd` with §3 (theta = 0 for the first two ticks) → must equal `phase_obs`
     while moving; during the stop, the corridor runs at `(0.12, 0, 0)` for 75 ticks and the pin then anneals
     over 50 ticks from the stand onset (`standing` goes true).
2. **Start-up.** At tick 0 all 10 slots of each term are identical.
3. **Backward.** Command `vx = -0.3`: `theta` must decrease; the robot walks backward with a normal stride.
4. **Direction flip mid-walk** (+0.3 → −0.3 → +0.3): no phase jump at the switch.
5. **Cadence.** Step rate ≈ 0.94 Hz at every commanded speed (the old adaptive builds varied 0.9–1.4).
6. **Stand/walk boundary** in both directions, as before.

---

## 8. Caveats

- **Flat ground only** so far, and pushes in training were light (3–8 N, under a second). These builds are
  **not hardened** — do not run push batteries on the first bundles.
- Speed tracking is currently about 80–85% of the command at the top of the forward/backward range.
- Sideways tops out at 0.13 m/s by design.
- This is a new training line. The previously shipped bundle stays the reference for standing and push work
  until one of these passes hardening.

---

*— Isaac side, 2026-10-02*
