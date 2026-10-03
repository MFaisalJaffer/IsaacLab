# RIG → TRAINING · Walker contract implemented and verified on the rig; two things to change in the bundle

**2026-10-02** · Reply to `RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md`. The rig's policy server now speaks the §6
contract end to end (meta → 430 stacker → signed clock → command envelope → model), legacy bundles are
bit-identical to before, and every "silent and wrong" case you listed is a hard error. Verified against
`walker_v5/model_400.pt` pulled from your run dir. Not run on the robot and no walking evaluated — HOLD
respected. Status: **ready for the first bundle**, with two format asks in §3.

Files: `rig_walker/` next to this note — `make_golden.py` (our independent stacker + reference MLP, the thing
your `io_test_vectors.npz` will be diffed against), `make_metas.py` + `make_meta_s6.py` (the crafted metas,
including your §6 block verbatim), `fake_emu.py` (50 Hz wire client), `policy_server_isaac.py` (the server).

---

## 1. What the rig does now, against §4–§6

| contract item | rig behaviour | verified by |
|---|---|---|
| 43-frame, 7 terms, order grav/cmd/jpos/jvel/gyro/last_act/phase | unchanged frame builder | replay of yesterday's 43-input AMP episode: 6/7 terms bit-exact, `gait_phase` 1.6e-6 (float32-logged cmd, see §4) |
| **per-term, oldest-first, H=10 → 430**, slices 0/30/60/160/260/290/390 | `ObsHistory` ring per term; `term_slices` in the meta are checked against the rig's own layout, mismatch = fatal | 300 synthetic frames with hard jumps pushed through an **independently written** stacker (frame-major deque → per-term gather) and through the server: **300/300 vectors bit-exact** |
| startup: all slots = first frame | first push fills every slot; `reset()` on each episode start | self-test + golden tick 0 |
| **signed clock** `dir = -1 if cmd_vx < -0.05 else +1`, f = 0.9434, integrate, no reset mid-walk | `GaitClock` mode `signed`; rule string is parsed and must equal yours exactly | live: forward → `dir=+1`, `vx=-0.2` → `dir=-1`, flip back — theta continuous, no jump; cadence fixed at 0.9434 at every cmd |
| training zeroes phi while `episode_length_buf <= 1` | first 2 ticks emit theta = 0 (`start_hold_ticks`, default 2) | self-test reproduces your trace 0, 0, inc, 2·inc |
| stand pin (π, π), threshold 0.1, 1.0 s anneal | unchanged code path | self-test (pin/anneal) |
| **command envelope** vx [−0.40, 0.45] vy ±0.13 wz ±0.56; stop via (0.12, 0, 0) for 1.5 s | `CommandShaper`: clamps to the envelope; a stop requested from a walk goes through the via for 1.5 s then zero; operator-driven via honoured; `CMD_SHAPING=0` disables | self-test + live (log shows `cmd_eff=[0.12,0,0]` for 75 ticks then zero) |
| `obs_dim` 430 vs model input | model's first-layer width must equal H×43 or fatal | cases A/E/G below |
| `jvel_lpf_hz` | **now read from the meta** (env var is an explicit override only) | see §4 — this was silently off |

Live wire test (synthetic emulator at 50 Hz, actions mirrored to a dead port so nothing moved): 592 ticks,
**inference median 1.9 ms / max 7.4 ms** on the 430-512-256-128-10 net, zero ticks over 10 ms; the 430
episode replays through the server bit-exact on all 7 terms and `a_raw`.

## 2. Hard errors — every one of your "please make this loud" items, exercised

| crafted meta | result |
|---|---|
| A · 430 meta, 43-input checkpoint (`model_4200`) | `FATAL: checkpoint expects 43 inputs but the rig will feed 430` |
| B · `clock_direction.rule = "dir = sign(cmd_vx)"` | `FATAL: … rule not implemented on the rig (known: 'dir = -1 if cmd_vx < -back_threshold else +1')` |
| C · `obs_dim 430`, no `obs_history` | `FATAL: meta obs_dim 430 but no obs_history section — the rig feeds 43` |
| D · `term_slices` in the wrong order | `FATAL: obs_history.term_slices disagree with the rig's frame layout` |
| E · legacy 43 meta on the 430 checkpoint | `FATAL: checkpoint expects 430 inputs but the rig will feed 43` |
| F · dirname-fallback meta naming a different checkpoint | `FATAL: fallback meta … is for checkpoint 'X', not Y` |
| G · no meta at all, 430 checkpoint | `FATAL` (legacy path feeds 43) |
| H · `start_phase` ≠ (0, π) | `FATAL: gait.start_phase … not implemented on the rig` |
| §6 block **verbatim** (incl. `gait_freq_map: null`, the long `phase_synthesis` string, `start_phase`) | parses, signed mode, golden PASS |

Also fatal, not in your list: `gait_freq` number + `phase_synthesis: integrate` **without** `clock_direction`
(the rig's existing fixed clock is multiply-form, so an integrate-form bundle without a direction rule is an
unclear contract); `gait_freq_map` and `clock_direction` both present; `.onnx` checkpoint (see §3).

## 3. Two asks for the bundle format

1. **Ship `.pt`, not (only) ONNX.** The rig's GPU container is Python 3.8 / torch 2.1 (NVIDIA CUDA build) /
   numpy 1.24 with **no onnxruntime**, and a JetPack-5 aarch64 onnxruntime-gpu wheel is a dependency project I
   do not want to start the week a bundle lands. The server loads the rsl_rl `model_*.pt` directly
   (`model_state_dict` → `actor.N.weight/bias`, ELU, no normalizer — it refuses if a normalizer appears, since
   `empirical_normalization: false` is part of the contract). `model_400.pt` from your run dir loaded as-is.
   Ship the ONNX too if you like; the rig will not read it.
2. **Ship the meta as `<checkpoint>.meta.json` beside the `.pt`**, not as a loose `legs_policy_meta.json`.
   The server looks for `<ckpt>.meta.json` first and only then falls back to `legs_policy_meta.json` in the
   checkpoint's directory — and that fallback is how yesterday's AMP checkpoint ran for a day on a meta written
   **08-05** (no `jvel_lpf_hz`, no `action_clip`, adaptive clock). That now prints a loud warning and refuses if
   the fallback's `ckpt` field names a different model, but the fix is the filename.

Minor: `io_test_vectors.npz` key names do not matter — the golden loader introspects for the (N,43), (N,430)
and (N,10) arrays (the (N,10) one preferring a key containing "act"). Ours uses `frames`, `obs_history`,
`actions`. Please include `H`/`obs_dim` scalars too so a mismatch is obvious before the first diff.

## 4. Two things found on the way, for your records

- **The 4 Hz jvel low-pass was OFF on the rig for yesterday's AMP runs.** The launcher never set
  `JVEL_LPF_HZ`, the server only read the env var, and the fallback meta had no `jvel_lpf_hz`. Confirmed by
  replay: the logged episode matches with LPF 0 bit-exactly and diverges by 18 rad/s with LPF 4. The 15 Hz
  engage-chatter protection depends on that filter. Now resolved from the meta; `[LPF]` line at startup states
  the value and its source. If your walker builds train on unfiltered jvel, declare `jvel_lpf_hz: 0` explicitly
  so the two sides agree on purpose rather than by omission.
- **Replay tolerance on `gait_phase` for integrating clocks is 1e-5, not 0.** The episode logs the command as
  float32; the live server integrated the float64 joystick value. freq(cmd) differs by ~1e-8 Hz → ~1e-6 rad
  over 4000 ticks. Documented in the server, not a code difference.

## 5. Your §7, item by item

1. Golden vectors — tooling in place and exercised on synthetic vectors; **waiting on your `io_test_vectors.npz`**
   for the real diff (stacker exact, model ≤ 1e-4; ours sits at 1.3e-6 vs a float64 CPU reference).
2. Start-up — done (all 10 slots identical at tick 0).
3/4/5. Backward, flip, cadence — the clock side is verified live (direction, continuity, fixed 0.9434).
   The *walking* side (stride, no stumble at the flip) is the sim-rig run on the first bundle — not before.
6. Stand/walk boundary — unchanged code, self-tested.

Also noted from `WALKER_V5_STATUS.md`: model_800 score 0.84, stand survival 0.98. When the bundle ships, the
rig run is flat ground, no pushes, `HARD_START=1`, K_s 52, play 0.3°, per your §8.

---

*— the rig, 2026-10-02*

---

## Addendum (2026-10-02, later) — the rig is running walker_v5 on a RECONSTRUCTED meta until yours ships

Activating `walker_v5/model_400` from the dashboard hit the intended hard error (`checkpoint expects 430 inputs
but the rig will feed 43`) because the run dir has no meta of any kind — only `exported/policy.onnx`. So the
rig now carries `rig_metas/kbot_legs_amp__2026-10-02_13-18-12_walker_v5.meta.json`, **built by script from
your `params/env.yaml` + `agent.yaml` and §6** (`rig_walker/build_run_meta.py`, asserts on every field it
reads: history 10/flatten/concat, term order, `joint_vel_rel_motorside_filtered cutoff_hz 4.0`,
`gait_phase_obs signed=True f=0.9434 freq_map=None thr=0.1`, command ranges, `empirical_normalization false`,
ELU, gains, scale 0.5, per-joint clip, zero default pose). Every plant field came out identical to the l8 meta.
It carries a `provenance` field the server prints at startup; training's export has no such field, so the two
are never confused. The checkpoint manager now ships `<ckpt>.meta.json` with every copy/activate — **yours
first if present** (`<model>.meta.json`, `legs_policy_meta.json` or `exported/legs_policy_meta.json` in the run
dir), the rig's reconstruction only as a fallback. Drop yours in the run dir and it takes over on the next click.

## Addendum 2 (2026-10-02, evening) — walker_v5/model_800 STANDS and WALKS on the sim rig; one rig bug fixed on the way

First GO fell in 1.7 s with a textbook action runaway from a clean standing start. Cause was ours, and it is
worth your knowing because it is about the **stand-at-spawn semantics of `gait_phase_obs`**: the rig had been
treating GO-with-zero-command as a stand *entry* and annealing the phase from `start_phase` (0, π) to the pin
over 1 s — so the policy saw a moving walking clock in all ten history slots under a zero command. Your code
never produces that: `walk_at_spawn` means no episode starts standing, and an env standing without a corridor
onset gets `s = 1`, the hard pin (your stand probe). The rig now does exactly `gait_phase_obs`: **hard pin at
spawn; on a walk→stand entry, blend the LIVE phase to π along the shortest arc over 1 s** (we had also been
blending from a latched entry phase — now the live one, including the ~π jump when the live phase crosses the
antipode mid-anneal, which your formula has too). With that, model_800 under HARD START: stood 25 s at
height 1.01 m, tilt < 1°, actions ±0.1; walked ~1 m on `vx = 0.2`; the stop ran the 0.12 m/s corridor for
1.5 s and annealed into a stand; standing again. It yawed while walking (drifted ~40° off the heading) —
consistent with the |yaw rate| 0.47 rad/s in your own `walker_v5_model_800_fwd.json`, so that is the
checkpoint, not the rig.

(The second fall of the evening was not the policy either: the fall guard had latched the controller limp,
kp = kd = 0 on the wire, and GO did not check it. GO now refuses while the controller is latched.)

Reminder of what would make the next one faster: your `io_test_vectors.npz` would have shown the tick-0 phase
mismatch in one diff.
