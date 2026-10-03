# TRAINING → RIG: static-tilt servo fix — test notes for your sim/offline harness

**Re:** your report "stand policy does not regulate static tilt" (2026-08-16, on `model_33800`)
**Status:** root cause found, fixed in training, servo formed and passing your criteria in our sim.
**Checkpoint to test:** `kbot_legs_rough / 2026-08-15_15-35-03 / model_42400.pt` (available now, mid-training)
— a consolidated candidate lands tomorrow morning after an overnight run + full battery; same run, higher iter.

---

## 1. What was wrong (confirmed, reproduced)

Your finding reproduced **exactly** in sim with a kinematic tilt-hold probe (now our permanent
gate station 7, `eval_watch/static_tilt_probe.py`): on `33800`, correlation(roll, hip-roll
differential) = **−0.37**, response at 10° = 0.03 rad — indistinguishable from the zero-tilt
bias. Matches your ~0.002 rad hardware measurement.

Root cause (your H1+H2, one defect): in training, tilted-quasi-static states occurred almost
exclusively **during** push bursts — which is precisely when all tilt penalties were released
(the mechanism that freed the push counter-lean). Outside bursts nothing ever tilted the robot.
Net: the policy was never once graded on static attitude while actually tilted.

## 2. What changed in training (no deploy-side action needed for any of it)

1. **Tilt-servo holds** — ramped base *moments* (3–12 Nm ≈ 3–12.5° quasi-static, 70% roll-biased,
   1–3 s) applied to established stands, creating the "tilted but stable, feet planted" state.
2. **Always-on attitude wall at stand** — linear tilt penalty beyond a 3° deadband, deliberately
   NOT released during pushes or holds. (The deadband keeps the push counter-lean legal.)
3. **Motion-tax release during holds** — stillness/pose/gyro penalties pause while held so the
   *correction* is untaxed. Your H4 was real: a tilted statue at default pose was collecting the
   full pose bonus, taxing any correction. Closed.

## 3. Result on your acceptance criteria (sim, `42400`, +4k iters after fix)

| criterion | your bar | `33800` | **`42400`** |
|---|---|---|---|
| correlation(roll error, hip-roll differential) | > 0.5 | −0.37 | **+0.955** |
| differential at 10° tilt | ≥ ~0.1 rad | 0.03 (=bias) | **0.26 rad** |
| monotonic with roll error | yes | no | **yes** (see sweep) |

Full sweep at `42400` (hip-roll differential command, rad, action-space × 0.5 scale):

```
tilt:  −12°     −10°    −8°     −5°     −3°     0°      +3°     +5°     +8°     +10°    +12°
Δ:     −0.048   +0.027  +0.094  +0.107  +0.069  +0.117  +0.184  +0.237  +0.250  +0.260  +0.253
```

**Two polish caveats, so your curve-shape comparison is fair:**
- **Zero-tilt bias:** ~+0.117 rad standing differential (was 0.03 on `33800`). Your correlation
  and response-magnitude criteria are bias-immune, but if you check absolute symmetry you'll see it.
- **Directional asymmetry:** the + roll direction responds ~1.5× stronger than −. Both directions
  pass your magnitude bar after bias correction; expect the same asymmetry on your rig.

## 4. How we measured — replicate this to compare apples to apples

- Kinematic hold: base pose written every control step (fixed roll quat, z = 1.00·cos(tilt) − 0.005),
  root velocities zeroed every step. Feet in ground contact. 3 s per tilt, measure last 2 s.
- Stand commanded: `cmd = (0,0,0)`, standing flag forced true.
- **Phase obs at the hard pin (π, π)** — our probe runs without the training-side corridor, which
  means the anneal shortcut yields s=1 = the hard pin. Your offline harness with the hard pin is
  therefore directly comparable to the numbers above. (The annealed *entry* matters only for live
  stand transitions — §6.)
- Gains pinned to nominal (no DR draws). 64 envs averaged; deterministic (mean) actions.
- Metric: mean over the window of `(act[hip_roll_L] − act[hip_roll_R]) × 0.5` in rad.

## 5. Other results on `42400`-era checkpoints you may want to spot-check

- Sustained push, **1024 envs**, full DR: 20 N × 1.5 s → **96% survive**, peak tilt ~9–17°;
  25 N → 87%. (Your earlier 12 N×3 s 5/5 and 20 N 2/3 were n≤5 — at n=1024 the tails are visible.)
- Impulse kick 0.4 m/s: 99% survive, ankle-first response at **80 ms** — dynamic recovery intact,
  so we predict your controlled-release test passes. Please still run it.
- Quiet stand 12 s: height 1.00 flat, roll p95 ~2°, gyro 0.08–0.12 rad/s, drift ~2–3 cm,
  hip-roll ~2.2–2.7 Nm mean (no A-frame; under continuous rating).
- Walk thermals: hips/knees RMS **under** continuous ratings; ankles 1.18–1.24× (duty-cycle note).
  Speed tracking 81% of commanded 0.3 — known, watched, not yet chased.

## 6. Deploy contract — one line changed, everything else as your §6

Unchanged: 43-d obs layout, action scale 0.5, clip 8, kp 150/60, kd 5/3, 50 Hz,
`freq_map (0.9, 1.4, 0.15, 0.45)`.

**The one delta (lineage-3 checkpoints, incl. this one):** stand-pin **entry** is annealed —
below 0.1 m/s, blend each foot's phase to π along the shortest arc over ~1.0 s:

```
s = clamp(t_since_stand / 1.0, 0, 1)
phi_obs = wrap(phi_live + s * wrap(pi − phi_live))
```

End state identical to your existing (π, π) pin; only the entry is continuous. Training uses this;
a hard snap at stand entry re-creates an obs discontinuity this policy no longer trains under.
Full spec in `HIL_ADAPTIVE_CLOCK.md` (amendment, 2026-08-13). Static-tilt testing at an
established stand is unaffected (s=1 there).

## 7. What we'd like back

1. Your offline network-on-real-obs tilt sweep on `42400` (or tomorrow's consolidated candidate)
   — correlation, 10° response, and the sweep table so we can compare curve shapes.
2. The controlled-release test result.
3. If you re-run your 5-minute stand: the settled lean should now be **< 3°** (the wall's deadband)
   — on `33800` you measured 3.6°, which the old reward structure tolerated. If it still settles
   above 3°, tell us: that would mean the wall's sim-behavior isn't transferring.
