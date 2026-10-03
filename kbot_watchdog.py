"""Training watchdog for the kbot-train service.

Polls the newest kbot_legs_rough run's TensorBoard events and detects the two
divergence modes measured on the 2026-07-08 warm-start failure:

  COLLAPSE: mean_episode_length stuck < 300 for >= 500 iters on a run that was
            once healthy (> 600). The 07-08 run sat at ~8 steps for 20k iters.
  LIMP:     adaptive-KL slammed the LR to its 1e-5 floor AND reward is stuck
            far below the run's own peak for >= 3000 iters. (The 07-08 run
            "recovered" episode length at 62-65k but reward was 5 vs 121 with
            LR pinned -> it was never going to heal.)

On detection: stop kbot-train (checkpoint-save-age guard first), archive the
diseased run dir to kbot_legs_rough_autoroll_<ts>/, stage the last-healthy
checkpoint (minus a safety margin) into kbot_legs_rough/rollback_<iter>/ so the
run_legs_train.sh resume-from-latest logic picks it up, restart the service.

Safety rails: 2 h cooldown between rollbacks, max 3 rollbacks per 24 h (then
HALT the trainer and leave a status file — something structural is wrong).
State + status live in eval_watch/ so the :8800 page can show them.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

IL = "/home/faisal/IsaacLab"
RUNS = f"{IL}/logs/rsl_rl/kbot_legs_rough"
OUT = f"{IL}/eval_watch"
STATE = f"{OUT}/watchdog_state.json"
STATUS = f"{OUT}/watchdog_status.txt"
LOG = f"{OUT}/watchdog.log"
POLL_SECS = 300

# collapse trigger
# 600 -> 400 (2026-08-05): disturbance-training runs legitimately cap out around
# 450-550 ep_len (sustained-push bursts end episodes early BY DESIGN), so the
# old bar was never cleared and the collapse guard stayed SILENTLY DISARMED for
# the whole run — it reported 'healthy' at ep_len 184 through a real cliff.
EP_HEALTHY = 400.0      # run must have reached this once (skips fresh-run warmup)
EP_DEAD = 250.0         # ...and then be stuck below this (was 300; keep it below
                        # the ~360-460 healthy band of disturbance runs)
# 500 -> 300 (2026-07-30): the 07-29 VF explosion needed manual rescue because
# 500 consecutive dead iters + 5-min poll = ~30 min blind window; real collapses
# sit FAR below 300 (159) while turbulent-but-alive runs hover 400+ — 300 iters
# (~16 min @3.3s) keeps the same separation with ~half the burn.
COLLAPSE_ITERS = 300    # ...for at least this many iters
# value-sickness trigger — EGREGIOUS-ONLY tail catcher. Measured: VF>10 samples are
# intermittent even in healthy runs (worst 300-iter frac 0.13 live; doomed run's
# pre-onset windows only hit 0.16-0.52), so there is NO clean early-warning band.
# The root fixes (clip_actions=8, LR cap 1e-3) carry the real weight; this fires
# only on a sustained full brew-up that nothing measured so far would trip.
VF_SICK = 10.0
VF_SICK_ITERS = 500
VF_SICK_FRAC = 0.7      # fraction of window samples above VF_SICK
# limp trigger
LR_FLOOR = 1.2e-5
LR_FLOOR_FRAC = 0.75    # fraction of window at floor (diseased run: 0.88; healthy
                        # runs' worst transient dip: 0.55 — threshold sits between)
LIMP_ITERS = 3000
REWARD_PEAK_MIN = 50.0  # run must have peaked above this for limp to apply
REWARD_FRACTION = 0.20  # stuck below 20% of peak
# rollback
HEALTHY_EP = 380.0      # rollback anchor: last iter with ep_len above this
                        # (was 800; same disarm bug — anchor() returned None so
                        #  detect() could never fire on a disturbance run)
MARGIN_ITERS = 1000     # step back this far before the anchor
COOLDOWN_S = 2 * 3600
MAX_ROLLBACKS_24H = 3


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    with open(STATUS, "w") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")


def load_series(run_dir: str):
    """Return {tag: [(step, value)]} for the tags we watch, from the freshest
    event file that actually has scalars."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags = ("Train/mean_episode_length", "Train/mean_reward", "Loss/learning_rate",
            "Loss/value_function")
    files = sorted(glob.glob(f"{run_dir}/*events*"), key=os.path.getmtime, reverse=True)
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        have = ea.Tags()["scalars"]
        if "Train/mean_episode_length" in have:
            return {t: [(s.step, s.value) for s in ea.Scalars(t)] for t in tags if t in have}
    return {}


def detect(series) -> tuple[str, int] | None:
    """Return (reason, rollback_iter) or None. Pure function -> unit-testable."""
    ep = series.get("Train/mean_episode_length", [])
    if len(ep) < 5:
        return None
    last_step = ep[-1][0]
    was_healthy = any(v > EP_HEALTHY for _, v in ep)
    rw = series.get("Train/mean_reward", [])

    def anchor() -> int | None:
        """Last iter that was healthy on BOTH ep-len and reward, minus margin.
        Reward matters: the 07-08 run kept ep_len ~890 through its limp phase
        (reward 5 vs peak 121) — rolling back only to an ep-healthy point would
        resume a sick run."""
        healthy = [s for s, v in ep if v > HEALTHY_EP]
        if not healthy:
            return None
        a = max(healthy)
        if rw:
            # bucket medians (250-iter bins): single-iter reward spikes must not
            # count as "healthy" (the 07-08 limp phase had lone spikes to 65+)
            buckets: dict[int, list[float]] = {}
            for s, v in rw:
                buckets.setdefault(s // 250, []).append(v)
            med = {b: sorted(vs)[len(vs) // 2] for b, vs in buckets.items()}
            peak = max(med.values())
            rw_healthy = [b for b, m in med.items() if m >= 0.5 * peak]
            if rw_healthy:
                a = min(a, (max(rw_healthy) + 1) * 250)
        return a - MARGIN_ITERS

    # --- collapse: everything in the trailing window is dead
    if was_healthy:
        window = [(s, v) for s, v in ep if s > last_step - COLLAPSE_ITERS]
        if len(window) >= 5 and all(v < EP_DEAD for _, v in window):
            a = anchor()
            if a is not None:
                return (f"COLLAPSE: ep_len<{EP_DEAD:.0f} for last {COLLAPSE_ITERS} iters", a)

    # --- VF-onset (added 2026-07-30 after the triple-explosion investigation):
    # the plain-resume explosion class starts SUDDENLY (VF 0.67 -> 7.8 within
    # ~250 iters from a calm 0.6 baseline) and ep_len collapse FOLLOWS it, so a
    # short sustained-VF window catches it ~5-8 min in vs the ep_len trigger's
    # ~20+. Target -1 = the pick_checkpoint oldest-fallback (resume snapshot):
    # the 199800 lesson — checkpoints near onset carry contamination even when
    # metrics still look healthy; for THIS class only the lineage anchor is safe.
    vf = series.get("Loss/value_function", [])
    if was_healthy and vf:
        vf_on = [v for st, v in vf if st > last_step - 100]
        if len(vf_on) >= 20 and sum(v > 5.0 for v in vf_on) / len(vf_on) >= 0.8:
            return ("VF-ONSET: VF>5 for >=80% of last 100 iters", -1)

    # --- value sickness: VF loss sustained way above healthy while the run was
    # once healthy — the measured pre-collapse window (catches it ~1k+ iters
    # before the action blowup, when a rollback loses almost nothing)
    if was_healthy and vf:
        vf_w = [v for st, v in vf if st > last_step - VF_SICK_ITERS]
        if len(vf_w) >= 50 and sum(v > VF_SICK for v in vf_w) / len(vf_w) >= VF_SICK_FRAC:
            a = anchor()
            if a is not None:
                return (f"VALUE-SICK: VF>{VF_SICK} for {VF_SICK_FRAC:.0%} of last "
                        f"{VF_SICK_ITERS} iters", a)

    # --- limp: LR at floor + reward far below the run's own peak, sustained
    lr = series.get("Loss/learning_rate", [])
    rw = series.get("Train/mean_reward", [])
    if lr and rw:
        peak = max(v for _, v in rw)
        if peak > REWARD_PEAK_MIN:
            lr_w = [(s, v) for s, v in lr if s > last_step - LIMP_ITERS]
            rw_w = sorted(v for s, v in rw if s > last_step - LIMP_ITERS)
            median_rw = rw_w[len(rw_w) // 2] if rw_w else float("inf")
            floor_frac = (
                sum(v <= LR_FLOOR for _, v in lr_w) / len(lr_w) if lr_w else 0.0
            )
            if (
                len(lr_w) >= 5
                and floor_frac >= LR_FLOOR_FRAC
                # median (not all): the 07-08 limp phase had reward SPIKES to ~51
                # amid a ~5 baseline vs a 121 peak — spikes must not mask the limp
                and median_rw < REWARD_FRACTION * peak
            ):
                a = anchor()
                if a is not None:
                    return (f"LIMP: lr<= {LR_FLOOR} and reward<{REWARD_FRACTION:.0%} of peak "
                            f"({peak:.0f}) for {LIMP_ITERS} iters", a)
    return None


def pick_checkpoint(run_dir: str, target_iter: int) -> str | None:
    """Newest model_<i>.pt with i <= target_iter; else the OLDEST in the dir.

    Fallback added 2026-07-30: on a warm-started run the computed anchor-margin
    often lands BELOW the run's first checkpoint (the resume snapshot), so the
    strict search returned None and the watchdog logged "no checkpoint <= N;
    not acting" while the fleet burned (07-19 x14, 07-24). The OLDEST checkpoint
    in a young dir IS the resume snapshot = last-known-good lineage — strictly
    better than not acting.
    """
    best, best_i = None, -1
    oldest, oldest_i = None, 10**12
    for p in glob.glob(f"{run_dir}/model_*.pt"):
        try:
            i = int(os.path.basename(p)[6:-3])
        except ValueError:
            continue
        if best_i < i <= target_iter:
            best, best_i = p, i
        if i < oldest_i:
            oldest, oldest_i = p, i
    if best is None and oldest is not None:
        log(f"pick_checkpoint: nothing <= {target_iter}; FALLBACK to oldest "
            f"{os.path.basename(oldest)} (resume snapshot)")
        return oldest
    return best


def read_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {"rollbacks": []}


def write_state(st: dict) -> None:
    with open(STATE, "w") as f:
        json.dump(st, f)


def sysctl(*args: str) -> int:
    return subprocess.call(["systemctl", "--user", *args])


def newest_ckpt_age(run_dir: str) -> float:
    cks = glob.glob(f"{run_dir}/model_*.pt")
    if not cks:
        return 1e9
    return time.time() - max(os.path.getmtime(p) for p in cks)


def verify_ckpt(path: str) -> bool:
    import torch  # deferred: heavy

    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        log(f"ckpt verify FAILED {path}: {e}")
        return False


def do_rollback(run_dir: str, reason: str, target_iter: int) -> None:
    st = read_state()
    now = time.time()
    recent = [t for t in st["rollbacks"] if now - t < 24 * 3600]
    if recent and now - max(recent) < COOLDOWN_S:
        status(f"trigger ({reason}) but in cooldown; watching")
        return
    if len(recent) >= MAX_ROLLBACKS_24H:
        log(f"HALT: {len(recent)} rollbacks in 24h — stopping trainer, not restarting")
        sysctl("stop", "kbot-train")
        status(f"HALTED after {len(recent)} rollbacks/24h — needs a human. last reason: {reason}")
        return

    ckpt = pick_checkpoint(run_dir, target_iter)
    if ckpt is None:
        log(f"trigger ({reason}) but no checkpoint <= {target_iter}; not acting")
        status(f"trigger ({reason}) but no rollback ckpt; watching")
        return
    if not verify_ckpt(ckpt):
        return
    log(f"ROLLBACK: {reason} -> {os.path.basename(ckpt)} (target iter {target_iter})")

    # never stop mid-save
    for _ in range(12):
        if newest_ckpt_age(run_dir) > 30:
            break
        time.sleep(10)
    sysctl("stop", "kbot-train")
    time.sleep(5)
    subprocess.call(["pkill", "-9", "-f", "train.py --task=Isaac-Velocity-Rough-KbotLegs"])
    time.sleep(3)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arch = f"{RUNS}_autoroll_{ts}"
    os.makedirs(arch, exist_ok=True)
    it = int(os.path.basename(ckpt)[6:-3])
    stage = f"{RUNS}/rollback_{it}"
    os.makedirs(stage, exist_ok=True)
    shutil.copy2(ckpt, stage)  # copy BEFORE moving the run dir
    shutil.move(run_dir, arch)
    log(f"archived {os.path.basename(run_dir)} -> {arch}; staged {stage}/model_{it}.pt")

    sysctl("start", "--no-block", "kbot-train")
    st["rollbacks"] = recent + [now]
    write_state(st)
    status(f"ROLLED BACK to iter {it} ({reason}); trainer restarted")


DISK_WARN_GB = 25
DISK_CRIT_GB = 10


def disk_guard() -> str:
    """Disk-full killed training silently for 6 h on 2026-07-14 (ckpts accumulate
    ~15-20 GB/day). Below CRIT: auto-thin non-current segment dirs to their newest
    checkpoint (same retention policy as the manual cleanup) + truncate the append
    log. Curated archives (hilvalidated*/statue*/best*) are never touched."""
    free_gb = shutil.disk_usage(RUNS).free / 1e9
    if free_gb >= DISK_WARN_GB:
        return ""
    note = f" DISK LOW: {free_gb:.0f}GB free"
    if free_gb < DISK_CRIT_GB:
        import re
        freed = 0
        dirs = sorted(glob.glob(f"{RUNS}/*/"), key=os.path.getmtime)
        for rd in dirs[:-1]:  # never the current segment (watchdog anchors)
            cks = sorted(glob.glob(rd + "model_*.pt"),
                         key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
            for p in cks[:-1]:
                freed += os.path.getsize(p)
                os.remove(p)
        biglog = f"{IL}/train_kbot_legs.log"
        if os.path.exists(biglog) and os.path.getsize(biglog) > 500e6:
            with open(biglog, "rb") as f:
                f.seek(-20_000_000, 2)
                tail = f.read()
            with open(biglog, "wb") as f:
                f.write(tail)
        log(f"DISK CRIT ({free_gb:.0f}GB): auto-thinned {freed/1e9:.1f}GB of old-segment ckpts")
        note += f" (auto-thinned {freed/1e9:.1f}GB)"
    return note


def main() -> None:
    log("watchdog started")
    while True:
        try:
            disk_note = disk_guard()
            dirs = sorted(glob.glob(f"{RUNS}/*/"), key=os.path.getmtime, reverse=True)
            if not dirs:
                status("no run dir; waiting")
            else:
                run_dir = dirs[0].rstrip("/")
                series = load_series(run_dir)
                trig = detect(series)
                if trig is None:
                    ep = series.get("Train/mean_episode_length", [])
                    tail = f"iter {ep[-1][0]} ep_len {ep[-1][1]:.0f}" if ep else "no scalars yet"
                    status(f"healthy: {os.path.basename(run_dir)} {tail}{disk_note}")
                else:
                    do_rollback(run_dir, *trig)
        except Exception as e:
            log(f"watchdog error: {e!r}")
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
