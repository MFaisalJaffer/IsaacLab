#!/usr/bin/env python3
"""Host-side checkpoint manager for the K-Bot HIL dashboard.

Runs on the Jetson HOST (needs docker + ssh-to-ubuntu). Serves :8090 with CORS.
Scans ALL experiment dirs under IsaacLab/logs/rsl_rl (kbot_legs_rough,
kbot_legs_rough_symmetry, ...), lets you copy a SPECIFIC checkpoint, and
activate any on-Jetson checkpoint in the HIL.

  GET  /api/state                       -> {experiments:[{exp,runs:[{run,latest,count,kp,kd}]}], jetson, active}
  POST /api/copy   {exp,run,model}      -> scp -> docker cp -> kbot-zed:/root/<exp>__<run>__<model>  (model="latest" ok)
  POST /api/activate {name}             -> restart isaac policy server (ISAAC_CKPT) + HIL loop
"""
import json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UBUNTU   = "faisal@100.95.25.109"
SSH_KEY  = os.path.expanduser("~/.ssh/ckpt_manager")
LOGS     = "IsaacLab/logs/rsl_rl"
MAC_IP   = os.environ.get("MAC_IP", "100.77.88.94")
PORT     = 8090
SSHOPTS  = ["-i", SSH_KEY, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new"]
_cache   = {"t": 0, "exps": []}
CACHE_TTL = 20   # seconds; the ubuntu scan is cheap but polled every few s


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def ssh(remote_cmd, timeout=30):
    return run(["ssh", *SSHOPTS, UBUNTU, remote_cmd], timeout=timeout)


def scan_experiments():
    if time.time() - _cache["t"] < CACHE_TTL and _cache["exps"]:
        return _cache["exps"]
    awk_kp = r'''awk '/^[[:space:]]*stiffness:[[:space:]]*$/{getline; if(match($0,/[0-9]+\.[0-9]+/)){print substr($0,RSTART,RLENGTH); exit}}' '''
    awk_kd = r'''awk '/^[[:space:]]*damping:[[:space:]]*$/{getline; if(match($0,/[0-9]+\.[0-9]+/)){print substr($0,RSTART,RLENGTH); exit}}' '''
    script = (
        f'for e in ~/{LOGS}/*/; do exp=$(basename "$e"); '
        f'  for d in "$e"*/; do [ -d "$d" ] || continue; '
        f'    m=$(ls -t "$d"model_*.pt 2>/dev/null | head -1); [ -z "$m" ] && continue; '
        f'    c=$(ls "$d"model_*.pt 2>/dev/null | wc -l); y="$d"params/env.yaml; '
        f'    kp=$({awk_kp} "$y" 2>/dev/null); kd=$({awk_kd} "$y" 2>/dev/null); '
        f'    echo "$exp|$(basename $d)|$(basename $m)|$c|$kp|$kd|$(stat -c %Y "$m")"; '
        f'  done; done'
    )
    r = ssh(script, timeout=40)
    exps = {}
    for line in r.stdout.strip().splitlines():
        p = line.split("|")
        if len(p) != 7:
            continue
        exp, runid, latest, count, kp, kd, mtime = p
        try:
            mt = int(mtime)
        except ValueError:
            mt = 0
        exps.setdefault(exp, []).append(
            {"run": runid, "latest": latest, "count": int(count or 0),
             "kp": kp or "?", "kd": kd or "?", "mtime": mt,
             "age_min": int(max(0, time.time() - mt) / 60) if mt else None})
    # sort by RECENCY: the run being trained right now must be first, whatever
    # it is named (name-sorting buried it under seed_adapt/rollback_* runs)
    out = [{"exp": e, "runs": sorted(rs, key=lambda x: x["mtime"], reverse=True)}
           for e, rs in exps.items()]
    out.sort(key=lambda x: (-max((r["mtime"] for r in x["runs"]), default=0), x["exp"]))
    _cache["t"], _cache["exps"] = time.time(), out
    return out


def list_jetson():
    r = run(["docker", "exec", "kbot-zed", "bash", "-lc", "ls -1 /root/*.pt 2>/dev/null"])
    out = []
    for path in r.stdout.strip().splitlines():
        name = os.path.basename(path)
        parts = name.split("__")
        if len(parts) >= 3:   exp, runid, model = parts[0], parts[1], "__".join(parts[2:])
        elif len(parts) == 2: exp, runid, model = "kbot_legs_rough", parts[0], parts[1]
        else:                 exp, runid, model = "(untagged)", "(untagged)", name
        try: it = int(model.replace("model_", "").replace(".pt", ""))
        except ValueError: it = 0
        out.append({"name": name, "exp": exp, "run": runid, "model": model, "iter": it})
    out.sort(key=lambda x: (x["exp"], x["run"], x["iter"]), reverse=True)
    return out


def active_ckpt():
    r = run(["docker", "exec", "kbot-zed", "bash", "-lc",
             "p=$(pgrep -f policy_server_isaac.py | head -1); "
             "[ -n \"$p\" ] && tr '\\0' '\\n' < /proc/$p/environ | grep '^ISAAC_CKPT=' | cut -d= -f2"])
    path = r.stdout.strip()
    return os.path.basename(path) if path else None


RIG_METAS = os.path.expanduser("~/rig_metas")   # host-side per-RUN metas, rig-reconstructed from env.yaml


def ship_meta(exp, runid, model, dest):
    """Put the bundle's meta beside the checkpoint on kbot-zed as <dest>.meta.json.

    The policy server reads <ckpt>.meta.json first; without it, it falls back to
    /root/legs_policy_meta.json — whatever bundle last dropped it there (an AMP
    checkpoint ran a day on a meta from 08-05 that way, 2026-10-02), and a
    history/430 bundle refuses to start at all. Order of preference:
      1. training's own export in the run dir (<model>.meta.json, legs_policy_meta.json,
         exported/legs_policy_meta.json)
      2. rig-reconstructed per-run meta ~/rig_metas/<exp>__<run>.meta.json (carries a
         `provenance` field; the server prints it)
      3. none -> say so loudly in the UI message."""
    rd = f"{LOGS}/{exp}/{runid}"; stem = model[:-3] if model.endswith(".pt") else model
    dmeta = (dest[:-3] if dest.endswith(".pt") else dest) + ".meta.json"
    local = f"/tmp/{dmeta}"; src = None
    for cand in (f"{stem}.meta.json", "legs_policy_meta.json", "exported/legs_policy_meta.json"):
        if "ok" in ssh(f'test -f ~/{rd}/{cand} && echo ok', timeout=15).stdout:
            r = run(["scp", *SSHOPTS, f"{UBUNTU}:{rd}/{cand}", local], timeout=60)
            if r.returncode == 0:
                src = f"training's {cand}"; break
    if src is None:
        rig = os.path.join(RIG_METAS, f"{exp}__{runid}.meta.json")
        if not os.path.exists(rig):
            return (f"NO META: training shipped none in {exp}/{runid} and no {rig} — the server will "
                    f"use the fallback /root/legs_policy_meta.json and WARN; a history (430) bundle will refuse")
        with open(rig) as fh:
            mj = json.load(fh)
        mj["ckpt"] = dest          # the run meta serves every iteration; stamp this one
        with open(local, "w") as fh:
            json.dump(mj, fh, indent=1)
        src = f"RIG-RECONSTRUCTED run meta {os.path.basename(rig)}"
    r2 = run(["docker", "cp", local, f"kbot-zed:/root/{dmeta}"], timeout=60)
    try: os.remove(local)
    except OSError: pass
    if r2.returncode != 0:
        return f"meta docker cp FAILED: {r2.stderr.strip()[:120]}"
    return f"meta: {src} -> {dmeta}"


def split_name(name):
    """<exp>__<run>__<model>.pt -> (exp, run, model) or None for untagged files."""
    parts = os.path.basename(name).split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], "__".join(parts[2:])
    return None


def do_copy(exp, runid, model):
    exp = os.path.basename(exp); runid = os.path.basename(runid)
    rd = f"{LOGS}/{exp}/{runid}"
    if not model or model == "latest":
        r0 = ssh(f'ls -t ~/{rd}/model_*.pt 2>/dev/null | head -1', timeout=20)
        model = os.path.basename(r0.stdout.strip())
        if not model: return False, f"no checkpoints in {exp}/{runid}"
    model = os.path.basename(model)
    if not model.endswith(".pt"): model += ".pt"
    chk = ssh(f'test -f ~/{rd}/{model} && echo ok', timeout=15)
    if "ok" not in chk.stdout:
        return False, f"{model} not found in {exp}/{runid}"
    dest = f"{exp}__{runid}__{model}"
    local = f"/tmp/{dest}"
    r1 = run(["scp", *SSHOPTS, f"{UBUNTU}:{rd}/{model}", local], timeout=180)
    if r1.returncode != 0:
        return False, f"scp failed: {r1.stderr.strip()[:200]}"
    r2 = run(["docker", "cp", local, f"kbot-zed:/root/{dest}"], timeout=60)
    try: os.remove(local)
    except OSError: pass
    if r2.returncode != 0:
        return False, f"docker cp failed: {r2.stderr.strip()[:200]}"
    return True, f"copied {exp}/{runid}/{model} -> {dest}; " + ship_meta(exp, runid, model, dest)


def do_activate(name, stand_pin="new", action_clip="8"):
    name = os.path.basename(name)
    # stand-pin convention: "new" = (pi,pi) for 07-09+ lineage; "old" = (pi/2,pi)
    # for pre-07-09 checkpoints (e.g. best54k). Passed through run_isaac_policy.sh.
    stand_pin = "old" if str(stand_pin).lower() == "old" else "new"
    chk = run(["docker", "exec", "kbot-zed", "bash", "-lc", f"test -f /root/{name} && echo ok"])
    if "ok" not in chk.stdout:
        return False, f"{name} not on kbot-zed"
    # checkpoints copied before metas travelled with them have no <ckpt>.meta.json; fetch
    # one now so the server does not start on the stale fallback (or refuse, for 430 bundles)
    meta_note = ""
    stem = name[:-3] if name.endswith(".pt") else name
    has_meta = run(["docker", "exec", "kbot-zed", "bash", "-lc", f"test -f /root/{stem}.meta.json && echo ok"])
    if "ok" not in has_meta.stdout:
        tag = split_name(name)
        meta_note = "; " + (ship_meta(*tag, name) if tag else "untagged checkpoint, no meta lookup possible")
    r1 = run(["docker", "exec", "-e", f"ISAAC_CKPT=/root/{name}", "-e", "POLICY_CMD=0,0,0",
              "-e", f"STAND_PIN={stand_pin}", "-e", f"ACTION_CLIP={float(action_clip or 0)}",
              "kbot-zed", "bash", "/root/run_isaac_policy.sh"], timeout=90)
    if "UP" not in r1.stdout:
        return False, f"server start failed: {(r1.stdout + r1.stderr)[-300:]}{meta_note}"
    # Preserve the operating mode: if the dashboard reports HARDWARE, do NOT
    # restart the sim loop — that would silently swap /imu and the world back
    # to sim while real motors may be armed (the fall-guard would watch a
    # simulated IMU). Policy server is already restarted with the new ckpt.
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8080/state.json", timeout=3) as f:
            live_mode = (json.load(f).get("mode") or {}).get("mode", "UNKNOWN")
    except Exception:
        live_mode = "UNKNOWN"
    if live_mode == "HARDWARE":
        return True, f"activated {name} (stand_pin={stand_pin}); HARDWARE mode — policy server restarted, sim loop untouched{meta_note}"
    # Preserve the ankle free-play band across the restart. loop_simimu.sh
    # defaults it to 0, so without this a checkpoint switch silently makes a
    # backlash sim rigid — invisibly invalidating whatever is being measured.
    play = "0"
    try:
        ps = run(["docker", "exec", "kbot-imu", "sh", "-c",
                  "ps -eo args | grep '[v]irtual_motor_node'"], timeout=15)
        m = re.search(r"--ankle-play-deg\s+([0-9.]+)", ps.stdout)
        if m:
            play = m.group(1)
    except Exception:
        pass
    r2 = run(["docker", "exec", "-e", f"MAC={MAC_IP}", "-e", f"ANKLE_PLAY_DEG={play}",
              "kbot-imu", "bash", "/root/loop_simimu.sh"], timeout=90)
    ok = "READY" in r2.stdout
    return ok, (f"activated {name} (stand_pin={stand_pin}, ankle_play={play} deg); "
                f"loop {'READY' if ok else 'restart FAILED'}{meta_note}")


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Methods", "GET,POST,OPTIONS"),
                     ("Access-Control-Allow-Headers", "Content-Type"),
                     ("Content-Length", str(len(body)))):
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self): self._send(200, {})
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                self._send(200, {"experiments": scan_experiments(),
                                 "jetson": list_jetson(), "active": active_ckpt()})
            except Exception as e:
                self._send(500, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try: b = json.loads(self.rfile.read(n) or "{}")
        except Exception: b = {}
        try:
            if self.path.startswith("/api/copy"):
                ok, msg = do_copy(b.get("exp", ""), b.get("run", ""), b.get("model", "latest"))
            elif self.path.startswith("/api/activate"):
                ok, msg = do_activate(b.get("name", ""), b.get("stand_pin", "new"), b.get("action_clip", "8"))
            else:
                ok, msg = False, "unknown endpoint"
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        except subprocess.TimeoutExpired:
            self._send(504, {"ok": False, "msg": "operation timed out"})
        except Exception as e:
            self._send(500, {"ok": False, "msg": str(e)})


if __name__ == "__main__":
    print(f"[ckpt_manager] listening on :{PORT} (ubuntu={UBUNTU}, MAC={MAC_IP})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
