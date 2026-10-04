#!/usr/bin/env python3
"""Sim stand matrix: restart the sim loop with a given ankle model / load, hard-start stand for SECS,
then run rock_analyze.py on the episode. Runs on the Jetson host.
usage: stand_matrix.py <label> <play_deg> <series_k> <roll_moment_Nm> <pitch_moment_Nm> [secs] [rotor_fc]"""
import json, socket, subprocess, sys, time, urllib.request
label, play, ks, mroll, mpitch = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
secs = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0
rfc = sys.argv[7] if len(sys.argv) > 7 else "0"
D = "http://127.0.0.1:8080"; EMU = ("127.0.0.1", 9994)
def g(p, t=8):
    with urllib.request.urlopen(D + p, timeout=t) as f: return json.loads(f.read() or b"{}")
def emu(m): socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(m.encode(), EMU)
def ctl(): p = g("/state.json").get("preflight") or {}; return p.get("state"), p.get("reason")
def sh(c): return subprocess.run(c, capture_output=True, text=True).stdout
sh(["docker", "exec", "-e", "MAC=100.77.88.94", "-e", "ANKLE_PLAY_DEG=" + play, "-e", "ANKLE_SERIES_K=" + ks,
    "-e", "FRICTION_SCALE=1.0", "-e", "ANKLE_ROTOR_FC=" + rfc, "kbot-imu", "bash", "-lc", "bash /root/loop_simimu.sh"])
time.sleep(4)
banner = sh(["docker", "exec", "kbot-imu", "bash", "-c", "grep -E 'SERIES|FREE-PLAY|ROTOR' /tmp/vmotor.log | head -3"]).strip().replace("\n", " | ")
for _ in range(20):
    try:
        if ctl()[0] == "ENGAGED": break
    except Exception: pass
    time.sleep(1)
g("/cmd/hardstart?on=1"); g("/cmd/imuoffset?roll=0&pitch=0&yaw=0"); g("/cmd/home?secs=4"); time.sleep(6)
if mroll:  emu("TILTHOLD:0:x"); time.sleep(0.2); emu("TILTHOLD:off"); time.sleep(0.2); emu("TILTM:%s" % mroll); time.sleep(0.6)
if mpitch: emu("TILTHOLD:0:y"); time.sleep(0.2); emu("TILTHOLD:off"); time.sleep(0.2); emu("TILTM:%s" % mpitch); time.sleep(0.6)
print("=== %s  [%s]  roll %.1f Nm pitch %.1f Nm" % (label, banner[:150], mroll, mpitch), flush=True)
g("/cmd/go"); t0 = time.time(); end = "stood %.0f s" % secs
while time.time() - t0 < secs:
    s, r = ctl()
    if s in ("FAULT_LATCHED", "ESTOP"): end = "%s at %.2f s (%s)" % (s, time.time() - t0, (r or "")[:40]); break
    time.sleep(0.1)
emu("TILTM:0"); g("/cmd/reset"); time.sleep(2.5)
wd = sh(["docker", "exec", "kbot-imu", "bash", "-c", "grep E-STOP /tmp/watchdog_events.log | tail -1"]).strip()
print("result: %s | emulator: %s" % (end, sh(["docker", "exec", "kbot-imu", "bash", "-c", "tail -1 /tmp/vmotor.log"]).strip()[:95]))
if "ESTOP" in end: print("watchdog:", wd[-110:])
ep = sh(["docker", "exec", "kbot-zed", "bash", "-c", "ls -t /root/policy_logs/ep_*.npz | head -1"]).strip()
print(sh(["docker", "exec", "kbot-zed", "bash", "-c", "python3 /root/walker_test/rock_analyze.py %s 0.6 | grep -E 'ticks|torso angle|dominant|ANKLES'" % ep]))
