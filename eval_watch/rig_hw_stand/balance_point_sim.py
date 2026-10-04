#!/usr/bin/env python3
"""Fair version of the sim home-hold test: find the sim robot's balance point the way the operator does on
the real robot, then see how long it stays.  The sim's zero pose is NOT its balance point (COM ~1 cm ahead of
the ankles), so a constant pitch moment on the torso is bisected until the robot falls neither forward nor
back; the stay time at the best balance is the result.  Controller holds the home pose, policy only shadows
(start the policy server with MIRROR_PORT != 9998).
usage: balance_point_sim.py <play_deg> <rotor_fc_Nm> [iterations=9] [tmax_s=30]"""
import json, socket, subprocess, sys, time, urllib.request
play, rfc = sys.argv[1], sys.argv[2]; iters = int(sys.argv[3]) if len(sys.argv) > 3 else 9; tmax = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
D = "http://127.0.0.1:8080"; EMU = ("127.0.0.1", 9994)
def g(p):
    with urllib.request.urlopen(D + p, timeout=8) as f: return json.loads(f.read() or b"{}")
def emu(m): socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(m.encode(), EMU)
def ctl():
    try: p = g("/state.json").get("preflight") or {}; return p.get("state"), p.get("reason")
    except Exception: return None, None
def start_loop():
    subprocess.run(["docker", "exec", "-e", "MAC=100.77.88.94", "-e", "ANKLE_PLAY_DEG=" + play, "-e", "ANKLE_SERIES_K=52", "-e", "FRICTION_SCALE=1.0",
                    "-e", "ANKLE_ROTOR_FC=" + rfc, "-e", "IMU_AGE_MS=0", "-e", "JNT_AGE_MS=0", "kbot-imu", "bash", "-lc", "bash /root/loop_simimu.sh"], capture_output=True)
    time.sleep(4)
    for _ in range(25):
        if ctl()[0] == "ENGAGED": break
        time.sleep(1)
    g("/cmd/hardstart?on=1"); g("/cmd/imuoffset?roll=0&pitch=0&yaw=0"); g("/cmd/home?secs=3"); time.sleep(5)
ANALYZE = '''
import numpy as np, sys
r = np.load(sys.argv[1], allow_pickle=True)["data"]; t = r[:, 0] - r[0, 0]; th = np.degrees(np.arcsin(np.clip(r[:, 30], -1, 1)))
i = np.flatnonzero(np.abs(th) > 5.0)
print(("%.2f %.2f" % ((t[i[0]], th[i[0]]) if len(i) else (-1.0, th[-1]))) + (" %.3f %.2f" % (np.ptp(th), t[-1])))
'''
def run(M):
    if ctl()[0] != "ENGAGED":
        g("/cmd/rearm"); time.sleep(1.0); g("/cmd/home?secs=2"); time.sleep(3.5)
        if ctl()[0] != "ENGAGED": start_loop()
    emu("TILTHOLD:0:y"); time.sleep(0.2); emu("TILTHOLD:off"); time.sleep(0.2); emu("TILTM:%.4f" % M); time.sleep(1.2)
    g("/cmd/go"); t0 = time.time()
    while time.time() - t0 < tmax:
        if ctl()[0] in ("FAULT_LATCHED", "ESTOP"): break
        time.sleep(0.1)
    g("/cmd/reset"); time.sleep(2.2)
    ep = subprocess.run(["docker", "exec", "kbot-zed", "bash", "-c", "ls -t /root/policy_logs/ep_*.npz | head -1"], capture_output=True, text=True).stdout.strip()
    o = subprocess.run(["docker", "exec", "-i", "kbot-zed", "python3", "-", ep], input=ANALYZE, capture_output=True, text=True).stdout.split()
    t5, th5, p2p, dur = float(o[0]), float(o[1]), float(o[2]), float(o[3])
    return t5, th5, p2p, dur
start_loop()
lo, hi = -0.5, 4.0; best = (0.0, None)      # in this emulator a NEGATIVE pitch moment tips the robot forward (+pitch)
print("sim balance search: play %s deg, rotor stiction %s Nm (pitch moment on the torso, Nm -> result)" % (play, rfc), flush=True)
for k in range(iters):
    M = 0.5 * (lo + hi); t5, th5, p2p, dur = run(M)
    if t5 < 0:
        print("  M %+.4f: stayed up %.1f s (pitch moved %.2f deg in total)" % (M, dur, p2p), flush=True); best = (dur, M); break
    print("  M %+.4f: past 5 deg %s after %.2f s" % (M, "forward" if th5 > 0 else "backward", t5), flush=True)
    if t5 > best[0]: best = (t5, M)
    if th5 > 0: lo = M
    else: hi = M
print("RESULT play %s, stiction %s: longest stay %.1f s at M %+.4f (bracket now %.4f Nm wide)" % (play, rfc, best[0], best[1], hi - lo), flush=True)
g("/cmd/reset")
