#!/usr/bin/env python3
"""Synthetic emulator: streams 27-f64 obs frames to the policy server at 50 Hz exactly as
virtual_motor_node.policy_loop does, reads the 24-f64 reply, and reports round-trip timing.
Obs = home-ish pose with a slow sway so the history is not constant. Does NOT touch the real
sim stack. Also drives the walk command over UDP :9992 (fwd / stop / back) so the signed
clock and the shaper are exercised live."""
import socket, struct, time, sys, math
import numpy as np
host = "127.0.0.1"; port = 9999; secs = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
s = socket.socket(); s.connect((host, port)); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rt = []; late = 0; n = 0; t_start = time.time(); period = 0.02
jp0 = np.zeros(10);
while time.time() - t_start < secs:
    t0 = time.time(); tt = t0 - t_start
    # command schedule: stand 0-3 s, fwd 0.3 at 3 s, back -0.2 at 6 s, stand at 9 s
    if n == 150: u.sendto(struct.pack("<3d", 0.3, 0.0, 0.0), (host, 9992))
    if n == 300: u.sendto(struct.pack("<3d", -0.2, 0.0, 0.0), (host, 9992))
    if n == 450: u.sendto(struct.pack("<3d", 0.0, 0.0, 0.0), (host, 9992))
    jp = jp0 + 0.05 * math.sin(2 * math.pi * 0.5 * tt)
    jv = np.full(10, 0.05 * 2 * math.pi * 0.5 * math.cos(2 * math.pi * 0.5 * tt))
    ang = 0.02 * math.sin(2 * math.pi * 0.3 * tt)             # small pitch sway, wxyz
    quat = [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0]
    gyro = [0.0, 0.02 * 2 * math.pi * 0.3 * math.cos(2 * math.pi * 0.3 * tt), 0.0]
    s.sendall(struct.pack("<27d", *jp, *jv, *quat, *gyro))
    need = 24 * 8; buf = b""
    while len(buf) < need:
        c = s.recv(need - len(buf))
        if not c: raise SystemExit("server closed")
        buf += c
    rt.append(time.time() - t0); n += 1
    rem = period - (time.time() - t0)
    if rem < 0: late += 1
    time.sleep(max(0.0, rem))
rt = np.array(rt) * 1e3
print("ticks %d in %.1f s (%.1f Hz)  round-trip ms: median %.2f  p95 %.2f  max %.2f  | over-budget ticks (>20 ms): %d"
      % (n, time.time() - t_start, n / (time.time() - t_start), np.median(rt), np.percentile(rt, 95), rt.max(), late))
a = np.frombuffer(buf, dtype="<f8"); print("last reply action20[:10] =", np.round(a[:10], 3), " quat =", np.round(a[20:], 3))
s.close()
