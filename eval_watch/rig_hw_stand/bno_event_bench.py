"""Bench for an event-driven BNO08x read (driver STOPPED; this script owns the sensor).
 1. flat-out drain with a leaner read (header + packet, no separate data-ready check): cost per packet, cost of an
    empty poll, when gyro and orientation reports arrive relative to each other, and how old each sample already
    is when the sensor offers it (SH-2 timebase + per-report delay).
 2. the loop the driver would run: sleep until just before the next report is due, poll until it arrives, hand it
    over at once.  Reports hand-overs per second, the wait before each report was picked up, and bus time used.
 3. how long the periodic feature refresh blocks the loop.
usage: bno_event_bench.py [quat_ms=20] [gyro_ms=20] [accel_ms=200] [guard_ms=2.0]"""
import struct, sys, time
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import PacketError, BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_GAME_ROTATION_VECTOR
import warnings; warnings.simplefilter("ignore")
qms, gms, ams = [float(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((1, 20.0), (2, 20.0), (3, 200.0))]
guard = (float(sys.argv[4]) if len(sys.argv) > 4 else 2.0) / 1000.0
Q, G, A = BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE, BNO_REPORT_ACCELEROMETER; NAMES = {Q: "quat", G: "gyro", A: "accel"}
i2c = I2C(8); time.sleep(1.0); bno = BNO08X_I2C(i2c, address=0x4B, debug=False); time.sleep(0.5)
try: bno.hard_reset()
except Exception as e: print("hard_reset failed (%s); soft_reset" % e); bno.soft_reset()
time.sleep(1.5)
def enable():
    for feat, ms in ((A, ams), (G, gms), (Q, qms)):
        for _ in range(5):
            try: bno.enable_feature(feat, int(ms * 1000)); break
            except Exception: time.sleep(0.4)
enable(); time.sleep(0.5)
pc = time.perf_counter
def poll():
    """one read attempt; returns list of (report_id, sensor_age_s) or None if nothing was waiting"""
    try: pkt = bno._read_packet()
    except PacketError: return None
    out = []
    d = pkt.data
    if pkt.channel_number == 3 and len(d) > 9 and d[0] == 0xFB:
        base = struct.unpack_from("<i", d, 1)[0] * 1e-4; rid = d[5]; delay = (((d[7] >> 2) << 8) | d[8]) * 1e-4
        out.append((rid, base + delay))
    bno._handle_packet(pkt); return out
# ---- 1. flat-out drain
arr = {Q: [], G: [], A: []}; age = {Q: [], G: [], A: []}; tp = []; te = []; t0 = pc()
while pc() - t0 < 4.0:
    s = pc(); r = poll(); e = pc()
    if r is None: te.append(e - s); continue
    tp.append(e - s)
    for rid, ag in r:
        if rid in arr: arr[rid].append(e); age[rid].append(ag)
import statistics as st
med = lambda x: st.median(x) if x else float("nan")
print("1) flat-out, lean read: packet %.2f ms (median), empty poll %.2f ms | reports/s: %s" % (1000 * med(tp), 1000 * med(te), " ".join("%s %.0f" % (NAMES[k], len(arr[k]) / 4.0) for k in arr)))
for k in (Q, G):
    iv = [b - a for a, b in zip(arr[k], arr[k][1:])]
    print("   %s: interval median %.2f ms (5-95%%: %.1f..%.1f) | sample already %.1f ms old when offered (5-95%%: %.1f..%.1f)"
          % (NAMES[k], 1000 * med(iv), 1000 * sorted(iv)[len(iv) // 20], 1000 * sorted(iv)[-len(iv) // 20 - 1], 1000 * med(age[k]), 1000 * sorted(age[k])[len(age[k]) // 20], 1000 * sorted(age[k])[-len(age[k]) // 20 - 1]))
ph = []
for tq in arr[Q]:
    prev = [tg for tg in arr[G] if tg <= tq]
    if prev: ph.append(tq - prev[-1])
print("   orientation report arrives %.1f ms after the latest gyro report (median; 5-95%%: %.1f..%.1f)" % (1000 * med(ph), 1000 * sorted(ph)[len(ph) // 20], 1000 * sorted(ph)[-len(ph) // 20 - 1]))
# ---- 2. the driver loop: predict, sleep, poll, hand over
last = {Q: pc(), G: pc()}; per = {Q: qms / 1000.0, G: gms / 1000.0}; hist = {Q: [], G: []}
hand = {Q: [], G: []}; waits = []; busy = 0.0; empties = 0; t0 = pc(); last_poll_end = pc(); DUR = 8.0
while pc() - t0 < DUR:
    now = pc(); nxt = min(last[k] + per[k] for k in (Q, G)); w = nxt - guard - now
    if w > 0.0003: time.sleep(w)
    s = pc(); r = poll(); e = pc(); busy += e - s
    if r is None: empties += 1; last_poll_end = e; continue
    for rid, ag in r:
        if rid in last:
            hist[rid].append(e - last[rid]); last[rid] = e; hand[rid].append(e); waits.append((s - last_poll_end, e - s, ag))
            if len(hist[rid]) >= 8: per[rid] = st.median(hist[rid][-25:])
    last_poll_end = e
print("2) driver loop (guard %.1f ms): hand-overs/s quat %.1f gyro %.1f | empty polls %.0f/s | I2C busy %.0f%% of the time"
      % (1000 * guard, len(hand[Q]) / DUR, len(hand[G]) / DUR, empties / DUR, 100 * busy / DUR))
gap = sorted(x[0] for x in waits); rd = sorted(x[1] for x in waits); tot = sorted(x[0] + x[1] for x in waits); ag = sorted(x[2] for x in waits)
pct = lambda v, p: 1000 * v[min(len(v) - 1, int(p * len(v)))]
print("   per report: unseen for at most %.1f ms (median; 95%%: %.1f, max %.1f) + read %.1f ms => handed over within %.1f ms of being offered (median; 95%%: %.1f, max %.1f)"
      % (pct(gap, 0.5), pct(gap, 0.95), 1000 * gap[-1], pct(rd, 0.5), pct(tot, 0.5), pct(tot, 0.95), 1000 * tot[-1]))
print("   sensor-internal age when offered: median %.1f ms | so sample -> hand-over is about %.1f ms (median), %.1f ms (95%%)" % (pct(ag, 0.5), pct(ag, 0.5) + pct(tot, 0.5), pct(ag, 0.5) + pct(tot, 0.95)))
for k in (Q, G):
    iv = sorted(b - a for a, b in zip(hand[k], hand[k][1:]))
    print("   %s hand-over interval: median %.2f ms, 5-95%%: %.1f..%.1f, max %.1f" % (NAMES[k], pct(iv, 0.5), pct(iv, 0.05), pct(iv, 0.95), 1000 * iv[-1]))
# ---- 3. feature refresh cost
s = pc(); enable_t = []
for feat, ms in ((A, ams), (G, gms), (Q, qms)):
    a = pc(); bno.enable_feature(feat, int(ms * 1000)); enable_t.append(pc() - a)
print("3) periodic refresh (re-enable 3 features) blocks the loop for %.0f ms (%s)" % (1000 * (pc() - s), " + ".join("%.0f" % (1000 * x) for x in enable_t)))
