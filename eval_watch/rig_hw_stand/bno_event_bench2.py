"""Bench of the cycle-anchored event-driven read (driver STOPPED; this script owns the sensor).
Each cycle: sleep until just before the next reports are due (anchored on when the previous cycle's first
packet was found), poll until a packet is there, read until gyro and orientation of this cycle are both in,
'publish', drain whatever else is waiting.  Lean read = header + packet (two I2C transfers) with length guards.
usage: bno_event_bench2.py [guard_ms ...]   (quat 20 ms, gyro 20 ms, accel 200 ms)"""
import struct, sys, time, warnings
warnings.simplefilter("ignore")
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import Packet, BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_GAME_ROTATION_VECTOR
Q, G, A = BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE, BNO_REPORT_ACCELEROMETER
i2c = I2C(8); time.sleep(1.0); bno = BNO08X_I2C(i2c, address=0x4B, debug=False); time.sleep(0.5)
try: bno.hard_reset()
except Exception as e: print("hard_reset failed (%s); soft_reset" % e); bno.soft_reset()
time.sleep(1.5)
for feat, us in ((A, 200000), (G, 20000), (Q, 20000)):
    for _ in range(5):
        try: bno.enable_feature(feat, us); break
        except Exception: time.sleep(0.4)
time.sleep(0.5)
pc = time.perf_counter; new = {Q: False, G: False}; sage = {}
orig = bno._process_report
def hook(rid, rb):
    orig(rid, rb)
    if rid in new: new[rid] = True
bno._process_report = hook
def poll():
    hdr = bno._read_header(); n = hdr.packet_byte_count
    if n == 0 or n == 0x7FFF or n > 512 or hdr.channel_number > 5: return False
    bno._sequence_number[hdr.channel_number] = hdr.sequence_number
    bno._read(n - 4); pkt = Packet(bno._data_buffer); bno._update_sequence_number(pkt)
    d = pkt.data
    if pkt.channel_number == 3 and len(d) > 9 and d[0] == 0xFB:
        sage[d[5]] = struct.unpack_from("<i", d, 1)[0] * 1e-4 + (((d[7] >> 2) << 8) | d[8]) * 1e-4
    bno._handle_packet(pkt); return True
P = 0.020
for gms in [float(x) for x in (sys.argv[1:] or ["2.0"])]:
    guard = gms / 1000.0; DUR = 10.0
    pubs = []; both = 0; lat = []; emp = 0; busy = 0.0; ages = {Q: [], G: []}; missed = 0
    t_cycle = pc(); t0 = pc()
    while pc() - t0 < DUR:
        w = t_cycle + P - guard - pc()
        if w > 0.0003: time.sleep(w)
        tp0 = pc()
        while True:
            s = pc(); ok = poll(); e = pc(); busy += e - s
            if ok: break
            emp += 1
            if e - tp0 > P: time.sleep(0.001)
        if s - t_cycle > 1.5 * P: missed += 1
        t_cycle = s; published = False
        while True:
            if new[Q] and new[G] and not published:
                tpub = pc(); pubs.append(tpub); lat.append(tpub - t_cycle); both += 1; published = True
                for k in (Q, G): ages[k].append(sage.get(k, 0.0) + (tpub - t_cycle))
            s2 = pc(); ok = poll(); busy += pc() - s2
            if not ok: emp += 1; break
        if not published and (new[Q] or new[G]): pubs.append(pc()); lat.append(pubs[-1] - t_cycle)
        new[Q] = new[G] = False
    iv = sorted(b - a for a, b in zip(pubs, pubs[1:])); lat.sort()
    pct = lambda v, p: 1000 * v[min(len(v) - 1, int(p * len(v)))]
    aq = sorted(ages[Q]); ag = sorted(ages[G])
    print("guard %.1f ms: %.1f publishes/s (%.0f%% with both gyro and orientation new) | publish interval median %.2f ms, 1-99%%: %.1f..%.1f, max %.1f | late cycles %d"
          % (gms, len(pubs) / DUR, 100.0 * both / max(1, len(pubs)), pct(iv, 0.5), pct(iv, 0.01), pct(iv, 0.99), 1000 * iv[-1], missed))
    print("   found -> published: median %.1f ms (99%%: %.1f) | sample age at publish: gyro %.1f ms (99%%: %.1f), orientation %.1f ms (99%%: %.1f) | empty polls %.1f per cycle | I2C busy %.0f%%"
          % (pct(lat, 0.5), pct(lat, 0.99), pct(ag, 0.5), pct(ag, 0.99), pct(aq, 0.5), pct(aq, 0.99), emp / max(1, len(pubs)), 100 * busy / DUR))
