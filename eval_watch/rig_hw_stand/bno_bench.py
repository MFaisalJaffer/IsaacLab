"""Bench the BNO08x link with the driver STOPPED (this script owns the sensor): for each report-interval
setting, count the reports that actually arrive per second and time the I2C work, first draining as fast
as possible, then in a 50 Hz loop that reads the three values the way the driver does.
usage: bno_bench.py "quat_ms,gyro_ms,accel_ms" [...]"""
import sys, time
from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_GAME_ROTATION_VECTOR
i2c = I2C(8); time.sleep(1.0); bno = BNO08X_I2C(i2c, address=0x4B, debug=False); time.sleep(0.5)
try: bno.hard_reset()
except Exception as e: print("hard_reset failed (%s); soft_reset" % e); bno.soft_reset()
time.sleep(1.5)
counts = {}; orig_pr = bno._process_report
def pr(report_id, report_bytes):
    counts[report_id] = counts.get(report_id, 0) + 1; return orig_pr(report_id, report_bytes)
bno._process_report = pr
npk = [0]; orig_rp = bno._read_packet
def rp(*a, **k):
    npk[0] += 1; return orig_rp(*a, **k)
bno._read_packet = rp
NAMES = {BNO_REPORT_ACCELEROMETER: "accel", BNO_REPORT_GYROSCOPE: "gyro", BNO_REPORT_GAME_ROTATION_VECTOR: "quat"}
def enable(q, g, a):
    for feat, ms in ((BNO_REPORT_ACCELEROMETER, a), (BNO_REPORT_GYROSCOPE, g), (BNO_REPORT_GAME_ROTATION_VECTOR, q)):
        for _ in range(5):
            try: bno.enable_feature(feat, int(ms * 1000)); break
            except Exception: time.sleep(0.4)
        else: print("could not enable", NAMES[feat])
    time.sleep(0.5)
def rates(dur):
    return " ".join("%s %.0f/s" % (NAMES[k], counts.get(k, 0) / dur) for k in NAMES)
for cfg in sys.argv[1:]:
    q, g, a = [float(x) for x in cfg.split(",")]; enable(q, g, a)
    counts.clear(); npk[0] = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < 4.0: bno._process_available_packets()
    d = time.perf_counter() - t0
    print("quat %g ms, gyro %g ms, accel %g ms | drain flat out: %s | %d packets/s" % (q, g, a, rates(d), npk[0] / d))
    counts.clear(); npk[0] = 0; ticks = []; nxt = time.perf_counter(); t0 = nxt
    while time.perf_counter() - t0 < 4.0:
        s = time.perf_counter(); _ = bno.acceleration; _ = bno.gyro; _ = bno.game_quaternion; ticks.append(time.perf_counter() - s)
        nxt += 0.02; dt = nxt - time.perf_counter()
        if dt > 0: time.sleep(dt)
        else: nxt = time.perf_counter()
    d = time.perf_counter() - t0; ticks.sort()
    print("    50 Hz driver-style loop: %s | %.1f loops/s | I2C work per loop: median %.1f ms, 95th pct %.1f ms, max %.1f ms"
          % (rates(d), len(ticks) / d, 1000 * ticks[len(ticks) // 2], 1000 * ticks[int(0.95 * len(ticks))], 1000 * ticks[-1]))
