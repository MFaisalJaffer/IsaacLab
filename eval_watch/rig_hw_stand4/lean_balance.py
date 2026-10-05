"""Is a stand free-standing?  Per bin: torso pitch, the ankle moment gravity needs to hold that lean (MGH x sin), the backward-pushing ankle
moment the drives actually apply (kp x (target - encoder), left minus right because the ankle axes are mirrored), and the difference = what
something else (a strap, a hand) must be supplying.  Also heading from the integrated yaw rate.
usage: lean_balance.py <ep.npz> [bin_s] [t1] [t2] [pitch_at_balance_deg]"""
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); r = z["data"]; t = r[:, 0] - r[0, 0]
B = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; T1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; T2 = float(sys.argv[4]) if len(sys.argv) > 4 else t[-1]
P0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3; MGH = 86.0; KP = 60.0
fr = r[:, 29:72]; pitch = np.degrees(np.arcsin(np.clip(fr[:, 1], -1, 1))); gy = fr[:, 26:29]; jp = fr[:, 6:16]; tg = 0.5 * r[:, 82:92]
head = np.degrees(np.cumsum(gy[:, 0]) * 0.02)
eL = tg[:, 8] - jp[:, 8]; eR = tg[:, 9] - jp[:, 9]; push = KP * (eL - eR)          # + = ankles push the body BACK (R ankle + = toe-up; sign confirmed by the 2026-10-05 hand test)
need = -MGH * np.sin(np.radians(pitch - P0))                                        # + = backward push needed (body leaning forward; gravity y < 0 = forward lean)
print("pitch + = BACKWARD lean (policy frame y = backward); balance assumed at pitch %+.1f deg; MGH %.0f Nm/rad" % (P0, MGH))
print("   t(s)      pitch   pitch rate  heading | ankle push back (Nm): L     R   total | gravity needs | extra (other support)")
for t0 in np.arange(T1, T2, B):
    m = (t >= t0) & (t < t0 + B)
    if m.sum() < 2: continue
    print(" %5.1f-%-5.1f %+5.2f   %+6.3f     %+5.1f  |                    %+5.2f %+5.2f  %+5.2f |    %+5.2f      |  %+5.2f" % (
        t0, t0 + B, pitch[m].mean(), -gy[m, 2].mean(), head[m].mean(), KP * eL[m].mean(), -KP * eR[m].mean(), push[m].mean(), need[m].mean(), push[m].mean() - need[m].mean()))
