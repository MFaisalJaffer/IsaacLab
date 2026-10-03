import numpy as np, sys
names={13:"L hip pitch",14:"L hip roll",15:"L hip yaw",16:"L knee",17:"L ankle",3:"R hip pitch",4:"R hip roll",5:"R hip yaw",6:"R knee",7:"R ankle"}
sh=np.load(sys.argv[1],allow_pickle=True); en=np.load(sys.argv[2],allow_pickle=True)
def ser(z,n,kind):
    w=z["wire"]; t0=z["data"][0,0]; m=(w[:,1].astype(int)==n)&(w[:,2].astype(int)==kind); return w[m,0]-t0, w[m,3], w[m,5]
thr=np.radians(0.10)
print("%-12s preload | breakaway: t      err(cmd-fb)  |tau| reported  kp  | tracking dev/err @0.3s @0.4s | note" % "joint")
for n in (13,3,14,4,15,5,16,6,17,7):
    t,p,tau=ser(sh,n,1); m=(t>2)&(t<30); pre=np.abs(tau[m]).mean()
    te,pe,taue=ser(en,n,1); tc,pc,kpc=ser(en,n,0); rest=np.median(pe[te<0.1])
    err=lambda x: np.interp(x,tc,pc)-np.interp(x,te,pe)
    dev=pe-rest; idx=np.where((np.abs(dev)>thr)&(te>0.12))[0]
    def frac(x):
        e=err(x); d=np.interp(x,te,pe)-rest; return (d/(d+e)) if abs(d+e)>1e-4 else float("nan")   # fraction of the way to the command
    note=""
    if n in (16,6): note="command stayed at 0 until the lurch; no small-signal probe"
    if n in (17,7):
        x=0.40; d=abs(np.interp(x,te,pe)-rest); note="@0.4s encoder moved %.1f deg = %.1f Nm of spring at K_s 52; reported %.1f Nm -> motion is spring wind-up, body not moving" % (np.degrees(d), 52*d, abs(np.interp(x,te,taue)))
    if len(idx):
        i=idx[0]; tm=te[i]; tq=np.abs(taue[max(0,i-2):i+1]).max()
        # direction: moving toward or away from the command?
        toward = np.sign(dev[i])==np.sign(err(tm)+dev[i]) if abs(err(tm))>1e-4 else True
        print("%-12s %4.2f Nm | %5.2f s  %+5.2f deg   %4.2f Nm       %3.0f | %4.2f  %4.2f | %s%s" % (names[n], pre, tm, np.degrees(err(tm)), tq, np.interp(tm,tc,kpc), frac(0.3), frac(0.4), "" if toward else "moved AWAY from command (pushed by load) ", note))
    else:
        print("%-12s %4.2f Nm |    no motion > 0.1 deg before 0.88 s; max |tau| %.2f Nm | %s" % (names[n], pre, np.abs(taue).max(), note))
