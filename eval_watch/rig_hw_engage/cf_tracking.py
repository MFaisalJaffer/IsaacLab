"""Closed-loop re-simulation of the policy's own loop with a toy ankle plant.
Everything except ankle jpos/jvel and last_action comes from the logged HW frames (static in
the first 0.4 s anyway). The toy ankle reaches `frac` of the commanded target one tick later:
frac=0 reproduces the hardware dead zone, frac~0.2 the sim-B response."""
import numpy as np, torch, sys
sd = torch.load(sys.argv[2], map_location="cpu"); sd = sd.get("model_state_dict", sd)
Ws = []; k = 0
while f"actor.{k}.weight" in sd:
    Ws.append((sd[f"actor.{k}.weight"].double().numpy(), sd[f"actor.{k}.bias"].double().numpy())); k += 2
def fwd(x):
    h = x.astype(np.float64)
    for j, (W, b) in enumerate(Ws):
        h = W @ h + b
        if j < len(Ws) - 1: h = np.where(h > 0, h, np.expm1(h))
    return h
dims = [3, 3, 10, 10, 3, 10, 4]; off = np.cumsum([0] + dims)
def stack(frames): return np.concatenate([np.concatenate([f[off[i]:off[i + 1]] for f in frames]) for i in range(7)])
F0 = np.load(sys.argv[1], allow_pickle=True)["data"][:, 29:72].copy()
def rollout(frac, nt=22):
    frames = []; la = np.zeros(10); ank = np.zeros(2); out = []
    for k in range(nt):
        f = F0[k].copy(); f[29:39] = la; f[14:16] = ank; f[24:26] = 0.0
        frames.append(f); H = [frames[max(0, j)] for j in range(k - 9, k + 1)]
        a = np.clip(fwd(stack(H)), -8, 8); la = a.copy(); out.append(a[8:10].copy())
        ank = frac * 0.5 * a[8:10]
    return np.array(out)
for frac in (0.0, 0.1, 0.2, 0.5, 1.0):
    o = rollout(frac)
    print("ankle reaches %3.0f%% of target: L ankle action at ticks 5/10/15/20 = %+.2f %+.2f %+.2f %+.2f  (R at t20 %+.2f)"
          % (100 * frac, o[5, 0], o[10, 0], o[15, 0], o[20, 0], o[20, 1]))
