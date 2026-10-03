#!/usr/bin/env python3
"""Independent golden-vector builder for the walker contract.

Deliberately NOT the server's ObsHistory: this keeps a frame-major deque of the last H
43-vectors and gathers per term by slicing, straight from the handoff text:

  "stacked PER TERM, oldest first ... startup fills all slots with the first frame"
  term order: grav3, cmd3, jpos10, jvel10, gyro3, last_action10, gait_phase4
  430 = sum over terms of H*term_dim ; term offsets 0/30/60/160/260/290/390

Actions come from an independently loaded torch MLP (actor.N.weight/bias, ELU).
Writes io_test_vectors.npz with keys: frames (N,43) f32, obs_history (N,430) f32,
actions (N,10) f32, plus H and dims as 0-d arrays.
"""
import sys, collections
import numpy as np

TERM_DIMS = [3, 3, 10, 10, 3, 10, 4]           # 43
H = 10
FRAME_DIM = sum(TERM_DIMS)
N = 300
ckpt = sys.argv[1] if len(sys.argv) > 1 else "walker_v5_model_400.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "io_test_vectors.npz"

rng = np.random.default_rng(20261002)
# frames: smooth-ish random walk so consecutive frames differ (catches stale-slot bugs),
# with a few hard jumps (catches ordering bugs: oldest-first vs newest-first).
frames = np.zeros((N, FRAME_DIM), np.float32)
x = rng.normal(0, 0.3, FRAME_DIM)
for i in range(N):
    x = 0.9 * x + rng.normal(0, 0.1, FRAME_DIM)
    if i in (7, 60, 199):
        x = rng.normal(0, 1.0, FRAME_DIM)
    frames[i] = x.astype(np.float32)
# make the gait_phase block look like a real signed clock so the model sees sane input
th = 0.0
for i in range(N):
    d = -1.0 if frames[i, 3] < -0.05 else 1.0
    th += d * 2 * np.pi * 0.9434 * 0.02
    if i < 2: th = 0.0
    frames[i, 39:43] = [np.sin(th), np.cos(th), np.sin(th + np.pi), np.cos(th + np.pi)]

offs = np.cumsum([0] + TERM_DIMS)                # 0,3,6,16,26,29,39,43
dq = collections.deque(maxlen=H)
stack = np.zeros((N, H * FRAME_DIM), np.float32)
for i in range(N):
    if not dq:
        for _ in range(H):
            dq.append(frames[i])                 # startup: every slot = first frame
    else:
        dq.append(frames[i])                     # deque drops the oldest
    hist = np.stack(list(dq))                    # (H,43), row 0 = oldest
    pieces = []
    for t in range(len(TERM_DIMS)):
        pieces.append(hist[:, offs[t]:offs[t + 1]].reshape(-1))   # H*dim, oldest first
    stack[i] = np.concatenate(pieces)
# sanity on the layout claim in the handoff
term_off = np.cumsum([0] + [H * d for d in TERM_DIMS])
assert list(term_off) == [0, 30, 60, 160, 260, 290, 390, 430], term_off
# oldest-first check on a jump row: at i=60 (jump), slot H-1 of each term is the new frame,
# slot 0 is frame 51
for t in range(len(TERM_DIMS)):
    d = TERM_DIMS[t]; base = term_off[t]
    assert np.array_equal(stack[60, base + (H - 1) * d: base + H * d], frames[60, offs[t]:offs[t + 1]])
    assert np.array_equal(stack[60, base: base + d], frames[51, offs[t]:offs[t + 1]])

import torch
sd = torch.load(ckpt, map_location="cpu")
sd = sd.get("model_state_dict", sd)
layers = []
k = 0
while f"actor.{k}.weight" in sd:
    layers.append((sd[f"actor.{k}.weight"].double().numpy(), sd[f"actor.{k}.bias"].double().numpy()))
    k += 2
assert layers and layers[0][0].shape[1] == H * FRAME_DIM, [l[0].shape for l in layers]
norm_keys = [kk for kk in sd if "normaliz" in kk.lower()]
assert not norm_keys, "unexpected normalizer in ckpt: %s" % norm_keys

def elu(v): return np.where(v > 0, v, np.expm1(v))
def fwd(v):
    h = v.astype(np.float64)
    for j, (W, b) in enumerate(layers):
        h = W @ h + b
        if j < len(layers) - 1: h = elu(h)
    return h
acts = np.stack([fwd(stack[i]) for i in range(N)]).astype(np.float32)
np.savez(out, frames=frames, obs_history=stack, actions=acts,
         H=np.int64(H), frame_dim=np.int64(FRAME_DIM), obs_dim=np.int64(H * FRAME_DIM))
print("wrote", out, "frames", frames.shape, "obs_history", stack.shape, "actions", acts.shape,
      "| widths", [l[0].shape[0] for l in layers], "| act range %.3f..%.3f" % (acts.min(), acts.max()))
