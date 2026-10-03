"""Reference-motion dataset: consecutive-frame transitions in the discriminator's feature layout.

Feature layout (must match the env's ``amp`` observation group, whose two terms each carry a
history of 2 flattened oldest-first):

    x = [ q_t (J) | q_t+1 (J) | qd_t (J) | qd_t+1 (J) ]        term-major, oldest first

Accepted npz layouts (both written at the policy rate, 50 Hz):
  * ``(T, n_env, J)`` — ``eval_watch/amp_track_eval.py --record``: per-env episodes of one
    policy, no resets inside; pairs never cross env boundaries;
  * ``(T, J)`` — a single clip (``amp_build_ref.py`` output or a Menlo-style file).

Mirror augmentation doubles the data with the left/right mirror used by the rsl_rl symmetry
loss (swap L/R pairs, negate — see symmetry.py). It also cancels any turning bias a recorded
policy carried, so the discriminator never learns "walking" == "drifting left".
"""
from __future__ import annotations

import numpy as np
import torch

# symmetry.py: joint order [Lhp,Rhp, Lhr,Rhr, Lhy,Rhy, Lk,Rk, La,Ra] -> swap each pair and negate
_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]


def mirror_transitions(x: torch.Tensor, n_joints: int = 10) -> torch.Tensor:
    """Mirror every J-block of a (N, 4J) transition tensor."""
    assert x.shape[-1] == 4 * n_joints and n_joints == len(_PERM), x.shape
    blocks = x.reshape(*x.shape[:-1], 4, n_joints)
    return (-blocks[..., _PERM]).reshape(x.shape)


def transitions_from_series(q: np.ndarray, qd: np.ndarray, done: np.ndarray | None = None) -> np.ndarray:
    """(T, J) series -> (T-1, 4J) transitions. With a (T,) done mask, pairs whose second frame is a
    post-reset frame (done[t] True) are dropped — recordings of episodic trackers restart mid-stream."""
    x = np.concatenate([q[:-1], q[1:], qd[:-1], qd[1:]], axis=1)
    if done is not None:
        x = x[~done[1:].astype(bool)]
    return x


class MotionDataset:
    def __init__(self, files: list[str], device: str, fps_expected: float | None = None, mirror: bool = True,
                 default_joint_pos: np.ndarray | None = None, dtype=torch.float32):
        chunks = []
        self.files = list(files)
        self.n_joints = None
        self.joint_names = None
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            fps = float(np.asarray(d["fps"]).reshape(-1)[0])
            if fps_expected is not None:
                assert abs(fps - fps_expected) < 1e-3, f"{f}: fps {fps} != policy rate {fps_expected}"
            q = np.asarray(d["joint_pos"], dtype=np.float32)
            qd = np.asarray(d["joint_vel"], dtype=np.float32)
            names = [str(s) for s in d["joint_names"]] if "joint_names" in d else None
            if self.joint_names is None:
                self.joint_names = names
            elif names is not None:
                assert names == self.joint_names, f"{f}: joint order differs from {self.files[0]}"
            if default_joint_pos is not None:
                q = q - default_joint_pos.reshape(1, -1) if q.ndim == 2 else q - default_joint_pos.reshape(1, 1, -1)
            done = np.asarray(d["done"]) if "done" in d else None
            if q.ndim == 3:  # (T, n_env, J): one series per env
                for e in range(q.shape[1]):
                    chunks.append(transitions_from_series(q[:, e], qd[:, e], done[:, e] if done is not None else None))
            elif q.ndim == 2:
                chunks.append(transitions_from_series(q, qd))
            else:
                raise ValueError(f"{f}: joint_pos has shape {q.shape}")
            j = q.shape[-1]
            assert self.n_joints in (None, j), "joint count differs between files"
            self.n_joints = j
        x = torch.tensor(np.concatenate(chunks, axis=0), dtype=dtype, device=device)
        self.n_real = x.shape[0]
        if mirror:
            x = torch.cat([x, mirror_transitions(x, self.n_joints)], dim=0)
        self.x = x
        self.device = device
        self.obs_dim = x.shape[1]
        self.mirrored = mirror

    def __len__(self) -> int:
        return self.x.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.x.shape[0], (n,), device=self.device)
        return self.x[idx]

    def describe(self) -> str:
        return (f"MotionDataset: {len(self.files)} file(s), {self.n_real} real transitions"
                f"{' + mirror' if self.mirrored else ''} = {len(self)} rows x {self.obs_dim} features, joints={self.n_joints}")
