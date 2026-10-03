"""Ring buffer of past policy transitions for discriminator training (Escontrela et al. 2022:
mixing old policy samples in stabilises the adversary against a fast-moving policy)."""
from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int, device: str):
        self.x = torch.zeros(capacity, obs_dim, device=device)
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

    def insert(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            self.x.copy_(x[-self.capacity:])
            self.ptr, self.size = 0, self.capacity
            return
        end = self.ptr + n
        if end <= self.capacity:
            self.x[self.ptr:end] = x
        else:
            k = self.capacity - self.ptr
            self.x[self.ptr:] = x[:k]
            self.x[: n - k] = x[k:]
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.size, (n,), device=self.device)
        return self.x[idx]

    def __len__(self) -> int:
        return self.size
