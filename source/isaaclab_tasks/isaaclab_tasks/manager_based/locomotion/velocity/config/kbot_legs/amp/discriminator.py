"""AMP discriminator: MLP over normalised transition features, LS-GAN targets ±1.

Ported in structure from menloresearch/isaac_asimov ``algorithms/discriminator.py``
(BSD-3-Clause, Menlo Research): running-statistics feature normaliser with clipping, a
plain MLP, a gradient penalty on expert samples, and the reward map
``r = clamp(1 - 0.25 (D - 1)^2, 0)`` (Peng et al. 2021, AMP).

Differences: the normaliser is updated only when ``update_normalizer`` is called (once per
discriminator step, on the mixed batch), never silently inside ``forward``; inputs are
nan-scrubbed the way this project's rsl_rl guards scrub storage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureNormalizer(nn.Module):
    """Running mean/variance normaliser (Welford), clipped output."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        if n == 0:
            return
        bm = x.mean(0)
        bv = x.var(0, unbiased=False)
        tot = self.count + n
        delta = bm - self.mean
        new_mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = bv * n
        new_var = (m_a + m_b + delta.pow(2) * self.count * n / tot) / tot
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return y.clamp(-self.clip, self.clip)


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 256), activation: str = "relu", device: str = "cpu",
                 normalize: bool = True, clip_obs: float = 10.0, reward_map: str = "log"):
        super().__init__()
        self.reward_map = reward_map
        act = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation]
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), act()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.normalizer = FeatureNormalizer(input_dim, clip=clip_obs) if normalize else nn.Identity()
        self.input_dim = input_dim
        self.to(device)
        self.device = device

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit D(x), shape (N,)."""
        return self.head(self.trunk(self.normalizer(self._prep(x)))).squeeze(-1)

    @torch.no_grad()
    def update_normalizer(self, x: torch.Tensor) -> None:
        if isinstance(self.normalizer, FeatureNormalizer):
            self.normalizer.update(self._prep(x))

    @torch.no_grad()
    def predict_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Style reward from the logit D.

        'amp'  (Peng 2021): clamp(1 - 0.25 (D - 1)^2, 0) — in [0, 1], but EXACTLY 0 for D <= -1: once
               the judge saturates on the policy (pilots 0b/0c: D_policy -0.94..-0.98) it has no slope.
        'log'  (GAIL / DeepMimic): -log(1 - sigmoid(D)). With the LS-GAN keeping D near +-1 this is
               bounded (0.31 at D=-1, 0.69 at 0, 1.31 at +1) and its slope sigmoid(D) never vanishes.
        """
        d = self.forward(x)
        if self.reward_map == "amp":
            return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
        return -torch.log((1.0 - torch.sigmoid(d.clamp(-5.0, 5.0))).clamp(min=1e-4))

    def grad_penalty(self, expert_x: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        """lambda * E[ ||d D/d x||^2 ] on expert samples (gradient taken w.r.t. the normalised input)."""
        x = self.normalizer(self._prep(expert_x)).detach().requires_grad_(True)
        d = self.head(self.trunk(x)).squeeze(-1)
        grad = torch.autograd.grad(d.sum(), x, create_graph=True)[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()
