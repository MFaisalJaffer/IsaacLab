"""AMPPPO: rsl_rl 2.3.3 PPO + an adversarial style reward (AMP_PLAN.md §2.2).

Hooks mirror the built-in RND path:
  * ``process_env_step``: reads the transition pair from ``infos["observations"]["amp"]`` and
    the walk gate from ``infos["observations"]["amp_gate"]``, adds
    ``style_weight * gate * r_style`` to the extrinsic reward, and stores the pair for
    discriminator training (masked on ``done`` steps — the pair there is the post-reset
    duplicate, and on gate == 0 — standing envs must never teach or be taught style);
  * ``update``: trains the discriminator (LS-GAN ±1, gradient penalty, replay mixing, expert
    mirror augmentation) BEFORE the PPO update, then defers to PPO. Everything PPO does —
    including the project's [kbot guard] NaN scrubs and poisoned-mini-batch skips — is
    inherited untouched.

Reward mixing is additive (r_task + w_s * dt * gate * r_s), not Menlo's lerp: the lineage's
standing/push economy stays exactly as tuned and w_s is the one knob (AMP_PLAN §2.2).

UNITS: ``style_weight`` is in the same units as a RewTerm weight. Isaac Lab's reward manager
multiplies every term by the step dt (0.02 s), so a task term is worth ~0.05/step; the style
bonus is therefore scaled by dt here too, exactly as rsl_rl scales its RND weight
(``rnd_cfg["weight"] *= step_dt`` in the runner). The first pilot added it unscaled and the
style bonus was ~40x the whole task reward (style_share = 1.00) — the policy was being paid
only to look like the reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import optim

from rsl_rl.algorithms.ppo import PPO

from .discriminator import AMPDiscriminator
from .motion_dataset import MotionDataset
from .replay_buffer import ReplayBuffer

AMP_DEFAULTS = dict(
    style_weight=6.0,             # RewTerm-style weight on the [0, 1] style reward (x dt at runtime)
    disc_hidden_dims=(256, 256),
    disc_activation="relu",
    disc_learning_rate=5e-5,      # pilot 0b: at 1e-4 x 16 steps the judge saturated at +-0.98 -> zero style gradient
    disc_weight_decay=1e-4,
    logit_reg=0.3,                # penalise D^2 so outputs stay off +-1 (0.05 balanced the LS loss only at |D|=0.95, pilot 0c)
    reward_map="log",             # 'log' = -log(1-sigmoid(D)), no dead zone; 'amp' = Peng's clamp (pilots 0..0c)
    grad_pen_lambda=10.0,
    replay_size=200_000,
    replay_frac=0.5,              # share of each policy batch drawn from the replay buffer
    disc_steps=8,                 # discriminator gradient steps per PPO update (16 in pilots 0/0b)
    disc_batch=2048,              # policy transitions per step (same number of expert ones)
    update_interval=1,            # train the discriminator every N PPO updates
    mirror_augment=True,
    motion_files=[],
    obs_dim=None,                 # filled by the runner from the env's `amp` group
    fps=None,                     # filled by the runner (1 / step_dt)
)


class AMPPPO(PPO):
    def __init__(self, policy, amp_cfg: dict, device: str = "cpu", **kwargs):
        super().__init__(policy, device=device, **kwargs)
        cfg = dict(AMP_DEFAULTS)
        cfg.update(amp_cfg)
        assert cfg["obs_dim"] is not None and cfg["fps"] is not None, "runner must fill obs_dim and fps"
        assert cfg["motion_files"], "amp_cfg.motion_files is empty"
        self.amp_cfg = cfg
        self.amp_obs_dim = int(cfg["obs_dim"])
        self.style_weight = float(cfg["style_weight"])
        self.reward_dt = 1.0 / float(cfg["fps"])  # Isaac Lab scales every reward term by step dt
        self.disc = AMPDiscriminator(self.amp_obs_dim, tuple(cfg["disc_hidden_dims"]), cfg["disc_activation"], device=device,
                                     reward_map=str(cfg["reward_map"]))
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=cfg["disc_learning_rate"], weight_decay=cfg["disc_weight_decay"])
        self.dataset = MotionDataset(list(cfg["motion_files"]), device=device, fps_expected=float(cfg["fps"]), mirror=bool(cfg["mirror_augment"]))
        assert self.dataset.obs_dim == self.amp_obs_dim, (
            f"dataset feature dim {self.dataset.obs_dim} != env amp obs dim {self.amp_obs_dim}: "
            "the env's `amp` group must be [joint_pos hist-2, joint_vel hist-2] over the same joints")
        self.replay = ReplayBuffer(self.amp_obs_dim, int(cfg["replay_size"]), device)
        print(f"[AMP] {self.dataset.describe()}; style_weight {self.style_weight} (x dt {self.reward_dt:.3f} = {self.style_weight * self.reward_dt:.3f}/step at r_s=1); disc {cfg['disc_hidden_dims']} "
              f"lr {cfg['disc_learning_rate']} wd {cfg['disc_weight_decay']} logit_reg {cfg['logit_reg']} gp {cfg['grad_pen_lambda']} reward_map {cfg['reward_map']} steps/update {cfg['disc_steps']} x {cfg['disc_batch']}")
        self._amp_buf = None
        self._amp_valid = None
        self._amp_step = 0
        self._n_updates = 0
        self._stats = self._zero_stats()
        self.amp_log: dict[str, float] = {}

    # ------------------------------------------------------------------ storage
    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape)
        self._amp_buf = torch.zeros(num_transitions_per_env, num_envs, self.amp_obs_dim, device=self.device)
        self._amp_valid = torch.zeros(num_transitions_per_env, num_envs, dtype=torch.bool, device=self.device)
        self._amp_step = 0

    @staticmethod
    def _zero_stats():
        return dict(style_raw_sum=0.0, style_w_sum=0.0, task_sum=0.0, gated=0, steps=0)

    # ------------------------------------------------------------------ rollout hook
    def process_env_step(self, rewards, dones, infos):
        amp_obs = infos["observations"]["amp"].to(self.device)
        gate = infos["observations"]["amp_gate"].to(self.device).reshape(-1)
        done = dones.to(self.device).bool()
        r_style = self.disc.predict_reward(amp_obs)  # (N,) in [0, 1]
        live = (gate > 0.5) & ~done
        style = self.style_weight * self.reward_dt * r_style * live.float()
        if self._amp_step < self._amp_buf.shape[0]:
            self._amp_buf[self._amp_step] = amp_obs
            self._amp_valid[self._amp_step] = live
        self._amp_step += 1
        n_live = int(live.sum())
        self._stats["style_raw_sum"] += float(r_style[live].sum()) if n_live else 0.0
        self._stats["style_w_sum"] += float(style.sum())
        self._stats["task_sum"] += float(rewards.to(self.device)[live].clamp(min=0.0).sum()) if n_live else 0.0
        self._stats["gated"] += n_live
        self._stats["steps"] += 1
        super().process_env_step(rewards + style.to(rewards.device), dones, infos)

    # ------------------------------------------------------------------ learning hook
    def update(self):
        self._n_updates += 1
        amp_losses = {}
        if self._n_updates % int(self.amp_cfg["update_interval"]) == 0:
            amp_losses = self._update_discriminator()
        loss_dict = super().update()
        s = self._stats
        g = max(s["gated"], 1)
        self.amp_log = {
            "style_reward_raw": s["style_raw_sum"] / g,
            "style_reward_weighted": s["style_w_sum"] / g,
            "task_reward_pos": s["task_sum"] / g,
            "style_share": s["style_w_sum"] / max(s["style_w_sum"] + s["task_sum"], 1e-6),
            "gated_frac": s["gated"] / max(s["steps"] * self._amp_buf.shape[1], 1),
            **amp_losses,
        }
        self._stats = self._zero_stats()
        self._amp_step = 0
        loss_dict.update({f"amp_{k}": v for k, v in amp_losses.items()})
        return loss_dict

    def _update_discriminator(self) -> dict:
        n_steps = min(self._amp_step, self._amp_buf.shape[0])
        buf = self._amp_buf[:n_steps]
        valid = self._amp_valid[:n_steps]
        x_new = buf[valid]
        m = x_new.shape[0]
        if m < 16:
            return {"disc_skipped_no_walkers": 1.0}
        self.replay.insert(x_new)
        cfg = self.amp_cfg
        B = int(cfg["disc_batch"])
        n_old = int(B * float(cfg["replay_frac"])) if len(self.replay) > 0 else 0
        n_new = B - n_old
        lam = float(cfg["grad_pen_lambda"])
        self.disc.train()
        acc = dict(loss=0.0, gp=0.0, expert_pred=0.0, policy_pred=0.0, acc_expert=0.0, acc_policy=0.0)
        done_steps = 0
        for _ in range(int(cfg["disc_steps"])):
            xp = x_new[torch.randint(0, m, (n_new,), device=self.device)]
            if n_old:
                xp = torch.cat([xp, self.replay.sample(n_old)], 0)
            xe = self.dataset.sample(B)
            self.disc.update_normalizer(torch.cat([xp, xe], 0))
            d_e = self.disc(xe)
            d_p = self.disc(xp)
            loss_e = F.mse_loss(d_e, torch.ones_like(d_e))
            loss_p = F.mse_loss(d_p, -torch.ones_like(d_p))
            gp = self.disc.grad_penalty(xe, lam)
            logit = float(cfg["logit_reg"]) * 0.5 * (d_e.pow(2).mean() + d_p.pow(2).mean())
            loss = 0.5 * (loss_e + loss_p) + gp + logit
            if not torch.isfinite(loss):
                print("[AMP] non-finite discriminator loss, skipping step")
                continue
            self.disc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
            self.disc_optimizer.step()
            acc["loss"] += float(loss)
            acc["gp"] += float(gp)
            acc["expert_pred"] += float(d_e.mean())
            acc["policy_pred"] += float(d_p.mean())
            acc["acc_expert"] += float((d_e > 0).float().mean())
            acc["acc_policy"] += float((d_p < 0).float().mean())
            done_steps += 1
        self.disc.eval()
        if done_steps == 0:
            return {"disc_skipped_nonfinite": 1.0}
        out = {k: v / done_steps for k, v in acc.items()}
        out["policy_transitions"] = float(m)
        return out
