"""KbotAmpRunner: rsl_rl 2.3.3 OnPolicyRunner that trains with AMPPPO.

Why a subclass: the stock runner hard-codes ``class_name in ("PPO", "Distillation")`` and
resolves the class with ``eval`` inside its own module, so an AMP algorithm cannot be
selected from config alone. The subclass lets the parent build the policy, storage and
loggers for a plain PPO, then swaps in AMPPPO (same policy object) sized from the env's
``amp`` observation group. Checkpoints additionally carry the discriminator and its
optimizer; TensorBoard gets an ``AMP/`` section.

At construction it also verifies, by stepping the env once, that the ``amp`` group is laid
out as the dataset assumes ([q_t, q_t+1, qd_t, qd_t+1] over the robot's joints).
"""
from __future__ import annotations

import math
import os

import torch

from rsl_rl.runners import OnPolicyRunner

from .amp_ppo import AMPPPO


class KbotAmpRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        alg_cfg = train_cfg["algorithm"]
        assert alg_cfg.get("class_name") == "AMPPPO", f"KbotAmpRunner expects algorithm.class_name == 'AMPPPO', got {alg_cfg.get('class_name')}"
        amp_cfg = dict(alg_cfg.pop("amp_cfg"))
        alg_cfg["class_name"] = "PPO"  # let the parent build policy/storage/normalisers for a plain PPO
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        obs, extras = self.env.get_observations()
        groups = extras["observations"]
        for key in ("amp", "amp_gate"):
            if key not in groups:
                raise ValueError(f"observation group '{key}' missing from infos['observations'] — use the AMP env config")
        amp_cfg["obs_dim"] = int(groups["amp"].shape[1])
        amp_cfg["fps"] = 1.0 / float(self.env.unwrapped.step_dt)
        amp_cfg["motion_files"] = [os.path.abspath(p) for p in amp_cfg.get("motion_files", [])]

        num_obs = obs.shape[1]
        num_privileged_obs = groups[self.privileged_obs_type].shape[1] if self.privileged_obs_type is not None else num_obs
        self.alg = AMPPPO(self.alg.policy, amp_cfg=amp_cfg, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        self.alg.init_storage(self.training_type, self.env.num_envs, self.num_steps_per_env, [num_obs], [num_privileged_obs], [self.env.num_actions])
        self._check_amp_layout()

    # ------------------------------------------------------------------ layout guard
    def _check_amp_layout(self) -> None:
        robot = self.env.unwrapped.scene["robot"]
        J = robot.num_joints
        assert self.alg.amp_obs_dim == 4 * J, f"amp obs dim {self.alg.amp_obs_dim} != 4 x {J} joints"
        with torch.inference_mode():
            q_before = (robot.data.joint_pos - robot.data.default_joint_pos).clone()
            _, _, dones, infos = self.env.step(torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device))
            amp = infos["observations"]["amp"]
            q_after = robot.data.joint_pos - robot.data.default_joint_pos
            qd_after = robot.data.joint_vel - robot.data.default_joint_vel
            ok = ~dones.bool()
            e_prev = (amp[ok, :J] - q_before[ok]).abs().max()
            e_cur = (amp[ok, J:2 * J] - q_after[ok]).abs().max()
            e_vel = (amp[ok, 3 * J:] - qd_after[ok]).abs().max()
        if max(float(e_prev), float(e_cur), float(e_vel)) > 1e-4:
            raise RuntimeError(
                f"amp observation layout mismatch: |prev q err| {float(e_prev):.2e}, |cur q err| {float(e_cur):.2e}, "
                f"|cur qd err| {float(e_vel):.2e}. Expected [q_t, q_t+1, qd_t, qd_t+1] (history-2 terms, oldest first).")
        print(f"[AMP] observation layout verified: [q_t | q_t+1 | qd_t | qd_t+1], {J} joints, dt {1.0 / self.alg.amp_cfg['fps']:.3f} s")

    # ------------------------------------------------------------------ logging
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        if self.writer is not None and not self.disable_logs:
            for k, v in self.alg.amp_log.items():
                self.writer.add_scalar(f"AMP/{k}", v, locs["it"])

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str, infos=None):
        super().save(path, infos)
        d = torch.load(path, weights_only=False)
        d["amp_disc_state_dict"] = self.alg.disc.state_dict()
        d["amp_disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        d["amp_cfg"] = {k: v for k, v in self.alg.amp_cfg.items()}
        torch.save(d, path)

    def load(self, path: str, load_optimizer: bool = True):
        result = super().load(path, load_optimizer)
        d = torch.load(path, weights_only=False)
        # Warm starts from a converged policy carry its collapsed exploration noise (the tracked
        # policy: std 0.035, frozen for 1000 AMP iterations in pilot 1 -> standing/turning barely
        # learned). KBOT_AMP_INIT_STD re-inflates the actor's noise after loading.
        init_std = os.environ.get("KBOT_AMP_INIT_STD")
        if init_std:
            target = float(init_std)
            policy = self.alg.policy
            with torch.no_grad():
                if hasattr(policy, "log_std"):
                    policy.log_std.fill_(math.log(target))
                elif hasattr(policy, "std"):
                    policy.std.fill_(target)
            print(f"[AMP] actor exploration noise re-inflated to std {target} after load (KBOT_AMP_INIT_STD)")
        if "amp_disc_state_dict" in d:
            self.alg.disc.load_state_dict(d["amp_disc_state_dict"])
            if load_optimizer and "amp_disc_optimizer_state_dict" in d:
                self.alg.disc_optimizer.load_state_dict(d["amp_disc_optimizer_state_dict"])
            print("[AMP] discriminator restored from checkpoint")
        else:
            print("[AMP] checkpoint has no discriminator (plain PPO weights) — starting the discriminator fresh")
        return result
