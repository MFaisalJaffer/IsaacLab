"""Unit tests for the AMP package (AMP_PLAN.md §2.3, tests 1 and the dataset checks).

No Isaac Sim: the package is loaded by path so the isaaclab_tasks import chain is skipped.
  python eval_watch/amp_unit_tests.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AMP_DIR = os.path.join(ROOT, "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/kbot_legs/amp")
DATA = os.path.join(ROOT, "eval_watch/amp_refs/asimov_tracked_kbot.npz")

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
p.add_argument("--steps", type=int, default=1500)
a = p.parse_args()
dev = a.device

spec = importlib.util.spec_from_file_location("kbot_amp", os.path.join(AMP_DIR, "__init__.py"), submodule_search_locations=[AMP_DIR])
amp = importlib.util.module_from_spec(spec)
sys.modules["kbot_amp"] = amp
spec.loader.exec_module(amp)

fails = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- 1. dataset layout
print("== dataset")
raw = np.load(DATA, allow_pickle=True)
q, qd = raw["joint_pos"], raw["joint_vel"]
ds = amp.MotionDataset([DATA], device=dev, fps_expected=50.0, mirror=True)
print("  ", ds.describe())
J = ds.n_joints
x = ds.x
check("feature dim = 4J", ds.obs_dim == 4 * J, f"({ds.obs_dim})")
check("real rows = n_env*(T-1)", ds.n_real == q.shape[1] * (q.shape[0] - 1), f"({ds.n_real})")
e0 = torch.tensor(np.concatenate([q[0, 0], q[1, 0], qd[0, 0], qd[1, 0]]), device=dev)
check("row 0 = [q_0, q_1, qd_0, qd_1] of env 0", torch.allclose(x[0], e0, atol=1e-6))
last_env0 = q.shape[0] - 2
e1 = torch.tensor(np.concatenate([q[last_env0, 0], q[last_env0 + 1, 0], qd[last_env0, 0], qd[last_env0 + 1, 0]]), device=dev)
check("last row of env 0 does not cross into env 1", torch.allclose(x[q.shape[0] - 2], e1, atol=1e-6))
m = amp.mirror_transitions(x[: ds.n_real], J)
check("mirror is an involution", torch.allclose(amp.mirror_transitions(m, J), x[: ds.n_real], atol=1e-6))
check("mirror rows appended", torch.allclose(x[ds.n_real:], m, atol=1e-6))
check("mirror swaps L/R hip pitch and negates", torch.allclose(m[:, 0], -x[: ds.n_real, 1]) and torch.allclose(m[:, 1], -x[: ds.n_real, 0]))
s = ds.sample(4096)
check("sample shape", tuple(s.shape) == (4096, 4 * J))

# ---------------------------------------------------------------- 2. replay buffer
print("== replay buffer")
rb = amp.ReplayBuffer(4 * J, 1000, dev)
rb.insert(x[:300]); rb.insert(x[300:1100])
check("wraps to capacity", len(rb) == 1000 and rb.ptr == 100)
check("keeps newest rows", torch.allclose(rb.x[99], x[1099]))
check("sample shape", tuple(rb.sample(64).shape) == (64, 4 * J))

# ---------------------------------------------------------------- 3. discriminator learns expert vs scrambled
print("== discriminator (expert = dataset, 'policy' = time-scrambled pairs)")
disc = amp.AMPDiscriminator(4 * J, (256, 256), "relu", device=dev, reward_map="amp")
opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
real = x[: ds.n_real]


PERIOD = 53  # frames per stride at 50 Hz (1.06 s)


def scrambled(n: int) -> torch.Tensor:
    """q_t paired with q_{t+k}, k drawn 8..45 frames ahead within the same env (same marginals,
    inconsistent dynamics; k avoids the near-consistent 0 / one-period shifts)."""
    i = torch.randint(0, real.shape[0] - PERIOD, (n,), device=dev)
    k = torch.randint(8, PERIOD - 8, (n,), device=dev)
    j = i + k
    return torch.cat([real[i, :J], real[j, J:2 * J], real[i, 2 * J:3 * J], real[j, 3 * J:]], 1)


t0 = time.time()
for step in range(a.steps):
    xe = ds.sample(1024)
    xp = scrambled(1024)
    disc.update_normalizer(torch.cat([xe, xp]))
    de, dp = disc(xe), disc(xp)
    loss = 0.5 * (torch.nn.functional.mse_loss(de, torch.ones_like(de)) + torch.nn.functional.mse_loss(dp, -torch.ones_like(dp))) + disc.grad_penalty(xe, 10.0)
    opt.zero_grad(); loss.backward(); opt.step()
disc.eval()
with torch.no_grad():
    xe, xp = ds.sample(8192), scrambled(8192)
    acc_e = float((disc(xe) > 0).float().mean()); acc_p = float((disc(xp) < 0).float().mean())
    r_e = float(disc.predict_reward(xe).mean()); r_p = float(disc.predict_reward(xp).mean())
gp = float(disc.grad_penalty(xe[:1024], 10.0))
print(f"  {a.steps} steps in {time.time() - t0:.1f} s: acc expert {acc_e:.3f} / scrambled {acc_p:.3f}; reward expert {r_e:.3f} / scrambled {r_p:.3f}; grad-pen {gp:.3f}")
check("expert accuracy > 0.95", acc_e > 0.95)
check("scrambled accuracy > 0.9", acc_p > 0.9)
check("style reward on expert > 0.85", r_e > 0.85)
check("style reward on scrambled < 0.3", r_p < 0.3)
check("gradient penalty finite", np.isfinite(gp))
disc.reward_map = "log"
with torch.no_grad():
    l_e = float(disc.predict_reward(xe).mean()); l_p = float(disc.predict_reward(xp).mean())
print(f"  log reward map: expert {l_e:.3f} / scrambled {l_p:.3f} (floor 0.31, ceiling 1.31)")
check("log map keeps a margin (expert - scrambled > 0.5)", l_e - l_p > 0.5)
check("log map has no dead zone (scrambled >= 0.3)", l_p >= 0.3)
disc.reward_map = "amp"
with torch.no_grad():
    check("nan input does not propagate", torch.isfinite(disc(torch.full((4, 4 * J), float("nan"), device=dev))).all().item())

# ---------------------------------------------------------------- 4. AMPPPO reward hook (no env: fake infos)
print("== AMPPPO process_env_step / update on a fake rollout")
from rsl_rl.modules import ActorCritic  # noqa: E402

N, T, OBS, ACT = 64, 24, 43, J
policy = ActorCritic(OBS, OBS, ACT, actor_hidden_dims=[64], critic_hidden_dims=[64], activation="elu", init_noise_std=1.0).to(dev)
alg = amp.AMPPPO(policy, amp_cfg={"motion_files": [DATA], "obs_dim": 4 * J, "fps": 50.0, "disc_steps": 4, "disc_batch": 256, "replay_size": 5000},
                 device=dev, num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-4, entropy_coef=0.0, gamma=0.99, lam=0.95)
alg.init_storage("rl", N, T, [OBS], [OBS], [ACT])
gate = torch.zeros(N, 1, device=dev); gate[: N // 2] = 1.0
for t in range(T):
    obs = torch.randn(N, OBS, device=dev)
    alg.act(obs, obs)
    dones = torch.zeros(N, dtype=torch.long, device=dev)
    if t == 0:
        dones[0] = 1
    infos = {"observations": {"amp": ds.sample(N), "amp_gate": gate}}
    alg.process_env_step(torch.ones(N, device=dev), dones, infos)
r_stored = alg.storage.rewards[:, :, 0]
check("style added only on gated envs", bool((r_stored[:, N // 2:] == 1.0).all()) and bool((r_stored[1:, 1:N // 2] > 1.0).all()), f"gated mean {float(r_stored[:, :N//2].mean()):.3f} vs {float(r_stored[:, N//2:].mean()):.3f}")
check("done env gets no style on its done step", bool(r_stored[0, 0] == 1.0) and bool(r_stored[1, 0] > 1.0))
check("valid mask excludes done + ungated", int(alg._amp_valid.sum()) == T * (N // 2) - 1)
alg.compute_returns(torch.randn(N, OBS, device=dev))
losses = alg.update()
print("  loss keys:", sorted(losses.keys()))
check("update ran the discriminator", "amp_loss" in losses and np.isfinite(losses["amp_loss"]))
check("amp_log populated", all(k in alg.amp_log for k in ("style_reward_raw", "style_share", "gated_frac", "acc_expert")))
check("gated_frac = 0.5", abs(alg.amp_log["gated_frac"] - (T * (N // 2) - 1) / (T * N)) < 1e-6, f"({alg.amp_log['gated_frac']:.3f})")

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else " — all AMP unit tests passed"))
sys.exit(1 if fails else 0)
