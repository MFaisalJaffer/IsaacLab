"""Warm-start checkpoint for the multi-cycle tracker v2 (actor with H frames of observation history).

Source: a 43-input tracker checkpoint (v1d). The history layout keeps every term's H frames contiguous,
oldest first, so the newest frame of term k sits in the LAST d_k columns of its block. The old first-layer
weights are copied onto those columns and the older frames start at zero: the padded policy acts exactly
like the source until it learns to use the history. Noise is set to --std, optimizer fresh.

Checks: (a) history order (the newest frame at step t is the second-newest at t+1), (b) the padded actor's
action equals the source actor's action on the newest frame.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_tm_prep_warm.py \
      --src <43-D ckpt> --out archive_anchors/trackmulti_v2_warm.pt --std 0.35 --headless
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--std", type=float, default=0.35)
parser.add_argument("--task", default="Isaac-TrackMulti-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

FRAME = [3, 3, 10, 10, 3, 10, 4]  # projgrav, velcmd, jointpos, jointvel, imu, actions, gaitphase


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    dev = uenv.device
    n = args.num_envs
    om = uenv.observation_manager
    names = om.active_terms["policy"]; dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    print(f"[prep] policy obs {env.num_obs}, critic {env.num_privileged_obs}; terms {list(zip(names, dims))}")
    assert len(dims) == len(FRAME) and all(d % f == 0 for d, f in zip(dims, FRAME)), "unexpected policy layout"
    H = dims[0] // FRAME[0]
    assert all(d == H * f for d, f in zip(dims, FRAME)) and env.num_obs == 43 * H
    # column map: old column -> newest-frame column of the same term
    col, off_old, off_new = [], 0, 0
    for f in FRAME:
        col += [off_new + (H - 1) * f + i for i in range(f)]
        off_old += f; off_new += H * f
    col = torch.tensor(col, device=dev)
    # ---- check (a): history order
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
        for _ in range(H + 2):
            prev = obs.clone()
            obs, _, dones, _ = env.step(0.1 * torch.randn(n, env.num_actions, device=dev))
        ok_env = ~dones.bool()
        bad, off = 0.0, 0
        for f in FRAME:
            a = prev[:, off:off + H * f].reshape(n, H, f); b = obs[:, off:off + H * f].reshape(n, H, f)
            bad = max(bad, float((b[ok_env, H - 2] - a[ok_env, H - 1]).abs().max()))
            off += H * f
    print(f"[prep] history {H} frames; order check (newest at t == second-newest at t+1): max diff {bad:.2e}")
    assert bad < 1e-5, "history is not oldest-first / term-contiguous"
    # ---- pad
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    src = torch.load(args.src, map_location=agent_cfg.device, weights_only=False)
    old = src["model_state_dict"]; new = runner.alg.policy.state_dict()
    assert old["actor.0.weight"].shape[1] == 43, old["actor.0.weight"].shape
    sd = {}
    for k_, v in new.items():
        if k_ == "actor.0.weight":
            w = torch.zeros_like(v); w[:, col] = old[k_]; sd[k_] = w
        elif k_ == "log_std":
            sd[k_] = torch.full_like(v, math.log(args.std))
        else:
            assert old[k_].shape == v.shape, (k_, old[k_].shape, v.shape)
            sd[k_] = old[k_].clone()
    runner.alg.policy.load_state_dict(sd)
    torch.save({"model_state_dict": runner.alg.policy.state_dict(), "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 0, "infos": None}, args.out)
    # ---- check (b): same action as the source on the newest frame
    runner2 = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner2.load(args.out)
    pol = runner2.get_inference_policy(device=dev)
    layers = []
    for i in (0, 2, 4, 6):
        lin = torch.nn.Linear(old[f"actor.{i}.weight"].shape[1], old[f"actor.{i}.weight"].shape[0]).to(dev)
        lin.weight.data.copy_(old[f"actor.{i}.weight"]); lin.bias.data.copy_(old[f"actor.{i}.bias"])
        layers += [lin] + ([torch.nn.ELU()] if i < 6 else [])
    old_actor = torch.nn.Sequential(*layers)
    with torch.inference_mode():
        diff = float((pol(obs) - old_actor(obs[:, col])).abs().max())
    print(f"[prep] padded actor vs source actor on the newest frame: max action diff {diff:.2e}")
    assert diff < 1e-4
    print(f"[prep] wrote {args.out}: actor.0 {tuple(sd['actor.0.weight'].shape)}, std {args.std}, entropy {agent_cfg.algorithm.entropy_coef}, "
          f"max_iterations {agent_cfg.max_iterations}, source iter {src.get('iter')}")
    rw = uenv.reward_manager
    print("[prep] rewards:", {t: rw.get_term_cfg(t).weight for t in rw.active_terms if t.startswith("track")}, "terminations:", uenv.termination_manager.active_terms)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
