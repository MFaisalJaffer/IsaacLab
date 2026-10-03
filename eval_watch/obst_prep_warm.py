"""Warm-start checkpoint for the obstacle course (stage 1): walker v5 + an unconnected height-map input.

Source: a walker checkpoint (430 inputs). The obstacle env's policy input is the same 430 values followed by
the height map (187). The actor's first layer gets the walker's weights on the first 430 columns and ZEROS on
the map columns, so the padded policy acts exactly like the walker until training connects the map. Critic
(288 inputs, unchanged), action noise and the style judge are copied; the PPO optimizer starts fresh.

Checks: (a) the first 430 inputs have the walker's layout (7 terms x H frames, oldest first), (b) the padded
actor's action equals the source actor's on those 430 values, whatever the map shows.

  source eval_watch/obstacle_env.sh; ./isaaclab.sh -p eval_watch/obst_prep_warm.py \
      --src archive_anchors/walker_v5_best.pt --out archive_anchors/obstacle_s1_warm_from_walker_v5.pt --headless
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--std", type=float, default=0.0, help="action noise to start from (0 = keep the source's)")
parser.add_argument("--task", default="Isaac-Velocity-Obstacle-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
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
    assert names[-1] == "height_map" and len(dims) == len(FRAME) + 1
    H = dims[0] // FRAME[0]
    assert all(d == H * f for d, f in zip(dims[:-1], FRAME)), "walker terms do not have the expected layout"
    n_walk = sum(dims[:-1]); n_map = dims[-1]
    # ---- check (a): history order of the walker terms
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
    print(f"[prep] walker part {n_walk} = {H} frames x 43; order check (newest at t == second-newest at t+1): max diff {bad:.2e}; map {n_map} values, "
          f"range on this tick [{float(obs[:, n_walk:].min()):+.2f}, {float(obs[:, n_walk:].max()):+.2f}]")
    assert bad < 1e-5, "history is not oldest-first / term-contiguous"
    # ---- pad
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    src = torch.load(args.src, map_location=agent_cfg.device, weights_only=False)
    old = src["model_state_dict"]; new = runner.alg.policy.state_dict()
    assert old["actor.0.weight"].shape[1] == n_walk, (old["actor.0.weight"].shape, n_walk)
    sd = {}
    for k_, v in new.items():
        if k_ == "actor.0.weight":
            w = torch.zeros_like(v); w[:, :n_walk] = old[k_]; sd[k_] = w
        elif k_ == "log_std" and args.std > 0:
            sd[k_] = torch.full_like(v, math.log(args.std))
        else:
            assert old[k_].shape == v.shape, (k_, old[k_].shape, v.shape)
            sd[k_] = old[k_].clone()
    runner.alg.policy.load_state_dict(sd)
    out = {"model_state_dict": runner.alg.policy.state_dict(), "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 0, "infos": None}
    if "amp_disc_state_dict" in src:   # the walker's style judge and its optimizer
        runner.alg.disc.load_state_dict(src["amp_disc_state_dict"])
        out["amp_disc_state_dict"] = runner.alg.disc.state_dict()
        if "amp_disc_optimizer_state_dict" in src:
            runner.alg.disc_optimizer.load_state_dict(src["amp_disc_optimizer_state_dict"])
            out["amp_disc_optimizer_state_dict"] = runner.alg.disc_optimizer.state_dict()
        out["amp_cfg"] = {k: v for k, v in runner.alg.amp_cfg.items()}
    torch.save(out, args.out)
    # ---- check (b): same action as the source on the walker values, whatever the map shows
    runner2 = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner2.load(args.out)
    pol = runner2.get_inference_policy(device=dev)
    layers = []
    for i in (0, 2, 4, 6):
        lin = torch.nn.Linear(old[f"actor.{i}.weight"].shape[1], old[f"actor.{i}.weight"].shape[0]).to(dev)
        lin.weight.data.copy_(old[f"actor.{i}.weight"]); lin.bias.data.copy_(old[f"actor.{i}.bias"])
        layers += [lin] + ([torch.nn.ELU()] if i < 6 else [])
    old_actor = torch.nn.Sequential(*layers)
    with torch.inference_mode():
        x = obs.clone(); x[:, n_walk:] = torch.randn(n, n_map, device=dev)   # any map
        diff = float((pol(x) - old_actor(obs[:, :n_walk])).abs().max())
    print(f"[prep] padded actor vs source actor (random map): max action diff {diff:.2e}")
    assert diff < 1e-4
    std = float(torch.exp(sd["log_std"]).mean()) if "log_std" in sd else float("nan")
    print(f"[prep] wrote {args.out}: actor.0 {tuple(sd['actor.0.weight'].shape)}, noise std {std:.3f}, judge {'copied' if 'amp_disc_state_dict' in out else 'none'}, "
          f"entropy {agent_cfg.algorithm.entropy_coef}, max_iterations {agent_cfg.max_iterations}, source iter {src.get('iter')}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
