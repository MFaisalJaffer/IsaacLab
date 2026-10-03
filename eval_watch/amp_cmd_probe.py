"""Probe the walker's command draw in the real env (axis_bias event): shares of standing / single-direction /
mixed commands, the size range per axis, that the policy's command input equals the masked command, and that
the gait clock runs (is not pinned to the stand value) for single-direction commands.

Zero actions; the commands are re-drawn at resets and at a forced resample (step --force_resample_at), which
also opens the lineage's stand-entry corridor for the envs that draw a stand mid-episode.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker train env vars> ./isaaclab.sh -p eval_watch/amp_cmd_probe.py --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--force_resample_at", type=int, default=300, help="expire every command timer at this step (zero-action robots rarely live to the 10 s resample)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def report(tag: str, uenv, obs) -> None:
    term = uenv.command_manager.get_term("base_velocity")
    c = uenv.command_manager.get_command("base_velocity")
    nz = c != 0
    stand = ~nz.any(1)
    pure = nz.sum(1) == 1
    mixed = nz.sum(1) == 3
    n = len(c)
    print(f"[probe] {tag}: standing {stand.float().mean():.3f} (flag {term.is_standing_env.float().mean():.3f}) | single-direction "
          f"{pure.float().mean():.3f} | mixed {mixed.float().mean():.3f} | other {(~stand & ~pure & ~mixed).float().mean():.3f}")
    for k, name in enumerate(("fwd/back", "lateral", "yaw")):
        v = c[pure & nz[:, k], k]
        if len(v):
            print(f"[probe]   {name:8s}: share {len(v) / n:.3f}, |cmd| min {v.abs().min():.3f} max {v.abs().max():.3f}, positive share {(v > 0).float().mean():.2f}")
    corr = getattr(uenv, "_stand_corridor_until", None)
    corr = (corr >= 0.0) if corr is not None else torch.zeros_like(stand)
    if corr.any():
        rows, counts = torch.unique((c[corr] * 1000).round() / 1000, dim=0, return_counts=True)
        print(f"[probe]   stand-entry corridor: {corr.float().mean():.3f} of envs, commands {[(tuple(round(float(x), 3) for x in r_), int(n_)) for r_, n_ in zip(rows, counts)][:6]}")
    below = (torch.norm(c, dim=1) < 0.1) & ~stand & ~corr
    print(f"[probe]   moving commands under the 0.1 stand threshold: {below.float().mean():.4f} of envs "
          f"(single-direction {(below & pure).float().mean():.4f}, mixed {(below & mixed).float().mean():.4f})")
    # policy input: every term's frames are contiguous, oldest first -> newest frame = last d columns of the block
    om = uenv.observation_manager
    names = om.active_terms["policy"]
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    off = 0
    for nm, d in zip(names, dims):
        if nm == "velocity_commands":
            newest = obs[:, off + d - 3:off + d]
            print(f"[probe]   policy command input vs command: max diff {float((newest - c).abs().max()):.2e} (term dim {d})")
        if nm == "gait_phase":
            newest = obs[:, off + d - 4:off + d]
            pinned = ((newest[:, 0] + 1).abs() < 1e-3) & (newest[:, 1].abs() < 1e-3) & ((newest[:, 2] + 1).abs() < 1e-3) & (newest[:, 3].abs() < 1e-3)
            print(f"[probe]   clock pinned to the stand value: standing envs {pinned[stand].float().mean():.2f}, single-direction envs "
                  f"{pinned[pure].float().mean():.2f}, mixed envs {pinned[mixed].float().mean():.2f}")
        off += d
    print(f"[probe]   policy terms {list(zip(names, dims))}")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    act = torch.zeros(args.num_envs, uenv.action_manager.total_action_dim, device=uenv.device)
    with torch.inference_mode():
        obs, _ = env.reset()
        c_prev = None
        changed = 0
        for t in range(args.steps):
            obs, _, term_, trunc_, _ = env.step(act)
            c = uenv.command_manager.get_command("base_velocity").clone()
            if c_prev is not None:
                changed += int(((c != c_prev).any(1) & ~(term_ | trunc_)).sum())
            c_prev = c
            if t == args.force_resample_at:
                uenv.command_manager.get_term("base_velocity").time_left[:] = 0.0
            if t in (60, args.force_resample_at + 12, args.steps - 1):
                report(f"step {t}", uenv, obs["policy"])
        print(f"[probe] command changes without a reset over {args.steps} steps: {changed} (expected ~ one per moving env per 10 s resample)")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
