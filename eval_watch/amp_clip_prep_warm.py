"""Prepare a warm-start checkpoint for the clip tracker whose policy obs grew by the clip_ref block.

The stride tracker (archive_anchors/amp_track_asimov_model_2598.pt) saw 43 policy / 288 critic
features; the clip env appends `clip_ref` (upcoming reference joints at +1 and +10 frames, 20-D)
to both groups. This script builds the clip env, checks the new observation, zero-pads the first
actor/critic layers for the new columns (so the warm policy behaves exactly as before until it
learns to use them), pairs the weights with a FRESH optimizer state, and verifies the result loads
through the stock runner and runs.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_clip_prep_warm.py \
      --src archive_anchors/amp_track_asimov_model_2598.pt --out archive_anchors/amp_track_asimov_model_2598_clipref.pt --headless
"""
from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="Isaac-TrackClip-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_clip  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def pad_state(old: dict, new: dict) -> dict:
    out = {}
    padded = []
    for k, v_new in new.items():
        v_old = old[k]
        if v_old.shape == v_new.shape:
            out[k] = v_old.clone()
        elif v_old.ndim == 2 and v_old.shape[0] == v_new.shape[0] and v_old.shape[1] < v_new.shape[1]:
            w = torch.zeros_like(v_new)
            w[:, : v_old.shape[1]] = v_old
            out[k] = w
            padded.append((k, tuple(v_old.shape), tuple(v_new.shape)))
        else:
            raise RuntimeError(f"{k}: cannot map {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
    assert set(old) == set(new), f"key mismatch: {set(old) ^ set(new)}"
    return out, padded


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    dev = uenv.device
    n = args.num_envs
    print(f"[prep] obs dims: policy {env.num_obs}  critic {env.num_privileged_obs}")
    terms = uenv.observation_manager.active_terms["policy"]
    dims = uenv.observation_manager.group_obs_term_dim["policy"]
    print("[prep] policy terms:", list(zip(terms, [tuple(d) for d in dims])))
    assert terms[-1] == "clip_ref" and tuple(dims[-1]) == (20,), "clip_ref must be the last policy term (20-D)"
    cterms = uenv.observation_manager.active_terms["critic"]
    assert cterms[-1] == "clip_ref", "clip_ref must be the last critic term"
    # --- the new observation after a warm-up step + reset (first-episode DR gap)
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, extras = env.reset()
    ref = obs[:, -20:]
    print(f"[prep] clip_ref at reset: |max| {float(ref.abs().max()):.3f} rad, mean|.| {float(ref.abs().mean()):.3f}, finite {bool(torch.isfinite(ref).all())}")
    print("[prep] env0 clip_ref(+1):", [round(float(x), 3) for x in ref[0, :10]])
    print("[prep] env0 clip_ref(+10):", [round(float(x), 3) for x in ref[0, 10:]])
    assert torch.isfinite(obs).all() and float(ref.abs().max()) < 3.0
    # --- pad the old checkpoint into the new shapes, fresh optimizer
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    old = torch.load(args.src, map_location=agent_cfg.device, weights_only=False)
    new_sd, padded = pad_state(old["model_state_dict"], runner.alg.policy.state_dict())
    for k, a, b in padded:
        print(f"[prep] padded {k}: {a} -> {b}")
    assert len(padded) == 2, padded
    runner.alg.policy.load_state_dict(new_sd)
    save = {"model_state_dict": runner.alg.policy.state_dict(), "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 0, "infos": None}
    torch.save(save, args.out)
    print(f"[prep] wrote {args.out} (fresh optimizer, lr {runner.alg.optimizer.param_groups[0]['lr']}); source iter {old.get('iter')}")
    # --- verify: stock load path + rollout
    runner2 = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner2.load(args.out)
    policy = runner2.get_inference_policy(device=dev)
    robot = uenv.scene["robot"]
    lib = mdp_clip.load_library(env_cfg.rewards.track_clip_pose.params["clip_dir"], dev)
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
        falls = 0; ends = 0; err = []
        for t in range(args.steps):
            actions = policy(obs)
            assert torch.isfinite(actions).all(), "non-finite actions"
            obs, rew, dones, infos = env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(rew).all(), "non-finite obs/reward"
            st = uenv._clip_state
            q_ref = lib["q"][st["clip"], mdp_clip._frame(uenv, lib)]
            err.append(float(((robot.data.joint_pos - q_ref) ** 2).mean(dim=1).sqrt().mean() * 180 / math.pi))
            ce = uenv.termination_manager.get_term("clip_end")
            ends += int((dones.bool() & ce).sum()); falls += int((dones.bool() & ~ce).sum())
    print(f"[prep] rollout {args.steps} steps x {n} envs with the padded policy: clip_ends {ends}, falls {falls}, rmse {sum(err) / len(err):.1f} deg (first {err[0]:.1f}, last {err[-1]:.1f})")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
