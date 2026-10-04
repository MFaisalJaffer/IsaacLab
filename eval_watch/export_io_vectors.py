"""Golden input/output vectors for a walker bundle (RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md §7.1).

One robot, nominal plant, no pushes, deterministic policy, scripted commands: start, forward, backward,
side-step, turn in place, mixed turn, then a stop (which goes through the training-side corridor and the
annealed stand pin, exactly as the rig should do it). Per 20 ms tick it records what the policy saw and did:

  input      (T, obs_dim)   the stacked policy input (10 frames per term, oldest first)
  frame      (T, frame_dim) the newest frame inside `input` (the 43 values the rig builds per tick)
  action     (T, 10)        the deterministic policy output for `input`
  cmd        (T, 3)         [vx, vy, wz] as seen by the policy that tick
  theta      (T,)           the signed gait clock (left-leg phase before wrapping), radians
  phase_obs  (T, 4)         [cos phiL, sin phiL, cos phiR, sin phiR] emitted (pinned/annealed while standing)
  standing   (T,) bool      stand flag;  corridor (T,) bool  stop corridor active;  t_s (T,) time
  onnx_action (T, 10)       the bundle's policy.onnx evaluated on `input` (if onnxruntime is importable)

Checks the rig can make without physics: stack our `frame` rows -> must equal `input` exactly; run policy.onnx
on `input` -> must equal `action` to 1e-4; synthesize the clock from `cmd` -> must equal `phase_obs`.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/export_io_vectors.py \
      --checkpoint <ckpt> --out <bundle dir>/io_test_vectors.npz --headless
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--out", required=True)
parser.add_argument("--onnx", default="", help="policy.onnx of the bundle, evaluated on the same inputs when onnxruntime is available")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# (start time s, vx, vy, wz, stand flag): held until the next entry; "stop" = stand flag -> the corridor opens
SCHEDULE = [(0.0, 0.3, 0.0, 0.0, False), (3.0, -0.3, 0.0, 0.0, False), (6.0, 0.0, 0.13, 0.0, False), (8.5, 0.0, 0.0, 0.5, False),
            (11.0, 0.3, 0.0, 0.3, False), (13.0, 0.0, 0.0, 0.0, True)]
T_END = 16.5


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.episode_length_s = T_END + 5.0
    # nominal plant, no pushes, no spawn-stand conversion; the stop corridor stays (it is part of the contract)
    for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints", "add_limb_masses",
               "randomize_rigid_body_material", "base_com", "add_base_mass", "push_robot", "sustained_push", "walk_at_spawn"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    c = env_cfg.commands.base_velocity
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1000.0, 1000.0)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    dev = uenv.device
    dt = float(uenv.step_dt)
    om = uenv.observation_manager
    names = list(om.active_terms["policy"])
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    H = int(getattr(env_cfg.observations.policy, "history_length", 0) or 0) or 1
    frame_dims = [d // H for d in dims]
    frame_dim = sum(frame_dims)
    term = uenv.command_manager.get_term("base_velocity")

    def newest_frame(x: torch.Tensor) -> torch.Tensor:  # the last frame of every term block
        out, off = [], 0
        for d, f in zip(dims, frame_dims):
            out.append(x[:, off + d - f:off + d])
            off += d
        return torch.cat(out, dim=1)

    def apply(entry):
        _, vx, vy, wz, stand = entry
        term.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=dev)
        term.is_standing_env[:] = stand

    T = int(round(T_END / dt))
    rec = {"input": [], "frame": [], "action": [], "cmd": [], "theta": [], "phase_obs": [], "standing": [], "corridor": [], "t_s": []}
    with torch.inference_mode():
        env.step(torch.zeros(1, env.num_actions, device=dev))   # warm-up (first-episode gap), then a clean reset
        apply(SCHEDULE[0])
        obs, _ = env.reset()
        apply(SCHEDULE[0])
        obs = uenv.observation_manager.compute()["policy"]        # first observation with the scripted command in it
        k = 1
        for t in range(T):
            now = t * dt
            if k < len(SCHEDULE) and now >= SCHEDULE[k][0] - 1e-9:
                apply(SCHEDULE[k]); k += 1
                obs = uenv.observation_manager.compute()["policy"]  # the policy sees the new command this tick (as on the rig)
            a = policy(obs)
            st = getattr(uenv, "_kbot_signed_phase", None)
            rec["input"].append(obs[0].cpu().numpy().astype(np.float32))
            rec["frame"].append(newest_frame(obs)[0].cpu().numpy().astype(np.float32))
            rec["action"].append(a[0].cpu().numpy().astype(np.float32))
            rec["cmd"].append(term.vel_command_b[0].cpu().numpy().astype(np.float32))
            rec["theta"].append(float(st["phi"][0]) if st is not None else 0.0)
            rec["phase_obs"].append(newest_frame(obs)[0, -4:].cpu().numpy().astype(np.float32))
            rec["standing"].append(bool(term.is_standing_env[0]))
            corr = getattr(uenv, "_stand_corridor_until", None)
            rec["corridor"].append(bool(corr[0] >= 0.0) if corr is not None else False)
            rec["t_s"].append(now)
            obs, _, dones, _ = env.step(a)
            if bool(dones[0]):
                print(f"[io] episode ended at t={now:.2f} s (fall?) — vectors truncated here")
                break
    R = {k_: np.asarray(v) for k_, v in rec.items()}
    # self-check: stacking our frames reproduces our inputs (per term, oldest first)
    T_ = len(R["frame"])
    hist = {i: [R["frame"][0, sum(frame_dims[:i]):sum(frame_dims[:i + 1])]] * H for i in range(len(dims))}
    worst = 0.0
    for t in range(T_):
        if t > 0:
            for i in range(len(dims)):
                hist[i] = hist[i][1:] + [R["frame"][t, sum(frame_dims[:i]):sum(frame_dims[:i + 1])]]
        x = np.concatenate([np.concatenate(hist[i]) for i in range(len(dims))])
        worst = max(worst, float(np.abs(x - R["input"][t]).max()))
    print(f"[io] stacking self-check over {T_} ticks: max |restacked - input| = {worst:.2e} (must be 0)")
    onnx_action = None
    if args.onnx:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            onnx_action = np.concatenate([sess.run(None, {name: R["input"][t:t + 1]})[0] for t in range(T_)], axis=0).astype(np.float32)
            print(f"[io] policy.onnx vs live policy over {T_} ticks: max |diff| = {float(np.abs(onnx_action - R['action']).max()):.2e}")
        except ImportError:
            print("[io] onnxruntime not importable here — onnx_action not recorded (the rig checks this)")
    seg = {"schedule": [{"t_s": e[0], "cmd": [e[1], e[2], e[3]], "stand_flag": e[4]} for e in SCHEDULE], "T_END": T_END}
    np.savez(args.out, **R, **({"onnx_action": onnx_action} if onnx_action is not None else {}),
             obs_terms=np.array(names), frame_dims=np.array(frame_dims), history_length=H, H=H, obs_dim=int(R["input"].shape[1]), frame_dim=frame_dim, control_dt=dt,
             layout="per_term_contiguous_oldest_first", schedule=json.dumps(seg),
             readme=("input[t] is what the policy saw at tick t (10 frames per term, oldest first); frame[t] is its newest frame; "
                     "action[t] = policy(input[t]); the stop at 13 s goes through the corridor (0.12,0,0) for 1.5 s then the annealed stand pin"))
    print(f"[io] wrote {args.out}: {T_} ticks, input {R['input'].shape}, frame {R['frame'].shape}; segments: forward, backward, side, pivot, mixed turn, stop")
    print(f"[io] theta at 2.9 s {R['theta'][int(2.9 / dt)]:+.3f}, at 5.9 s {R['theta'][int(5.9 / dt)]:+.3f} (must have decreased while walking backward); "
          f"corridor ticks {int(R['corridor'].sum())}, standing ticks {int(R['standing'].sum())}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
