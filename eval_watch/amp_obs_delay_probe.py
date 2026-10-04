"""Check the walker-v7 sensing path in the real env: the delayed / sample-held observation terms (mdp_amp.DelayedObs)
and the frozen-command episode (mdp_amp.action_hold).

Random small actions, sensor noise off. For every policy step it recomputes, from the env's own fresh values and
the per-robot draws in env._sense, what each wrapped term SHOULD show (the sample taken `delay` steps ago by a
sensor that refreshes every `hold` steps) and compares it with the newest frame the policy group delivers.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> KBOT_AMP_OBS_DELAY=0:1:1:3:3 KBOT_AMP_ACT_HOLD=5:6 \
      ./isaaclab.sh -p eval_watch/amp_obs_delay_probe.py --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=160)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_amp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FRAME = {"projected_gravity": 3, "velocity_commands": 3, "joint_pos": 10, "joint_vel": 10, "imu_ang_vel": 3, "actions": 10, "gait_phase": 4}
CHECK = ("projected_gravity", "imu_ang_vel", "joint_pos")   # joint_vel's inner function is a stateful filter: not re-evaluated here


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = 1.6                               # several resets inside the probe
    inner = {}
    for name in CHECK:
        term = getattr(env_cfg.observations.policy, name)
        assert term.func is mdp_amp.DelayedObs, f"{name} is not wrapped — is KBOT_AMP_OBS_DELAY set?"
        inner[name] = (term.params["inner"], term.params["inner_params"], term.params["sensor"])
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    n, dev = args.num_envs, uenv.device
    om = uenv.observation_manager
    names = list(om.active_terms["policy"])
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    off = {nm: (sum(dims[:i]) + dims[i] - FRAME[nm], sum(dims[:i]) + dims[i]) for i, nm in enumerate(names)}
    sense = uenv._sense
    robot = uenv.scene["robot"]
    act0 = next(a for a in robot.actuators.values() if getattr(a, "_series_k", 0.0) == 0.0)
    j0 = robot.find_joints(act0.joint_names)[0][0]
    D = 4
    held = {k: None for k in CHECK}
    line = {k: None for k in CHECK}                               # sensor output k steps ago
    err = {k: 0.0 for k in CHECK}
    cnt = {k: 0 for k in CHECK}
    held_ok = held_n = moved_n = 0
    prev_held = None
    with torch.inference_mode():
        obs, _ = env.reset()
        for t in range(args.steps):
            a = 0.3 * torch.randn(n, uenv.action_manager.total_action_dim, device=dev)
            obs, _, term_, trunc_, _ = env.step(a)
            x = obs["policy"]
            ep = uenv.episode_length_buf.clone()
            first = ep == 0                                       # just reset: the wrapper refills with the first sample
            for k in CHECK:
                fn, prm, sensor = inner[k]
                v = fn(uenv, **prm)
                s = sense[sensor]
                if held[k] is None:
                    held[k] = v.clone(); line[k] = v.unsqueeze(1).repeat(1, D + 1, 1)
                fresh = (((ep + s["phase"]) % s["hold"]) == 0) | first
                held[k] = torch.where(fresh.unsqueeze(1), v, held[k])
                line[k] = torch.cat([held[k].unsqueeze(1), line[k][:, :-1]], dim=1)
                line[k][first] = held[k][first].unsqueeze(1)
                want = line[k][torch.arange(n, device=dev), s["delay"]]
                got = x[:, off[k][0]:off[k][1]]
                if t >= D + 1:                                    # let the probe's own delay line fill
                    err[k] = max(err[k], float((got - want).abs().max()))
                    cnt[k] += n
            # frozen command: while the hold is active the actuator's target must not move
            m = act0._hold_mask[:, 0] if act0._hold_mask is not None else torch.zeros(n, dtype=torch.bool, device=dev)
            tgt = act0._held_target[:, 0].clone() if act0._held_target is not None else None
            if prev_held is not None and tgt is not None:
                both = m & prev_m
                held_n += int(both.sum()); held_ok += int((both & ((tgt - prev_held).abs() < 1e-7)).sum())
                free = ~m & ~prev_m
                moved_n += int((free & ((tgt - prev_held).abs() > 1e-6)).sum())
            prev_held, prev_m = tgt, m
    for sensor, s in sense.items():
        print(f"[delay] sensor '{sensor}': delay steps {torch.bincount(s['delay'], minlength=4).tolist()} | hold steps {torch.bincount(s['hold'], minlength=4).tolist()} (counts of robots per value 0,1,2,3)")
    for k in CHECK:
        print(f"[delay] {k:18s} policy input vs expected delayed/held sample: max |diff| = {err[k]:.2e} over {cnt[k]} robot-steps")
    print(f"[delay] frozen command: target unchanged on {held_ok}/{held_n} robot-steps inside a hold; changed on {moved_n} robot-steps outside (random actions)")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
