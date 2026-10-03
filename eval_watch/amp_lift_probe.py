"""Measure the walker-v5 stepping anchor (mdp_amp.ref_foot_lift) for a checkpoint, per direction, in the walker
env. One run: the envs are split into six groups with fixed commands (forward, backward, side-left, side-right,
pivot-left, pivot-right). Reports per group: survival, achieved speed, the anchor's mean value (with and without
its command-tracking gate) and the measured foot lift next to the reference cycle's.

A policy that steps like the reference should score high, one that slides its feet low — if not, the anchor's
timing or height assumptions are wrong and it must not be trained on.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 KBOT_AMP_HIST=10 KBOT_AMP_SIGNED_CLOCK=1 ... ./isaaclab.sh -p \
      eval_watch/amp_lift_probe.py --checkpoint <ckpt> --tag lift_probe --headless
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=384)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--rel_sigma", type=float, default=0.5, help="sigma as a share of each cycle's peak lift")
parser.add_argument("--speed_scale", type=float, default=1.0, help="commands = the cycles' own speeds x this")
parser.add_argument("--tag", default="lift_probe")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_amp, mdp_trackmulti  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(OUT, "amp_refs", "multicycle_v1.npz")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.episode_length_s = args.seconds + 1.0
    c = env_cfg.commands.base_velocity
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1000.0, 1000.0)
    for name in ("walk_at_spawn", "stand_corridor"):  # these rewrite commands at spawn / on transitions
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    n = args.num_envs
    T = int(args.seconds / uenv.step_dt)
    freq = float(env_cfg.observations.policy.gait_phase.params["gait_freq"])
    lib, prof = mdp_amp._lift_profile(LIB, dev)
    names = lib["names"]
    K = len(names)
    group = torch.arange(n, device=dev) % K
    table = lib["cmd"] * args.speed_scale
    term = uenv.command_manager.get_term("base_velocity")
    feet = robot.find_bodies(mdp_trackmulti.FEET, preserve_order=True)[0]
    planted = prof.amin(dim=1)  # (K, 2) reference foot height when planted
    print(f"[lift] clock {freq:.4f} Hz, sigma {args.rel_sigma} x peak lift, reference planted foot height [R, L] per cycle (m): "
          f"{[[round(float(x), 3) for x in r] for r in planted]}, reference peak lift (cm): "
          f"{[[round(float(x) * 100, 1) for x in r] for r in (prof.amax(dim=1) - planted)]}")

    def set_cmd():
        term.vel_command_b[:] = table[group]
        term.is_standing_env[:] = False

    with torch.inference_mode():  # warm-up + re-reset (first-episode DR gap)
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
        set_cmd()
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        R = {k: torch.zeros(T, n, device=dev) for k in ("free", "gated", "v", "cyc_ok")}
        Z = torch.zeros(T, n, 2, device=dev)
        for t in range(T):
            obs, _, dones, _ = env.step(policy(obs))
            set_cmd()
            fell |= dones.bool()
            R["free"][t] = mdp_amp.ref_foot_lift(uenv, LIB, freq, rel_sigma=args.rel_sigma)
            R["gated"][t] = mdp_amp.ref_foot_lift(uenv, LIB, freq, rel_sigma=args.rel_sigma, track_gate_lin=0.25)
            R["cyc_ok"][t] = (mdp_amp.cmd_cycle(term.vel_command_b, lib) == group).float()
            vb = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            v3 = torch.stack([vb[:, 0], vb[:, 1], robot.data.root_ang_vel_b[:, 2]], dim=1)
            axis = table[group].abs().argmax(dim=1)
            R["v"][t] = v3.gather(1, axis.unsqueeze(1)).squeeze(1)
            Z[t] = robot.data.body_pos_w[:, feet, 2] - uenv.scene.env_origins[:, 2:3]
    skip = int(2.0 / uenv.step_dt)
    ok = ~fell
    res = {"checkpoint": args.checkpoint, "rel_sigma": args.rel_sigma, "speed_scale": args.speed_scale, "cycles": {}}
    print(f"{'cycle':12s} {'surv':>5s} {'cmd':>6s} {'got':>6s} {'anchor':>7s} {'gated':>6s} {'lift R/L cm (p95)':>18s} {'ref lift R/L cm':>16s} {'planted R/L cm (p5)':>20s}")
    for k, name in enumerate(names):
        m = (group == k) & ok
        g_all = group == k
        if not m.any():
            print(f"{name:12s} {0.0:5.2f}  every robot fell")
            res["cycles"][name] = {"survival": 0.0}
            continue
        z = Z[skip:, m]  # (T', n_k, 2)
        hi = torch.quantile(z.reshape(-1, 2), 0.95, dim=0)
        lo = torch.quantile(z.reshape(-1, 2), 0.05, dim=0)
        ref_lift = (prof[k].amax(dim=0) - planted[k]) * 100
        axis = int(table[k].abs().argmax())
        r = {"survival": float(ok[g_all].float().mean()), "cmd": float(table[k, axis]), "achieved": float(R["v"][skip:, m].mean()),
             "anchor": float(R["free"][skip:, m].mean()), "anchor_gated": float(R["gated"][skip:, m].mean()),
             "lift_cm_p95": [float(x) * 100 for x in (hi - planted[k])], "ref_lift_cm": [float(x) for x in ref_lift],
             "planted_cm_p5": [float(x) * 100 for x in lo], "cycle_selected_ok": float(R["cyc_ok"][skip:, m].mean())}
        res["cycles"][name] = r
        print(f"{name:12s} {r['survival']:5.2f} {r['cmd']:6.2f} {r['achieved']:6.2f} {r['anchor']:7.2f} {r['anchor_gated']:6.2f} "
              f"{r['lift_cm_p95'][0]:8.1f} /{r['lift_cm_p95'][1]:5.1f}   {r['ref_lift_cm'][0]:7.1f} /{r['ref_lift_cm'][1]:5.1f}   "
              f"{r['planted_cm_p5'][0]:9.1f} /{r['planted_cm_p5'][1]:5.1f}   (cycle picked from the command: {r['cycle_selected_ok']:.2f})")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
