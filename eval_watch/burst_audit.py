"""AUDIT: is the sustained_push burst event ACTUALLY applying force in training?

Builds the env with the TRAINING config untouched (bursts ENABLED) and reports,
over a rollout: how often a nonzero external wrench is present, its magnitude
distribution, and what it does to tilt / falls. If the applied force reads ~0,
the event is a no-op and every conclusion drawn from 'training looked fine' is
void.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--no_burst", action="store_true", help="CONTROL arm: null the event")
parser.add_argument("--stochastic", action="store_true",
                    help="sample actions like TRAINING does (vs deterministic mean action)")
parser.add_argument("--stand_frac", type=float, default=None,
                    help="override rel_standing_envs (None = training default)")
parser.add_argument("--out", type=str, default="eval_watch/burst_audit.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    # TRAINING CONFIG AS-IS — do NOT null sustained_push; that's the point.
    if args_cli.no_burst and getattr(env_cfg.events, 'sustained_push', None) is not None:
        env_cfg.events.sustained_push = None
    if args_cli.stand_frac is not None:
        env_cfg.commands.base_velocity.rel_standing_envs = args_cli.stand_frac
    # keep TRAINING terminations ON — we want the fall rate, not a survival illusion
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    alg_policy = runner.alg.policy
    def act_fn(o):
        if args_cli.stochastic:
            return alg_policy.act(o)      # sampled, exploration noise ON (training-like)
        return policy(o)                   # deterministic mean action (probe default)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    ev = uenv.event_manager
    lines = [f"burst audit: {args_cli.num_envs} envs x {args_cli.steps} steps  "
         f"[actions: {'STOCHASTIC (training-like)' if args_cli.stochastic else 'deterministic'}]"]
    lines.append(f"  event terms present: {[t for t in ev.active_terms.get('interval', [])]}")
    lines.append(f"  body[0] name = {robot.data.body_names[0]}  (force target; must be the base/torso)")
    term_cfg = None
    try:
        term_cfg = ev.get_term_cfg("sustained_push")
        lines.append(f"  sustained_push params: {term_cfg.params}")
    except Exception as e:
        lines.append(f"  sustained_push NOT REGISTERED: {e}")

    mags, tilts = [], []
    wrench_on = 0
    dones_total = 0
    term_counts = {}
    nonzero_steps = 0
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            act = act_fn(obs)
            obs, _, _, _ = env.step(act)
            # read the ACTUAL external-force buffer the sim applies
            fb = getattr(robot, "_external_force_b", None)
            if fb is not None:
                m = fb[:, 0, :].norm(dim=-1)          # base-link force magnitude, per env
                mags.append(m.cpu().clone())
                nonzero_steps += int((m > 1e-3).any())
            pg = robot.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
            tilts.append((torch.asin(pg) * 180 / math.pi).cpu())
            wrench_on += int(bool(getattr(robot, 'has_external_wrench', False)))
            dones_total += int(uenv.termination_manager.dones.sum())
            for k_ in uenv.termination_manager.active_terms:
                term_counts[k_] = term_counts.get(k_, 0) + int(
                    uenv.termination_manager.get_term(k_).sum())

    if mags:
        M = torch.stack(mags)                          # (T, n)
        active = M > 1e-3
        frac_env_steps = active.float().mean().item() * 100
        lines.append(f"  APPLIED FORCE: nonzero on {frac_env_steps:.1f}% of env-steps "
                     f"({nonzero_steps}/{len(mags)} steps had >=1 env pushed)")
        if active.any():
            av = M[active]
            lines.append(f"    magnitude when active: mean={av.mean():.1f} N  "
                         f"p50={av.median():.1f}  p90={av.quantile(0.9):.1f}  max={av.max():.1f}")
            per_env = active.float().mean(dim=0)
            lines.append(f"    per-env active fraction: min={per_env.min()*100:.0f}% "
                         f"mean={per_env.mean()*100:.0f}% max={per_env.max()*100:.0f}%")
        else:
            lines.append("    *** ZERO FORCE EVER APPLIED — THE EVENT IS A NO-OP ***")
    else:
        lines.append("  could not read robot._external_force_b (API mismatch)")

    TL = torch.stack(tilts)
    lines.append(f"  *** has_external_wrench TRUE on {wrench_on}/{args_cli.steps} steps "
                 f"({wrench_on/args_cli.steps*100:.1f}%) — if LOW, forces are being CLOBBERED ***")
    lines.append(f"  TERMINATIONS: {dones_total} total over {args_cli.steps} steps x "
                 f"{args_cli.num_envs} envs = {dones_total/(args_cli.steps*args_cli.num_envs)*100:.3f}% of env-steps")
    lines.append(f"    by cause: {term_counts}")
    lines.append(f"  tilt: mean={TL.mean():.1f} p95={TL.quantile(0.95):.1f} max={TL.max():.1f} deg; "
                 f"env-steps >25deg: {(TL>25).float().mean()*100:.1f}%  >45deg: {(TL>45).float().mean()*100:.1f}%")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
