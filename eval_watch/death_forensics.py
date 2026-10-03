"""DEATH FORENSICS: when in the episode do robots die, and from what?

Training config AS-IS (events, curriculum, terminations all live). Tracks a
private per-env step clock (episode_length_buf is already zeroed for done envs
when step() returns) and, at each done, records (death_step, cause, base
height on the step before). Answers: is the base_height death mass the
SPAWN-SETTLE crumple (deaths <150 steps) or mid-episode collapses?
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--stochastic", action="store_true", help="sample actions like training")
parser.add_argument("--no_joint_draw", action="store_true", help="ISOLATION: spawn at default joint pose")
parser.add_argument("--no_base_vel_draw", action="store_true", help="ISOLATION: spawn with zero base velocity")
parser.add_argument("--no_gain_draw", action="store_true", help="ISOLATION: no actuator-gain redraw")
parser.add_argument("--no_tilt_draw", action="store_true", help="ISOLATION: no spawn roll/pitch")
parser.add_argument("--low_spawn", action="store_true", help="ISOLATION: spawn at standing height (minimal drop)")
parser.add_argument("--no_friction_draw", action="store_true", help="ISOLATION: no per-episode friction reroll")
parser.add_argument("--all_stand", action="store_true", help="ISOLATION: every env commanded to stand (identical episodes)")
parser.add_argument("--no_sustained", action="store_true", help="ISOLATION: sustained_push event fully removed")
parser.add_argument("--out", type=str, default="eval_watch/death_forensics.txt")
parser.add_argument("--label", type=str, default="forensics")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
    if args_cli.no_joint_draw:
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    if args_cli.no_base_vel_draw:
        env_cfg.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if args_cli.no_gain_draw:
        for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
            if getattr(env_cfg.events, ev, None) is not None:
                setattr(env_cfg.events, ev, None)
    if args_cli.no_tilt_draw:
        pr = dict(env_cfg.events.reset_base.params["pose_range"])
        pr["roll"] = (0.0, 0.0); pr["pitch"] = (0.0, 0.0)
        env_cfg.events.reset_base.params["pose_range"] = pr
    if args_cli.no_friction_draw:
        env_cfg.events.physics_material = None
    if args_cli.no_sustained:
        env_cfg.events.sustained_push = None
        if getattr(env_cfg.curriculum, "sustained_push_level", None) is not None:
            env_cfg.curriculum.sustained_push_level = None
    if args_cli.all_stand:
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    if args_cli.low_spawn:
        pr = dict(env_cfg.events.reset_base.params["pose_range"])
        pr["z"] = (-0.09, -0.02)
        env_cfg.events.reset_base.params["pose_range"] = pr
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    alg_policy = runner.alg.policy

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    tm = uenv.termination_manager
    causes = [c for c in tm.active_terms if c != "time_out"] + ["time_out"]
    clock = torch.zeros(uenv.num_envs, dtype=torch.long, device=uenv.device)
    prev_z = robot.data.root_pos_w[:, 2].clone()
    deaths = {c: [] for c in causes}          # (step, height) tuples
    settle = []                                # t=0 cohort: mean/p5 base z per step
    per_env_deaths = torch.zeros(uenv.num_envs, dtype=torch.long)
    early_deaths_standing = 0
    early_deaths_walking = 0
    # respawn settle: base z indexed by steps-since-reset, aggregated over resets
    respawn_z = [[] for _ in range(31)]
    cmd_mgr = uenv.command_manager
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            act = alg_policy.act(obs) if args_cli.stochastic else policy(obs)
            obs, _, _, _ = env.step(act)
            clock += 1
            if t < 30:
                z = robot.data.root_pos_w[:, 2]
                settle.append((z.mean().item(), z.quantile(0.05).item(), z.quantile(0.5).item()))
            # respawn-cohort settle: any env with clock<=30 (recent reset, incl. t=0)
            if t >= 100:                       # mid-run only (exclude t=0 cohort)
                zz = robot.data.root_pos_w[:, 2]
                for s in range(1, 31):
                    m = clock == s
                    if m.any():
                        respawn_z[s].extend(zz[m].tolist())
            dones = tm.dones
            if dones.any():
                cmd = cmd_mgr.get_command("base_velocity")
                standing = torch.norm(cmd[:, :3], dim=1) < 0.1
                for c in causes:
                    flags = tm.get_term(c) & dones
                    if flags.any():
                        ids = flags.nonzero(as_tuple=False).squeeze(-1)
                        for e in ids.tolist():
                            deaths[c].append((int(clock[e]), float(prev_z[e])))
                            if c == "base_height" and clock[e] <= 50:
                                per_env_deaths[e] += 1
                                if standing[e]: early_deaths_standing += 1
                                else: early_deaths_walking += 1
                clock[dones] = 0
            prev_z = robot.data.root_pos_w[:, 2].clone()

    lines = [f"{args_cli.label}: {args_cli.num_envs} envs x {args_cli.steps} steps "
             f"[{'STOCHASTIC' if args_cli.stochastic else 'deterministic'}]"]
    bins = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 600), (600, 999), (999, 10_000)]
    hdr = "  ".join(f"{a}-{b if b < 999 else 'end'}" for a, b in bins)
    lines.append(f"  death-step histogram        {hdr}")
    for c in causes:
        d = deaths[c]
        if not d:
            continue
        row = []
        for a, b in bins:
            row.append(sum(1 for s, _ in d if a <= s < b))
        med = sorted(s for s, _ in d)[len(d) // 2]
        mh = sum(h for _, h in d) / len(d)
        lines.append(f"  {c:16} n={len(d):4}  " + "  ".join(f"{r:5}" for r in row)
                     + f"   median_step={med}  mean_height_before={mh:.2f} m")
    tot = sum(len(v) for v in deaths.values())
    lines.append(f"  total episode endings observed: {tot}")
    if settle:
        lines.append("  t=0 cohort settle (base z): step: mean / p50 / p5   [kill line 0.55]")
        for i in (0, 2, 5, 9, 14, 19, 29):
            if i < len(settle):
                m, p5, p50 = settle[i]
                lines.append(f"    step {i+1:2}: {m:.3f} / {p50:.3f} / {p5:.3f}")
    import torch as _t
    d = per_env_deaths
    lines.append(f"  DOOM-LOOP check (early bh deaths per env): envs with 0: {(d==0).sum()}, "
                 f"1-2: {((d>=1)&(d<=2)).sum()}, 3-9: {((d>=3)&(d<=9)).sum()}, 10+: {(d>=10).sum()}, max {d.max()}")
    lines.append(f"  early bh deaths by command: STANDING {early_deaths_standing}  WALKING {early_deaths_walking}")
    rs = [(s, sum(v)/len(v), sorted(v)[max(0,len(v)//20)]) for s, v in enumerate(respawn_z) if v and s > 0]
    if rs:
        lines.append("  MID-RUN respawn settle (base z): step: mean / p5")
        for s, m, p5 in rs:
            if s in (1, 3, 6, 10, 15, 20, 30):
                lines.append(f"    step {s:2}: {m:.3f} / {p5:.3f}")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
