"""TILT-DECAY probe (2026-08-16) — the GROUND-TRUTH servo test.

The kinematic static_tilt_probe pins the base, so (a) the policy's correction
can never succeed and (b) at larger roll angles the uphill foot unloads
(contact forensics 08-16: L foot airborne 88-97% at +8..12 deg) — it measures
command twitches in a regime the robot is never actually in.

This probe instead applies a REAL attitude step: trained stand entry
(walk -> corridor decel -> stand, settled), then rotate the root quat by a
body-frame roll offset with rates zeroed and let the base run FREE.
The servo works iff the tilt DECAYS. Reports the mean |roll| trajectory,
recovery fraction below 2 deg, and falls.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--walk_in", type=int, default=150)
parser.add_argument("--decel_steps", type=int, default=75)
parser.add_argument("--decel_speed", type=float, default=0.12)
parser.add_argument("--settle", type=int, default=150)
parser.add_argument("--offsets_deg", type=float, nargs="+", default=[8.0, -8.0])
parser.add_argument("--axis", type=str, default="roll", choices=["roll", "pitch"])
parser.add_argument("--lean_sweep", action="store_true",
                    help="rig handoff #3 6.5: measure d(body_pitch)/d(ankle_encoder) "
                         "over a slow 1-2.5 deg lean (hardware measured 3.6)")
parser.add_argument("--ankle_play_deg", type=float, default=0.0,
                    help="fixed backlash band (deg) forced onto ankle actuators")
parser.add_argument("--free_steps", type=int, default=150)
parser.add_argument("--resettle", type=int, default=100)
parser.add_argument("--label", type=str, default="tilt_decay")
parser.add_argument("--out", type=str, default="eval_watch/tilt_decay.txt")
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
from isaaclab.utils.math import quat_mul
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for ev in ("push_robot", "sustained_push", "walk_at_spawn", "stand_corridor",
               "randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.episode_length_s = 120.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    dev = uenv.device
    n = uenv.num_envs
    dt = uenv.step_dt

    def set_cmd(vx, standing):
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1:] = 0.0
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = standing

    def roll_deg():
        pg = robot.data.projected_gravity_b
        i = 1 if args_cli.axis == "roll" else 0
        return torch.asin(pg[:, i].clamp(-1, 1)) * 180 / math.pi

    if args_cli.ankle_play_deg > 0.0:
        with torch.inference_mode():
            env.step(torch.zeros(n, uenv.action_manager.total_action_dim, device=dev))
        band = math.radians(args_cli.ankle_play_deg)
        for name, act in robot.actuators.items():
            if getattr(act, "_play", None) is not None and "ankle" in name:
                act._play[:] = band
        print(f"[play] ankle band forced to {args_cli.ankle_play_deg} deg")
    lines = [f"{args_cli.label}: FREE-BASE {args_cli.axis}-step decay, {args_cli.num_envs} envs, "
             f"trained entry (walk {args_cli.walk_in} -> decel {args_cli.decel_steps} -> settle {args_cli.settle}), gains pinned"]
    obs, _ = env.get_observations()
    if args_cli.lean_sweep:
        # quasi-static lean sweep, base pitched by hand exactly like the rig's
        # dual-stream test; report body tilt vs ANKLE ENCODER (motor side) —
        # the only quantity both worlds can measure (their 6.5).
        with torch.inference_mode():
            for t in range(args_cli.walk_in + args_cli.decel_steps + args_cli.settle):
                v = 0.25 if t < args_cli.walk_in else (
                    args_cli.decel_speed if t < args_cli.walk_in + args_cli.decel_steps else 0.0)
                set_cmd(v, v == 0.0)
                obs, _, _, _ = env.step(policy(obs))
            jn = list(robot.data.joint_names)
            aidx = [i for i, nm in enumerate(jn) if "ankle" in nm]
            hold0 = torch.zeros(n, uenv.action_manager.total_action_dim, device=dev)
            base_pose = robot.data.root_state_w[:, :7].clone()
            base_pose[:, 3:7] = robot.data.default_root_state[:, 3:7]
            zv = torch.zeros(n, 6, device=dev)
            for _ in range(60):              # settle at ZERO lean, position hold
                robot.write_root_pose_to_sim(base_pose)
                robot.write_root_velocity_to_sim(zv)
                set_cmd(0.0, True)
                obs, _, _, _ = env.step(hold0)
            enc0 = torch.stack([a.motor_pos.mean(dim=1) for nm, a in robot.actuators.items()
                                if "ankle" in nm]).mean(dim=0).clone()
            pg0 = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1, 1)).clone()
            rows = []
            for deg in (0.5, 1.0, 1.5, 2.0, 2.5):
                th = math.radians(deg)
                root = robot.data.root_state_w.clone()
                q_off = torch.tensor([math.cos(th/2), 0.0, math.sin(th/2), 0.0],
                                     device=dev).repeat(n, 1)
                # BRIDGE-HELD, like the rig's hand-lean test: the body angle is
                # EXTERNALLY IMPOSED and held every step while the joints settle.
                # (First version wrote the pose once and let the base run free —
                # it measured the robot falling away: a 0.5 deg command produced
                # 2.75 deg of body motion and a meaningless ~1.2 ratio.)
                pose = root[:, :7].clone()
                pose[:, 3:7] = quat_mul(robot.data.default_root_state[:, 3:7], q_off)
                zero_vel = torch.zeros(n, 6, device=dev)
                # POSITION HOLD, not live policy: the rig leaned a robot that
                # was holding position, so the rotor stays where commanded and
                # the SPRING absorbs the lean. With the policy live its active
                # ankle commands move the rotor and swamp the deflection (the
                # encoder deltas came out pure noise: +1.27/-0.72/+0.55/-1.02).
                hold = torch.zeros(n, uenv.action_manager.total_action_dim, device=dev)
                for _ in range(60):
                    robot.write_root_pose_to_sim(pose)
                    robot.write_root_velocity_to_sim(zero_vel)
                    set_cmd(0.0, True)
                    obs, _, _, _ = env.step(hold)
                enc = torch.stack([a.motor_pos.mean(dim=1) for nm, a in robot.actuators.items()
                                   if "ankle" in nm]).mean(dim=0)
                pg = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1, 1))
                d_body = (pg - pg0).abs().mean().item()
                d_enc = (enc - enc0).abs().mean().item()
                ratio = d_body / d_enc if d_enc > 1e-6 else float("nan")
                rows.append(f"  lean {deg:.1f} deg: d(body)={math.degrees(d_body):.2f} "
                            f"d(ankle_enc)={math.degrees(d_enc):.3f} RATIO={ratio:.2f}")
            out = (f"{args_cli.label}: LEAN SWEEP (rig 6.5; hardware ratio 3.6, rig sim 3.1)\n"
                   + "\n".join(rows))
            print(out)
            with open(args_cli.out, "a") as f:
                f.write(out + "\n\n")
        env.close(); return
    with torch.inference_mode():
        for t in range(args_cli.walk_in + args_cli.decel_steps + args_cli.settle):
            if t < args_cli.walk_in:
                set_cmd(0.25, False)
            elif t < args_cli.walk_in + args_cli.decel_steps:
                set_cmd(args_cli.decel_speed, False)
            else:
                set_cmd(0.0, True)
            obs, _, _, _ = env.step(policy(obs))
        pre = roll_deg()
        lines.append(f"  pre-step quiet stand: roll mean={pre.mean():+.2f} deg  p95(|.|)={pre.abs().quantile(0.95):.2f}")

        for off in args_cli.offsets_deg:
            th = math.radians(off)
            if args_cli.axis == "roll":
                q_off = torch.tensor([math.cos(th / 2), math.sin(th / 2), 0.0, 0.0],
                                     device=dev).repeat(n, 1)
            else:  # pitch: rotate about body y
                q_off = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                                     device=dev).repeat(n, 1)
            root = robot.data.root_state_w.clone()
            # body-frame roll step, rates zeroed ("placed tilted"), pose otherwise kept
            root[:, 3:7] = quat_mul(root[:, 3:7], q_off)
            root[:, 7:13] = 0.0
            robot.write_root_pose_to_sim(root[:, :7])
            robot.write_root_velocity_to_sim(root[:, 7:13])
            set_cmd(0.0, True)
            traj = []
            jerks = []          # ||delta action|| per step (rig jerk metric)
            prev_a = None
            for k in range(args_cli.free_steps):
                act = policy(obs)
                if prev_a is not None and k < 50:   # first 1 s = correction window
                    jerks.append((act - prev_a).norm(dim=1).cpu())
                prev_a = act.clone()
                obs, _, _, _ = env.step(act)
                traj.append(roll_deg().cpu())
            T = torch.stack(traj)                     # [steps, envs]
            h = robot.data.root_pos_w[:, 2]
            fallen = (h < 0.55)
            # SURVIVOR-MASKED trajectory (2026-08-17): fallen envs lie at
            # extreme angles and were dragging the mean rows (25.6k battery
            # read +11 deg at 3 s with 9% fallen). Means over survivors only.
            alive = (~fallen).cpu()
            Ta = T[:, alive] if alive.any() else T
            sgn = 1.0 if off >= 0 else -1.0
            marks = [(0.1, 4), (0.5, 24), (1.0, 49), (2.0, 99), (3.0, 149)]
            row = "  ".join(f"+{s}s:{(Ta[i] * sgn).mean():+.1f}" for s, i in marks if i < Ta.shape[0])
            rec = ((T[-25:].abs().mean(0) < 2.0) & ~fallen.cpu()).float().mean() * 100
            J = torch.stack(jerks)[:, alive] if alive.any() else torch.stack(jerks)
            lines.append(f"  step {off:+.1f} deg -> signed {args_cli.axis} (mean): {row}   "
                         f"recovered<2deg: {rec:.0f}%  fallen: {fallen.float().mean()*100:.0f}%  "
                         f"jerk(1s): mean {J.mean():.3f} p95 {J.reshape(-1).quantile(0.95):.3f}")
            for t in range(args_cli.resettle):
                obs, _, _, _ = env.step(policy(obs))

    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
