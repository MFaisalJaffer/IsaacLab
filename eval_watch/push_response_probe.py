"""PUSH-RESPONSE probe: does the policy make MICRO-adjustments to a push, or
only react once the body tilts severely?

Protocol: settle, record a pre-push action baseline, then at t0 apply a known
lateral (or backward) velocity kick to every env simultaneously (same mechanism
as the training push event). Record per policy step: action deviation from
baseline (z-score per joint class), torso tilt, base velocity, applied torque.

Verdict logic:
  MICRO-ADJUSTING: action z-score crosses ~3 within a few steps (<100 ms) while
    tilt is still small (<3 deg), response grows smoothly with the disturbance.
  LATE GROSS CORRECTION (user's hypothesis): action stays near baseline until
    tilt is large (>8-10 deg), then jumps.
"""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--settle", type=int, default=200)
parser.add_argument("--walk_in", type=int, default=0,
                    help="TRAINED ENTRY: steps of commanded walking before the stand")
parser.add_argument("--decel_steps", type=int, default=0)
parser.add_argument("--decel_speed", type=float, default=0.12)
parser.add_argument("--baseline", type=int, default=50, help="pre-push steps for the action baseline")
parser.add_argument("--post", type=int, default=150, help="steps recorded after the push")
parser.add_argument("--vx", type=float, default=0.0, help="commanded speed (0 = stand)")
parser.add_argument("--kick", type=float, default=0.4, help="velocity kick m/s")
parser.add_argument("--axis", type=str, default="y", choices=["x", "y", "-x", "-y"])
parser.add_argument("--force_n", type=float, default=None,
                    help="SUSTAINED force mode: apply this many newtons (body frame) instead of a kick")
parser.add_argument("--force_steps", type=int, default=75, help="sustained-force duration in policy steps")
parser.add_argument("--ankle_play_deg", type=float, default=0.0,
                    help="fixed backlash band (deg) forced onto ankle actuators")
parser.add_argument("--ramp_ms", type=float, default=0.0,
                    help="force builds linearly over this many ms (0 = instant-on; matches training ramp_range)")
parser.add_argument("--series_k", type=float, default=None,
                    help="pin ankle K_s (Nm/rad). Lineage 9+ randomises it 20-120; a single-point "
                         "probe must say which point it ran. Default None = cfg nominal (23).")
parser.add_argument("--label", type=str, default="push")
parser.add_argument("--pin_gains", action="store_true")
parser.add_argument("--com_center", action="store_true", help="DIAGNOSTIC: centre the torso COM (null the -2.4 cm y correction) to test left/right push asymmetry")
parser.add_argument("--out", type=str, default="eval_watch/push_response.txt")
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

# joint-class indices in the 10-d action (fixed ordering: hip_pitch L/R=0,1,
# hip_roll L/R=2,3, hip_yaw=4,5, knee=6,7, ankle=8,9)
GROUPS = {"hip_pitch": [0, 1], "hip_roll": [2, 3], "hip_yaw": [4, 5],
          "knee": [6, 7], "ankle": [8, 9]}


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.com_center and getattr(env_cfg.events, "correct_torso_com", None) is not None:
        env_cfg.events.correct_torso_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        print("[probe] torso COM CENTRED (diagnostic)")
    if args_cli.series_k is not None:
        for _jn, _ac in env_cfg.scene.robot.actuators.items():
            if 'ankle' in _jn:
                _ac.series_k = args_cli.series_k
        print(f'[plant] ankle K_s pinned at {args_cli.series_k} Nm/rad')
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    v = args_cli.vx
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0 if v == 0.0 else 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    # EVAL HYGIENE (2026-08-05): the KBOT_ADAPT config carries the sustained_push
    # burst event — it MUST be nulled in probes or it contaminates measurements
    # (caught when a 'baseline' read 21 deg tilt: a training burst mid-probe).
    if getattr(env_cfg.events, "sustained_push", None) is not None:
        env_cfg.events.sustained_push = None          # OUR kick is the only disturbance
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    if getattr(env_cfg.curriculum, "sustained_push_level", None) is not None:
        env_cfg.curriculum.sustained_push_level = None   # curriculum writes into the (nulled) event
    # PLANT curricula must be nulled too, or the probe measures the RAMP value
    # instead of the deploy plant (2026-08-24: a battery silently ran at
    # K_s=150 instead of 23 — deflection 2.4 deg with tau_s 6.3 Nm gave it away).
    for _cu in ("plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, _cu, None) is not None:
            setattr(env_cfg.curriculum, _cu, None)
    env_cfg.episode_length_s = 60.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    if args_cli.pin_gains:
        for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
            if getattr(env_cfg.events, ev, None) is not None:
                setattr(env_cfg.events, ev, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.ankle_play_deg > 0.0:
        import math as _m
        uenv0 = env.unwrapped
        with torch.inference_mode():
            env.step(torch.zeros(uenv0.num_envs, uenv0.action_manager.total_action_dim, device=uenv0.device))
        for _nm, _act in uenv0.scene["robot"].actuators.items():
            if getattr(_act, "_play", None) is not None and "ankle" in _nm:
                _act._play[:] = _m.radians(args_cli.ankle_play_deg)
        print(f"[play] ankle band forced to {args_cli.ankle_play_deg} deg")

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    cmd_term = uenv.command_manager.get_term("base_velocity")
    dt = uenv.step_dt
    # STEP-OUT tracking (2026-08-07): does the robot ever LIFT a foot and widen
    # its stance under sustained force, or does it statue-and-tip? If stepping
    # never happens, the stand-shaping rewards are suppressing the one strategy
    # that survives large sustained pushes — the plateau's structural suspect.
    _FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
    cs = uenv.scene["contact_forces"]
    cf_ids = [cs.body_names.index(f) for f in _FEET]
    foot_body_ids = [list(robot.data.body_names).index(f) for f in _FEET]

    axis = {"x": (0, +1), "y": (1, +1), "-x": (0, -1), "-y": (1, -1)}[args_cli.axis]

    acts, tilts, rolls, vys, tqs = [], [], [], [], []
    contacts, widths, fzs, leans = [], [], [], []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        # TRAINED ENTRY (2026-08-16, lineage-4 battery fix): spawn-commanded
        # stands are an UNTRAINED path (walk_at_spawn converts them in training;
        # forensics proved the spawn-stand death loop) — newborn policies enter
        # the push already mid-collapse (measured baseline tilt 16.5 deg vs 7 deg
        # in the corridor-entry stand battery). Walk -> decel -> stand first.
        if args_cli.walk_in > 0:
            with torch.inference_mode():
                for _t in range(args_cli.walk_in + args_cli.decel_steps):
                    _v = 0.25 if _t < args_cli.walk_in else args_cli.decel_speed
                    cmd_term.vel_command_b[:, 0] = _v
                    cmd_term.vel_command_b[:, 1:] = 0.0
                    if hasattr(cmd_term, "is_standing_env"):
                        cmd_term.is_standing_env[:] = False
                    obs, _, _, _ = env.step(policy(obs))

        total = args_cli.settle + args_cli.baseline + args_cli.post
        t_push = args_cli.settle + args_cli.baseline
        for t in range(total):
            cmd_term.vel_command_b[:, 0] = v
            cmd_term.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = (v == 0.0)
            if args_cli.force_n is None:
                if t == t_push:
                    # the training-style push: an instantaneous velocity kick
                    rv = robot.data.root_vel_w.clone()
                    rv[:, axis[0]] += axis[1] * args_cli.kick
                    robot.write_root_velocity_to_sim(rv)
            else:
                # SUSTAINED force on the base link (the "leaning on it" case —
                # quasi-static, no velocity spike for the obs to catch)
                f = torch.zeros(uenv.num_envs, 1, 3, device=uenv.device)
                if t_push <= t < t_push + args_cli.force_steps:
                    ramp_steps = max(args_cli.ramp_ms / (dt * 1000), 1e-6)
                    scale = min(1.0, (t - t_push + 1) / ramp_steps) if args_cli.ramp_ms > 0 else 1.0
                    f[:, 0, axis[0]] = axis[1] * args_cli.force_n * scale
                tz = torch.zeros_like(f)
                robot.set_external_force_and_torque(f, tz, body_ids=[0])
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args_cli.settle:
                continue
            acts.append(act.cpu().clone())
            pg = robot.data.projected_gravity_b
            tilts.append((torch.asin(pg[:, :2].norm(dim=-1).clamp(max=1.0)) * 180 / math.pi).cpu())
            rolls.append((torch.asin(pg[:, 1].clamp(-1, 1)) * 180 / math.pi).cpu())
            vys.append(robot.data.root_lin_vel_b[:, axis[0]].cpu())
            tq = torch.zeros(uenv.num_envs, 10, device=uenv.device)
            for a in robot.actuators.values():
                tq[:, a.joint_indices] = a.applied_effort
            tqs.append(tq.cpu())
            contacts.append((cs.data.net_forces_w[:, cf_ids, :].norm(dim=-1) > 1.0).cpu())
            fzs.append(cs.data.net_forces_w[:, cf_ids, 2].clamp(min=0.0).cpu())
            # upwind CoM lean: root vs feet midpoint, world y (push is along +/-y);
            # upwind = opposite the push direction
            mid_y = robot.data.body_pos_w[:, foot_body_ids, 1].mean(dim=1)
            sgn = -1.0 if args_cli.axis == "y" else 1.0   # upwind sign vs push
            leans.append((sgn * (robot.data.root_pos_w[:, 1] - mid_y)).cpu())
            # stance width: |y_L - y_R| of the feet in world (yaw drift is
            # negligible over the probe window; width is yaw-invariant enough)
            fp = robot.data.body_pos_w[:, foot_body_ids, :2]
            widths.append((fp[:, 0, :] - fp[:, 1, :]).norm(dim=-1).cpu())

    A = torch.stack(acts)      # (B+P, n, 10)
    TL = torch.stack(tilts)
    RL = torch.stack(rolls)
    VY = torch.stack(vys)
    TQ = torch.stack(tqs)
    B = args_cli.baseline
    base_mean = A[:B].mean(dim=0)                      # (n, 10)
    base_std = A[:B].std(dim=0).clamp(min=1e-3)        # (n, 10)
    tq_base = TQ[:B].mean(dim=0)

    dist = (f"SUSTAINED {args_cli.force_n} N x {args_cli.force_steps*dt*1000:.0f} ms"
            if args_cli.force_n else f"kick {args_cli.kick} m/s")
    lines = [f"{args_cli.label}: {dist} along {args_cli.axis}, cmd_vx={v}, "
             f"{args_cli.num_envs} envs, dt={dt*1000:.0f} ms"]
    base_tilt = TL[:B].mean()
    lines.append(f"  pre-push baseline: tilt {base_tilt:.1f} deg, action std/joint "
                 + " ".join(f"{g}:{base_std[:, i].mean():.3f}" for g, i in
                            [(g, idx) for g, idx in GROUPS.items()][:3]))
    # timeline (post-push): action z per group, tilt, lateral vel, torque delta
    lines.append(f"  {'t(ms)':>6} {'z_hip_roll':>10} {'z_ankle':>8} {'z_hip_pitch':>11} "
                 f"{'z_knee':>7} | {'tilt':>5} {'roll':>5} {'vel':>6} | {'dTq_hr':>7}")
    z_first = {}
    for k in range(B, A.shape[0]):
        z = ((A[k] - base_mean).abs() / base_std)      # (n, 10)
        zg = {g: z[:, idx].mean().item() for g, idx in GROUPS.items()}
        for g, zv in zg.items():
            if g not in z_first and zv > 3.0:
                z_first[g] = (k - B) * dt * 1000
        step_ms = (k - B) * dt * 1000
        if (k - B) <= 15 or (k - B) % 10 == 0:
            dtq = (TQ[k] - tq_base)[:, GROUPS["hip_roll"]].abs().mean().item()
            lines.append(f"  {step_ms:>6.0f} {zg['hip_roll']:>10.1f} {zg['ankle']:>8.1f} "
                         f"{zg['hip_pitch']:>11.1f} {zg['knee']:>7.1f} | "
                         f"{TL[k].mean():>5.1f} {RL[k].mean():>+5.1f} {VY[k].mean():>6.2f} | {dtq:>7.1f}")
    peak_tilt_ms = (TL[B:].mean(dim=1).argmax().item()) * dt * 1000
    lines.append(f"  FIRST z>3 latency: " + "  ".join(f"{g}={z_first.get(g, float('nan')):.0f}ms"
                 for g in ("hip_roll", "ankle", "hip_pitch", "knee")))
    lines.append(f"  tilt at those instants: " + "  ".join(
        f"{g}:{TL[B + int(z_first[g] / (dt*1000))].mean():.1f}deg" for g in z_first))
    lines.append(f"  peak tilt {TL[B:].mean(dim=1).max():.1f} deg at +{peak_tilt_ms:.0f} ms; "
                 f"kick velocity decays to half at "
                 f"+{(VY[B:].mean(dim=1).abs() < args_cli.kick/2).float().argmax().item() * dt * 1000:.0f} ms")

    # ---- STEP-OUT verdict: statue vs stepping under the disturbance ----
    CT = torch.stack(contacts)                     # (B+P, n, 2) feet contact
    WD = torch.stack(widths)                       # (B+P, n)
    post = CT[B:]
    # a LIFT = a foot airborne >=3 consecutive steps (>=60 ms; filters chatter)
    air = (~post).float()
    lifted = torch.zeros(post.shape[1], dtype=torch.bool)
    for foot in range(2):
        a = air[:, :, foot]
        run = torch.zeros_like(a[0])
        best = torch.zeros_like(a[0])
        for k in range(a.shape[0]):
            run = (run + 1) * a[k]
            best = torch.maximum(best, run)
        lifted |= (best >= 3)
    w0 = WD[:B].mean(dim=0)
    wmax = WD[B:].max(dim=0).values
    fell = (TL[B:].max(dim=0).values > 45.0)
    surv = ~fell
    lines.append(f"  STEP-OUT: envs lifting a foot >=60ms during response: "
                 f"{int(lifted.sum())}/{post.shape[1]}"
                 f" | stance width base={w0.mean()*100:.1f}cm"
                 f" max-during={wmax.mean()*100:.1f}cm (delta {100*(wmax-w0).mean():+.1f}cm)")
    if fell.any() and surv.any():
        lines.append(f"    fallers: {int(fell.sum())} (lifted: {int((lifted & fell).sum())})"
                     f" | survivors: {int(surv.sum())} (lifted: {int((lifted & surv).sum())})")
    lines.append(f"    verdict: {'STEPPING response present' if lifted.float().mean() > 0.3 else 'STATUE — no protective step; stand-shaping likely suppressing it'}")

    # ---- BRACE verdict: does weight shift to the DOWNWIND foot, and WHEN? ----
    # probe pushes along +axis; for '+y' the downwind foot is L (index 0)
    if fzs and args_cli.axis in ("y", "-y"):
        FZ = torch.stack(fzs)                      # (B+P, n, 2)  [L, R]
        dwn = 0 if args_cli.axis == "y" else 1
        upw = 1 - dwn
        tot = FZ.sum(dim=2).clamp(min=1.0)
        asym = (FZ[:, :, dwn] - FZ[:, :, upw]) / tot          # + = correct brace
        base_asym = asym[:B].mean()
        marks = [100, 200, 400, 800]
        vals = []
        for ms in marks:
            k = B + int(ms / (dt * 1000))
            if k < asym.shape[0]:
                vals.append(f"+{ms}ms:{asym[k].mean():+.2f}")
        lines.append(f"  BRACE: load asym toward downwind foot (baseline {base_asym:+.2f}; "
                     f"+1.0 = full correct transfer)")
        lines.append(f"    {'  '.join(vals)}")
        if fell.any() and surv.any():
            a200 = asym[B + int(0.2 / dt)]
            lines.append(f"    asym@200ms — survivors: {a200[surv].mean():+.2f}  fallers: {a200[fell].mean():+.2f}")

    # ---- LEAN verdict (v2 metric): upwind CoM displacement vs feet midpoint ----
    if leans:
        LN = torch.stack(leans)                    # (B+P, n)  + = upwind (into push)
        base_lean = LN[:B].mean()
        vals = []
        for ms in [100, 200, 400, 800]:
            k = B + int(ms / (dt * 1000))
            if k < LN.shape[0]:
                vals.append(f"+{ms}ms:{LN[k].mean()*100:+.1f}cm")
        lines.append(f"  LEAN: upwind CoM offset (baseline {base_lean*100:+.1f}cm; survivors lean INTO the push)")
        lines.append(f"    {'  '.join(vals)}")
        if fell.any() and surv.any():
            l200 = LN[B + int(0.2 / dt)]
            lines.append(f"    lean@200ms — survivors: {l200[surv].mean()*100:+.1f}cm  fallers: {l200[fell].mean()*100:+.1f}cm")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "a") as f:
        f.write(out + "\n\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
