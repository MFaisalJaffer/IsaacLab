# Walk sim-to-real gap probe (2026-07-17). Rig walk @ vx=0.3: 0.13 m/s actual
# (43% tracking) + torso tilt_mx 22 deg; sim same ckpt/cmd: 0.374 (125%) /
# 12.9 deg. This probe applies ONE candidate physical condition per run and
# measures the walk signature — the arm that reproduces the rig numbers names
# the domain gap (the stand campaign found its gap this way: delay DR @ 81k).
# Usage: env KBOT_FLAT=1 kbot_env/bin/python eval_watch/walk_gap_probe.py \
#   --checkpoint <ckpt> --vx 0.3 [--delay_pin N] [--ankle_fric X] \
#   [--hipknee_fric X] [--mass_add KG] [--mu X] [--kp_scale X] --label NAME
import argparse

from isaaclab.app import AppLauncher

import sys
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--settle", type=int, default=100)
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--label", type=str, default="base")
parser.add_argument("--delay_pin", type=int, default=None, help="pin actuator delay to N physics steps (5 ms each)")
parser.add_argument("--delay_range", type=str, default=None, help="'min,max' physics-step delay jitter band")
parser.add_argument("--ankle_fric", type=float, default=None, help="pin ankle joint friction (Nm)")
parser.add_argument("--hipknee_fric", type=float, default=None, help="pin hip/knee joint friction (Nm)")
parser.add_argument("--mass_add", type=float, default=None, help="pin base added mass (kg)")
parser.add_argument("--mu", type=float, default=None, help="pin contact friction")
parser.add_argument("--kp_scale", type=float, default=None, help="scale all actuator stiffness")
parser.add_argument("--physics_hz", type=int, default=None, help="physics rate override (policy stays 50Hz)")
parser.add_argument("--pin_gains", action="store_true", help="disable gain-DR events (nominal kp/kd = rig conditions)")
parser.add_argument("--gait_freq", type=float, default=None, help="override the gait clock (Hz) — freq-map study knob")
parser.add_argument("--joint_offset", type=float, default=None,
                    help="reset joints with +/- this offset (rad) — bent-start hardware-risk probe")
parser.add_argument("--out", type=str, default="eval_watch/walk_gap_results.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
import faulthandler; faulthandler.enable()
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

    # fixed straight-walk command, no pushes, no curriculum, no resets mid-window
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.events.push_robot = None
    # EVAL HYGIENE (2026-08-05): the KBOT_ADAPT config carries the sustained_push
    # burst event — it MUST be nulled in probes or it contaminates measurements
    # (caught when a 'baseline' read 21 deg tilt: a training burst mid-probe).
    if getattr(env_cfg.events, "sustained_push", None) is not None:
        env_cfg.events.sustained_push = None
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

    # ---- the ONE condition under test ----
    applied = []
    if args_cli.delay_pin is not None:
        for name, act in env_cfg.scene.robot.actuators.items():
            act.min_delay = args_cli.delay_pin
            act.max_delay = args_cli.delay_pin
        applied.append(f"delay={args_cli.delay_pin*5}ms")
    if args_cli.delay_range is not None:
        lo, hi = (int(x) for x in args_cli.delay_range.split(","))
        for name, act in env_cfg.scene.robot.actuators.items():
            act.min_delay = lo
            act.max_delay = hi
        applied.append(f"delay={lo*5}-{hi*5}ms")
    if args_cli.ankle_fric is not None:
        env_cfg.events.randomize_joint_friction_ankles.params["friction_distribution_params"] = (
            args_cli.ankle_fric, args_cli.ankle_fric)
        applied.append(f"ankle_fric={args_cli.ankle_fric}")
    if args_cli.hipknee_fric is not None:
        env_cfg.events.randomize_joint_friction_hips_knees.params["friction_distribution_params"] = (
            args_cli.hipknee_fric, args_cli.hipknee_fric)
        applied.append(f"hipknee_fric={args_cli.hipknee_fric}")
    if args_cli.mass_add is not None:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
            args_cli.mass_add, args_cli.mass_add)
        applied.append(f"mass+{args_cli.mass_add}kg")
    if args_cli.mu is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (args_cli.mu, args_cli.mu)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.8 * args_cli.mu, 0.8 * args_cli.mu)
        applied.append(f"mu={args_cli.mu}")
    if args_cli.kp_scale is not None:
        for name, act in env_cfg.scene.robot.actuators.items():
            act.stiffness = {k: v * args_cli.kp_scale for k, v in act.stiffness.items()}
        applied.append(f"kp x{args_cli.kp_scale}")
    if args_cli.physics_hz is not None:
        # HIL trace-diff verdict (2026-07-17): sub-tick servo limit cycle on the
        # low-inertia joints (yaw/ankle) aliases differently per integration
        # rate. If the walk changes materially at 400 Hz, the gait exploits an
        # under-resolved artifact. NB delay_pin is in PHYSICS steps — rescale
        # externally (15 ms = 3 steps @200 Hz, 6 steps @400 Hz).
        env_cfg.sim.dt = 1.0 / args_cli.physics_hz
        env_cfg.decimation = int(round(args_cli.physics_hz / 50))
        applied.append(f"physics {args_cli.physics_hz}Hz (decim {env_cfg.decimation})")

    if args_cli.gait_freq is not None:
        for _t in (env_cfg.observations.policy.gait_phase, env_cfg.rewards.feet_phase,
                   env_cfg.rewards.feet_alternation, env_cfg.rewards.knee_swing):
            _t.params["gait_freq"] = args_cli.gait_freq
            _t.params.pop("freq_map", None)
        applied.append(f"clock={args_cli.gait_freq}Hz")
    if args_cli.pin_gains:
        for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
            if getattr(env_cfg.events, ev, None) is not None:
                setattr(env_cfg.events, ev, None)
        applied.append("gains=nominal")
    if args_cli.joint_offset is not None:
        # NB training's reset_joints_by_scale is a SILENT NO-OP on this robot
        # (multiplicative on an all-zero default pose) — this probe injects the
        # additive offsets training never had, to measure bent-start behavior.
        import isaaclab.envs.mdp as _mdp
        env_cfg.events.reset_robot_joints.func = _mdp.reset_joints_by_offset
        env_cfg.events.reset_robot_joints.params = {
            "position_range": (-args_cli.joint_offset, args_cli.joint_offset),
            "velocity_range": (0.0, 0.0),
        }
        applied.append(f"joint_offset=+/-{args_cli.joint_offset}rad")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    n = env.unwrapped.num_envs
    dev = env.unwrapped.device

    vx_sum_s = 0.0                                     # alive-masked scalar accumulators
    trans_sum_s = 0.0
    duty_sum_s = 0.0
    tilt_sum = torch.zeros(n, device=dev)
    tilt_max = torch.zeros(n, device=dev)
    fallen = torch.zeros(n, dtype=torch.bool, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)   # sticky: tilt<=25 so far
    # T-V saturation accounting (HIL 2026-07-17: emulator walk = ankle 83-85%
    # at cap, hip roll 55% @T-V / 93%/65% even at STAND; Isaac must be compared
    # on the same statistic — applied |effort| within 2% of its live limit).
    sat_hits = {}
    sat3 = {}   # STEP-0 (2026-07-27): at-clamp events split motoring/braking/stall
    sat_steps = 0
    # chatter metric (HIL convention): fraction of consecutive policy ticks
    # where a joint's qd flips sign. Their baseline on our trace: yaw 42.6%,
    # ankle 35%, hips 20-25%.
    prev_qd = None
    flip_sum = torch.zeros(10, device=dev)
    flip_cnt = 0
    # foot-force FLICKER metric (HIL rec 2026-07-18): PhysX loads/unloads feet
    # ~5.8x/s (duty 55-58%) where MuJoCo holds smooth 0.52s stances (duty 78%)
    # — the contact-microstructure half of the 289400 speed gap. A transferable
    # gait should load cleanly in BOTH engines.
    csensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids_s = [csensor.body_names.index(nm) for nm in robot.body_names if nm.endswith("FOOT")]
    prev_contact = None
    trans_sum = torch.zeros(n, device=dev)
    duty_sum = torch.zeros(n, device=dev)
    duty_cnt = 0
    # per-joint |torque| profile (thermal audit 2026-07-20: hip rolls hold
    # ~18.5 Nm STATICALLY at stand = 2-3x typical continuous rating — the
    # narrow default stance's 0.14 m lever is the root cause)
    tq_sum = torch.zeros(10, device=dev)
    tq_sq = torch.zeros(10, device=dev)
    tq_cnt = 0

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            # hold the walk command against any resampling/standing logic
            cmd_term.vel_command_b[:, 0] = args_cli.vx
            cmd_term.vel_command_b[:, 1:] = 0.0
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = False
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step < args_cli.settle:
                continue
            # ---- ALIVE MASK (2026-07-31; THRESHOLD FIXED 2026-08-03) ----
            # Purpose: terminations are disabled here, so a fallen env grinds on
            # the floor forever and poisons the torque stats (that produced a
            # bogus "hips stalled at 20 Nm" reading). Exclude an env from the
            # step it first fails.
            # BUG (found by the user from the render): the original threshold was
            # STICKY total-tilt > 25 deg, which a *posture* trips as easily as a
            # fall. The attempt-3 policy WALKS with a steady +36 deg forward lean
            # (roll ~0, base height constant, 32/32 upright, translating at 82% of
            # command) -> every env was masked out on step 1 and the probe
            # reported "ALL FELL". Now use the ENV'S OWN fall criteria, which are
            # posture-agnostic: base height (termination minimum_height 0.55, with
            # margin) and the bad_orientation limit (1.0 rad = 57 deg).
            pg = robot.data.projected_gravity_b[:, :2].norm(dim=-1).clamp(max=1.0)
            tilt = torch.asin(pg) * 180.0 / math.pi
            height = robot.data.root_pos_w[:, 2]
            alive &= (height > 0.50) & (tilt <= 57.0)
            fallen |= (height <= 0.50) | (tilt > 57.0)
            tilt_sum += tilt
            tilt_max = torch.maximum(tilt_max, tilt)
            if alive.sum() == 0:
                continue
            vx_sum_s += robot.data.root_lin_vel_b[alive, 0].mean().item()
            # per-actuator saturation: |applied| >= 98% of the live motoring limit
            sat_steps += 1
            for aname, act in robot.actuators.items():
                if getattr(act, "tv_motoring_limit", None) is None:
                    continue
                eff = act.applied_effort.abs()[alive]
                lim = torch.minimum(act.tv_motoring_limit, torch.full_like(act.tv_motoring_limit, act._braking_torque))[alive]
                at_clamp = eff >= 0.98 * lim
                sat_hits[aname] = sat_hits.get(aname, 0.0) + at_clamp.float().mean().item()
                # STEP-0 split: tv_headroom is MOTORING-gated, so only the motoring
                # share of at-clamp time is priceable by it. Classify at the same read:
                #   motoring = same-sign torque/velocity while moving; stall = |qd|<0.05
                #   (stalls read as braking to the actuator — the A-frame blind spot).
                seff = act.applied_effort[alive]
                qd_j = robot.data.joint_vel[:, act.joint_indices][alive]
                moving = qd_j.abs() >= 0.05
                mot = (torch.sign(seff) == torch.sign(qd_j)) & moving
                d = sat3.setdefault(aname, {"mot": 0.0, "brk": 0.0, "stall": 0.0})
                d["mot"] += (at_clamp & mot).float().mean().item()
                d["brk"] += (at_clamp & ~mot & moving).float().mean().item()
                d["stall"] += (at_clamp & ~moving).float().mean().item()
            qd_now = robot.data.joint_vel
            if prev_qd is not None:
                flip_sum += ((qd_now * prev_qd) < 0)[alive].float().mean(dim=0)
                flip_cnt += 1
            prev_qd = qd_now.clone()
            tq_step = torch.zeros(n, 10, device=dev)
            for aname, act in robot.actuators.items():
                tq_step[:, act.joint_indices] = act.applied_effort.abs()
            tq_sum += tq_step[alive].mean(dim=0)
            tq_sq += (tq_step[alive] ** 2).mean(dim=0)
            tq_cnt += 1
            contact_now = csensor.data.net_forces_w[:, foot_ids_s, :].norm(dim=-1) > 1.0  # (n, feet)
            if prev_contact is not None:
                trans_sum_s += (contact_now != prev_contact)[alive].float().sum(dim=1).mean().item()
            duty_sum_s += contact_now[alive].float().mean().item()
            duty_cnt += 1
            prev_contact = contact_now.clone()

    meas = args_cli.steps - args_cli.settle
    up = alive
    if sat_steps == 0 or up.sum() == 0:
        line = (f"{args_cli.label:12s} [{', '.join(applied) or 'baseline'}] ALL FELL "
                f"(alive 0/{n}; fallen>45deg {int(fallen.sum())}/{n}) — no walking-fraction stats")
    else:
        vx_act = vx_sum_s / sat_steps          # mean over alive envs, per accumulated step
        tl = (tilt_sum / meas)[up]
        tm = tilt_max[up]
        line = (f"{args_cli.label:12s} [{', '.join(applied) or 'baseline'}] "
                f"vx={vx_act:.3f}/{args_cli.vx} ({vx_act/args_cli.vx*100:.0f}%)  "
                f"tilt mean={tl.mean():.1f} p95(max)={tm.quantile(0.95):.1f} max={tm.max():.1f}  "
                f"alive={int(up.sum())}/{n} (fell-in-window={n-int(up.sum())}) "
                f"[stats = alive envs only]")
    print("[probe] " + line)
    with open(args_cli.out, "a") as f:
        f.write(line + "\n")
        if sat_hits:
            # condense: joints sorted by saturation fraction, worst first
            sats = sorted(((v / sat_steps, k) for k, v in sat_hits.items()), reverse=True)
            f.write("    T-V saturation%: " + "  ".join(
                f"{k.replace('dof_','')}={v*100:.0f}" for v, k in sats) + "\n")
        if sat3 and sat_steps:
            # STEP-0: motoring share decides the tv_headroom dose (see calm-walk plan:
            # mot<30% of at-clamp -> drop headroom; 30-40% -> 0.90/-3.0; >=40% -> 0.88/-4.0)
            order = sorted(sat3.items(), key=lambda kv: -sum(kv[1].values()))
            f.write("    sat split mot/brk/stall %: " + "  ".join(
                f"{k.replace('dof_','')}={v['mot']/sat_steps*100:.0f}/{v['brk']/sat_steps*100:.0f}/{v['stall']/sat_steps*100:.0f}"
                for k, v in order if sum(v.values()) / sat_steps > 0.01) + "\n")
        if flip_cnt > 0:
            fr = (flip_sum / flip_cnt).cpu()
            jn = robot.data.joint_names
            f.write("    qd tick-flip%: " + "  ".join(
                f"{n.replace('dof_','')}={fr[i]*100:.0f}" for i, n in enumerate(jn)) + "\n")
        if duty_cnt > 0:
            # transitions/s per foot (load+unload both counted; /2 feet), sampled at
            # policy rate — same 50 Hz sampling as the trace the HIL's 5.8/s came from
            flick = (trans_sum_s / duty_cnt) / (2 * env.unwrapped.step_dt)
            duty = duty_sum_s / duty_cnt
            f.write(f"    contact: flicker={flick:.1f} trans/s/foot  stance-duty={duty*100:.0f}%"
                    f"  (MuJoCo ref: smooth 0.52s stances, duty 78%; Isaac@200Hz was ~5.8)\n")
        if tq_cnt > 0:
            m = (tq_sum / tq_cnt).cpu()
            rms = (tq_sq / tq_cnt).sqrt().cpu()
            jn = robot.data.joint_names
            f.write("    |torque| mean(rms) Nm: " + "  ".join(
                f"{nm.replace('dof_','')}={m[i]:.1f}({rms[i]:.1f})" for i, nm in enumerate(jn)) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
