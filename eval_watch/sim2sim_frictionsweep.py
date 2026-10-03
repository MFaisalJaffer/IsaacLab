"""SIM2SIM BATTERY 3 — constant-velocity friction sweep, Isaac side.

Reimplementation of the rig's `rig_sim2sim_frictionsweep.py` per
SIM2SIM_BATTERY3_SPEC.md. After Test B was retracted on both sides, the ~4 Nm of
dynamic Coulomb friction that a policy actually fights had never been compared.

METHOD (theirs, unchanged). Drive one joint at constant velocity across +-SPAN
and measure what it takes, in BOTH directions. Gravity and inertia are identical
either way; Coulomb friction flips sign:

    friction(v) = (ctrl_fwd - ctrl_bwd) / 2     <- gravity cancels exactly
    gravity     = (ctrl_fwd + ctrl_bwd) / 2     <- cross-check, must be dir-independent

Fit friction vs |v|: intercept = Coulomb, slope = viscous.

MEASURE `ctrl`, NOT NET TORQUE (their §"we got this wrong first time"). At
constant velocity the net applied torque just balances gravity whatever the
friction -- friction sits inside it and the PD pays for it in tracking error. We
compute ctrl from logged tracking error exactly as they do, rather than reading
our actuator's internal PD, so the two sides share the method and the recovery
ratio stays comparable.

ANKLES are read MOTOR-side (matching their `/joint_states`); joint-side is
reported alongside. Their ankle PD closes on the rotor and so does ours.

One env per (joint, speed, direction) = 80 envs, one pass.
"""
import argparse
import functools
import json
import math
import sys

print = functools.partial(print, flush=True)  # noqa: A001
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--series_k", type=float, default=52.0)
parser.add_argument("--play_deg", type=float, default=0.3)
parser.add_argument("--pin", type=str, default="fixed", choices=["write", "fixed"],
                    help="how the base is held. fixed: PhysX fix_root_link with the "
                         "spawn raised by --lift, no per-step interference. write: "
                         "re-write root pose every physics step -- CONTAMINATING, it "
                         "suppresses joint dynamics; kept only to reproduce the bug.")
parser.add_argument("--lift", type=float, default=0.6)
parser.add_argument("--span", type=float, default=12.0)
parser.add_argument("--speeds", type=str, default="5,10,20,40",
                    help="deg/s. Rig REPLY3 §2: 40/80 never reach steady state on a +-12 deg span, so the "
                         "10-80 fit was acceleration transient on BOTH sides. Use <=20 (40 kept as the check).")
parser.add_argument("--delay_steps", type=int, default=-1,
                    help=">=0 pins EVERY joint's command delay to this many 5 ms steps (latency calibration); "
                         "-1 = each family's configured midpoint.")
parser.add_argument("--out", type=str, default="eval_watch/sim2sim_out/frictionsweep_metrics.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
          "dof_right_knee_04", "dof_right_ankle_02",
          "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
          "dof_left_knee_04", "dof_left_ankle_02"]
KP = [150.0, 150.0, 60.0, 150.0, 60.0, 150.0, 150.0, 60.0, 150.0, 60.0]
KD = [2.5, 1.5, 1.0, 1.0, 0.5, 2.5, 1.5, 1.0, 1.0, 0.5]
NOMINAL = {"hip_pitch": 4.0, "hip_roll": 4.0, "hip_yaw": 0.4, "knee": 0.6, "ankle": 0.1}
SPEEDS = [float(x) for x in args_cli.speeds.split(",")]
D2R = math.pi / 180.0


def main():
    span = args_cli.span
    n_sp, n_env = len(SPEEDS), len(JOINTS) * len(SPEEDS) * 2
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    for jn, acfg in env_cfg.scene.robot.actuators.items():
        acfg.tv_randomization = 0.0
        m = (acfg.min_delay + acfg.max_delay) // 2
        if args_cli.delay_steps >= 0:
            m = args_cli.delay_steps
        acfg.min_delay, acfg.max_delay = m, m
        if "ankle" in jn:
            acfg.series_k = args_cli.series_k
            acfg.play_range = (args_cli.play_deg * D2R, args_cli.play_deg * D2R)
        else:
            acfg.play_range = (0.0, 0.0)
    # null every model randomiser -- see sim2sim_assets.py for why this matters
    for name in list(vars(env_cfg.events)):
        t = getattr(env_cfg.events, name, None)
        if t is None or not hasattr(t, "func"):
            continue
        fn = getattr(t.func, "__name__", "")
        if any(k in fn or k in name for k in ("mass", "material", "gains", "friction",
                                              "play", "inertia", "push", "walk", "corridor")):
            setattr(env_cfg.events, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level",
               "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
    env_cfg.episode_length_s = 1e4

    if args_cli.pin == "fixed":
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        ip = list(env_cfg.scene.robot.init_state.pos)
        ip[2] += args_cli.lift
        env_cfg.scene.robot.init_state.pos = tuple(ip)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    uenv.reset()
    robot = uenv.scene["robot"]
    sim, scene, dev = uenv.sim, uenv.scene, uenv.device
    phys_dt = sim.get_physics_dt()
    names = list(robot.data.joint_names)
    JI = [names.index(j) for j in JOINTS]
    nj = len(names)
    ank = {j: robot.actuators[j] for j in JOINTS if "ankle" in j}
    ank_cols = list(ank.keys())

    # PLANT CONTROL: prove the configuration we think we are running is the one running.
    print("PLANT CONTROL (what is actually in the loop):")
    for j in JOINTS:
        A = robot.actuators[j]
        print(f"  {j:26s} kp={float(A.stiffness[0,0]):6.1f} kd={float(A.damping[0,0]):5.2f} "
              f"Fc={A._fc_nominal:5.2f} b={A._viscous_b:4.2f} K_s={A._series_k:5.1f} "
              f"delay={A.cfg.min_delay}-{A.cfg.max_delay}")

    # env e = ji*(n_sp*2) + si*2 + di ; di 0 = forward(+v), 1 = backward(-v)
    def unpack(e):
        return e // (n_sp * 2), (e % (n_sp * 2)) // 2, e % 2

    sgn = torch.zeros(n_env, device=dev)
    vcmd = torch.zeros(n_env, device=dev)
    start = torch.zeros(n_env, device=dev)
    dur = torch.zeros(n_env, device=dev)
    for e in range(n_env):
        ji, si, di = unpack(e)
        s = 1.0 if di == 0 else -1.0
        sgn[e], vcmd[e] = s, SPEEDS[si] * s
        start[e] = -span * s
        dur[e] = 2.0 * span / SPEEDS[si]
    max_dur = float(dur.max())

    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    pose = root[:, :7].clone()
    pose[:, 2] += args_cli.lift
    zv6 = torch.zeros(n_env, 6, device=dev)
    zq = torch.zeros(n_env, nj, device=dev)
    rows = torch.arange(n_env, device=dev)
    jcol = torch.tensor([JI[unpack(e)[0]] for e in range(n_env)], device=dev)

    def step_sim(ptgt, vtgt):
        if args_cli.pin == "write":
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(zv6)
        robot.set_joint_position_target(ptgt)
        robot.set_joint_velocity_target(vtgt)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(phys_dt)

    ptgt = torch.zeros(n_env, nj, device=dev)
    vtgt = torch.zeros(n_env, nj, device=dev)
    with torch.inference_mode():
        robot.write_joint_state_to_sim(zq, zq)
        ptgt[rows, jcol] = start * D2R                      # move to start and settle
        for _ in range(int(1.2 / phys_dt)):
            step_sim(ptgt, vtgt)
        rec_t, rec_q, rec_qd, rec_tg, rec_tau, rec_mq = [], [], [], [], [], []
        el = 0.0
        for _ in range(int(round(max_dur / phys_dt))):
            prog = torch.clamp(torch.full_like(dur, el), max=dur)
            tgt_deg = start + sgn * vcmd.abs() * prog
            ptgt.zero_()
            vtgt.zero_()
            ptgt[rows, jcol] = tgt_deg * D2R
            vtgt[rows, jcol] = torch.where(torch.full_like(dur, el) < dur,
                                           vcmd * D2R, torch.zeros_like(vcmd))
            step_sim(ptgt, vtgt)
            el += phys_dt
            rec_t.append(el)
            # ALIGNMENT FIX (2026-09-29): the state read below is the state AFTER this physics
            # step, i.e. at time el. Pairing it with the target computed BEFORE the step (time
            # el - dt) under-reads tracking error by v*dt and biased every latency estimate by
            # exactly -5 ms (hip slope read ~0, below the configured viscous; latency came out
            # negative). Pair the state with the ramp evaluated at the SAME timestamp.
            tgt_now = start + sgn * vcmd.abs() * torch.clamp(torch.full_like(dur, el), max=dur)
            rec_tg.append(tgt_now.clone())
            rec_q.append(robot.data.joint_pos[rows, jcol].clone())
            rec_qd.append(robot.data.joint_vel[rows, jcol].clone())
            rec_tau.append(torch.stack([robot.actuators[j].applied_effort[:, 0]
                                        for j in JOINTS], 1).clone())
            rec_mq.append(torch.stack([ank[j].motor_pos[:, 0] for j in ank], 1).clone())

    T = np.array(rec_t)
    Q = torch.stack(rec_q).cpu().numpy() / D2R              # (steps, envs) deg
    QD = torch.stack(rec_qd).cpu().numpy() / D2R
    TG = torch.stack(rec_tg).cpu().numpy()
    TAU = torch.stack(rec_tau).cpu().numpy()                # (steps, envs, 10)
    MQ = torch.stack(rec_mq).cpu().numpy() / D2R            # (steps, envs, n_ank)

    def window(e):
        d = float(dur[e])
        s = min(0.4, 0.25 * d)                              # skip accel/decel ends
        return (T >= s) & (T <= d - s)

    def ctrl_of(e):
        """kp*perr + kd*verr in Nm, their formula, from logged tracking error."""
        ji, si, di = unpack(e)
        w = window(e)
        q = Q[w, e]
        if JOINTS[ji] in ank_cols:                          # ankles: motor side, as theirs
            q = MQ[w, e, ank_cols.index(JOINTS[ji])]
        perr = (TG[w, e] - q) * D2R
        verr = (float(vcmd[e]) - QD[w, e]) * D2R
        return (float(np.mean(KP[ji] * perr + KD[ji] * verr)),
                float(np.mean(TG[w, e] - q)),
                float(np.mean(np.abs(QD[w, e]))),
                float(np.mean(TAU[w, e, ji])),
                float(np.mean(Q[w, e] - TG[w, e])))

    print("\nBATTERY 3 — constant-velocity friction sweep (Isaac)")
    print("ctrl computed from tracking error, gravity removed by +/- differencing\n")
    print("%-22s %6s %9s %9s %10s %9s %8s" %
          ("joint", "speed", "ctrl_fwd", "ctrl_bwd", "FRICTION", "gravity", "vel ok"))
    results = {}
    for ji, jn in enumerate(JOINTS):
        pts, vs, fr = [], [], []
        for si, v in enumerate(SPEEDS):
            ef = ji * (n_sp * 2) + si * 2
            eb = ef + 1
            cp, epf, vp, taup, jerrf = ctrl_of(ef)
            cn, epb, vn, taun, jerrb = ctrl_of(eb)
            f_tau = 0.5 * (cp - cn)
            g_tau = 0.5 * (cp + cn)
            vach = 0.5 * (abs(vp) + abs(vn))
            ok = vach > 0.6 * v
            pts.append({"speed_deg_s": v, "ctrl_fwd_Nm": round(cp, 4), "ctrl_bwd_Nm": round(cn, 4),
                        "friction_Nm": round(f_tau, 4), "gravity_Nm": round(g_tau, 4),
                        "net_applied_tau_fwd_Nm": round(taup, 4),
                        "track_err_fwd_deg": round(epf, 3), "track_err_bwd_deg": round(epb, 3),
                        "vel_achieved_deg_s": round(vach, 2), "control_pass": bool(ok)})
            print("%-22s %5.0f° %8.3f %9.3f %10.3f %9.3f %7.0f%%%s"
                  % (jn.replace("dof_", "") if si == 0 else "", v, cp, cn, f_tau, g_tau,
                     100 * vach / v, "" if ok else "  <- DROPPED"))
            if ok:
                vs.append(v)
                fr.append(abs(f_tau))
        m = {"points": pts}
        if len(vs) >= 2:
            slope, icpt = np.polyfit(vs, fr, 1)
            nom = [w for k, w in NOMINAL.items() if k in jn]
            m.update({"coulomb_Nm": round(float(icpt), 4),
                      "viscous_Nm_per_deg_s": round(float(slope), 6),
                      "nominal_Fc_Nm": nom[0] if nom else None,
                      "recovery_pct": round(100 * float(icpt) / nom[0], 1) if nom and nom[0] else None})
            print("%-24s  -> Coulomb %.3f Nm (configured %.1f, recovery %.0f%%), "
                  "viscous %.4f Nm/(deg/s)\n"
                  % ("", icpt, nom[0] if nom else float("nan"),
                     100 * icpt / nom[0] if nom and nom[0] else float("nan"), slope))
        results[jn] = m

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    json.dump({"schema": "sim2sim_frictionsweep_v1", "side": "isaac",
               "speeds_deg_s": SPEEDS, "span_deg": span, "kp": KP, "kd": KD,
               "joint_order": JOINTS, "rate_hz": 1.0 / phys_dt,
               "delay_steps_pinned": args_cli.delay_steps,
               "alignment": "state and target paired at the same timestamp (post-step)",
               "base": f"pinned+lifted {args_cli.lift} m", "contact": "none",
               "method": "tau_friction = (ctrl_fwd - ctrl_bwd)/2 at matched |v|; gravity cancels",
               "torque_source": "ctrl RECONSTRUCTED from logged tracking error (kp*perr + kd*verr), "
                                "their method, not our actuator's internal PD; "
                                "net_applied_tau_fwd_Nm is our actuator's post-clamp output "
                                "for cross-check",
               "ankle_readout": "MOTOR side (matches your /joint_states)",
               "results": results}, open(args_cli.out, "w"), indent=1)
    print(f"\nwrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
