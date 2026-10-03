"""SIM2SIM BATTERY 2 — friction/stiction, Isaac side.

Reimplementation of the rig's `rig_sim2sim_friction.py` per SIM2SIM_FRICTION_SPEC.md.

A. AMPLITUDE LADDER (0.5 .. 20 deg): steady-state error vs step size separates
   the mechanisms by SHAPE. intercept -> Coulomb (tau_c = radians(intercept)*kp),
   slope -> compliance. One env per (joint, amplitude) = 60 envs, one pass.
   NB their intercepts sit inside their 16-bit encoder quantum (0.0219 deg) and
   are therefore UPPER BOUNDS. We have no quantiser (float32), so our ladder
   should resolve real friction where theirs cannot — a genuine intercept from
   us against their noise floor is EXPECTED, not a plant difference.

B. FREE DECAY (kp=kd=0, release from 15 deg): envelope shape. Linear -> Coulomb,
   exponential -> viscous. The actuator is switched off, so this is the one test
   that compares PLANTS rather than plant-plus-actuator. One env per joint.

Base pinned AND LIFTED clear of the floor (their §4 contact trap).
"""
import argparse
import functools
import json
import math
import sys

print = functools.partial(print, flush=True)  # noqa: A001
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--series_k", type=float, default=52.0)
parser.add_argument("--play_deg", type=float, default=0.3)
parser.add_argument("--lift", type=float, default=0.6)
parser.add_argument("--decay_s", type=float, default=6.0)
parser.add_argument("--skip_ladder", action="store_true")
parser.add_argument("--actuator_off", type=str, default="gains",
                    choices=["gains", "no_coulomb", "plant_only"],
                    help="what \"kp=kd=0\" means on our side. gains: only the PD is off, our "
                         "Coulomb+viscous stay live (they are actuator-side, in Nm). no_coulomb: "
                         "also drop Coulomb. plant_only: drop Coulomb AND viscous -> the true "
                         "bare-plant arm, the closest analogue to the rig's Test B.")
parser.add_argument("--dump", type=str, default="eval_watch/sim2sim_out/decay_traces.tsv")
parser.add_argument("--out", type=str, default="eval_watch/sim2sim_out/friction_metrics.json")
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
LADDER = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SIGN = {"dof_right_hip_pitch_04": +1, "dof_right_hip_roll_04": -1, "dof_right_hip_yaw_03": +1,
        "dof_right_knee_04": -1, "dof_right_ankle_02": +1,
        "dof_left_hip_pitch_04": +1, "dof_left_hip_roll_04": +1, "dof_left_hip_yaw_03": +1,
        "dof_left_knee_04": +1, "dof_left_ankle_02": -1}
DECAY_START = 15.0
D2R = math.pi / 180.0


def main():
    n_env = len(JOINTS) * len(LADDER)          # 60: ladder needs the most
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    for jn, acfg in env_cfg.scene.robot.actuators.items():
        acfg.tv_randomization = 0.0
        m = (acfg.min_delay + acfg.max_delay) // 2
        acfg.min_delay, acfg.max_delay = m, m
        if "ankle" in jn:
            acfg.series_k = args_cli.series_k
            acfg.play_range = (args_cli.play_deg * D2R, args_cli.play_deg * D2R)
        else:
            acfg.play_range = (0.0, 0.0)
        if args_cli.actuator_off in ("no_coulomb", "plant_only"):
            acfg.coulomb_fc = 0.0
        if args_cli.actuator_off == "plant_only":
            acfg.viscous_b = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e4, 1e4)
    for ev in ("push_robot", "sustained_push", "walk_at_spawn", "stand_corridor",
               "randomize_actuator_gains", "randomize_gains_small_joints",
               "randomize_gains_04_joints", "randomize_joint_play",
               "randomize_joint_friction_hip_pitch_roll", "randomize_joint_friction_knees",
               "randomize_joint_friction_yaws"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "plant_friction_level",
               "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.episode_length_s = 1e4

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    uenv.reset()
    robot = uenv.scene["robot"]
    sim, scene, dev = uenv.sim, uenv.scene, uenv.device
    phys_dt = sim.get_physics_dt()
    rate_hz = 1.0 / phys_dt
    names = list(robot.data.joint_names)
    JI = [names.index(j) for j in JOINTS]
    nj = len(names)

    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    pose = root[:, :7].clone()
    pose[:, 2] += args_cli.lift
    zvel = torch.zeros(n_env, 6, device=dev)
    zero_q = torch.zeros(n_env, nj, device=dev)

    def step_sim(targets):
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(zvel)
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(phys_dt)

    results = {j: {} for j in JOINTS}

    # ================= TEST A: amplitude ladder =================
    # env index e = ji*len(LADDER) + ai  -> joint ji held at LADDER[ai]*SIGN
    if not args_cli.skip_ladder:
        tgt = torch.zeros(n_env, nj, device=dev)
        for ji, jn in enumerate(JOINTS):
            for ai, a in enumerate(LADDER):
                tgt[ji * len(LADDER) + ai, JI[ji]] = a * SIGN[jn] * D2R
        ank_act = {j: robot.actuators[j] for j in JOINTS if "ankle" in j}
        with torch.inference_mode():
            robot.write_joint_state_to_sim(zero_q, zero_q)
            for _ in range(int(1.0 / phys_dt)):
                step_sim(zero_q)
            for _ in range(int(2.0 / phys_dt)):
                step_sim(tgt)
            tail, tail_m = [], []
            for _ in range(int(0.3 / phys_dt)):        # their 30-sample @100 Hz window
                step_sim(tgt)
                tail.append(robot.data.joint_pos.clone())
                tail_m.append(torch.stack([ank_act[j].motor_pos[:, 0] for j in ank_act], 1))
        TAIL = torch.stack(tail).mean(0).cpu() / D2R   # (envs, nj) deg  -- JOINT side
        TAILM = torch.stack(tail_m).mean(0).cpu() / D2R  # (envs, n_ank)  -- MOTOR side
        ank_cols = list(ank_act.keys())

        print("=== A. AMPLITUDE LADDER (steady-state error vs step size) ===")
        print("%-24s %7s %10s %11s" % ("joint", "amp", "reached", "err(deg)"))
        for ji, jn in enumerate(JOINTS):
            amps, errs, errs_m = [], [], []
            for ai, a in enumerate(LADDER):
                got = float(TAIL[ji * len(LADDER) + ai, JI[ji]])
                err = abs(a) - abs(got)                # positive = fell short
                amps.append(abs(a))
                errs.append(err)
                if jn in ank_cols:
                    gm = float(TAILM[ji * len(LADDER) + ai, ank_cols.index(jn)])
                    errs_m.append(abs(a) - abs(gm))
                print("%-24s %6.1f° %9.3f° %10.4f°"
                      % (jn.replace("dof_", "") if ai == 0 else "", abs(a), abs(got), err))
            slope, icpt = np.polyfit(np.array(amps), np.array(errs), 1)
            tau_c = math.radians(icpt) * KP[ji]        # DEGREES -> rad before x kp
            results[jn].update({
                "ladder_amp_deg": amps, "ladder_err_deg": [round(e, 5) for e in errs],
                "fit_slope": round(float(slope), 6), "fit_intercept_deg": round(float(icpt), 5),
                "implied_coulomb_Nm": round(float(tau_c), 4),
                "intercept_in_lsb": None, "quantisation_limited": False,
                "implied_series_frac": round(float(slope), 5)})
            if errs_m:
                sm, im = np.polyfit(np.array(amps), np.array(errs_m), 1)
                results[jn]["motor_side"] = {
                    "ladder_err_deg": [round(e, 5) for e in errs_m],
                    "fit_slope": round(float(sm), 6), "fit_intercept_deg": round(float(im), 5),
                    "implied_coulomb_Nm": round(float(math.radians(im) * KP[ji]), 4),
                    "note": "directly comparable to YOUR ankle rows, which are motor side"}
                print("%-24s  motor-side fit: err = %.5f*amp %+.5f deg\n" % ("", sm, im))
            print("%-24s  fit: err = %.5f*amp %+.5f deg -> Coulomb %.4f Nm, compliance %.3f%% of amp\n"
                  % ("", slope, icpt, abs(tau_c), 100 * slope))

    # ================= TEST B: free decay (actuator OFF) =================
    # env index e = ji  -> joint ji limp, all others held at 0 with full gains.
    print("=== B. FREE DECAY (kp=kd=0; plant only, no actuator) ===")
    print("%-24s %8s %9s %10s %10s %s" % ("joint", "peaks", "lin_r2", "exp_r2", "halflife", "verdict"))
    tgt = torch.zeros(n_env, nj, device=dev)
    for ji, jn in enumerate(JOINTS):
        tgt[ji, JI[ji]] = DECAY_START * SIGN[jn] * D2R
    with torch.inference_mode():
        robot.write_joint_state_to_sim(zero_q, zero_q)
        for _ in range(int(2.5 / phys_dt)):            # drive to the release point
            step_sim(tgt)
        # go limp: zero kp/kd for env ji on joint ji only. Actuator groups are
        # per joint name, so stiffness/damping are (num_envs, 1) tensors.
        saved = {}
        for ji, jn in enumerate(JOINTS):
            act = robot.actuators[jn]
            saved[jn] = (act.stiffness.clone(), act.damping.clone())
            act.stiffness[ji, :] = 0.0
            act.damping[ji, :] = 0.0
        rec, recv, rect = [], [], []
        for _ in range(int(args_cli.decay_s / phys_dt)):
            step_sim(tgt)
            rec.append(robot.data.joint_pos.clone())
            recv.append(robot.data.joint_vel.clone())
            rect.append(torch.stack([robot.actuators[j].applied_effort[:, 0]
                                     for j in JOINTS], 1).clone())
        for jn, (s, d) in saved.items():
            robot.actuators[jn].stiffness.copy_(s)
            robot.actuators[jn].damping.copy_(d)
    R = torch.stack(rec).cpu() / D2R                   # (steps, envs, nj) deg
    RV = torch.stack(recv).cpu() / D2R                 # deg/s
    RT = torch.stack(rect).cpu()                       # Nm, (steps, envs, 10)
    tvec = np.arange(R.shape[0]) * phys_dt

    with open(args_cli.dump, "w") as fh:      # raw traces, so the fit is auditable
        fh.write("# t_s\t" + "\t".join(JOINTS) + "\n")
        for i in range(R.shape[0]):
            fh.write("%.4f\t" % tvec[i]
                     + "\t".join("%.5f" % float(R[i, k, JI[k]]) for k in range(len(JOINTS))) + "\n")

    for ji, jn in enumerate(JOINTS):
        y = R[:, ji, JI[ji]].numpy().astype(float)
        # HYSTERESIS extremum detector. The naive "sign of the first difference
        # flipped" test fired on 830-1195 of 1200 samples on the first run --
        # it was counting numerical dither, not swings, and would have handed
        # the rig a half-life fitted to noise. Confirm an extremum only after
        # the signal reverses by more than `prom` from the running extreme.
        y_end = float(y[-1])
        prom = max(0.05, 0.02 * abs(float(y[0]) - y_end))
        pk, ext_i, rising = [], 0, None
        for i in range(1, len(y)):
            if rising is None:
                if abs(y[i] - y[ext_i]) > prom:
                    rising = y[i] > y[ext_i]
                continue
            if (rising and y[i] > y[ext_i]) or (not rising and y[i] < y[ext_i]):
                ext_i = i
            elif abs(y[i] - y[ext_i]) > prom:
                pk.append((tvec[ext_i], abs(y[ext_i] - y_end)))
                ext_i, rising = i, not rising
        pk = [p for p in pk if p[1] > 0.05]
        # CONTROLS. A limp joint that neither sticks nor swings looks like a
        # decaying signal to any envelope fit, so report the evidence that
        # separates a real swing from a per-timestep limit cycle:
        #   qd_signflip_frac ~1.0  -> the velocity reverses EVERY step (dither)
        #   |tau| with gains at 0  -> what is still driving the joint
        qd = RV[:, ji, JI[ji]].numpy().astype(float)
        tau = RT[:, ji, ji].numpy().astype(float)
        flips = int(np.sum(qd[1:] * qd[:-1] < 0))
        m_ctrl = {"qd_signflip_frac": round(flips / max(len(qd) - 1, 1), 4),
                  "abs_tau_mean_Nm_gains_zero": round(float(np.abs(tau).mean()), 4),
                  "qd_abs_max_deg_s": round(float(np.abs(qd).max()), 2)}
        # CONTROL: an extremum every few samples is dither, not oscillation.
        dither = len(pk) > 0.25 * len(y) or m_ctrl["qd_signflip_frac"] > 0.5
        m = {"decay_start_deg": DECAY_START * SIGN[jn], "n_peaks": len(pk),
             "peak_prominence_deg": round(prom, 4),
             "osc_freq_hz": round(len(pk) / (2.0 * max(args_cli.decay_s, 1e-9)), 3),
             "peak_to_peak_first_0p5s_deg": round(float(y[:int(0.5 / phys_dt)].ptp()), 4),
             "detector_dither_flag": bool(dither), **m_ctrl}
        if dither:
            m["envelope_verdict"] = ("REJECTED - joint dithers rather than swings "
                                     "(qd reverses %.0f%% of steps, |tau|=%.2f Nm with gains at 0)"
                                     % (100 * m_ctrl["qd_signflip_frac"],
                                        m_ctrl["abs_tau_mean_Nm_gains_zero"]))
            print("%-24s %8d %9s %10s %10s  %s"
                  % (jn.replace("dof_", ""), len(pk), "-", "-", "-", m["envelope_verdict"]))
            m["settled_deg"] = round(float(y[-1]), 4)
            m["residual_from_zero_deg"] = round(float(abs(y[-1])), 4)
            results[jn].update(m)
            continue
        if len(pk) >= 3:
            tp = np.array([p[0] for p in pk])
            ap = np.array([p[1] for p in pk])
            lin = np.polyfit(tp, ap, 1)
            r2l = 1 - np.sum((ap - np.polyval(lin, tp)) ** 2) / max(np.sum((ap - ap.mean()) ** 2), 1e-12)
            lg = np.log(np.maximum(ap, 1e-6))
            ex = np.polyfit(tp, lg, 1)
            r2e = 1 - np.sum((lg - np.polyval(ex, tp)) ** 2) / max(np.sum((lg - lg.mean()) ** 2), 1e-12)
            hl = math.log(2) / abs(ex[0]) if abs(ex[0]) > 1e-9 else float("inf")
            verdict = "COULOMB (linear)" if r2l > r2e + 0.05 else \
                      ("VISCOUS (exponential)" if r2e > r2l + 0.05 else "mixed/ambiguous")
            m.update({"linear_r2": round(float(r2l), 4), "exp_r2": round(float(r2e), 4),
                      "halflife_s": round(float(hl), 3), "envelope_verdict": verdict})
            print("%-24s %8d %9.3f %10.3f %9.2fs  %s"
                  % (jn.replace("dof_", ""), len(pk), r2l, r2e, hl, verdict))
        else:
            m["envelope_verdict"] = "no oscillation (overdamped or joint did not swing)"
            print("%-24s %8d %9s %10s %10s  %s"
                  % (jn.replace("dof_", ""), len(pk), "-", "-", "-", m["envelope_verdict"]))
        m["settled_deg"] = round(float(y[-1]), 4)
        m["residual_from_zero_deg"] = round(float(abs(y[-1])), 4)
        results[jn].update(m)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    json.dump({"schema": "sim2sim_friction_v1", "side": "isaac", "ladder_deg": LADDER,
               "decay_start_deg": DECAY_START, "kp": KP, "joint_order": JOINTS,
               "rate_hz": rate_hz, "base": f"pinned+lifted {args_cli.lift} m", "contact": "none",
               "actuator_off_mode": args_cli.actuator_off,
               "position_quantum_deg": 0.0,
               "note": "no encoder quantiser on this side (float32), so ladder intercepts are "
                       "MEASUREMENTS not upper bounds; friction model is -fc*tanh(qd/0.02), "
                       "i.e. NO stiction, same gap as yours",
               "results": results}, open(args_cli.out, "w"), indent=1)
    print(f"\nwrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
