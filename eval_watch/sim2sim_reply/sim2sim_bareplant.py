"""BARE-PLANT SWING — the replacement for the retracted Test B.

Both sides' "kp=kd=0" left friction live, so neither free-decay measured a plant.
This is the arm we both agreed on: gains zero AND friction off, so nothing is in
the loop but gravity and inertia.

Reports the rig's three columns (net travel, qd reversal fraction, peak |qd|)
plus SWING PERIOD, which is the cleanest inertia comparison available -- for a
gravity pendulum it depends only on sqrt(I / m g l) and on nothing else in
either stack. If yaw's period differs, that is the 25-vs-40 ms rise-time gap
showing up in a measurement with no actuator in it.

Every model randomiser is nulled (mass, COM, gains, material) and the plant
control below PRINTS what is actually in the loop -- an earlier pass of this
battery read a domain-randomised draw as the asset.
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
parser.add_argument("--lift", type=float, default=0.6)
parser.add_argument("--decay_s", type=float, default=6.0)
parser.add_argument("--start_deg", type=float, default=15.0)
parser.add_argument("--pin", type=str, default="fixed", choices=["write", "fixed"],
                    help="how the base is held. write: re-write root pose+zero root vel every "
                         "physics step. fixed: PhysX fix_root_link, no per-step interference "
                         "-- the honest reading of \"base pinned\".")
parser.add_argument("--friction", type=str, default="off", choices=["off", "on"],
                    help="off = the bare-plant arm. on = the control arm that reproduces dither.")
parser.add_argument("--out", type=str, default="eval_watch/sim2sim_out/bareplant_metrics.json")
parser.add_argument("--dump", type=str, default="eval_watch/sim2sim_out/bareplant_traces.tsv")
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
SIGN = {"dof_right_hip_pitch_04": +1, "dof_right_hip_roll_04": -1, "dof_right_hip_yaw_03": +1,
        "dof_right_knee_04": -1, "dof_right_ankle_02": +1,
        "dof_left_hip_pitch_04": +1, "dof_left_hip_roll_04": +1, "dof_left_hip_yaw_03": +1,
        "dof_left_knee_04": +1, "dof_left_ankle_02": -1}
D2R = math.pi / 180.0


def main():
    n = len(JOINTS)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    for jn, acfg in env_cfg.scene.robot.actuators.items():
        acfg.tv_randomization = 0.0
        m = (acfg.min_delay + acfg.max_delay) // 2
        acfg.min_delay, acfg.max_delay = m, m
        acfg.play_range = (0.0, 0.0)
        if args_cli.friction == "off":
            acfg.coulomb_fc = 0.0
            acfg.viscous_b = 0.0
    for name in list(vars(env_cfg.events)):
        t = getattr(env_cfg.events, name, None)
        if t is None or not hasattr(t, "func"):
            continue
        fn = getattr(t.func, "__name__", "")
        if any(k in fn or k in name for k in ("mass", "com", "material", "gains", "friction",
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

    print(f"PLANT CONTROL (friction={args_cli.friction}, pin={args_cli.pin}):")
    ctl = {}
    for j in JOINTS:
        A = robot.actuators[j]
        ctl[j] = {"Fc_nominal": A._fc_nominal, "viscous_b": A._viscous_b,
                  "series_k": A._series_k, "kp": float(A.stiffness[0, 0])}
        print(f"  {j:26s} kp={float(A.stiffness[0,0]):6.1f} Fc={A._fc_nominal:5.2f} "
              f"b={A._viscous_b:4.2f} K_s={A._series_k:5.1f}")
    tot = float(robot.root_physx_view.get_masses()[0].sum())
    print(f"  total mass {tot:.4f} kg (nominal 13.06 if randomisers are truly off)")

    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    pose = root[:, :7].clone()
    pose[:, 2] += args_cli.lift
    zv6 = torch.zeros(n, 6, device=dev)
    zq = torch.zeros(n, nj, device=dev)

    def step_sim(t):
        if args_cli.pin == "write":
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(zv6)
        robot.set_joint_position_target(t)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(phys_dt)

    tgt = torch.zeros(n, nj, device=dev)
    for ji, jn in enumerate(JOINTS):
        tgt[ji, JI[ji]] = args_cli.start_deg * SIGN[jn] * D2R
    with torch.inference_mode():
        robot.write_joint_state_to_sim(zq, zq)
        for _ in range(int(2.5 / phys_dt)):
            step_sim(tgt)
        held = [float(robot.data.joint_pos[ji, JI[ji]]) / D2R for ji in range(n)]
        for ji, jn in enumerate(JOINTS):                    # release: env ji, joint ji
            robot.actuators[jn].stiffness[ji, :] = 0.0
            robot.actuators[jn].damping[ji, :] = 0.0
        rq, rv, rt = [], [], []
        for _ in range(int(args_cli.decay_s / phys_dt)):
            step_sim(tgt)
            rq.append(robot.data.joint_pos.clone())
            rv.append(robot.data.joint_vel.clone())
            rt.append(torch.stack([robot.actuators[j].applied_effort[:, 0]
                                   for j in JOINTS], 1).clone())
    Q = torch.stack(rq).cpu().numpy() / D2R
    V = torch.stack(rv).cpu().numpy() / D2R
    TAU = torch.stack(rt).cpu().numpy()
    t = np.arange(Q.shape[0]) * phys_dt

    with open(args_cli.dump, "w") as fh:
        fh.write("# t_s\t" + "\t".join(JOINTS) + "\n")
        for i in range(Q.shape[0]):
            fh.write("%.4f\t" % t[i] + "\t".join("%.5f" % Q[i, k, JI[k]]
                                                 for k in range(n)) + "\n")

    print(f"\nBARE-PLANT SWING — released from {args_cli.start_deg} deg, "
          f"gains 0, friction {args_cli.friction}, {args_cli.decay_s} s\n")
    print("%-22s %9s %10s %10s %10s %9s %s"
          % ("joint", "held@rel", "travel", "qd rev", "peak|qd|", "period", "|tau|"))
    results = {}
    for ji, jn in enumerate(JOINTS):
        y = Q[:, ji, JI[ji]].astype(float)
        qd = V[:, ji, JI[ji]].astype(float)
        tau = TAU[:, ji, ji].astype(float)
        flips = int(np.sum(qd[1:] * qd[:-1] < 0))
        frac = flips / max(len(qd) - 1, 1)
        # period from successive velocity zero-crossings, with a prominence guard
        # so per-step dither cannot masquerade as a swing.
        prom = max(0.05, 0.02 * abs(y[0] - y[-1]))
        ext, e_i, rising = [], 0, None
        for i in range(1, len(y)):
            if rising is None:
                if abs(y[i] - y[e_i]) > prom:
                    rising = y[i] > y[e_i]
                continue
            if (rising and y[i] > y[e_i]) or (not rising and y[i] < y[e_i]):
                e_i = i
            elif abs(y[i] - y[e_i]) > prom:
                ext.append(t[e_i])
                e_i, rising = i, not rising
        period = 2.0 * float(np.mean(np.diff(ext))) if len(ext) >= 3 else None
        m = {"held_at_release_deg": round(held[ji], 3),
             "net_travel_deg": round(float(y[-1] - y[0]), 3),
             "qd_reversal_frac": round(frac, 4),
             "peak_abs_qd_deg_s": round(float(np.abs(qd).max()), 2),
             "n_extrema": len(ext),
             "swing_period_s": round(period, 4) if period else None,
             "abs_tau_mean_Nm": round(float(np.abs(tau).mean()), 4),
             "verdict": ("real swing" if frac < 0.1 else
                         ("dither / limit cycle" if frac > 0.3 else "mixed"))}
        results[jn] = m
        print("%-22s %8.2f° %+9.2f° %9.0f%% %9.1f %8s %8.3f  %s"
              % (jn.replace("dof_", ""), held[ji], m["net_travel_deg"], 100 * frac,
                 m["peak_abs_qd_deg_s"],
                 ("%.3fs" % period) if period else "-", m["abs_tau_mean_Nm"], m["verdict"]))

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    json.dump({"schema": "sim2sim_bareplant_v1", "side": "isaac",
               "friction_arm": args_cli.friction, "base_pin_mode": args_cli.pin, "start_deg": args_cli.start_deg,
               "decay_s": args_cli.decay_s, "joint_order": JOINTS,
               "rate_hz": 1.0 / phys_dt, "base": f"pinned+lifted {args_cli.lift} m",
               "total_mass_kg": round(tot, 4), "plant_control": ctl,
               "results": results}, open(args_cli.out, "w"), indent=1)
    print(f"\nwrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
