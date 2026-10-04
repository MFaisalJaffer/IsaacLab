"""Parked-robot test (rig RIG_REPLY_HW_STAND_SENSING.md §3): does the plant stay put at its balance point with NO
policy — a plain home hold?

The real robot does: balanced by hand "like a pencil on its tip", it stays 21-32 s under a home hold with the
torso within 0.15 deg, so the real ankle has a narrow sticky band. The rig's emulator (tanh friction, no static
regime), balanced by bisection to 0.001 Nm, stays at most 3.3-4.0 s. This asks what our plant does, with and
without the two sticky mechanisms it has: the PhysX ankle joint friction that training randomizes (0.03-0.18)
and the ankle rotor Coulomb friction with a true static regime (TVCurveActuator._rotor_fc).

Every robot: zero actions (joint targets = the home pose), nominal gains and masses, ankle spring 52 Nm/rad, at
rest. Each plant variant gets a sweep of constant torso pitch moments; the one that balances the robot is where
it stays longest. "Parked" = torso pitch within --band deg of where it was at 1 s.
Reported per variant: the longest park, and how wide the range of moments is that parks >= 10 s and >= 20 s
(a pencil on its tip has width 0; a sticky ankle has a band).

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/amp_balance_point.py --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--per_variant", type=int, default=96)
parser.add_argument("--seconds", type=float, default=32.0)
parser.add_argument("--band", type=float, default=0.5)
parser.add_argument("--centers", default="", help="comma list of sweep centres (Nm), one per variant; empty = 0 for all")
parser.add_argument("--span", type=float, default=4.0, help="the sweep covers centre +- span")
parser.add_argument("--play_deg", type=float, default=2.0)
parser.add_argument("--tag", default="balance_point")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
# name, PhysX ankle joint friction coefficient, ankle rotor stiction Nm
VARIANTS = [
    ("no ankle friction, no rotor stiction", 0.0, 0.0),
    ("ankle joint friction 0.03 (low end of training)", 0.03, 0.0),
    ("ankle joint friction 0.10", 0.10, 0.0),
    ("ankle joint friction 0.18 (high end of training)", 0.18, 0.0),
    ("rotor stiction 1.0 Nm only", 0.0, 1.0),
    ("ankle joint friction 0.10 + rotor stiction 1.0 Nm", 0.10, 1.0),
]


def main() -> int:
    per = args.per_variant
    n = per * len(VARIANTS)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 5.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0; c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    for name in ("walk_at_spawn", "stand_corridor", "push_robot", "randomize_actuator_gains", "randomize_gains_small_joints",
                 "randomize_gains_04_joints", "add_limb_masses", "randomize_joint_properties", "randomize_imu_mount",
                 "randomize_joint_friction_ankles", "randomize_joint_play", "amp_unanswered", "amp_axis_bias"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "series_k_band", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    # ONE plant for every robot of a variant: the torso CoM randomization (a startup event, a different offset per
    # robot, worth up to +-0.9 Nm of balance moment) is pinned to its centre — the first runs of this test swept the
    # moment across robots that did not share a balance point, and their results were noise.
    if getattr(ev, "correct_torso_com", None) is not None:
        cr = ev.correct_torso_com.params["com_range"]
        ev.correct_torso_com.params["com_range"] = {k: (0.5 * (v[0] + v[1]), 0.5 * (v[0] + v[1])) for k, v in cr.items()}
    ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
    ev.sustained_push.interval_range_s = (0.02, 0.02)
    for name in list(vars(env_cfg.terminations)):
        if name != "time_out" and getattr(env_cfg.terminations, name, None) is not None and not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    dt = float(uenv.step_dt)
    var = torch.arange(n, device=dev) // per
    centers = [float(x) for x in args.centers.split(",")] if args.centers else [0.0] * len(VARIANTS)
    assert len(centers) == len(VARIANTS)
    frac = (torch.arange(n, device=dev) % per).float() / (per - 1) * 2.0 - 1.0                # -1 .. +1 inside each variant
    moment = torch.tensor(centers, device=dev)[var] + frac * args.span
    fric = torch.tensor([v[1] for v in VARIANTS], device=dev)[var]
    rotor = torch.tensor([v[2] for v in VARIANTS], device=dev)[var]
    ank = robot.find_joints(["dof_left_ankle_02", "dof_right_ankle_02"])[0]
    load = torch.zeros(n, 2, device=dev); load[:, 1] = moment

    def set_plant():
        uenv._standing_moment_override = load
        robot.write_joint_friction_coefficient_to_sim(fric.unsqueeze(1).expand(n, len(ank)).clone(), joint_ids=ank)
        for act in robot.actuators.values():
            shape = (n, len(act.joint_names))
            if getattr(act, "_series_k", 0.0) > 0.0:
                act._series_k_env = torch.full(shape, 52.0, device=dev)
                act._play = torch.full(shape, math.radians(args.play_deg), device=dev)
                act._rotor_fc = rotor.unsqueeze(1).expand(shape).clone()
            else:
                act._play = torch.zeros(shape, device=dev)

    T = int(args.seconds / dt)
    zero = torch.zeros(n, uenv.action_manager.total_action_dim, device=dev)
    pitch = torch.zeros(T, n, device=dev)
    with torch.inference_mode():
        env.reset()
        env.step(zero)
        set_plant()
        env.reset()
        set_plant()
        for t in range(T):
            env.step(zero)
            pitch[t] = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1.0, 1.0)) * 180.0 / math.pi
    P = pitch.cpu().numpy()
    t1 = int(1.0 / dt)
    dev_ = np.abs(P - P[t1][None, :])
    out = dev_ > args.band
    out[:t1] = False
    left = np.where(out.any(axis=0), out.argmax(axis=0) * dt, args.seconds)                    # when each robot left the band
    M = moment.cpu().numpy()
    res = {"band_deg": args.band, "seconds": args.seconds, "play_deg": args.play_deg, "variants": {}}
    print(f"\n[park] home hold with no policy, ankle spring 52 Nm/rad, play {args.play_deg} deg, {per} torso moments per variant over centre +- {args.span} Nm, {args.seconds} s; parked = pitch within {args.band} deg of its value at 1 s")
    print(f"{'variant':52s} {'longest park':>12s} {'at moment':>10s} {'pitch there':>12s} | moments that park >= 10 s {'':3s} >= 20 s {'':3s} whole run")
    for i, (name, *_r) in enumerate(VARIANTS):
        m = (var == i).cpu().numpy()
        tt, mm = left[m], M[m]
        j = int(tt.argmax())
        step = (mm.max() - mm.min()) / (per - 1)
        w = lambda thr: float((tt >= thr).sum() * step)
        r = {"longest_s": float(tt[j]), "moment_nm": float(mm[j]), "pitch_deg_at_1s": float(P[t1][m][j]), "width_10s_nm": w(10.0), "width_20s_nm": w(20.0),
             "width_full_nm": w(args.seconds - 1e-6), "sweep": [float(mm.min()), float(mm.max())], "resolution_nm": float(step)}
        res["variants"][name] = r
        print(f"{name:52s} {r['longest_s']:10.1f} s {r['moment_nm']:+9.3f} {r['pitch_deg_at_1s']:+10.2f} deg | {r['width_10s_nm']:14.3f} Nm {r['width_20s_nm']:10.3f} Nm {r['width_full_nm']:10.3f} Nm   (sweep step {step:.4f} Nm)")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
