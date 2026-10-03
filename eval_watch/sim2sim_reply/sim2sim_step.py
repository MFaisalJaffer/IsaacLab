"""SIM2SIM BATTERY 1 — per-joint step response, Isaac side.

Reimplementation of the rig's `rig_sim2sim_step.py` (ROS2+MuJoCo, not runnable
here) per SIM2SIM_STEP_SPEC.md. Amplitudes/gains/timing are COPIED from their
script, never derived locally: deriving from each sim's own limits would let a
USD/MJCF limit difference silently make the two sides run different tests.

Full stack: the TV actuator is in the loop (T-V ceiling, per-family command
delay, Coulomb+viscous friction, ankle series spring). Base pinned AND LIFTED
(their §4: "base pinned" is not "no contact" — their feet cleared the floor by
3.7 mm at the standing pin height and ~1.6 deg of ankle rotation planted them,
contaminating an entire dataset). We lift 0.6 m and REPORT our own clearance.

One joint per env, so all ten run concurrently in one pass.

Also emits the asset facts their §6 asks for (mass, COM, foot polygon, joint
limits vs their MJCF table) since it needs the same scene.
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
parser.add_argument("--series_k", type=float, default=52.0, help="ankle K_s; rig ran 52")
parser.add_argument("--play_deg", type=float, default=0.3, help="ankle play band; rig ran 0.3")
parser.add_argument("--delay", type=str, default="mid", choices=["mid", "zero"],
                    help="mid = our measured per-family latency at its midpoint; "
                         "zero = latency removed, to separate plant from transport lag")
parser.add_argument("--pin", type=str, default="fixed", choices=["write", "fixed"],
                    help="how the base is held. fixed: PhysX fix_root_link with the "
                         "spawn raised by --lift, no per-step interference. write: "
                         "re-write root pose every physics step -- CONTAMINATING, it "
                         "suppresses joint dynamics; kept only to reproduce the bug.")
parser.add_argument("--lift", type=float, default=0.6)
parser.add_argument("--out", type=str, default="eval_watch/sim2sim_out/step_metrics.json")
parser.add_argument("--assets", action="store_true", help="also dump asset facts (their §6)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
AMP = {  # (positive_deg, negative_deg); 0 = direction outside the joint's range
    "dof_right_hip_pitch_04": (20.0, -20.0), "dof_right_hip_roll_04": (7.0, -20.0),
    "dof_right_hip_yaw_03": (20.0, -20.0), "dof_right_knee_04": (0.0, -20.0),
    "dof_right_ankle_02": (20.0, -7.0),
    "dof_left_hip_pitch_04": (20.0, -20.0), "dof_left_hip_roll_04": (20.0, -7.0),
    "dof_left_hip_yaw_03": (20.0, -20.0), "dof_left_knee_04": (20.0, 0.0),
    "dof_left_ankle_02": (7.0, -20.0),
}
MJCF_RANGE = {  # their table, for the limit cross-check they asked for
    "dof_right_hip_pitch_04": (-127, 60), "dof_right_hip_roll_04": (-130, 12),
    "dof_right_hip_yaw_03": (-90, 90), "dof_right_knee_04": (-155, 0),
    "dof_right_ankle_02": (-13, 72),
    "dof_left_hip_pitch_04": (-60, 127), "dof_left_hip_roll_04": (-12, 130),
    "dof_left_hip_yaw_03": (-90, 90), "dof_left_knee_04": (0, 155),
    "dof_left_ankle_02": (-72, 13),
}
HOLD_S, SETTLE_S = 2.0, 1.5
_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
D2R = math.pi / 180.0


def metrics(t, y, y0, tgt):
    """Their metrics(), unchanged, so the two files mean the same thing."""
    step = tgt - y0
    if abs(step) < 1e-6:
        return {}

    def frac(fr):
        want = y0 + fr * step
        for i, v in enumerate(y):
            if (step > 0 and v >= want) or (step < 0 and v <= want):
                return t[i]
        return None
    t10, t90 = frac(0.10), frac(0.90)
    peak = max(y, key=lambda v: (v - y0) * (1 if step > 0 else -1))
    tol = 0.02 * abs(step)
    settle = None
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - tgt) > tol:
            settle = t[i + 1] if i + 1 < len(t) else None
            break
        settle = t[i]
    tail = y[max(0, len(y) - 20):]
    return {"rise_10_90_ms": None if (t10 is None or t90 is None) else round(1000 * (t90 - t10), 1),
            "overshoot_pct": round(100.0 * ((peak - y0) / step - 1.0), 2),
            "settle_ms": None if settle is None else round(1000 * settle, 1),
            "steady_state_err_deg": round(sum(tail) / len(tail) - tgt, 4)}


def main():
    n = len(JOINTS)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)

    # --- plant: pin everything that would otherwise vary env to env ---
    for jn, acfg in env_cfg.scene.robot.actuators.items():
        acfg.tv_randomization = 0.0                       # deterministic T-V ceiling
        if args_cli.delay == "zero":
            acfg.min_delay, acfg.max_delay = 0, 0
        else:                                             # midpoint of our measured band
            m = (acfg.min_delay + acfg.max_delay) // 2
            acfg.min_delay, acfg.max_delay = m, m
        if "ankle" in jn:
            acfg.series_k = args_cli.series_k
            acfg.play_range = (args_cli.play_deg * D2R, args_cli.play_deg * D2R)
        else:
            acfg.play_range = (0.0, 0.0)
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

    if args_cli.pin == "fixed":
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        ip = list(env_cfg.scene.robot.init_state.pos)
        ip[2] += args_cli.lift
        env_cfg.scene.robot.init_state.pos = tuple(ip)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    uenv.reset()
    robot = uenv.scene["robot"]
    sim, scene = uenv.sim, uenv.scene
    dev = uenv.device
    phys_dt = sim.get_physics_dt()
    rate_hz = 1.0 / phys_dt
    names = list(robot.data.joint_names)
    JI = [names.index(j) for j in JOINTS]
    nj = len(names)

    # ---------- asset facts (their §6) ----------
    facts = {}
    masses = robot.root_physx_view.get_masses()[0]
    body_names = list(robot.data.body_names)
    facts["total_mass_kg"] = round(float(masses.sum()), 4)
    facts["link_mass_kg"] = {b: round(float(m), 4) for b, m in zip(body_names, masses)}
    lim = robot.data.joint_limits[0]
    facts["joint_limits_deg"] = {}
    facts["joint_limit_mismatch_vs_mjcf"] = {}
    for j in JOINTS:
        lo, hi = float(lim[names.index(j), 0]) / D2R, float(lim[names.index(j), 1]) / D2R
        facts["joint_limits_deg"][j] = [round(lo, 2), round(hi, 2)]
        mlo, mhi = MJCF_RANGE[j]
        if abs(lo - mlo) > 1.0 or abs(hi - mhi) > 1.0:
            facts["joint_limit_mismatch_vs_mjcf"][j] = {"usd": [round(lo, 2), round(hi, 2)],
                                                        "mjcf": [mlo, mhi]}

    # zero-pose COM and foot clearance: hold all joints at 0, settle, measure.
    zero_q = torch.zeros(uenv.num_envs, nj, device=dev)
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    stand_pose = root[:, :7].clone()
    lifted_pose = stand_pose.clone()
    lifted_pose[:, 2] += args_cli.lift
    zvel = torch.zeros(uenv.num_envs, 6, device=dev)

    def hold_base(pose):
        if args_cli.pin == "write":
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(zvel)

    def step_sim(targets, pose):
        if args_cli.pin == "write":
            hold_base(pose)
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(phys_dt)

    with torch.inference_mode():
        robot.write_joint_state_to_sim(zero_q, zero_q)
        for _ in range(int(1.5 / phys_dt)):
            step_sim(zero_q, stand_pose)
        # clearance AT THE STANDING PIN HEIGHT — the exact thing that bit them
        fid = [body_names.index(f) for f in _FEET]
        foot_z = robot.data.body_pos_w[:, fid, 2]
        # lowest collision point of the foot, from the physx bounding box
        try:
            from pxr import Usd, UsdGeom
            stage = uenv.sim.stage
            prim_path = robot.cfg.prim_path.replace("env_.*", "env_0")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "physics", "guide"])
            ext = {}
            for f in _FEET:
                pr = stage.GetPrimAtPath(f"{prim_path}/{f}")
                bb = cache.ComputeWorldBound(pr).ComputeAlignedRange()
                ext[f] = {"min": [round(float(v), 4) for v in bb.GetMin()],
                          "max": [round(float(v), 4) for v in bb.GetMax()]}
            facts["foot_world_bbox_at_zero_pose_m"] = ext
            floor_gap = min(ext[f]["min"][2] for f in _FEET)
            facts["foot_to_floor_clearance_at_standing_pin_mm"] = round(1000 * floor_gap, 2)
        except Exception as e:  # noqa: BLE001
            facts["foot_bbox_error"] = str(e)
            facts["foot_to_floor_clearance_at_standing_pin_mm"] = round(
                1000 * float(foot_z.min()), 2)
        com_w = (robot.data.body_pos_w[0] * masses.to(dev).unsqueeze(1)).sum(0) / masses.sum().to(dev)
        ank = body_names.index(_FEET[0])
        rootp = robot.data.root_pos_w[0]
        facts["com_minus_ankle_axis_x_m_zero_pose"] = round(
            float(com_w[0] - robot.data.body_pos_w[0, ank, 0]), 5)
        facts["com_x_rel_root_m_zero_pose"] = round(float(com_w[0] - rootp[0]), 5)
        facts["ankle_axis_x_rel_root_m_zero_pose"] = round(
            float(robot.data.body_pos_w[0, ank, 0] - rootp[0]), 5)
        facts["foot_polygon_x_rel_ankle_axis_m"] = [
            round(float(min(ext[f]["min"][0] for f in _FEET)
                        - robot.data.body_pos_w[0, ank, 0]), 4),
            round(float(max(ext[f]["max"][0] for f in _FEET)
                        - robot.data.body_pos_w[0, ank, 0]), 4)] if "foot_world_bbox_at_zero_pose_m" in facts else None
        facts["com_correction_applied"] = "YES - body-frame y -0.0243 m on Torso_Side_Right"
        facts["position_quantum_deg"] = 0.0
        facts["position_quantum_note"] = "float32, no encoder quantiser anywhere in the chain"
        facts["stiction_model"] = "NONE - identical -fc*tanh(qd/0.02) form to yours; zero at qd=0"
        facts["ankle_readout"] = "BOTH reported: joint-side (pos_*) and motor-side (motorpos_*)"

    print("ASSET FACTS:", json.dumps(facts, indent=1))

    # ---------- step battery ----------
    print(f"\nSIM2SIM step | full stack | base pinned+lifted {args_cli.lift} m | "
          f"delay={args_cli.delay} | K_s={args_cli.series_k} | {rate_hz:.0f} Hz\n")
    amps_per = [[a for a in AMP[j] if abs(a) > 1e-6] for j in JOINTS]
    max_amps = max(len(a) for a in amps_per)
    # phases: 1.0 s zero, then per amp slot: HOLD at amp, SETTLE at zero
    phases = [("zero", 1.0)] + sum([[("amp%d" % k, HOLD_S), ("zero", SETTLE_S)]
                                    for k in range(max_amps)], [])

    ank_act = {j: robot.actuators[j] for j in JOINTS if "ankle" in j}
    rec_t, rec_tgt, rec_pos, rec_vel, rec_mot = [], [], [], [], []
    tgt = torch.zeros(uenv.num_envs, nj, device=dev)
    t_el = 0.0
    with torch.inference_mode():
        robot.write_joint_state_to_sim(zero_q, zero_q)
        for pname, dur in phases:
            tgt.zero_()
            cur = torch.zeros(uenv.num_envs, device=dev)
            if pname != "zero":
                k = int(pname[3:])
                for e, j in enumerate(JOINTS):
                    a = amps_per[e][k] if k < len(amps_per[e]) else 0.0
                    tgt[e, JI[e]] = a * D2R
                    cur[e] = a
            for _ in range(int(round(dur / phys_dt))):
                step_sim(tgt, lifted_pose)
                t_el += phys_dt
                rec_t.append(t_el)
                rec_tgt.append(cur.clone())
                rec_pos.append(robot.data.joint_pos[:, JI].clone())
                rec_vel.append(robot.data.joint_vel[:, JI].clone())
                rec_mot.append(torch.stack([ank_act[j].motor_pos[:, 0] for j in ank_act], 1)
                               if all(ank_act[j].motor_pos is not None for j in ank_act)
                               else torch.zeros(uenv.num_envs, len(ank_act), device=dev))

    T = torch.tensor(rec_t)
    TG = torch.stack(rec_tgt).cpu()                    # (steps, envs)
    P = torch.stack(rec_pos).cpu() / D2R               # (steps, envs, 10) deg
    V = torch.stack(rec_vel).cpu() / D2R
    M = torch.stack(rec_mot).cpu() / D2R               # (steps, envs, n_ankle) deg
    ank_cols = list(ank_act.keys())

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    results = {}
    print("%-26s %16s %10s %s" % ("joint", "reached", "leak", "reach control"))
    for e, jn in enumerate(JOINTS):
        m = {"amplitudes_deg": AMP[jn]}
        reach_ok, reached = True, []
        # per-env trace of ITS OWN test joint; motor-side too if it is an ankle
        own = P[:, e, e]
        own_mot = M[:, e, ank_cols.index(jn)] if jn in ank_cols else None
        for a in amps_per[e]:
            sel = (TG[:, e] - a).abs() < 1e-6
            if not bool(sel.any()):
                continue
            idx = torch.nonzero(sel).squeeze(1)
            yy = own[idx].tolist()
            tt = (T[idx] - T[idx[0]]).tolist()
            got = yy[-1]
            reached.append(got)
            if abs(got - a) > max(2.0, 0.15 * abs(a)):
                reach_ok = False
            lbl = "pos" if a > 0 else "neg"
            m[lbl] = metrics(tt, yy, yy[0], a)
            m[lbl]["target_deg"] = a
            m[lbl]["reached_deg"] = round(got, 3)
            m[lbl]["peak_vel_deg_s"] = round(float(V[idx, e, e].abs().max()), 2)
            if own_mot is not None:
                ym = own_mot[idx].tolist()
                mm = metrics(tt, ym, ym[0], a)
                m[lbl]["motor_side"] = {"reached_deg": round(ym[-1], 3),
                                        "rise_10_90_ms": mm.get("rise_10_90_ms"),
                                        "overshoot_pct": mm.get("overshoot_pct")}
        held = [k for k in range(len(JOINTS)) if k != e]
        leak_v, leak_k = 0.0, ""
        mx = P[:, e, :].abs().max(dim=0).values
        for k in held:
            if float(mx[k]) > leak_v:
                leak_v, leak_k = float(mx[k]), JOINTS[k]
        m["max_leak_deg"] = round(leak_v, 3)
        m["worst_leak_joint"] = leak_k
        m["reach_control_pass"] = reach_ok
        results[jn] = m
        print("%-26s %16s %9.2f  %s" % (jn, "/".join("%+.1f" % r for r in reached), leak_v,
                                        "OK" if reach_ok else "!! FAIL - exclude"))

    json.dump({"schema": "sim2sim_step_v2", "side": "isaac",
               "rate_hz": rate_hz, "kp": KP, "kd": KD, "joint_order": JOINTS,
               "amplitudes_deg": AMP, "hold_s": HOLD_S, "settle_s": SETTLE_S,
               "base": f"pinned+lifted {args_cli.lift} m", "contact": "none", "gravity_z": -9.81,
               "delay_arm": args_cli.delay,
               "stack": f"FULL: TVCurveActuator (T-V ceiling, per-family command delay, "
                        f"Coulomb+viscous friction in Nm, ankle series K_s={args_cli.series_k} "
                        f"b=0.07 play {args_cli.play_deg} deg)",
               "ankle_readout": "joint-side primary; motor_side sub-dict added per direction",
               "asset_facts": facts, "results": results},
              open(args_cli.out, "w"), indent=1)
    print(f"\nwrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
