"""Engage test: reproduce the first hardware engage (rig RIG_HW_ENGAGE_FINDINGS.md, 2026-10-03) in Isaac.

Every robot is spawned at rest at the zero pose with a stand command from tick 0 (hard phase pin), no pushes —
the way the rig engages a policy. PLANT: by default the training randomization and sensor noise stay ON
(--plant trained). CORRECTION 2026-10-03: the first version of this test stripped them (--plant nominal), which
also removes the ankle joint friction every policy was trained with; on that never-seen plant even an unloaded
v5 stand drifts and falls (34% in 20 s), and the "falls under the standing load" numbers reported that day were
that artifact. On the trained plant v5 holds -3..+3 Nm for 20 s without a fall. Always read the control row
("no load, hard start") at the same duration before trusting any other row.
The envs are split into groups that differ only in:
  load   constant moment on the torso (Nm), the measured standing load of the real robot (~2.5 Nm at the ankles)
  ramp   the rig's old engage crossfade: targets scaled 0 -> 1 and gains 0.5 -> 1 over ramp seconds (0 = hard start)
  rotor  Coulomb stiction on the ankle rotor (Nm)
  dead   output dead band on the rigid joints (Nm)
Per group: ankle action at tick 20 (0.4 s), peak ankle action / joint speed / torso tilt in the first 2 s, and
the share of robots that fell within the test. A policy that does not integrate on unanswered commands keeps
its tick-20 ankle action small in every group (hardware: -1.42 / +1.25, violent motion, power cut at 0.88 s).

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/amp_engage_test.py \
      --checkpoint <ckpt> --tag engage_v5 --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--per_group", type=int, default=32)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--series_k", type=float, default=52.0, help="ankle spring for every env (rig: loaded ankle 52 Nm/rad)")
parser.add_argument("--play_deg", type=float, default=1.0, help="ankle free play for every env (rig: ~1 deg loaded)")
parser.add_argument("--load", type=float, default=2.5)
parser.add_argument("--tag", default="engage_test")
parser.add_argument("--plant", default="trained", choices=("nominal", "trained"), help="nominal = one fixed plant, no sensor noise (NB: also removes the ankle joint friction the policy was trained with); trained = the training randomization (gains, masses, friction, ankle play and spring, IMU mount) and sensor noise stay on")
parser.add_argument("--keep", default="", help="nominal plant only: comma list of randomization events to leave ON (to find which ingredient of the trained plant matters)")
parser.add_argument("--drop", default="", help="trained plant only: comma list of randomization events to switch OFF")
parser.add_argument("--noise", type=int, default=-1, help="sensor noise: -1 = as the plant (nominal off, trained on), 0 = off, 1 = on")
parser.add_argument("--sweep", type=int, default=0, help="1 = load sweep instead of the 15 conditions: pitch and roll moments from -3 to +3 Nm, hard start, to find how much standing load a policy tolerates")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_amp  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
M = args.load
# name, (mx, my) torso moment, ramp seconds, ankle rotor stiction, rigid-joint dead band
GROUPS = [
    ("no load, hard start", (0.0, 0.0), 0.0, 0.0, 0.0),
    ("no load, 1 s ramp", (0.0, 0.0), 1.0, 0.0, 0.0),
    (f"load x+{M}, hard start", (M, 0.0), 0.0, 0.0, 0.0),
    (f"load x-{M}, hard start", (-M, 0.0), 0.0, 0.0, 0.0),
    (f"load y+{M}, hard start", (0.0, M), 0.0, 0.0, 0.0),
    (f"load y-{M}, hard start", (0.0, -M), 0.0, 0.0, 0.0),
    (f"load x+{M}, 1 s ramp", (M, 0.0), 1.0, 0.0, 0.0),
    (f"load x-{M}, 1 s ramp", (-M, 0.0), 1.0, 0.0, 0.0),
    (f"load y+{M}, 1 s ramp", (0.0, M), 1.0, 0.0, 0.0),
    (f"load y-{M}, 1 s ramp", (0.0, -M), 1.0, 0.0, 0.0),
    ("rotor stiction 1.2, hard start", (0.0, 0.0), 0.0, 1.2, 0.0),
    (f"load x+{M} + rotor 1.2 + dead band 0.7, hard", (M, 0.0), 0.0, 1.2, 0.7),
    (f"load x-{M} + rotor 1.2 + dead band 0.7, hard", (-M, 0.0), 0.0, 1.2, 0.7),
    (f"load y+{M} + rotor 1.2 + dead band 0.7, hard", (0.0, M), 0.0, 1.2, 0.7),
    (f"load y-{M} + rotor 1.2 + dead band 0.7, hard", (0.0, -M), 0.0, 1.2, 0.7),
]

if args.sweep == 2:   # short form: no load and the measured load both ways, hard start
    GROUPS = [(f"pitch load {m:+.1f} Nm", (0.0, m), 0.0, 0.0, 0.0) for m in (-2.5, 0.0, 2.5)]
elif args.sweep:
    GROUPS = [(f"pitch load {m:+.1f} Nm", (0.0, m), 0.0, 0.0, 0.0) for m in (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)] + \
             [(f"roll load {m:+.1f} Nm", (m, 0.0), 0.0, 0.0, 0.0) for m in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]


def main() -> int:
    n = args.per_group * len(GROUPS)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 2.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0
    c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    off = ["walk_at_spawn", "stand_corridor", "push_robot", "amp_unanswered", "amp_axis_bias"]
    keep = [x for x in args.keep.split(",") if x]
    if args.plant == "nominal":
        off += [e for e in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints", "add_limb_masses",
                            "randomize_joint_properties", "randomize_imu_mount", "randomize_joint_friction_ankles", "randomize_joint_play", "physics_material") if e not in keep]
    else:
        off += [x for x in args.drop.split(",") if x]
    for name in off:
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "series_k_band", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    # at rest at the zero pose
    ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    # the push term stays as the owner of the external wrench, with no pushes
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
    ev.sustained_push.interval_range_s = (0.02, 0.02)
    ev.engage_probe_ramp = EventTerm(func=mdp_amp.engage_gain_ramp, mode="interval", interval_range_s=(0.02, 0.02))
    noise_on = (args.plant == "trained") if args.noise < 0 else bool(args.noise)
    env_cfg.observations.policy.enable_corruption = noise_on
    fixed_ankle = args.plant == "nominal" and "randomize_joint_play" not in keep
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    jn = list(robot.joint_names)
    aL, aR = jn.index("dof_left_ankle_02"), jn.index("dof_right_ankle_02")
    grp = torch.arange(n, device=dev) // args.per_group
    load = torch.tensor([g[1] for g in GROUPS], device=dev)[grp]              # (n, 2)
    ramp = torch.tensor([g[2] for g in GROUPS], device=dev)[grp]
    rotor = torch.tensor([g[3] for g in GROUPS], device=dev)[grp]
    dead = torch.tensor([g[4] for g in GROUPS], device=dev)[grp]
    act_term = uenv.action_manager.get_term("joint_pos")
    scale0 = act_term._scale if torch.is_tensor(act_term._scale) else torch.full((n, len(jn)), float(act_term._scale), device=dev)

    def set_plant():
        uenv._standing_moment_override = load
        uenv._engage_g0 = torch.where(ramp > 0, torch.full_like(ramp, 0.5), torch.ones_like(ramp))
        uenv._engage_T = ramp
        for act in robot.actuators.values():
            shape = (n, len(act.joint_names))
            if getattr(act, "_series_k", 0.0) > 0.0:
                if fixed_ankle:
                    act._series_k_env = torch.full(shape, args.series_k, device=dev)
                    act._play = torch.full(shape, math.radians(args.play_deg), device=dev)
                act._rotor_fc = rotor.unsqueeze(1).expand(shape).clone()
            else:
                if fixed_ankle:
                    act._play = torch.zeros(shape, device=dev)
                act._deadband = dead.unsqueeze(1).expand(shape).clone()

    T = int(args.seconds / uenv.step_dt)
    A = torch.zeros(T, n, len(jn), device=dev)
    QD = torch.zeros(T, n, device=dev)
    TILT = torch.zeros(T, n, device=dev)
    QA = torch.zeros(T, n, 2, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_at = torch.full((n,), float("inf"), device=dev)
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        set_plant()
        obs, _ = env.reset()
        set_plant()
        for t in range(T):
            now = uenv.episode_length_buf.float() * uenv.step_dt
            alpha = torch.where(ramp > 0, (now / ramp.clamp(min=1e-6)).clamp(0.0, 1.0), torch.ones_like(ramp))
            act_term._scale = scale0 * alpha.unsqueeze(1)                      # the crossfade scales the TARGETS too
            a = policy(obs)
            A[t] = a
            obs, _, dones, _ = env.step(a)
            newly = dones.bool() & ~fell
            fell_at[newly] = t * uenv.step_dt
            fell |= dones.bool()
            QD[t] = robot.data.joint_vel.abs().amax(dim=1)
            QA[t, :, 0] = robot.data.joint_pos[:, aL]; QA[t, :, 1] = robot.data.joint_pos[:, aR]
            TILT[t] = torch.asin(robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)) * 180.0 / math.pi
    w2 = int(2.0 / uenv.step_dt)
    res = {"checkpoint": args.checkpoint, "series_k": args.series_k, "play_deg": args.play_deg, "groups": {}}
    print(f"\n[engage] {args.checkpoint} | plant: {args.plant}" + (f" (stripped; kept on: {keep or 'nothing'}; sensor noise {'on' if noise_on else 'off'})" if args.plant == "nominal" else f" (training randomization on, minus {args.drop or 'nothing'}; sensor noise {'on' if noise_on else 'off'})") + f" | {args.per_group} robots per group, {args.seconds} s")
    print(f"{'group':46s} {'ankle act @0.4s L/R':>20s} {'peak |ankle act| 2s':>20s} {'peak joint speed':>17s} {'peak tilt':>10s} {'fell':>6s}")
    for g, (name, *_rest) in enumerate(GROUPS):
        m = grp == g
        alive = ~(fell_at[m] < 0.4)
        a20 = A[20][m]
        pk = A[:w2, m][:, :, [aL, aR]].abs().amax(dim=0).amax(dim=1)          # per robot
        r = {"ankle_action_t20": [float(a20[:, aL].mean()), float(a20[:, aR].mean())], "peak_ankle_action_2s": float(pk.mean()),
             "peak_any_action_2s": float(A[:w2, m].abs().amax(dim=0).amax(dim=1).mean()),
             "peak_joint_speed_2s": float(QD[:w2, m].amax(dim=0).mean()), "peak_tilt_deg_2s": float(TILT[:w2, m].amax(dim=0).mean()),
             "fell": float(fell[m].float().mean()), "fell_before_2s": float((fell_at[m] < 2.0).float().mean())}
        res["groups"][name] = r
        print(f"{name:46s} {r['ankle_action_t20'][0]:+9.2f} /{r['ankle_action_t20'][1]:+6.2f}   {r['peak_ankle_action_2s']:14.2f}       {r['peak_joint_speed_2s']:10.1f} rad/s {r['peak_tilt_deg_2s']:8.1f} deg {r['fell']:6.2f}")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
