"""Stand pose of a checkpoint in Isaac (rig ask, RIG stand #4 note 2026-10-04): mean joint angles and policy targets
over a flat stand at zero command, and how mirror-symmetric they are.

Hard-start stand (rest at the zero pose, cmd 0, hard pin), no pushes, plant as trained (randomization and sensor
noise on) unless --plant nominal. Per joint: the mean over 2 s .. end and over all robots of the link angle, the
motor-side (encoder) angle, and the policy's target = clip(0.5 x action, clip table); plus the spread across robots
of each robot's own time-mean (is a lean a property of the policy or of each robot's plant draw?).
Mirror symmetry: mirroring swaps left and right and negates every joint, so a mirror-symmetric pose has L + R = 0
for every pair. The script prints L + R per pair.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/amp_stand_pose.py \
      --checkpoint <ckpt> --tag stand_pose_v7_800 --headless
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
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--plant", default="trained", choices=("trained", "nominal"))
parser.add_argument("--tag", default="stand_pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("hip pitch", "dof_left_hip_pitch_04", "dof_right_hip_pitch_04"), ("hip roll", "dof_left_hip_roll_04", "dof_right_hip_roll_04"),
         ("hip yaw", "dof_left_hip_yaw_03", "dof_right_hip_yaw_03"), ("knee", "dof_left_knee_04", "dof_right_knee_04"),
         ("ankle", "dof_left_ankle_02", "dof_right_ankle_02")]


def main() -> int:
    n = args.num_envs
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 2.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0; c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    off = ["walk_at_spawn", "stand_corridor", "push_robot", "amp_unanswered", "amp_axis_bias"]
    if args.plant == "nominal":
        off += ["randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints", "add_limb_masses",
                "randomize_joint_properties", "randomize_imu_mount", "randomize_joint_play", "physics_material"]
        env_cfg.observations.policy.enable_corruption = False
    for name in off:
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
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
    act_term = uenv.action_manager.get_term("joint_pos")
    clip = act_term._clip[0] if getattr(act_term, "_clip", None) is not None else None      # (J, 2) on the PD target
    scale = act_term._scale if torch.is_tensor(act_term._scale) else torch.full((n, len(jn)), float(act_term._scale), device=dev)
    motor = {}
    for a in robot.actuators.values():
        for k, name in enumerate(a.joint_names):
            motor[name] = (a, k)
    T = int(args.seconds / uenv.step_dt)
    s0 = int(2.0 / uenv.step_dt)
    q = torch.zeros(n, len(jn), device=dev); qm = torch.zeros_like(q); tg = torch.zeros_like(q); raw = torch.zeros_like(q)
    g = torch.zeros(n, 2, device=dev); yawrate = torch.zeros(n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    k = 0
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
        for t in range(T):
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            fell |= dones.bool()
            if t >= s0:
                tgt = a * scale + robot.data.default_joint_pos
                if clip is not None:
                    tgt = torch.maximum(torch.minimum(tgt, clip[:, 1]), clip[:, 0])
                q += robot.data.joint_pos; tg += tgt; raw += a
                mp = torch.stack([motor[name][0].motor_pos[:, motor[name][1]] if motor[name][0].motor_pos is not None else robot.data.joint_pos[:, i] for i, name in enumerate(jn)], dim=1)
                qm += mp
                g += robot.data.projected_gravity_b[:, :2]; yawrate += robot.data.root_ang_vel_b[:, 2]
                k += 1
    up = ~fell
    Q, QM, TG, RAW = [np.degrees((x[up] / k).cpu().numpy()) for x in (q, qm, tg, raw / 1.0)]
    RAW = (raw[up] / k).cpu().numpy()
    G = (g[up] / k).cpu().numpy()
    # each robot's own proportional gain (randomized per robot): kp x (target - encoder) = the motor's standing torque
    KP = torch.stack([motor[name][0].stiffness[:, motor[name][1]] for name in jn], dim=1)[up].cpu().numpy()
    TQ = KP * np.radians(TG - QM)
    res = {"checkpoint": args.checkpoint, "plant": args.plant, "seconds": args.seconds, "robots": int(up.sum()), "fell": int(fell.sum()), "pairs": {}}
    print(f"\n[pose] {args.checkpoint} | plant {args.plant} | flat stand, cmd 0, hard start | {int(up.sum())} robots up of {n}, means over 2-{args.seconds:.0f} s; degrees")
    print(f"[pose] torso lean (mean over robots): pitch {math.degrees(math.asin(float(G[:, 0].mean()))):+.2f} deg, roll {math.degrees(math.asin(float(G[:, 1].mean()))):+.2f} deg; "
          f"spread across robots {math.degrees(float(G[:, 0].std())):.2f} / {math.degrees(float(G[:, 1].std())):.2f} deg")
    print(f"{'joint':10s} | {'link angle L / R':>18s} {'L+R':>6s} | {'encoder L / R':>16s} {'L+R':>6s} | {'policy target L / R':>20s} {'L+R':>6s} | {'raw action L / R':>17s} | spread across robots of the target (L / R)")
    for name, jl, jr in PAIRS:
        il, ir = jn.index(jl), jn.index(jr)
        r = {"link": [float(Q[:, il].mean()), float(Q[:, ir].mean())], "encoder": [float(QM[:, il].mean()), float(QM[:, ir].mean())],
             "target": [float(TG[:, il].mean()), float(TG[:, ir].mean())], "raw_action": [float(RAW[:, il].mean()), float(RAW[:, ir].mean())],
             "target_spread": [float(TG[:, il].std()), float(TG[:, ir].std())]}
        res["pairs"][name] = r
        print(f"{name:10s} | {r['link'][0]:+8.2f} /{r['link'][1]:+7.2f} {r['link'][0] + r['link'][1]:+6.2f} | {r['encoder'][0]:+7.2f} /{r['encoder'][1]:+7.2f} {r['encoder'][0] + r['encoder'][1]:+6.2f} | "
              f"{r['target'][0]:+9.2f} /{r['target'][1]:+7.2f} {r['target'][0] + r['target'][1]:+6.2f} | {r['raw_action'][0]:+7.2f} /{r['raw_action'][1]:+7.2f} | {r['target_spread'][0]:.2f} / {r['target_spread'][1]:.2f}")
        # the population one real robot is compared with: spread across robots (std and 5-95 %) of each robot's own mean
        pct = lambda x: [float(np.percentile(x, 5)), float(np.percentile(x, 95))]
        sm = TG[:, il] + TG[:, ir]
        r.update({"encoder_spread": [float(QM[:, il].std()), float(QM[:, ir].std())], "link_spread": [float(Q[:, il].std()), float(Q[:, ir].std())],
                  "target_p05_p95": [pct(TG[:, il]), pct(TG[:, ir])], "encoder_p05_p95": [pct(QM[:, il]), pct(QM[:, ir])],
                  "kp": [float(KP[:, il].mean()), float(KP[:, ir].mean())],
                  "motor_torque_nm": [float(TQ[:, il].mean()), float(TQ[:, ir].mean())], "motor_torque_spread_nm": [float(TQ[:, il].std()), float(TQ[:, ir].std())],
                  "motor_torque_p05_p95_nm": [pct(TQ[:, il]), pct(TQ[:, ir])],
                  "target_sum": float(sm.mean()), "target_sum_spread": float(sm.std()), "target_sum_p05_p95": pct(sm),
                  "robots_with_target_sum_within_1deg": int((np.abs(sm) < 1.0).sum()),
                  "robots_with_same_sign_as_mean": [int((np.sign(TG[:, il]) == np.sign(TG[:, il].mean())).sum()), int((np.sign(TG[:, ir]) == np.sign(TG[:, ir].mean())).sum())]})
    print(f"\n[pose] the population (each robot's own 2-{args.seconds:.0f} s mean; std and 5-95 % across the {int(up.sum())} robots)")
    print(f"{'joint':10s} | {'target 5-95 % L':>18s} {'R':>16s} | {'encoder std L / R':>18s} | {'kp x (target - encoder), Nm: mean L / R':>40s} {'std L / R':>12s} | {'L+R of target: mean, std, 5-95 %':>36s} | robots with |L+R| < 1 deg | same sign as the mean (L / R)")
    for name, _jl, _jr in PAIRS:
        r = res["pairs"][name]
        print(f"{name:10s} | {r['target_p05_p95'][0][0]:+7.2f} ..{r['target_p05_p95'][0][1]:+6.2f} {r['target_p05_p95'][1][0]:+7.2f} ..{r['target_p05_p95'][1][1]:+6.2f} | {r['encoder_spread'][0]:8.2f} /{r['encoder_spread'][1]:5.2f} | "
              f"{r['motor_torque_nm'][0]:+20.2f} /{r['motor_torque_nm'][1]:+6.2f} {r['motor_torque_spread_nm'][0]:7.2f} /{r['motor_torque_spread_nm'][1]:5.2f} | "
              f"{r['target_sum']:+10.2f} {r['target_sum_spread']:6.2f} {r['target_sum_p05_p95'][0]:+7.2f} ..{r['target_sum_p05_p95'][1]:+6.2f} | {r['robots_with_target_sum_within_1deg']:8d} | {r['robots_with_same_sign_as_mean'][0]:6d} /{r['robots_with_same_sign_as_mean'][1]:4d}")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
