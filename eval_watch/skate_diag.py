# Skating diagnostic (2026-07-15): stand_foot_slide has sat flat at ~-0.19/s for
# 27k iters (176k->203k) despite the -3.0 penalty. Hypothesis (from the reward-term
# comments): the policy braces feet OUTWARD at stand (constant action = free under
# stand_action_rate), and on low-friction DR draws the bracing force overcomes grip
# -> feet skate. This script measures, per env at a zero command stand:
#   - slide speed of in-contact feet (the exact reward quantity)
#   - per-env robot material friction (read back from PhysX)
#   - tangential/normal GRF ratio (bracing indicator: sustained ratio ~= mu means
#     the foot rides the friction cone edge -> slides on low-mu envs)
#   - stance width drift (are feet displacing outward over the stand?)
# Correlating slide vs friction decides the lever: slide concentrated on low-mu
# envs => bracing-vs-ice => penalize lateral GRF; slide everywhere => posture term.
# Pushes are DISABLED (steady-state bracing measurement, not push recovery).
#
# Run (alongside training; ~6 GB VRAM headless):
#   env KBOT_FLAT=1 kbot_env/bin/python eval_watch/skate_diag.py \
#       --checkpoint <model_*.pt> --num_envs 256 --steps 600 --headless

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.append("scripts/reinforcement_learning/rsl_rl")  # for cli_args, like play.py
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stand-skating diagnostic.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
# NB --checkpoint comes from cli_args.add_rsl_rl_args (defining it here collides)
parser.add_argument("--steps", type=int, default=600, help="stand steps @50Hz (600 = 12 s)")
parser.add_argument("--settle", type=int, default=100, help="steps to skip before measuring")
parser.add_argument("--out", type=str, default="eval_watch/skate_diag.json")
parser.add_argument(
    "--mu_override", type=float, default=None,
    help="pin static friction of ALL envs to this value (dynamic = 0.8x) — interventional runs",
)
parser.add_argument(
    "--usd_override", type=str, default=None,
    help="alternate robot USD (e.g. robot_torsional.usd) — contact-physics experiments",
)
parser.add_argument(
    "--corr_dist", type=float, default=None,
    help="override physx friction_correlation_distance (default 0.025)",
)
parser.add_argument(
    "--pin_gains", action="store_true",
    help="disable gain-DR events (eval at nominal kp/kd — comparable to pre-DR records)",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
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

    # Zero-command stand for every env, never resampled during the measurement.
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    # Steady-state bracing measurement: no pushes (transient slides would pollute it).
    # Must drop the push curriculum too — it writes push_robot.params on every reset.
    env_cfg.events.push_robot = None
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    # Keep episodes longer than the measurement window.
    env_cfg.episode_length_s = 60.0
    # Drift metrics need an unbroken window — a mid-window reset teleports the
    # root and fakes meters of drift. Same terminations-off approach as
    # stand_analysis.py (natural boot, one continuous rollout, no manual reset).
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    # Contact-physics experiment knobs (2026-07-15 drift investigation)
    if args_cli.usd_override is not None:
        env_cfg.scene.robot.spawn.usd_path = args_cli.usd_override
    if args_cli.corr_dist is not None:
        env_cfg.sim.physx.friction_correlation_distance = args_cli.corr_dist
    if args_cli.pin_gains:
        for ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
            if getattr(env_cfg.events, ev, None) is not None:
                setattr(env_cfg.events, ev, None)
    # Interventional friction pin: same policy, same everything, only mu changes.
    if args_cli.mu_override is not None:
        m = args_cli.mu_override
        env_cfg.events.physics_material.params["static_friction_range"] = (m, m)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.8 * m, 0.8 * m)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[diag] loading {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    sensor = scene.sensors["contact_forces"]
    foot_ids_robot, foot_names = robot.find_bodies(".*FOOT")
    # contact sensor body indexing can differ from the articulation's — resolve by name
    foot_ids_sensor = [sensor.body_names.index(n) for n in foot_names]
    print(f"[diag] feet: {foot_names} robot_ids={foot_ids_robot} sensor_ids={foot_ids_sensor}")

    # Per-env, per-SHAPE robot material friction as randomized at startup. Mean over
    # shapes washes out the per-shape draws (all-shape mean ~ 0.98 +/- 0.03 for a
    # 0.4-1.4 range) — keep the full matrix and identify the contact-relevant (foot)
    # shape columns data-driven, by which columns' mu correlates with slide.
    mats = robot.root_physx_view.get_material_properties()  # (envs, shapes, 3)
    static_mu_shapes = mats[..., 0].cpu()  # (envs, shapes)
    static_mu = static_mu_shapes.mean(dim=1)
    dynamic_mu = mats[..., 1].mean(dim=1).cpu()

    n = env.unwrapped.num_envs
    dev = env.unwrapped.device
    from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

    slide_sum = torch.zeros(n, device=dev)      # summed in-contact foot xy speed (reward quantity)
    # Bracing indicator: slide velocity component pointing AWAY from the body
    # midline (per foot, in the root-yaw frame). Consistently positive = both feet
    # pushing outward = snowplow bracing. (Contact-sensor forces are normal-only,
    # so a force-based tangential test is blind — direction of motion is not.)
    outward_sum = torch.zeros(n, device=dev)
    outward_cnt = torch.zeros(n, device=dev)
    gyro_sum = torch.zeros(n, device=dev)
    width_first = torch.zeros(n, device=dev)
    width_last = torch.zeros(n, device=dev)
    got_first = False
    # WORLD-FRAME drift (rig instrument of record, 2026-07-15): the rig's dominant
    # skating mode is COHERENT WHOLE-ROBOT drift — both feet translate the same
    # way, so stance width and outward-coherence are blind to it, and its speed
    # (rig: 0.2-0.5 cm/s) is far below the instantaneous-slide noise floor.
    # Cumulative world-frame displacement over the window is the right metric.
    root_first = torch.zeros(n, 2, device=dev)
    root_last = torch.zeros(n, 2, device=dev)
    feet_first = torch.zeros(n, 2, 2, device=dev)
    feet_last = torch.zeros(n, 2, 2, device=dev)
    # OSCILLATORY-GLIDE metrics (2026-07-16, user: feet still glide in/out on
    # the rig; nets to ~0 so every net metric — sim AND rig gate — is blind).
    # path = cumulative in-contact xy travel (glide as MOTION, not displacement)
    # width samples -> oscillation amplitude; foot roll/pitch rate while in
    # contact separates ROCKING on the convex sole (angular ~ linear) from
    # friction-limited SLIPPING (linear, no angular).
    path_sum = torch.zeros(n, device=dev)
    width_sq_sum = torch.zeros(n, device=dev)
    width_sum = torch.zeros(n, device=dev)
    foot_rot_sum = torch.zeros(n, device=dev)
    foot_rot_cnt = torch.zeros(n, device=dev)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step < args_cli.settle:
                continue
            forces = sensor.data.net_forces_w[:, foot_ids_sensor, :]  # (envs, 2, 3)
            contact = forces.norm(dim=-1) > 1.0
            foot_vel_xy = robot.data.body_lin_vel_w[:, foot_ids_robot, :2]
            slide = (foot_vel_xy.norm(dim=-1) * contact.float()).sum(dim=1)
            slide_sum += slide
            # outward slide component in the root-yaw frame
            yq = yaw_quat(robot.data.root_quat_w)  # (envs, 4)
            root_pos = robot.data.root_pos_w
            for k in range(len(foot_ids_robot)):
                rel = robot.data.body_pos_w[:, foot_ids_robot[k], :] - root_pos
                rel_b = quat_rotate_inverse(yq, rel)
                vel3 = robot.data.body_lin_vel_w[:, foot_ids_robot[k], :]
                vel_b = quat_rotate_inverse(yq, vel3)
                side = torch.sign(rel_b[:, 1])  # which side of the midline this foot is on
                outward = side * vel_b[:, 1] * contact[:, k].float()
                outward_sum += outward
                outward_cnt += contact[:, k].float()
            gyro_sum += robot.data.root_ang_vel_b.norm(dim=-1)
            feet_pos = robot.data.body_pos_w[:, foot_ids_robot, :2]
            width = (feet_pos[:, 0, :] - feet_pos[:, 1, :]).norm(dim=-1)
            # cumulative glide path + width oscillation + rocking signature
            path_sum += slide * env.unwrapped.step_dt
            width_sum += width
            width_sq_sum += width * width
            foot_ang = robot.data.body_ang_vel_w[:, foot_ids_robot, :2].norm(dim=-1)  # roll/pitch rate
            foot_rot_sum += (foot_ang * contact.float()).sum(dim=1)
            foot_rot_cnt += contact.float().sum(dim=1)
            if not got_first:
                width_first = width.clone()
                root_first = robot.data.root_pos_w[:, :2].clone()
                feet_first = feet_pos.clone()
                got_first = True
            width_last = width
            root_last = robot.data.root_pos_w[:, :2].clone()
            feet_last = feet_pos.clone()

    steps_meas = args_cli.steps - args_cli.settle
    slide_mean = (slide_sum / steps_meas).cpu()            # m/s, summed over both feet
    outward_mean = (outward_sum / outward_cnt.clamp(min=1)).cpu()  # m/s per foot-contact, +=outward
    gyro_mean = (gyro_sum / steps_meas).cpu()
    width_drift = (width_last - width_first).cpu()
    win_s = steps_meas * env.unwrapped.step_dt
    scale30 = 30.0 / win_s  # normalize drift to the rig's per-30s convention
    base_drift = ((root_last - root_first).norm(dim=-1) * scale30).cpu()
    foot_net = ((feet_last - feet_first).norm(dim=-1) * scale30).cpu()  # (envs, 2)
    glide_path = (path_sum * scale30).cpu()  # cumulative in-contact travel, m/30s (both feet)
    wm = width_sum / steps_meas
    width_std = (width_sq_sum / steps_meas - wm * wm).clamp(min=0).sqrt().cpu()
    foot_rot = (foot_rot_sum / foot_rot_cnt.clamp(min=1)).cpu()  # rad/s roll+pitch while planted

    out = {
        "checkpoint": resume_path,
        "num_envs": n,
        "steps_measured": steps_meas,
        "mu_override": args_cli.mu_override,
        "static_mu": static_mu.tolist(),
        "static_mu_shapes": static_mu_shapes.tolist(),
        "dynamic_mu": dynamic_mu.tolist(),
        "slide_mean_mps": slide_mean.tolist(),
        "outward_vel_mps": outward_mean.tolist(),
        "gyro_mean": gyro_mean.tolist(),
        "stance_width_drift_m": width_drift.tolist(),
        "window_s": win_s,
        "base_drift_m_per30s": base_drift.tolist(),
        "foot_net_m_per30s": foot_net.tolist(),
        "glide_path_m_per30s": glide_path.tolist(),
        "width_std_m": width_std.tolist(),
        "foot_rot_radps": foot_rot.tolist(),
    }
    with open(args_cli.out, "w") as f:
        json.dump(out, f)

    # ---- summary (also mirrored to a .txt because Isaac's logger eats stdout) ----
    lines = []
    lines.append(f"==== SKATE DIAGNOSTIC @ {resume_path.split('/')[-1]} ====")
    lines.append(f"envs={n} window={steps_meas} steps ({steps_meas/50:.0f}s stand, pushes off)"
                 + (f" MU PINNED = {args_cli.mu_override}" if args_cli.mu_override is not None else " (training DR mu)"))
    lines.append(f"slide (both feet, m/s): mean={slide_mean.mean():.4f} p50={slide_mean.median():.4f}"
                 f" p90={slide_mean.quantile(0.9):.4f} max={slide_mean.max():.4f}")
    lines.append(f"outward slide vel: mean={outward_mean.mean():+.4f} m/s"
                 f" (+ = away from midline = bracing; envs outward: {(outward_mean > 0).float().mean()*100:.0f}%)")
    lines.append(f"width drift: mean={width_drift.mean()*100:+.2f} cm p90={width_drift.quantile(0.9)*100:+.2f} cm")
    lines.append(f"WORLD-FRAME (rig instrument, normalized /30s, window {win_s:.0f}s):")
    lines.append(f"  base_drift cm/30s: mean={base_drift.mean()*100:.1f} p50={base_drift.median()*100:.1f}"
                 f" p90={base_drift.quantile(0.9)*100:.1f} max={base_drift.max()*100:.1f}")
    lines.append(f"  foot net cm/30s: L={foot_net[:,0].mean()*100:.1f} R={foot_net[:,1].mean()*100:.1f}"
                 f" (rig 204800: base 17.2, feet 15/16; rig 171500: base 6.8, feet 6/5)")
    lines.append(f"OSCILLATORY GLIDE (nets to ~0, invisible to the above):")
    lines.append(f"  cumulative path cm/30s (both feet): mean={glide_path.mean()*100:.0f}"
                 f" p50={glide_path.median()*100:.0f} p90={glide_path.quantile(0.9)*100:.0f}")
    lines.append(f"  stance width oscillation std: mean={width_std.mean()*100:.2f} cm"
                 f" p90={width_std.quantile(0.9)*100:.2f} cm")
    lines.append(f"  planted-foot roll/pitch rate: mean={foot_rot.mean():.3f} rad/s"
                 f" (high + tracks glide = ROCKING on hull; low = true slip)")
    lines.append(f"gyro mean={gyro_mean.mean():.3f} rad/s (statue check)")
    if args_cli.mu_override is None:
        # data-driven: which shape's mu explains slide? (foot shapes should pop)
        sm = static_mu_shapes  # (envs, shapes)
        varying = sm.std(dim=0) > 1e-4
        corrs = []
        for j in range(sm.shape[1]):
            if not varying[j]:
                continue
            c = torch.corrcoef(torch.stack([sm[:, j], slide_mean]))[0, 1].item()
            corrs.append((c, j))
        corrs.sort()
        lines.append(f"per-shape corr(mu_j, slide): strongest negative {corrs[:3]}")
        lines.append(f"                             strongest positive {corrs[-3:]}")
        best = corrs[0][1] if corrs else 0
        mu_f = sm[:, best]
        q = torch.quantile(mu_f, torch.tensor([0.25, 0.5, 0.75]))
        bins = [(mu_f <= q[0]), (mu_f > q[0]) & (mu_f <= q[1]), (mu_f > q[1]) & (mu_f <= q[2]), (mu_f > q[2])]
        for name, m in zip(["muQ1(low)", "muQ2", "muQ3", "muQ4(high)"], bins):
            lines.append(f"  shape{best} {name:10s} mu={mu_f[m].mean():.2f} slide={slide_mean[m].mean():.4f}"
                         f" outward={outward_mean[m].mean():+.4f} drift={width_drift[m].mean()*100:+.1f}cm")
    txt = "\n".join(lines)
    print("\n" + "\n".join(f"[diag] {l}" for l in lines))
    with open(args_cli.out.replace(".json", ".txt"), "w") as f:
        f.write(txt + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
