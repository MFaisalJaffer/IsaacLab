"""Deep-dive on STANDING quality: per-foot lift events, L/R asymmetry, settle time.

One CONTINUOUS no-reset rollout (manual env.reset() corrupts the eval -- envs
fall even on flat; boot naturally like stand_hold.py does):
  A) stand from the boot init state, cmd=0, stand_steps.
  B) walk vx for walk_steps, then cmd=0 -> the walk->stand transition (the real
     deployment case): settle time + quietness after settling.

MUST run with KBOT_FLAT=1 (training terrain) -- without it the policy is dropped
on rough terrain it never trained on and every number is garbage.

Key output: per-foot lift counts while standing. The gait_phase obs pins the
stand phase to (phi_l=pi/2, phi_r=pi) -- pi/2 is the "mid-swing ~6cm" phase, so
the hypothesis is the LEFT foot does most of the micro-stepping. This measures it.
"""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--walk_steps", type=int, default=500)   # 10s walk in scenario B
parser.add_argument("--stand_steps", type=int, default=1500)  # 30s stand window
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--label", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import glob
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def analyze_stand(con, jvel, bh, dt, foot_names, lines, tag, fz=None, gyro=None, tilt=None):
    """con: (T, N, 2) contact bools; jvel: (T, N, J) joint vel; bh: (T, N);
    fz: (T, N, 2) foot heights; gyro: (T, N) |base ang vel|; tilt: (T, N) deg —
    gyro/tilt reported in the SAME units as the HIL rig's stillness tracking."""
    T, N, _ = con.shape
    upright = bh > 0.6
    # only count envs that stayed upright the whole window (fallers pollute lift stats)
    ok = upright.all(axis=0)
    lines.append(f"[{tag}] upright whole window: {ok.sum()}/{N} envs")
    if ok.sum() == 0:
        return
    c = con[:, ok, :]  # (T, M, 2)
    both = (c.sum(2) == 2).mean() * 100
    # lift EVENTS: contact 1 -> 0 transitions per foot
    lifts = ((c[:-1] == 1) & (c[1:] == 0)).sum(axis=(0, 1))  # per foot
    dur = T * dt
    m = ok.sum()
    lines.append(f"[{tag}] both-feet-down: {both:.0f}%   window {dur:.0f}s x {m} envs")
    lines.append(
        f"[{tag}] lift events/env/min:  {foot_names[0]}: {lifts[0]/m/(dur/60):.1f}   "
        f"{foot_names[1]}: {lifts[1]/m/(dur/60):.1f}   (L/R ratio {lifts[0]/max(lifts[1],1):.2f})"
    )
    # airborne fraction per foot
    air = (1 - c).mean(axis=(0, 1)) * 100
    lines.append(f"[{tag}] airborne %:  {foot_names[0]}: {air[0]:.1f}%   {foot_names[1]}: {air[1]:.1f}%")
    jv = np.abs(jvel[:, ok, :])
    lines.append(f"[{tag}] mean |joint_vel| rad/s: {jv.mean():.2f}  max joint: {jv.mean(axis=(0,1)).max():.2f}")
    if gyro is not None:
        g = gyro[:, ok]
        lines.append(f"[{tag}] STILLNESS |body gyro| rad/s: mean {g.mean():.2f}  p95 {np.percentile(g, 95):.2f}  max {g.max():.2f}")
    if tilt is not None:
        tl = tilt[:, ok]
        lines.append(f"[{tag}] STILLNESS tilt deg: mean {tl.mean():.1f}  p95 {np.percentile(tl, 95):.1f}  max {tl.max():.1f}")
    if fz is not None:
        # how HIGH do feet actually rise during no-contact samples? baseline =
        # per-env-per-foot 5th-percentile height (firmly planted level)
        z = fz[:, ok, :]
        base = np.percentile(z, 5, axis=0, keepdims=True)
        lift_h = (z - base)[(1 - c).astype(bool)]
        if lift_h.size:
            lines.append(
                f"[{tag}] lift HEIGHT during no-contact: median {np.median(lift_h)*1000:.1f}mm  "
                f"p95 {np.percentile(lift_h, 95)*1000:.1f}mm  max {lift_h.max()*1000:.1f}mm"
            )


def main():
    task = args_cli.task
    ckpt = args_cli.checkpoint
    if ckpt is None:
        cks = glob.glob("logs/rsl_rl/kbot_legs_rough/*/model_*.pt")
        ckpt = max(cks, key=lambda p: int(p.split("model_")[1].split(".")[0]))
    print(f"[stand-analysis] checkpoint: {ckpt}")

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    # no pushes/curriculum; NEVER reset
    if getattr(env_cfg.curriculum, "velocity_push_curriculum", None) is not None:
        env_cfg.curriculum.velocity_push_curriculum = None
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 10000.0
    for term in ("time_out", "base_contact", "base_height", "bad_orientation"):
        if getattr(env_cfg.terminations, term, None) is not None:
            setattr(env_cfg.terminations, term, None)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task.split(":")[-1], args_cli)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    dev = env.unwrapped.device
    cs = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = [i for i, n in enumerate(cs.body_names) if n.endswith("FOOT")]
    foot_names = [cs.body_names[i] for i in foot_ids]
    print(f"[stand-analysis] feet: {foot_names}")

    def force_cmd(vx, standing):
        cmd_term.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=dev)
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = bool(standing)

    robot_foot_ids = [i for i, n in enumerate(robot.data.body_names) if n.endswith("FOOT")]

    def rollout(steps, vx, standing):
        con, jv, bh, fz, gy, tl = [], [], [], [], [], []
        with torch.inference_mode():
            obs, _ = env.get_observations()
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                force_cmd(vx, standing)
                f = cs.data.net_forces_w_history[:, :, foot_ids, :].norm(dim=-1).max(dim=1)[0]
                con.append((f > 1.0).float().cpu().numpy())
                jv.append(robot.data.joint_vel.cpu().numpy())
                bh.append(robot.data.root_pos_w[:, 2].cpu().numpy())
                fz.append(robot.data.body_pos_w[:, robot_foot_ids, 2].cpu().numpy())
                gy.append(robot.data.root_ang_vel_b.norm(dim=1).cpu().numpy())
                pg = robot.data.projected_gravity_b
                tl.append(torch.rad2deg(torch.atan2(pg[:, :2].norm(dim=1), -pg[:, 2])).cpu().numpy())
        return np.array(con), np.array(jv), np.array(bh), np.array(fz), np.array(gy), np.array(tl)

    dt = float(env.unwrapped.step_dt)
    lines = [f"=== stand-analysis {args_cli.label} ckpt={ckpt} envs={args_cli.num_envs} ==="]

    # -------- scenario A: stand from the natural boot init, cmd = 0 --------
    with torch.inference_mode():
        force_cmd(0.0, True)
    con, jv, bh, fz, gy, tl = rollout(args_cli.stand_steps, 0.0, True)
    # skip the first 3s settle (boot init has random pose/velocity)
    s = int(3.0 / dt)
    analyze_stand(con[s:], jv[s:], bh[s:], dt, foot_names, lines, "A stand-from-init", fz[s:], gy[s:], tl[s:])

    # -------- scenario B: walk vx then stop (continuous, no reset) --------
    with torch.inference_mode():
        force_cmd(args_cli.vx, False)
    _c, _j, _b, _f, _g, _t = rollout(args_cli.walk_steps, args_cli.vx, False)
    walked_ok = (_b[-1] > 0.6)
    lines.append(f"[B] still upright after {args_cli.walk_steps*dt:.0f}s walk @vx={args_cli.vx}: {walked_ok.sum()}/{args_cli.num_envs}")
    con, jv, bh, fz, gy, tl = rollout(args_cli.stand_steps, 0.0, True)
    # settle time: last lift event per env in the stand window
    T, N, _ = con.shape
    lifts_t = ((con[:-1] == 1) & (con[1:] == 0))  # (T-1, N, 2)
    last_lift = np.zeros(N)
    for e in range(N):
        w = np.where(lifts_t[:, e, :].any(axis=1))[0]
        last_lift[e] = (w[-1] + 1) * dt if len(w) else 0.0
    upright_end = bh[-1] > 0.6
    lines.append(f"[B walk->stop] upright at end: {upright_end.sum()}/{N}")
    lines.append(f"[B walk->stop] last foot-lift after stop cmd (s): median {np.median(last_lift):.1f}  p90 {np.percentile(last_lift, 90):.1f}")
    s = int(5.0 / dt)  # analyze quietness after 5s settle
    analyze_stand(con[s:], jv[s:], bh[s:], dt, foot_names, lines, "B post-settle", fz[s:], gy[s:], tl[s:])

    out = "\n".join(lines)
    print(out)
    with open("/tmp/stand_analysis_result.txt", "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
