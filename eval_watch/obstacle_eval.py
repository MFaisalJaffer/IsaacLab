"""Obstacle course test (stage 1): crossing success per kind of obstacle and per height.

Robots are spread evenly over every (kind, height) tile and told to walk straight ahead at --vx; curriculum off.
A robot "crossed" when it got further than the obstacle's outer edge + 0.5 m from its tile centre without
falling first. Flat tiles are the control (same distance). The training-time "fell behind the command"
termination is OFF here: on flat ground it already ends about a quarter of the walker's straight walks
(heading drift), which would hide what the obstacle does. A robot that neither crossed nor fell by the end is
"stuck" (stopped or wandering in front of the obstacle).

Per (kind, height): share that crossed / fell before crossing / stuck, time to cross, steps with a toe or leg
hit before crossing, speed, heading change.

  --blind   zero the height map before it reaches the policy (does the policy use the map?)
  --yaw0    every robot starts facing +x: head-on approach (default: random heading -> approach up to 45 deg off)
  --render_seconds S --render_kind platform --render_level 4   render one robot (side + rear view) -> <tag>.gif

  source eval_watch/obstacle_env.sh; ./isaaclab.sh -p eval_watch/obstacle_eval.py --checkpoint <ckpt> --tag obst_eval --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Obstacle-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=1200)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--vx", type=float, default=0.35)
parser.add_argument("--blind", action="store_true")
parser.add_argument("--yaw0", action="store_true")
parser.add_argument("--no_push", action="store_true")
parser.add_argument("--cross_margin", type=float, default=0.5)
parser.add_argument("--tag", default="obst_eval")
parser.add_argument("--render_seconds", type=float, default=0.0)
parser.add_argument("--render_kind", default="platform")
parser.add_argument("--render_level", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.render_seconds > 0
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_obstacle  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
FALL_TERMS = ("base_contact", "bad_orientation", "base_height")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    n = args.num_envs
    env_cfg.episode_length_s = args.seconds + 2.0
    env_cfg.curriculum.obstacle_levels = None
    env_cfg.terminations.root_drift = None      # see the module docstring
    c = env_cfg.commands.base_velocity
    c.obstacle_vx = (args.vx, args.vx)
    c.ranges.lin_vel_x = (args.vx, args.vx); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)   # flat tiles: the control
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1000.0, 1000.0)
    for name in ("walk_at_spawn", "stand_corridor", "amp_axis_bias"):   # these rewrite commands
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    if args.yaw0:
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    if args.no_push:
        for name in ("push_robot", "sustained_push"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        for name in ("velocity_push_curriculum", "sustained_push_level"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)
    gen = env_cfg.scene.terrain.terrain_generator
    rows = gen.num_rows
    col_kind = mdp_obstacle.column_kinds(gen)
    env_col = np.floor(np.arange(n) / (n / gen.num_cols)).astype(int)
    env_level = np.arange(n) % rows
    cam_env = 0
    if args.render_seconds > 0:
        match = [i for i in range(n) if col_kind[env_col[i]] == args.render_kind and env_level[i] == args.render_level]
        assert match, f"no env on a {args.render_kind} tile at level {args.render_level}"
        cam_env = match[0]
        env_cfg.viewer.origin_type = "asset_root"; env_cfg.viewer.asset_name = "robot"; env_cfg.viewer.env_index = cam_env
        env_cfg.viewer.resolution = (960, 540); env_cfg.viewer.eye = (0.15, -2.15, 0.35); env_cfg.viewer.lookat = (0.0, 0.0, -0.3)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.render_seconds > 0 else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    dev = uenv.device
    robot = uenv.scene["robot"]
    terrain = uenv.scene.terrain
    course = mdp_obstacle.course(uenv)
    assert np.array_equal(terrain.terrain_types.cpu().numpy(), env_col), "column assignment differs from the generator's rule"
    level = torch.tensor(env_level, device=dev)
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
    kind = mdp_obstacle.env_kind(uenv)
    plat = [i for i, s in enumerate(course["names"]) if s == "platform"]
    outer = torch.where(course["kind_is_obstacle"][kind], course["kind_outer"][kind], course["kind_outer"][plat[0]] if plat else torch.full_like(course["kind_outer"][kind], 2.3))
    goal = outer + args.cross_margin
    n_map = int(uenv.observation_manager.group_obs_term_dim["policy"][-1][0])
    rw = uenv.reward_manager
    T = int(args.seconds / uenv.step_dt)

    def term_call(name):
        cfg_ = rw.get_term_cfg(name)
        return cfg_.func(uenv, **cfg_.params)

    views = {"side": ((0.15, -2.15, 0.35), (0.0, 0.0, -0.3)), "rear34": ((-1.55, 1.25, 0.6), (0.0, 0.0, -0.35))}
    frames = {k: [] for k in views}
    n_render = int(args.render_seconds / uenv.step_dt)

    def capture():
        uenv.sim.render(); uenv.sim.render()
        return uenv.render(recompute=True)

    with torch.inference_mode():  # warm-up + re-reset (first-episode gap), now on the assigned tiles
        env.step(torch.zeros(n, env.num_actions, device=dev))
        obs, _ = env.reset()
        cmd0 = uenv.command_manager.get_command("base_velocity").clone()
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        crossed_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        end_kind = torch.zeros(n, dtype=torch.long, device=dev)      # 0 none, 1 fall
        yaw_start = torch.atan2(2.0 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3] + robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
                                1.0 - 2.0 * (robot.data.root_quat_w[:, 2] ** 2 + robot.data.root_quat_w[:, 3] ** 2))
        turn = torch.zeros(n, device=dev)
        end_dist = torch.zeros(n, device=dev)
        hits = torch.zeros(n, device=dev); lift = torch.zeros(n, device=dev); near = torch.zeros(n, device=dev); steps = torch.zeros(n, device=dev)
        gate = torch.zeros(n, device=dev); vfwd = torch.zeros(n, device=dev); base_z = torch.zeros(n, device=dev); map_abs = torch.zeros(n, device=dev)
        dist = (robot.data.root_pos_w[:, :2] - uenv.scene.env_origins[:, :2]).abs().amax(dim=1)
        if n_render > 0:
            for _ in range(8):
                capture()
        for t in range(T):
            x = obs
            if args.blind:
                x = obs.clone(); x[:, -n_map:] = 0.0
            obs, _, dones, infos = env.step(policy(x))
            tm_ = uenv.termination_manager
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            for name in FALL_TERMS:
                if name in tm_.active_terms:
                    fell |= tm_.get_term(name)
            ended = alive & dones.bool()
            end_kind[ended] = 1
            end_dist[ended] = dist[ended]            # position one tick before the reset
            alive &= ~dones.bool()
            dist = (robot.data.root_pos_w[:, :2] - uenv.scene.env_origins[:, :2]).abs().amax(dim=1)
            newly = alive & (crossed_at < 0) & (dist > goal)
            crossed_at[newly] = t
            m = alive & (crossed_at < 0)             # still on the way to / over the obstacle
            hits += m.float() * ((term_call("feet_stumble") + term_call("leg_contact")) > 0).float()
            lift += m.float() * term_call("ref_foot_lift")
            near += m.float() * mdp_obstacle.near_obstacle(uenv).float()
            gate += m.float() * infos["observations"]["amp_gate"][:, 0]
            vfwd += m.float() * quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, 0]
            base_z += m.float() * (robot.data.root_pos_w[:, 2] - mdp_obstacle.foot_ground(uenv)["under"].mean(dim=1))
            map_abs += m.float() * obs[:, -n_map:].abs().amax(dim=1)
            steps += m.float()
            q_ = robot.data.root_quat_w
            yaw_now = torch.atan2(2.0 * (q_[:, 0] * q_[:, 3] + q_[:, 1] * q_[:, 2]), 1.0 - 2.0 * (q_[:, 2] ** 2 + q_[:, 3] ** 2))
            turn = torch.where(m, torch.atan2(torch.sin(yaw_now - yaw_start), torch.cos(yaw_now - yaw_start)).abs(), turn)
            if t < n_render and t % 2 == 0:
                for name, (eye, lookat) in views.items():
                    uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                    frames[name].append(capture())
    ok = (crossed_at >= 0).cpu().numpy()
    K = kind.cpu().numpy(); L = env_level
    R = {k_: v.cpu().numpy() for k_, v in dict(end_kind=end_kind, end_dist=end_dist, hits=hits, lift=lift, near=near, steps=steps, gate=gate, vfwd=vfwd,
                                               base_z=base_z, crossed_at=crossed_at, map_abs=map_abs, turn=turn).items()}
    st = np.maximum(R["steps"], 1.0)
    heights = [round(float(h) * 100, 1) for h in course["heights"]]
    res = {"checkpoint": args.checkpoint, "num_envs": n, "seconds": args.seconds, "vx": args.vx, "blind": bool(args.blind or mdp_obstacle.BLIND),
           "yaw0": bool(args.yaw0), "heights_cm": heights, "kinds": {}, "cmd_check": [round(float(v), 3) for v in cmd0.abs().amax(dim=0)]}
    print(f"[eval] {n} robots, {args.seconds:.0f} s, forward {args.vx} m/s, map {'ZEROED' if res['blind'] else 'on'}, heading {'head-on' if args.yaw0 else 'random'}; "
          f"commands |max| {res['cmd_check']}")
    print(f"{'kind':>9s} {'cm':>5s} {'n':>4s} {'crossed':>8s} {'fell':>6s} {'stuck':>6s} {'t_cross':>8s} {'hit steps':>10s} {'speed':>6s} {'turn deg':>9s} {'lift rew':>9s} {'near':>5s} {'style gate':>11s}")
    for i, name in enumerate(course["names"]):
        per = []
        levels = [None] if not bool(course["kind_is_obstacle"][i]) else list(range(rows))
        for lv in levels:
            m = (K == i) if lv is None else ((K == i) & (L == lv))
            if not m.any():
                continue
            before = m & ~ok
            r = {"height_cm": 0.0 if lv is None else heights[lv], "n": int(m.sum()), "crossed": float(ok[m].mean()),
                 "fell_before": float((before & (R["end_kind"] == 1)).sum() / m.sum()), "stuck": float((before & (R["end_kind"] == 0)).sum() / m.sum()),
                 "turn_deg": float(np.degrees(R["turn"][m]).mean()),
                 "t_cross_s": float((R["crossed_at"][m & ok] * uenv.step_dt).mean()) if (m & ok).any() else None,
                 "hit_steps": float(R["hits"][m].mean()), "speed": float((R["vfwd"][m] / st[m]).mean()), "lift_reward": float((R["lift"][m] / st[m]).mean()),
                 "near_share": float((R["near"][m] / st[m]).mean()), "style_gate": float((R["gate"][m] / st[m]).mean()),
                 "base_above_feet_ground_m": float((R["base_z"][m] / st[m]).mean()), "map_abs_max": float((R["map_abs"][m] / st[m]).mean())}
            per.append(r)
            tc = "   n/a" if r["t_cross_s"] is None else f"{r['t_cross_s']:6.1f}"
            print(f"{name:>9s} {r['height_cm']:5.1f} {r['n']:4d} {r['crossed']:8.2f} {r['fell_before']:6.2f} {r['stuck']:6.2f} {tc:>8s} {r['hit_steps']:10.1f} "
                  f"{r['speed']:6.2f} {r['turn_deg']:9.0f} {r['lift_reward']:9.2f} {r['near_share']:5.2f} {r['style_gate']:11.2f}")
        res["kinds"][name] = per
    fl = res["kinds"].get("flat", [{}])[0]
    print(f"[eval] flat tiles: base {fl.get('base_above_feet_ground_m', float('nan')):.3f} m above the ground under the feet; largest |map value| {fl.get('map_abs_max', float('nan')):.2f} "
          f"(x 1/5 = {fl.get('map_abs_max', float('nan')) / 5 * 100:.1f} cm)")
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    if n_render > 0:
        import subprocess

        import imageio.v2 as imageio
        FF = os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.11", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-linux-x86_64-v7.0.2")
        for name in views:
            imageio.mimwrite(os.path.join(OUT, f"{args.tag}_{name}.mp4"), frames[name], fps=25, codec="libx264", quality=7, macro_block_size=None)
        gif = os.path.join(OUT, f"{args.tag}.gif")
        subprocess.run([FF, "-y", "-i", os.path.join(OUT, f"{args.tag}_side.mp4"), "-i", os.path.join(OUT, f"{args.tag}_rear34.mp4"), "-filter_complex",
                        "[0:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[a];[1:v]crop=iw*0.5:ih:iw*0.25:0,scale=320:-1,hqdn3d=10:8:16:12[b];"
                        "[a][b]hstack,fps=10,split[s0][s1];[s0]palettegen=max_colors=32:stats_mode=diff[p];[s1][p]paletteuse=dither=none", gif],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        idx = [int(k * (len(frames["side"]) - 1) / 7) for k in range(8)]
        rows_ = []
        for name in views:
            tiles = []
            for i in idx:
                img = frames[name][i]; hh = 360; ww = int(img.shape[1] * hh / img.shape[0])
                ys = (np.arange(hh) * img.shape[0] / hh).astype(int); xs = (np.arange(ww) * img.shape[1] / ww).astype(int)
                tiles.append(img[ys][:, xs])
            rows_.append(np.concatenate(tiles, axis=1))
        imageio.imwrite(os.path.join(OUT, f"{args.tag}_filmstrip.png"), np.concatenate(rows_, axis=0))
        print(f"[render] {gif} ({os.path.getsize(gif) / 1e6:.1f} MB): env {cam_env}, {args.render_kind} {heights[args.render_level]} cm, "
              f"crossed {bool(ok[cam_env])}, fell {bool(R['end_kind'][cam_env])}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
