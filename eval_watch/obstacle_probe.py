"""Static checks of the obstacle course (stage 1) in the real env — run before training on it.

  1. policy input layout (walker terms x 10 frames, then the height map) and the course (kind per column, height per row);
  2. height-map cell order vs the mirror function (symmetry.MAP_ROWS/COLS): mirrored cell = same x, opposite y;
  3. course geometry seen by the sensors: robots are placed at known offsets from their tile centre and the
     base scanner's centre ray and the foot scanners must read the expected ground height;
  4. foot frame: which end of the sole is the toe, where the sole is under the foot origin;
  5. commands: obstacle tiles forward-only, flat tiles the walker's mix; contact-penalty bodies.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> KBOT_OBST_INIT_LEVEL=9 ./isaaclab.sh -p eval_watch/obstacle_probe.py --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Obstacle-KbotLegs-AMP-v0")
parser.add_argument("--num_envs", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_obstacle, mdp_trackmulti, symmetry  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.curriculum.obstacle_levels = None      # keep the random start levels
    env_cfg.terminations.root_drift = None         # robots are placed by hand below; that is not 'falling behind the command'
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    dev = uenv.device
    n = args.num_envs
    robot = uenv.scene["robot"]
    terrain = uenv.scene.terrain
    ok = True

    # ---- 1. layout + course
    om = uenv.observation_manager
    names = list(om.active_terms["policy"]); dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    print(f"[probe] policy obs {sum(dims)}: {list(zip(names, dims))}")
    print(f"[probe] critic obs {sum(int(d[0]) for d in om.group_obs_term_dim['critic'])}")
    c = mdp_obstacle.course(uenv)
    kinds = mdp_obstacle.env_kind(uenv)
    print(f"[probe] kinds {c['names']} | columns {[c['names'][int(i)] for i in c['col_kind']]}")
    print(f"[probe] heights per level (cm): {[round(float(h) * 100, 1) for h in c['heights']]}")
    print(f"[probe] envs per kind: {[int((kinds == i).sum()) for i in range(len(c['names']))]} | levels used: {sorted(set(terrain.terrain_levels.tolist()))}")
    ok &= dims[-1] == symmetry.MAP_ROWS * symmetry.MAP_COLS and names[-1] == "height_map" and sum(dims[:-1]) % 43 == 0

    # ---- 2. map cell order vs the mirror
    scan = uenv.scene.sensors["height_scanner"]
    xy = scan.ray_starts[0, :, :2]
    grid = xy.reshape(symmetry.MAP_ROWS, symmetry.MAP_COLS, 2)
    rows_y = grid[:, 0, 1]; cols_x = grid[0, :, 0]
    mirrored = symmetry._mirror_map(torch.arange(xy.shape[0], device=dev).float().unsqueeze(0))[0].long()
    good = torch.allclose(xy[mirrored], xy * torch.tensor([1.0, -1.0], device=dev), atol=1e-6)
    print(f"[probe] map grid: y rows {rows_y[0]:+.2f}..{rows_y[-1]:+.2f}, x cols {cols_x[0]:+.2f}..{cols_x[-1]:+.2f}; "
          f"rows constant in y {bool(torch.allclose(grid[:, :, 1], rows_y[:, None].expand(-1, symmetry.MAP_COLS)))}; mirror = same x, opposite y: {good}")
    ok &= good
    centre = int(((xy ** 2).sum(dim=1)).argmin())

    # ---- 3. geometry by placing robots at known offsets
    with torch.inference_mode():
        env.reset()
        act = torch.zeros(n, uenv.action_manager.total_action_dim, device=dev)
        heights = c["heights"][terrain.terrain_levels]
        is_ring = c["kind_is_obstacle"][kinds]
        inner = c["kind_inner"][kinds]; outer = c["kind_outer"][kinds]
        print(f"{'offset':>7s} {'kind':>9s} {'envs':>5s} {'expect = level height?':>24s} {'base ray err (mm)':>18s} {'foot rays err (mm)':>19s}")
        for dx in (0.0, 1.45, 1.8, 2.6, 3.5):
            root = robot.data.default_root_state.clone()
            root[:, :3] += uenv.scene.env_origins
            root[:, 0] += dx
            on = is_ring & (dx > inner) & (dx < outer)
            expect = torch.where(on, heights, torch.zeros_like(heights))
            root[:, 2] = 1.02 + expect
            root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            robot.write_root_pose_to_sim(root[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
            env.step(act)
            base_hit = scan.data.ray_hits_w[:, centre, 2]
            g = mdp_obstacle.foot_ground(uenv)
            feet = robot.data.body_pos_w[:, robot.find_bodies(mdp_trackmulti.FEET, preserve_order=True)[0]]
            fdx = feet[..., 0] - uenv.scene.env_origins[:, None, 0]
            fdy = (feet[..., 1] - uenv.scene.env_origins[:, None, 1]).abs()
            f_on = is_ring[:, None] & (fdx > inner[:, None]) & (fdx < outer[:, None]) & (fdy < inner[:, None])
            f_expect = torch.where(f_on, heights[:, None].expand(-1, 2), torch.zeros(n, 2, device=dev))
            for i, name in enumerate(c["names"]):
                m = kinds == i
                e_base = (base_hit[m] - expect[m]).abs().max() * 1000
                e_foot = (g["under"][m] - f_expect[m]).abs().max() * 1000
                print(f"{dx:7.2f} {name:>9s} {int(m.sum()):5d} {str(bool(on[m].any())):>24s} {float(e_base):18.2f} {float(e_foot):19.2f}")
                ok &= float(e_base) < 1.0 and float(e_foot) < 1.0
            if dx == 1.45:  # on the ring's inner part: the patch around the feet must see the step
                near = mdp_obstacle.near_obstacle(uenv)
                print(f"[probe]   near_obstacle at 1.45 m: flat {float(near[~is_ring].float().mean()):.2f} (want 0), rings {float(near[is_ring].float().mean()):.2f} (want 1); "
                      f"patch max-under (cm) on rings: {float(((g['max'] - g['under'])[is_ring]).mean() * 100):.1f}")
                ok &= float(near[~is_ring].float().mean()) == 0.0 and float(near[is_ring].float().mean()) == 1.0

        # ---- 4. foot frame (robot upright, yaw 0, default joints)
        env.reset()
        root = robot.data.default_root_state.clone(); root[:, :3] += uenv.scene.env_origins; root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        robot.write_root_pose_to_sim(root[:, :7]); robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        env.step(act)
        ids = robot.find_bodies(mdp_trackmulti.FEET, preserve_order=True)[0]
        q = robot.data.body_quat_w[:, ids]
        for label, p in (("sole end A (local x=-0.135)", (-0.135, 0.047, 0.0)), ("sole end B (local x=+0.077)", (0.077, 0.047, 0.0))):
            v = quat_apply(q.reshape(-1, 4), torch.tensor(p, device=dev).expand(n * 2, 3)).reshape(n, 2, 3)
            print(f"[probe] foot {label}: offset from the foot origin in the world (robot faces +x), R {[round(float(x), 3) for x in v[:, 0].mean(0)]} L {[round(float(x), 3) for x in v[:, 1].mean(0)]}")
        print(f"[probe] base height at the default pose {float(robot.data.root_pos_w[:, 2].mean() - uenv.scene.env_origins[:, 2].mean()):.3f} m; "
              f"foot origin height {float((robot.data.body_pos_w[:, ids, 2]).mean()):.3f} m")
        lib = mdp_trackmulti.load_lib(mdp_obstacle.os.path.join(mdp_obstacle.mdp_amp.__file__.rsplit('/source/', 1)[0], "eval_watch", "amp_refs", "multicycle_v1.npz"), dev)
        print(f"[probe] reference cycles' base height (m): {[round(float(z), 3) for z in lib['base_z'].mean(dim=1)]} for {lib['names']}")

        # ---- 5. commands and penalty bodies
        env.reset()
        for _ in range(3):
            env.step(act)
        term = uenv.command_manager.get_term("base_velocity")
        cmd = term.vel_command_b
        ob = mdp_obstacle.obstacle_env_mask(uenv)
        print(f"[probe] obstacle tiles: vx {float(cmd[ob, 0].min()):.2f}..{float(cmd[ob, 0].max()):.2f}, |vy| max {float(cmd[ob, 1].abs().max()):.3f}, "
              f"|wz| max {float(cmd[ob, 2].abs().max()):.3f}, standing {float(term.is_standing_env[ob].float().mean()):.2f}")
        cf = cmd[~ob]; nz = cf != 0
        print(f"[probe] flat tiles: standing {float((~nz.any(1)).float().mean()):.2f}, single-direction {float((nz.sum(1) == 1).float().mean()):.2f}, "
              f"mixed {float((nz.sum(1) == 3).float().mean()):.2f}, vx {float(cf[:, 0].min()):.2f}..{float(cf[:, 0].max()):.2f}")
        ok &= float(cmd[ob, 1:].abs().max()) == 0.0 and not bool(term.is_standing_env[ob].any())
        rw = uenv.reward_manager
        for t in ("feet_stumble", "leg_contact"):
            cfg_ = rw.get_term_cfg(t)
            sensor = uenv.scene.sensors["contact_forces"]
            print(f"[probe] {t}: weight {cfg_.weight}, bodies {[sensor.body_names[i] for i in cfg_.params['sensor_cfg'].body_ids]}")
        print(f"[probe] ref_foot_lift -> {rw.get_term_cfg('ref_foot_lift').func.__name__}, gate -> {uenv.observation_manager.cfg.amp_gate.gate.func.__name__}")
    print(f"[probe] RESULT: {'ALL CHECKS PASSED' if ok else 'A CHECK FAILED'}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
