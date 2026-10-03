"""Puppet pass for a single-cycle reference (asimov_walk_cycle layout): ground-follow base height per
phase sample and the body velocity the cycle kinematically implies on the K-Bot (stance-foot motion with
the base held fixed, negated). Writes `cycle_base_z` and `kin_vel` (vx, vy, wz) into the npz; with
--set_vel the planar part also becomes `vel_b` (the command the track env gives).

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_cycle_ground.py \
      --cycle eval_watch/amp_refs/sidestep_left_cycle.npz --set_vel --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cycle", required=True, nargs="+")
parser.add_argument("--set_vel", action="store_true")
parser.add_argument("--feet_only", action="store_true", help="only add cycle_feet_rel (feet [R, L] relative to the base); leave base_z / velocities untouched")
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.physics_dt
    dev = uenv.device
    with torch.inference_mode():
        env.reset()
    jn = robot.joint_names
    J = len(jn)
    bn = robot.body_names
    fi = [[i for i, n in enumerate(bn) if "501R" in n][0], [i for i, n in enumerate(bn) if "501L" in n][0]]
    sensor = uenv.scene.sensors["contact_forces"]
    s_feet = [i for i, n in enumerate(sensor.body_names) if n.endswith("FOOT")]
    zero = torch.zeros(1, J, device=dev)
    zero6 = torch.zeros(1, 6, device=dev)
    root_hi = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=dev)

    def puppet(q, root):
        robot.write_root_pose_to_sim(root); robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(q, zero); uenv.sim.step(render=False); uenv.scene.update(dt)

    with torch.inference_mode():
        puppet(zero, root_hi)
        foot_rel_z0 = float((robot.data.body_pos_w[0, fi, 2] - robot.data.root_pos_w[0, 2]).mean())
        h_stand = None
        z = -foot_rel_z0 + 0.06
        while z > -foot_rel_z0 - 0.06:
            puppet(zero, torch.tensor([[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]], device=dev))
            if float(sensor.data.net_forces_w[0, s_feet].norm(dim=-1).max()) > 5.0:
                h_stand = z; break
            z -= 0.001
        assert h_stand is not None
        print(f"[ground] standing base height {h_stand:.3f} m, foot link {-foot_rel_z0:.3f} m below base")
        for path in args.cycle:
            d = dict(np.load(path, allow_pickle=True))
            assert [str(s) for s in d["joint_names"]] == jn, f"{path}: joint order"
            q_cyc = torch.tensor(d["joint_pos"] if "joint_pos" in d else d["cycle_q"], device=dev)
            N = q_cyc.shape[0]; P = float(d["period_s"]); fps = N / P
            q_all = torch.cat([q_cyc, q_cyc], dim=0)  # two periods for the velocity estimate (wrap-around)
            T = q_all.shape[0]
            rel = torch.zeros(T, 2, 3, device=dev); yaw = torch.zeros(T, 2, device=dev)
            for t in range(T):
                puppet(q_all[t:t + 1], root_hi)
                rel[t] = robot.data.body_pos_w[0, fi] - robot.data.root_pos_w[0]
                qf = robot.data.body_quat_w[0, fi]
                yaw[t] = torch.atan2(2 * (qf[:, 0] * qf[:, 3] + qf[:, 1] * qf[:, 2]), 1 - 2 * (qf[:, 2] ** 2 + qf[:, 3] ** 2))
            p = rel.cpu().numpy(); y = np.unwrap(yaw.cpu().numpy(), axis=0)
            zf = p[:N, :, 2]
            base_z = h_stand + foot_rel_z0 - zf.min(axis=1)
            stance = np.argmin(p[:, :, 2], axis=1)
            idx = np.arange(T - 1)
            v_imp = -(p[1:] - p[:-1])[idx, stance[:-1]] * fps
            w_imp = -(y[1:] - y[:-1])[idx, stance[:-1]] * fps
            kin = np.array([v_imp[:, 0].mean(), v_imp[:, 1].mean(), w_imp.mean()], np.float32)
            clear = (zf - zf.min(axis=1, keepdims=True)).max() * 100
            width = np.abs(p[:N, 0, 1] - p[:N, 1, 1])
            d["cycle_feet_rel"] = p[:N].astype(np.float32)  # (N, 2, 3) feet [R, L] relative to the base, base upright
            if args.feet_only:
                np.savez(path, **d)
                print(f"[cycle] {os.path.basename(path)}: cycle_feet_rel added; x range R {p[:N, 0, 0].min():.3f}..{p[:N, 0, 0].max():.3f}, "
                      f"y range R {p[:N, 0, 1].min():.3f}..{p[:N, 0, 1].max():.3f}, z {p[:N, :, 2].min():.3f}..{p[:N, :, 2].max():.3f}")
                continue
            d["cycle_base_z"] = base_z.astype(np.float32)
            d["kin_vel"] = kin
            old = d.get("vel_b", np.array([float(d.get("speed_mps", 0.0)), 0.0], np.float32))
            if args.set_vel:
                d["vel_b"] = kin[:2].astype(np.float32)
                d["speed_mps"] = float(kin[0])
                d["wz"] = float(kin[2])
            np.savez(path, **d)
            print(f"[cycle] {os.path.basename(path)}: base_z {base_z.min():.3f}-{base_z.max():.3f} m, clearance {clear:.1f} cm, "
                  f"stance width {width.min() * 100:.1f}-{width.max() * 100:.1f} cm, kinematic vel (vx {kin[0]:.3f}, vy {kin[1]:.3f}, wz {kin[2]:.3f}) "
                  f"[vel_b before: {np.asarray(old).round(3).tolist()}{' -> set' if args.set_vel else ''}]")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
