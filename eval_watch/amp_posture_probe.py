"""Are the hips over the feet? Puppet pass with the base held UPRIGHT (as the tracker's flat-base
policy keeps it): mean foot position relative to the base in the base frame (x forward, y left), for
both feet and for the lower (stance) foot. A reference whose feet sit well in front of the hips is a
'sitting back' crouch: a tracker following it must over-step to balance. Also tries hip-pitch offsets
(left += d, right -= d) to show which sign brings the feet back under the hips.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_posture_probe.py \
      --files eval_watch/amp_refs/asimov_walk_cycle.npz eval_watch/amp_refs/lafan1/clips_v2/backward_0.npz --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", required=True)
parser.add_argument("--offsets_deg", default="-20,-10,0,10,20")
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
    J = len(robot.joint_names)
    bn = robot.body_names
    fi = [[i for i, n in enumerate(bn) if "501R" in n][0], [i for i, n in enumerate(bn) if "501L" in n][0]]
    zero = torch.zeros(1, J, device=dev)
    zero6 = torch.zeros(1, 6, device=dev)
    root_hi = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=dev)
    offs = [float(x) for x in args.offsets_deg.split(",")]

    def puppet(q):
        robot.write_root_pose_to_sim(root_hi); robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(q, zero); uenv.sim.step(render=False); uenv.scene.update(dt)

    def feet_rel(q_all):
        T = q_all.shape[0]
        rel = torch.zeros(T, 2, 3, device=dev)
        for t in range(T):
            puppet(q_all[t:t + 1])
            rel[t] = robot.data.body_pos_w[0, fi] - robot.data.root_pos_w[0]
        return rel.cpu().numpy()

    with torch.inference_mode():
        puppet(zero)
        rp = robot.data.root_pos_w[0]
        print("[links] body z below base at the zero pose (m):", {n: round(float(robot.data.body_pos_w[0, i, 2] - rp[2]), 3) for i, n in enumerate(bn)})
    print(f"{'file':28s} {'off':>5s} {'feet x':>7s} {'stance x':>9s} {'feet y':>7s} {'z_lo':>6s}   (m; x>0 = feet IN FRONT of hips)")
    with torch.inference_mode():
        for path in args.files:
            d = np.load(path, allow_pickle=True)
            q = d["joint_pos"] if "joint_pos" in d else d["cycle_q"]
            q = np.asarray(q, np.float32)
            step = max(1, q.shape[0] // 300)
            q = q[::step]
            for off in offs:
                qo = q.copy(); qo[:, 0] += np.radians(off); qo[:, 1] -= np.radians(off)
                p = feet_rel(torch.tensor(qo, device=dev))
                st = np.argmin(p[:, :, 2], axis=1)
                sx = p[np.arange(len(p)), st, 0]
                print(f"{os.path.basename(path)[:28]:28s} {off:5.0f} {p[:, :, 0].mean():7.3f} {sx.mean():9.3f} {p[:, :, 1].mean():7.3f} {p[:, :, 2].min():6.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
