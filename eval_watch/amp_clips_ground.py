"""One Isaac puppet pass over the clip library: body height per frame (ground-follow) + foot metrics.

The time-indexed tracker starts episodes mid-clip (RSI) and needs the base height that puts the
lowest sole on the ground for that frame. Same method as amp_replay_clip.py: joints written with
the base fixed in the air, feet read back, base_z = h_stand + foot_rel_z0 - min(foot_rel_z).
Writes `base_z` (and clearance / width stats) into every clips/<name>.npz.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_clips_ground.py --headless
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clips", default="/home/faisal/IsaacLab/eval_watch/amp_refs/lafan1/clips")
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
        summary = {}
        for path in sorted(glob.glob(os.path.join(args.clips, "*.npz"))):
            d = dict(np.load(path, allow_pickle=True))
            assert [str(s) for s in d["joint_names"]] == jn, f"{path}: joint order"
            q_all = torch.tensor(d["joint_pos"], device=dev)
            T = q_all.shape[0]
            rel = torch.zeros(T, 2, 3, device=dev)
            for t in range(T):
                puppet(q_all[t:t + 1], root_hi)
                rel[t] = robot.data.body_pos_w[0, fi] - robot.data.root_pos_w[0]
            p = rel.cpu().numpy(); zf = p[:, :, 2]
            base_z = h_stand + foot_rel_z0 - zf.min(axis=1)
            h = zf - zf.min(axis=1, keepdims=True)
            width = np.abs(p[:, 0, 1] - p[:, 1, 1])
            d["base_z"] = base_z.astype(np.float32)
            d["h_stand"] = h_stand
            np.savez(path, **d)
            summary[os.path.basename(path)[:-4]] = {"base_z_min": round(float(base_z.min()), 3), "clearance_cm": round(float(h.max()) * 100, 1), "width_cm": round(float(np.median(width)) * 100, 1)}
        base = os.path.basename(os.path.normpath(args.clips)); json.dump(summary, open(os.path.join(args.clips, "..", "clip_ground.json" if base == "clips" else f"clip_ground_{base}.json"), "w"), indent=1)
        print(f"[ground] {len(summary)} clips updated; clearance range {min(v['clearance_cm'] for v in summary.values())}-{max(v['clearance_cm'] for v in summary.values())} cm, "
              f"width {min(v['width_cm'] for v in summary.values())}-{max(v['width_cm'] for v in summary.values())} cm")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
