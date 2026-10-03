"""Check the robot for a LATERAL (left/right) center-of-mass asymmetry at the
default symmetric pose — a systematic reason it would always circle one way."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=4)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # disable random resets/DR so we read the NOMINAL default pose
    for ev in ("push_robot",):
        if getattr(cfg.events, ev, None) is not None:
            setattr(cfg.events, ev, None)
    if getattr(cfg.curriculum, "velocity_push_curriculum", None) is not None:
        cfg.curriculum.velocity_push_curriculum = None

    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    # HOLD the symmetric default pose: reset randomizes joints, so drive zero action
    # (= default joint targets) for a bit so the robot settles to joints~0 before we
    # read geometry. Otherwise a random asymmetric pose fakes a COM offset.
    act = torch.zeros((env.unwrapped.num_envs, robot.data.joint_pos.shape[1]), device=robot.device)
    for _ in range(60):
        env.step(act)
    names = robot.data.body_names
    m = robot.data.default_mass[0].to("cpu")                       # (B,)
    coms_local = robot.root_physx_view.get_coms()[0, :, :3].to("cpu")   # (B,3) local COM offset
    bp = robot.data.body_pos_w[0].to("cpu")                        # (B,3) body origins (world)
    bq = robot.data.body_quat_w[0].to("cpu")                       # (B,4)
    base_p = robot.data.root_pos_w[0].to("cpu")
    base_q = robot.data.root_quat_w[0].to("cpu")

    # world COM of each body -> overall world COM
    world_com = bp + quat_apply(bq, coms_local)                    # (B,3)
    total = m.sum()
    com_w = (m[:, None] * world_com).sum(0) / total                # (3,) overall world COM

    # foot support center (world) = midpoint of the two foot bodies
    foot_idx = [i for i, n in enumerate(names) if n.endswith("FOOT")]
    foot_mid_w = bp[foot_idx].mean(0)                              # (3,)

    # COM offset from support center, expressed in the base (heading) frame:
    # lateral(y) is what makes it topple/circle; forward(x) relates to lean.
    off_base = quat_apply_inverse(base_q, (com_w - foot_mid_w))    # (3,)

    lines = [f"base height (settled) = {base_p[2]:.3f} m  (fell if < ~0.6)"]
    lines.append("=== foot body positions in the BASE frame (y = left/right) ===")
    for i in foot_idx:
        fb = quat_apply_inverse(base_q, bp[i] - base_p)
        lines.append(f"  {names[i]:26s} base x={fb[0]:+.3f} y={fb[1]:+.3f} z={fb[2]:+.3f}")
    fmb = quat_apply_inverse(base_q, foot_mid_w - base_p)
    lines.append(f"  foot MIDPOINT (base)       x={fmb[0]:+.3f} y={fmb[1]:+.3f} z={fmb[2]:+.3f}")
    cmb = quat_apply_inverse(base_q, com_w - base_p)
    lines.append(f"  overall COM  (base)        x={cmb[0]:+.3f} y={cmb[1]:+.3f} z={cmb[2]:+.3f}")
    lines.append("")
    lines.append(f">>> COM vs FOOT-CENTER  forward(x)={off_base[0]*1000:+.0f} mm  LATERAL(y)={off_base[1]*1000:+.0f} mm")
    lines.append("    (lateral 0 = balanced; a persistent lateral offset -> topple/circle)")
    out = "\n".join(lines)
    print(out)
    with open("/tmp/com_result.txt", "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
