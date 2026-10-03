"""Extract each joint's WORLD rotation axis (from the contact Jacobian) at the
neutral pose, then decide the L/R mirror sign for each pair: reflect the left
axis across the sagittal plane (y->-y) and check if it matches +right (=> swap+
negate in joint space) or -right (=> swap only)."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=2)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    # hold neutral pose briefly
    act = torch.zeros((env.unwrapped.num_envs, robot.data.joint_pos.shape[1]), device=robot.device)
    for _ in range(30):
        env.step(act)

    jnames = list(robot.data.joint_names)
    bnames = list(robot.data.body_names)
    base_q = robot.data.root_quat_w[0]
    J = robot.root_physx_view.get_jacobians()  # (N, num_bodies?, 6, num_dofs(+6))
    lines = [f"jacobian shape: {tuple(J.shape)}   #joints={len(jnames)}  #bodies={len(bnames)}"]
    ndof = len(jnames)
    dof_off = J.shape[-1] - ndof  # floating base cols in front, if any
    lines.append(f"dof col offset (floating base) = {dof_off}")

    def world_axis_for(joint_name):
        # use the foot on that joint's side; its angular Jacobian column = joint world axis
        side = "L_LEG_FOOT" if "left" in joint_name else "R_LEG_FOOT"
        bi = [i for i, n in enumerate(bnames) if n.endswith(side)][0]
        ji = jnames.index(joint_name)
        col = J[0, bi, 3:6, dof_off + ji]  # angular part (wx,wy,wz) in world
        # express in base frame so y is cleanly lateral
        return quat_apply_inverse(base_q, col)

    pairs = [("dof_left_hip_pitch_04", "dof_right_hip_pitch_04"),
             ("dof_left_hip_roll_04", "dof_right_hip_roll_04"),
             ("dof_left_hip_yaw_03", "dof_right_hip_yaw_03"),
             ("dof_left_knee_04", "dof_right_knee_04"),
             ("dof_left_ankle_02", "dof_right_ankle_02")]
    R = torch.tensor([1.0, -1.0, 1.0], device=robot.device)  # reflect y in base frame
    lines.append(f"\n{'pair':14s} {'axis_L(base)':>22s} {'axis_R(base)':>22s}  {'R*aL vs aR':>12s}  SIGN")
    for ln, rn in pairs:
        aL = world_axis_for(ln); aR = world_axis_for(rn)
        aLn = aL / (aL.norm() + 1e-9); aRn = aR / (aR.norm() + 1e-9)
        reflAL = R * aLn
        dot = float((reflAL * aRn).sum())
        sign = "swap+NEGATE (flip)" if dot > 0 else "swap only (no flip)"
        lines.append(f"{ln.replace('dof_left_',''):14s} [{aLn[0]:+.2f},{aLn[1]:+.2f},{aLn[2]:+.2f}]      "
                     f"[{aRn[0]:+.2f},{aRn[1]:+.2f},{aRn[2]:+.2f}]   dot={dot:+.2f}  {sign}")
    out = "\n".join(lines)
    print(out)
    with open("/tmp/axis_result.txt", "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
