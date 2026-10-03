"""Print the robot's nominal per-body masses + the randomized total-mass spread
the policy actually trains on (after the startup mass-DR events)."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import torch  # noqa
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("\n=== mass-DR events on the env cfg ===")
    for e in ("add_base_mass", "base_com", "add_limb_masses"):
        term = getattr(cfg.events, e, "MISSING")
        if term is None:
            print(f"  {e}: DISABLED (None)")
        elif term == "MISSING":
            print(f"  {e}: not present")
        else:
            print(f"  {e}: ACTIVE params={term.params}")

    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    names = robot.data.body_names
    nominal = robot.data.default_mass[0]                 # (num_bodies,) nominal
    actual = robot.root_physx_view.get_masses()          # (N, num_bodies) after DR
    tot = actual.sum(dim=1)

    lines = ["=== nominal per-body mass (kg) ==="]
    for n, m in zip(names, nominal.tolist()):
        lines.append(f"  {n:32s} {m:7.4f}")
    lines.append(f"  {'TOTAL nominal':32s} {nominal.sum().item():7.3f} kg")
    lines.append(f"\n=== randomized TOTAL mass across {args_cli.num_envs} envs (what it trains on) ===")
    lines.append(f"  min {tot.min():.3f}  mean {tot.mean():.3f}  max {tot.max():.3f} kg  "
                 f"(+/-{(tot.max()-tot.min())/2/tot.mean()*100:.0f}% around mean)")
    out = "\n".join(lines)
    print(out)
    with open("/tmp/mass_result.txt", "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
