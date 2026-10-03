"""Decisive test: (A) does _resample_command actually flag ~20% standing?
(B) if so, how long do standing episodes SURVIVE vs walking ones? (survivorship
would explain why standing env-steps are ~1% despite 20% sampling)."""
import argparse
import sys
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", type=str, default="eval_watch/stand_sampling_test.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    n = args_cli.num_envs
    lines = [f"rel_standing_envs (live cfg) = {cmd_term.cfg.rel_standing_envs}"]

    # ---- (A) does the resample logic flag ~20%? force-resample all envs 5x ----
    ids = torch.arange(n, device=env.unwrapped.device)
    fr = []
    for _ in range(5):
        cmd_term._resample_command(ids)
        fr.append(float(cmd_term.is_standing_env.float().mean()))
    lines.append(f"(A) forced resample x5 -> standing fraction: "
                 + ", ".join(f"{f*100:.0f}%" for f in fr)
                 + f"   [expect ~{cmd_term.cfg.rel_standing_envs*100:.0f}%]")

    # ---- (B) survivorship: track how long standing vs walking episodes last ----
    alive_steps = torch.zeros(n, dtype=torch.long, device=env.unwrapped.device)
    stand_life, walk_life = [], []
    stand_steps = walk_steps = 0
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            was_standing = cmd_term.is_standing_env.clone()
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
            alive_steps += 1
            stand_steps += int(was_standing.sum())
            walk_steps += int((~was_standing).sum())
            d = dones.nonzero(as_tuple=False).flatten()
            for i in d.tolist():
                (stand_life if bool(was_standing[i]) else walk_life).append(int(alive_steps[i]))
            alive_steps[d] = 0
    def stats(v):
        return f"n={len(v)} mean={sum(v)/len(v):.0f} steps" if v else "n=0"
    lines.append(f"(B) episode lifetime — STANDING-commanded: {stats(stand_life)}")
    lines.append(f"    episode lifetime — WALKING-commanded : {stats(walk_life)}")
    tot = stand_steps + walk_steps
    lines.append(f"    env-step share: standing {stand_steps/tot*100:.1f}%  walking {walk_steps/tot*100:.1f}%"
                 f"   [sampling says {cmd_term.cfg.rel_standing_envs*100:.0f}% / "
                 f"{(1-cmd_term.cfg.rel_standing_envs)*100:.0f}%]")
    out = "\n".join(lines)
    print(out)
    with open(args_cli.out, "w") as f:
        f.write(out + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
