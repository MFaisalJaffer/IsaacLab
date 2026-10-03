"""Ground-truth obs/state dump for the kbot_legs policy (for the Mac MuJoCo port)."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0-Play")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.assets import retrieve_file_path

def r(x): return [round(float(v),4) for v in x]

def main():
    task=args_cli.task.split(":")[-1]
    env_cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg=cli_args.parse_rsl_rl_cfg(task, args_cli)
    resume_path=retrieve_file_path(args_cli.checkpoint)
    env=gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env=RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner=OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy=runner.get_inference_policy(device=env.unwrapped.device)
    u=env.unwrapped; om=u.observation_manager; robot=u.scene["robot"]
    obs,_=env.get_observations()
    print("##DUMP_START##")
    print("JOINT_NAMES", list(robot.data.joint_names))
    print("DEFAULT_JOINT_POS", r(robot.data.default_joint_pos[0]))
    try: print("TERM_NAMES", om.active_terms["policy"])
    except Exception as e: print("TERM_NAMES_ERR", e)
    try: print("TERM_DIMS", om.group_obs_term_dim["policy"])
    except Exception as e: print("TERM_DIMS_ERR", e)
    imu=None
    try: imu=u.scene["imu"]
    except Exception as e: print("NO_IMU_SENSOR", e)
    for s in range(4):
        with torch.inference_mode(): a=policy(obs)
        print(f"##STEP {s}")
        print(" root_quat_w", r(robot.data.root_quat_w[0]))
        print(" joint_pos", r(robot.data.joint_pos[0]))
        print(" joint_vel", r(robot.data.joint_vel[0]))
        try: print(" cmd", r(u.command_manager.get_command("base_velocity")[0]))
        except Exception as e: print(" cmd_err", e)
        if imu is not None:
            try: print(" imu_quat_w", r(imu.data.quat_w[0]))
            except Exception as e: print(" imu_quat_err", e)
            for attr in ("projected_gravity_b","ang_vel_b","lin_acc_b"):
                try: print(f" imu_{attr}", r(getattr(imu.data,attr)[0]))
                except Exception as e: print(f" imu_{attr}_err", e)
        print(" obs_flat", r(obs[0]))
        print(" action", r(a[0]))
        with torch.inference_mode(): obs,_,_,_=env.step(a)
    print("##DUMP_END##")
    env.close(); simulation_app.close()

main()
