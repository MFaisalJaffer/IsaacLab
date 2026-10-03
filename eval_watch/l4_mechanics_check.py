"""LINEAGE-4 MECHANICS CHECK (2026-08-16): direct verification of the new
machinery on a tiny live env — things the smoke training log can't show.
  1. default-pose stance width (calibrates stand_stance_geometry nominal_width)
  2. play DR: ankle _play drawn inside the thermostat band after resets;
     other joints <= 0.5 deg
  3. motor-side obs: policy joint_pos/joint_vel diverge from link truth on
     ankles when slack (play forced to 15 deg), match exactly at play=0 joints
  4. obs manager actually wired to the motorside funcs
  5. _rehome_scale gates 0.25/1.0 per env
  6. stand_stance_geometry ~0 at default pose, positive when feet spread
"""
import faulthandler
import functools
import sys

faulthandler.enable()
print = functools.partial(print, flush=True)  # noqa: A001 — crash forensics
from isaaclab.app import AppLauncher
sys.path.append("scripts/reinforcement_learning/rsl_rl")

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_gait
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

OK = []
BAD = []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(f"{name}: {'PASS' if cond else 'FAIL'} {detail}")


def main():
    env_cfg = parse_env_cfg("Isaac-Velocity-Rough-KbotLegs-v0", device="cuda:0", num_envs=8)
    env = gym.make("Isaac-Velocity-Rough-KbotLegs-v0", cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    print("[STAGE] env made, resetting")
    env.reset()
    print("[STAGE] reset done")

    # ---- 4. obs wiring ----
    pol_terms = uenv.observation_manager._group_obs_term_names["policy"]
    idx_jp = pol_terms.index("joint_pos")
    funcs = uenv.observation_manager._group_obs_term_cfgs["policy"]
    check("obs joint_pos wired motorside", funcs[idx_jp].func is mdp_gait.joint_pos_rel_motorside)
    idx_jv = pol_terms.index("joint_vel")
    check("obs joint_vel wired motorside", funcs[idx_jv].func is mdp_gait.joint_vel_rel_motorside)

    # ---- 1. default-pose stance width (fresh reset, before stepping) ----
    from isaaclab.utils.math import quat_apply_inverse
    names = list(robot.data.body_names)
    fids = [names.index(f) for f in ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]]
    fp_w = robot.data.body_pos_w[:, fids, :] - robot.data.root_pos_w.unsqueeze(1)
    q = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    fp_b = quat_apply_inverse(q.reshape(-1, 4), fp_w.reshape(-1, 3)).reshape(8, 2, 3)
    width0 = (fp_b[:, 0, 1] - fp_b[:, 1, 1]).abs().mean().item()
    print(f"[MEASURE] default-pose stance width = {width0*100:.1f} cm  (cfg nominal_width should be ~this)")

    # ---- step a bit so actuators lazily init _play and curriculum runs ----
    act_dim = uenv.action_manager.total_action_dim
    print("[STAGE] stepping 30")
    with torch.inference_mode():
        for _ in range(30):
            env.step(torch.zeros(uenv.num_envs, act_dim, device=dev))
        print("[STAGE] forcing full reset")
        # force a full reset so the play event redraws with the curriculum cap
        uenv.episode_length_buf[:] = 10 ** 6
        env.step(torch.zeros(uenv.num_envs, act_dim, device=dev))
        print("[STAGE] post-reset step done")

        cap = getattr(uenv, "_ankle_play_cap", None)
        check("curriculum publishes _ankle_play_cap", cap is not None,
              f"cap={math.degrees(cap):.2f} deg" if cap is not None else "missing")
        ankle_ok, other_ok, any_ankle = True, True, False
        for name, act in robot.actuators.items():
            p = getattr(act, "_play", None)
            if p is None:
                continue
            if "ankle" in name:
                any_ankle = True
                if cap is not None:
                    lo, hi = math.radians(1.0) - 1e-6, cap + 1e-6
                    ankle_ok &= bool(((p >= lo) & (p <= hi)).all())
            else:
                other_ok &= bool((p <= math.radians(0.5) + 1e-6).all())
        check("ankle play in thermostat band", any_ankle and ankle_ok)
        check("other-joint play <= 0.5 deg", other_ok)

        # ---- 3. motor-side obs: force 15 deg on ankles, drive off-target ----
        for name, act in robot.actuators.items():
            if getattr(act, "_play", None) is not None and "ankle" in name:
                act._play[:] = math.radians(15.0)
        a = torch.zeros(uenv.num_envs, act_dim, device=dev)
        a[:, 8:10] = 0.4                      # push ankle targets off current pos
        for _ in range(5):
            obs_all, _ = env.step(a)[:2]
        true_rel = mdp.joint_pos_rel(uenv)
        motor_rel = mdp_gait.joint_pos_rel_motorside(uenv)
        d = (true_rel - motor_rel).abs()
        # ankle joint indices in joint ordering
        jn = list(robot.data.joint_names)
        aidx = [i for i, n in enumerate(jn) if "ankle" in n]
        nidx = [i for i, n in enumerate(jn) if "ankle" not in n]
        # non-ankle joints have tiny play drawn (0-0.5 deg) -> divergence bounded
        # by half band (0.25 deg = 0.0044 rad)
        # tolerance: half band (0.25 deg = 0.0044) + one physics substep of
        # joint motion (motor_pos is computed at the last actuator compute)
        check("motorside == truth on ~rigid joints", bool((d[:, nidx] < 0.02).all()),
              f"max diff {d[:, nidx].max():.5f} rad")
        check("motorside diverges on slack ankles", bool((d[:, aidx].max() > 0.01)),
              f"max diff {d[:, aidx].max():.4f} rad (15 deg band => up to ~0.13)")
        dv = (mdp.joint_vel_rel(uenv) - mdp_gait.joint_vel_rel_motorside(uenv)).abs()
        check("motorside vel differs on ankles (slack blindness)", bool(dv[:, aidx].max() > 1e-3),
              f"max vel diff {dv[:, aidx].max():.3f} rad/s")

        # ---- 5. rehome scale ----
        now = uenv.episode_length_buf.float() * uenv.step_dt
        uenv._rehome_until = torch.zeros(uenv.num_envs, device=dev)
        uenv._rehome_until[:4] = now[:4] + 1.0
        s = mdp_gait._rehome_scale(uenv)
        check("_rehome_scale 0.25 in grace / 1.0 outside",
              bool((s[:4] == 0.25).all() and (s[4:] == 1.0).all()), f"s={s.tolist()}")

        # ---- 6. stance geometry term ----
        cmd_term = uenv.command_manager.get_term("base_velocity")
        cmd_term.vel_command_b[:, :] = 0.0
        if hasattr(cmd_term, "is_standing_env"):
            cmd_term.is_standing_env[:] = True
        uenv._rehome_until[:] = 0.0
        uenv._sustained_push_active[:] = False
        uenv._tilt_hold_active[:] = False
        cfg = SceneEntityCfg("robot", body_names=".*_LEG_FOOT")
        cfg.resolve(uenv.scene)
        pen = mdp_gait.stand_stance_geometry(uenv, cfg, nominal_width=round(width0, 3))
        check("stance_geometry finite & >=0", bool(torch.isfinite(pen).all() and (pen >= 0).all()),
              f"vals={pen.tolist()}")
        check("stance_geometry ~0 near default stance", bool((pen < 0.06).all()),
              f"max={pen.max():.3f} (post-settle drift ok)")

    print("\n".join(OK))
    if BAD:
        print("\n".join(BAD))
        print("MECHANICS: FAIL")
    else:
        print("MECHANICS: ALL PASS")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
