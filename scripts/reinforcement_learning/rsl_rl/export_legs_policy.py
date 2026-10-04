# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export the trained LEGS-ONLY policy to ONNX + a metadata JSON, so it can be
replayed in a plain MuJoCo viewer on a Mac (see ksim-kbot/view_mac_legs_isaac.py).

rsl_rl already exports `policy.onnx` (the 39-obs -> 10-action actor MLP). The one
thing the Mac viewer cannot know without Isaac is the exact joint DOF order and
the per-joint kp/kd the policy was trained with — this script dumps those (and
the obs layout / action scale / timing) into `legs_policy_meta.json`.

Run on the training box:
    cd /home/faisal/IsaacLab && source kbot_env/bin/activate
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/export_legs_policy.py \
        --task Isaac-Velocity-Rough-KbotLegs-v0 --num_envs 1 [--checkpoint /abs/model_X.pt]

Outputs (next to the checkpoint, under .../exported/):
    policy.onnx            - actor MLP, raw 39-d obs -> 10-d action (deterministic)
    policy.pt             - TorchScript equivalent
    legs_policy_meta.json - joint order, obs layout, action scale, kp/kd, timing
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  (local module living next to play.py)

parser = argparse.ArgumentParser(description="Export the legs policy to ONNX + metadata.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--trained_sensing", type=str, default=None, help="JSON describing the sensing path the policy was TRAINED on (delays, holds); copied into the meta as information — the export env itself always uses fresh sensing")
parser.add_argument("--out_dir", type=str, default=None, help="where to write the bundle (default: <checkpoint dir>/exported)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import inspect
import json
import os

import gymnasium as gym
import torch  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner


# The actor observation group (PolicyCfg) — order MUST match the env cfg.
# (config/kbot/rough_env_cfg.py :: KBotObservations.PolicyCfg)
OBS_TERMS = [
    {"name": "imu_projected_gravity", "dim": 3, "note": "gravity unit vec in IMU frame"},
    {"name": "velocity_commands", "dim": 3, "note": "[vx, vy, wz]"},
    {"name": "joint_pos_rel", "dim": 10, "note": "q - q_default, in joint_names order"},
    {"name": "joint_vel_rel", "dim": 10, "note": "qd, in joint_names order"},
    {"name": "imu_ang_vel", "dim": 3, "note": "angular vel in IMU frame"},
    {"name": "last_action", "dim": 10, "note": "previous raw policy output"},
    {"name": "gait_phase", "dim": 4,
     "note": "[cos phiL, sin phiL, cos phiR, sin phiR]; anti-phase gait clock at gait_freq"},
]

# Gait-clock params (must match config/kbot_legs/mdp_gait.py) — the Mac viewer
# needs these to reconstruct the gait_phase observation.
# stand_phase corrected 2026-07-17 (HIL caught the stale constant): the pin has
# been (pi, pi) "both planted" since the mirror-invariant fix — (pi/2, pi) was
# the OLD asymmetric pin that caused the 385-taps/min right-foot fidget. The
# meta had been declaring the old pin while every build since ran the new one.
# gait_freq: MUST match _GAIT_FREQ in kbot_legs/rough_env_cfg.py — the rig
# synthesizes phase obs from this; a mismatch feeds the policy a clock it never
# trained on. (1.15 cadence experiment 2026-07-28 REVERTED same day: stride-
# length coupling blew up hip_pitch — see rough_env_cfg.py _GAIT_FREQ note.)
GAIT = {"gait_freq": 1.4, "stand_still_threshold": 0.1,
        "start_phase": [0.0, 3.141592653589793],
        "stand_phase": [3.141592653589793, 3.141592653589793]}

# SPEED-ADAPTIVE CLOCK (2026-08-04): when exporting a KBOT_ADAPT-trained build,
# the rig MUST reproduce the same speed->frequency law or per-speed gait timing
# breaks. Deploy-side spec (the ONLY deploy-side change of the adaptive design):
#   f(cmd)  = clip(f_min + (f_max-f_min)*(|cmd_vx|-v_lo)/(v_hi-v_lo), f_min, f_max)
#             computed from the COMMANDED vx (not measured velocity)
#   phase   : INTEGRATED, phi += 2*pi*f(cmd)*dt per 20 ms tick, free-running
#             (do NOT use ticks*f — f varies, the product form jumps on speed change)
#   standing: |cmd| < stand_still_threshold -> pin the OBS to stand_phase
#             (integrator keeps running underneath, exactly as in sim)
if os.environ.get("KBOT_ADAPT") == "1":
    GAIT["gait_freq_map"] = {"f_min": 0.9, "f_max": 1.4, "v_lo": 0.15, "v_hi": 0.45}
    GAIT["phase_synthesis"] = ("integrate: phi += 2*pi*f(|cmd_vx|)*dt each tick; "
                               "f = clip(f_min + (f_max-f_min)*(|cmd_vx|-v_lo)/(v_hi-v_lo), f_min, f_max); "
                               "pin obs to stand_phase when |cmd| < stand_still_threshold")
    GAIT["gait_freq"] = None   # explicit: fixed-clock field is NOT the truth for this build


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Export NOMINAL gains (2026-07-18): gain-DR events fire on the export env's
    # reset and were baking a random kp/kd draw into the meta (caught on the
    # 289400 bundle: yaw/ankle kp 50-58 instead of 60). The meta is the rig's
    # deploy truth — it must carry the nominal spec, never a DR sample.
    for _ev in ("randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints"):
        if getattr(env_cfg.events, _ev, None) is not None:
            setattr(env_cfg.events, _ev, None)
    # PLANT CURRICULA MUST BE NULLED (rig ask 2026-08-24): they run on reset,
    # so a fresh export env would initialise the ankle-stiffness ramp at its
    # START value (150) and stamp THAT into the meta as `series_k` — declaring
    # a plant neither the policy trained on nor the robot has. Nulling them
    # makes the meta report DEPLOY TRUTH (K_s 23, fitted friction, full play).
    # NB the meta cannot know where the ramp stood when the checkpoint was
    # saved (curriculum state is not checkpointed) — see `plant_ramp_warning`;
    # the shipping rule is: only export candidates whose training ramp has
    # reached level 1.0.
    for _cu in ("sustained_push_level", "velocity_push_curriculum",
                "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, _cu, None) is not None:
            setattr(env_cfg.curriculum, _cu, None)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # Export the NOMINAL robot: disable domain randomizations that would otherwise
    # bake an asymmetric sample into the meta. Gain DR randomizes kp/kd per joint,
    # so a captured env-0 sample is L/R-asymmetric -> the viewer's PD controller
    # would circle. Disabling it makes the captured gains nominal & symmetric.
    for ev in ("randomize_actuator_gains", "add_limb_masses", "randomize_rigid_body_material",
               "base_com", "add_base_mass"):
        if getattr(env_cfg.events, ev, None) is not None:
            setattr(env_cfg.events, ev, None)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[export] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    # AMP builds (walker v4/v5, 2026-10-02) train with KbotAmpRunner; the stock runner rejects AMPPPO.
    if getattr(agent_cfg.algorithm, "class_name", "PPO") == "AMPPPO":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner
        runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    export_dir = args_cli.out_dir or os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=runner.obs_normalizer, path=export_dir, filename="policy.onnx")
    print(f"[export] wrote policy.onnx + policy.pt to {export_dir}")

    # ---- gather metadata from the LIVE env (authoritative) ----
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)  # Isaac DOF order
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()

    # kp/kd live on the ACTUATOR objects, not robot.data.joint_stiffness — the
    # T-V actuator does PD in Python (effort control), so the PhysX drive gains
    # are 0. Pull the real per-joint gains from each actuator group.
    kp_map, kd_map = {}, {}
    for act in robot.actuators.values():
        for k, jn in enumerate(act.joint_names):
            kp_map[jn] = float(act.stiffness[0, k].detach().cpu())
            kd_map[jn] = float(act.damping[0, k].detach().cpu())
    kp = [kp_map[jn] for jn in joint_names]
    kd = [kd_map[jn] for jn in joint_names]
    try:
        armature = robot.data.joint_armature[0].detach().cpu().tolist()
    except Exception:
        armature = None
    try:
        effort_limit = robot.data.joint_effort_limits[0].detach().cpu().tolist()
    except Exception:
        effort_limit = None

    # Per-joint ACTION CLIP table (added 2026-07-30 — flagged as must-ship since
    # the stop-pressing fix): the env clamps the PROCESSED action, i.e. the PD
    # target AFTER raw*scale + default_offset, per joint, in radians
    # (isaaclab joint_actions.py:134-139). The rig/port MUST replicate this
    # clamp or hardware can command targets past joint stops (knee hyperextension
    # brace class). Read from the LIVE action term = training-time truth.
    act_term = env.unwrapped.action_manager.get_term("joint_pos")
    clip_t = getattr(act_term, "_clip", None)
    action_clip = None
    if clip_t is not None:
        c = clip_t[0].detach().cpu()  # (action_dim, 2) — env 0, DR-free export env
        action_clip = {jn: [float(c[i, 0]), float(c[i, 1])] for i, jn in enumerate(joint_names)}

    # ---- observation layout from the LIVE observation manager ----
    # Walker v4/v5 builds stack H frames per term (every term's frames contiguous, oldest first); the
    # per-frame content (OBS_TERMS, 43 values) is unchanged. See eval_watch/RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md.
    om = env.unwrapped.observation_manager
    live_names = list(om.active_terms["policy"])
    live_dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    H = int(getattr(env_cfg.observations.policy, "history_length", 0) or 0) or 1
    frame_dim = sum(t["dim"] for t in OBS_TERMS)
    assert len(live_dims) == len(OBS_TERMS) and all(d == H * t["dim"] for d, t in zip(live_dims, OBS_TERMS)), \
        f"policy obs layout {list(zip(live_names, live_dims))} does not match OBS_TERMS x history {H}"
    obs_dim = sum(live_dims)
    slices, off = [], 0
    for t, d in zip(OBS_TERMS, live_dims):
        slices.append({"name": t["name"], "offset": off, "length": d, "frame_dim": t["dim"]})
        off += d
    obs_history = None
    if H > 1:
        obs_history = {"length": H, "layout": "per_term_contiguous_oldest_first",
                       "startup": "fill every slot with the first frame",
                       "note": "input = for each term in obs_terms order: that term's last H frames, oldest first; the newest frame is the last frame_dim values of each block",
                       "term_slices": slices}

    # ---- gait clock from the LIVE gait_phase term ----
    gp = dict(env_cfg.observations.policy.gait_phase.params)
    gait = dict(GAIT)
    gait["stand_still_threshold"] = float(gp.get("stand_still_threshold", 0.1))
    if gp.get("freq_map"):
        fm = gp["freq_map"]
        gait["gait_freq"] = None
        gait["gait_freq_map"] = {"f_min": fm[0], "f_max": fm[1], "v_lo": fm[2], "v_hi": fm[3]} if not isinstance(fm, dict) else dict(fm)
        gait["phase_synthesis"] = ("integrate: phi += 2*pi*f(|cmd_vx|)*dt each tick; "
                                   "f = clip(f_min + (f_max-f_min)*(|cmd_vx|-v_lo)/(v_hi-v_lo), f_min, f_max); "
                                   "pin obs to stand_phase when |cmd| < stand_still_threshold")
    else:
        gait["gait_freq"] = float(gp.get("gait_freq", GAIT["gait_freq"]))
        gait["gait_freq_map"] = None
        gait["phase_synthesis"] = ("integrate: theta += dir * 2*pi*gait_freq*dt each tick (dir from clock_direction, +1 if absent); "
                                   "phi_L = wrap(theta), phi_R = wrap(theta + pi); pin obs to stand_phase when |cmd| < stand_still_threshold")
    if gp.get("signed"):
        from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import mdp_gait as _mg
        bt = float(inspect.signature(_mg._phase_signed).parameters["back_threshold"].default)
        gait["clock_direction"] = {"rule": "dir = -1 if cmd_vx < -back_threshold else +1 (COMMANDED vx)", "back_threshold": bt}
    if getattr(env_cfg.events, "stand_corridor", None) is not None:
        gait["stand_pin_anneal_s"] = 1.0  # gait_phase_obs anneal_s: phase glides to the pin over 1 s after the stand onset

    # ---- command envelope (trained ranges) and the stop path ----
    rg = env_cfg.commands.base_velocity.ranges
    command_envelope = {"vx": [float(rg.lin_vel_x[0]), float(rg.lin_vel_x[1])], "vy": [float(rg.lin_vel_y[0]), float(rg.lin_vel_y[1])],
                        "wz": [float(rg.ang_vel_z[0]), float(rg.ang_vel_z[1])],
                        "stand_when": f"norm(vx, vy, wz) < {gait['stand_still_threshold']}"}
    corr = getattr(env_cfg.events, "stand_corridor", None)
    if corr is not None:
        cp = dict(corr.params)
        command_envelope["stop_via"] = [float(cp.get("decel_speed", 0.12)), 0.0, 0.0]
        command_envelope["stop_via_s"] = float(cp.get("decel_s", 1.5))
        command_envelope["stop_note"] = "training enters every stand through stop_via for stop_via_s seconds, then (0,0,0)"

    meta = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "obs_dim": obs_dim,
        "frame_dim": frame_dim,
        "obs_history": obs_history,            # None for single-frame builds
        "action_dim": len(joint_names),
        "obs_terms": OBS_TERMS,                 # ONE frame
        "command_envelope": command_envelope,
        "train_env_vars": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KBOT_")},
        "trained_sensing": json.loads(args_cli.trained_sensing) if args_cli.trained_sensing else None,   # information only
        "joint_names": joint_names,                 # <-- Isaac DOF order (key!)
        "default_joint_pos": default_joint_pos,     # zero pose
        "kp": dict(zip(joint_names, kp)),
        "kd": dict(zip(joint_names, kd)),
        # RIG ASK 2026-08-20 (§5.2): JOINT_ORDER-aligned arrays so the bridge
        # can diff its /root/policy_gains.json against the checkpoint's truth
        # at deploy time — the inherited-gain bug survived months unchecked.
        # rig handoff #3B §4.1: stamp PLANT assumptions too, so the bridge can
        # verify compliance the same way it verifies gains.
        "series_k": {jn: float(getattr(env.unwrapped.scene["robot"].actuators.get(jn, None), "_series_k", 0.0))
                     for jn in joint_names},
        "series_b": {jn: float(getattr(env.unwrapped.scene["robot"].actuators.get(jn, None), "_series_b", 0.0))
                     for jn in joint_names},
        "jvel_lpf_hz": 4.0,
        "plant_ramp_warning": ("series_k/friction/play here are DEPLOY TRUTH (curricula nulled at "
                               "export). Training may use a ramp toward these values; a candidate "
                               "is only valid if its training ramp had reached level 1.0."),
        "kp_array": list(kp),
        "kd_array": list(kd),
        "armature": dict(zip(joint_names, armature)) if armature else None,
        "effort_limit": dict(zip(joint_names, effort_limit)) if effort_limit else None,
        "action_scale": 0.5,            # JointPositionActionCfg.scale
        "use_default_offset": True,     # target = scale*action + default
        "action_clip": action_clip,     # per-joint [lo, hi] clamp on the PD TARGET
        "action_clip_semantics": "clamp(raw_action*action_scale + default_joint_pos, lo, hi) per joint, radians — applied AFTER scale+offset; rig must replicate",
        "sim_dt": float(env.unwrapped.physics_dt),
        "decimation": int(env.unwrapped.cfg.decimation),
        "control_dt": float(env.unwrapped.step_dt),
        "imu_site": "imu_site",         # MJCF site to read gravity/ang-vel from
        "spawn_height": 1.05,
        "gait": gait,                   # gait clock params for reconstructing gait_phase obs (from the live term)
    }
    meta_path = os.path.join(export_dir, "legs_policy_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {meta_path}")
    print(f"[export] joint order: {joint_names}")
    print(f"[export] obs_dim={obs_dim} (frame {frame_dim} x history {H}) action_dim={len(joint_names)} ctrl_dt={meta['control_dt']}")
    print(f"[export] gait: {json.dumps(gait)}")
    print(f"[export] command envelope: {json.dumps(command_envelope)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
