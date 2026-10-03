"""Stage-2 feasibility probe (AMP_PLAN §5): can OUR actuators track a reference clip?

Base pinned (fix_root_link, raised) so the legs swing in the air; the clip's joint
positions are fed as PD targets through the live actuator model — kp/kd, command delay
buffer, T-V torque envelope, Coulomb/viscous friction and the ankle series-elastic
rotor — with the delay and the ankle stiffness K_s PINNED per env, every other
randomiser nulled (probe hygiene). Four envs = four (delay, K_s) corners.

What a pinned probe can and cannot say: tracking error and phase lag per joint are
real (they are set by delay, bandwidth and the T-V envelope); torques are the SWING-LEG
numbers only — nobody carries body weight here. Stance loads need the RL tracking run.

Usage:
  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_track_pinned.py \
      --clip eval_watch/amp_refs/asimov_walk_kbot.npz --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="eval_watch/amp_refs/asimov_walk_kbot.npz", help="npz in OUR joint order (amp_build_ref.py)")
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--configs", default="2:52,5:52,2:120,5:120", help="delay_steps:K_s per env")
parser.add_argument("--window", default="12:32", help="seconds of the clip to score")
parser.add_argument("--tag", default="amp_track_pinned_asimov")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

CONTINUOUS = {"_04": 7.5, "_03": 5.0, "_02": 5.0}  # Nm, memory: kbot-motor-ratings
OUT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    cfgs = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.configs.split(",")]
    n = len(cfgs)
    clip = np.load(args.clip, allow_pickle=True)
    fps = float(clip["fps"])
    clip_names = [str(s) for s in clip["joint_names"]]
    q_clip = clip["joint_pos"].astype(np.float32)
    lo, hi = (float(x) for x in args.window.split(":"))
    t0, t1 = int(lo * fps), int(hi * fps)

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    # pin the base
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    ip = list(env_cfg.scene.robot.init_state.pos)
    ip[2] += 0.30
    env_cfg.scene.robot.init_state.pos = tuple(ip)
    # probe hygiene: null every randomiser and curriculum; pushes off
    for name in ("add_limb_masses", "randomize_actuator_gains", "randomize_gains_small_joints", "randomize_gains_04_joints",
                 "randomize_joint_properties", "randomize_joint_friction_ankles", "randomize_joint_play", "push_robot",
                 "sustained_push", "walk_at_spawn", "stand_corridor", "physics_material", "reset_robot_joints", "reset_base"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in list(vars(env_cfg.curriculum)):
        if not name.startswith("_"):
            setattr(env_cfg.curriculum, name, None)
    env_cfg.episode_length_s = 1000.0
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.physics_dt
    dev = uenv.device
    with torch.inference_mode():
        env.reset()
    jn = robot.joint_names
    assert jn == clip_names, f"joint order mismatch:\n ours {jn}\n clip {clip_names}"
    J = len(jn)
    q_ref_all = torch.tensor(q_clip, device=dev)

    # pin delay + K_s per env (after one compute so the lazy per-env tensors exist)
    with torch.inference_mode():
        robot.set_joint_position_target(q_ref_all[t0:t0 + 1].expand(n, J))
        robot.write_joint_state_to_sim(q_ref_all[t0:t0 + 1].expand(n, J), torch.zeros(n, J, device=dev))
        robot.write_data_to_sim()
        uenv.sim.step(render=False)
        uenv.scene.update(dt)
        ankle_acts = []
        for act in robot.actuators.values():
            lags = torch.tensor([c[0] for c in cfgs], device=dev, dtype=torch.int)
            act.positions_delay_buffer.set_time_lag(lags, torch.arange(n, device=dev))
            if getattr(act, "_series_k", 0.0) > 0.0:
                if act._series_k_env is None:
                    act._series_k_env = torch.full((n, len(act.joint_indices)), act._series_k, device=dev)
                act._series_k_env[:] = torch.tensor([c[1] for c in cfgs], device=dev).unsqueeze(1)
                ankle_acts.append(act)
            act._tv_rand = 0.0  # probe hygiene: no per-step T-V envelope randomisation
        lag_check = [int(v) for v in next(iter(robot.actuators.values())).positions_delay_buffer.time_lags]
        print(f"[pin] delay steps per env {lag_check}; K_s per env {[c[1] for c in cfgs]}; ankle actuators {len(ankle_acts)}")

    # settle at the first frame's pose for 0.5 s so the rotor states line up
    with torch.inference_mode():
        for _ in range(100):
            robot.set_joint_position_target(q_ref_all[t0:t0 + 1].expand(n, J))
            robot.write_data_to_sim()
            uenv.sim.step(render=False)
            uenv.scene.update(dt)

    sub = int(round((1.0 / fps) / dt))  # physics steps per clip frame (4)
    T = t1 - t0
    rec_q = torch.zeros(T, n, J, device=dev)
    rec_tau = torch.zeros(T, n, J, device=dev)
    rec_tau_cmd = torch.zeros(T, n, J, device=dev)
    rec_lim = torch.zeros(T, n, J, device=dev)
    rec_qd = torch.zeros(T, n, J, device=dev)
    rec_defl = torch.zeros(T, n, 2, device=dev)
    ankle_idx = [i for i, name in enumerate(jn) if "ankle" in name]
    with torch.inference_mode():
        for k in range(T):
            tgt = q_ref_all[t0 + k:t0 + k + 1].expand(n, J)
            for _ in range(sub):
                robot.set_joint_position_target(tgt)
                robot.write_data_to_sim()
                uenv.sim.step(render=False)
                uenv.scene.update(dt)
            rec_q[k] = robot.data.joint_pos
            rec_tau[k] = robot.data.applied_torque
            rec_tau_cmd[k] = robot.data.computed_torque
            rec_qd[k] = robot.data.joint_vel
            for act in robot.actuators.values():
                lim = getattr(act, "tv_motoring_limit", None)
                if lim is not None:
                    rec_lim[k][:, act.joint_indices] = lim
            for a_i, act in enumerate(ankle_acts):
                if act.motor_pos is not None:
                    rec_defl[k, :, a_i] = (act.motor_pos - robot.data.joint_pos[:, act.joint_indices]).squeeze(-1)

    q = rec_q.cpu().numpy()
    tau = rec_tau.cpu().numpy()
    tau_cmd = rec_tau_cmd.cpu().numpy()
    lim = rec_lim.cpu().numpy()
    qd = rec_qd.cpu().numpy()
    defl = rec_defl.cpu().numpy()
    ref = q_clip[t0:t1]
    skip = int(2 * fps)  # drop the first 2 s
    results = {}
    print(f"\n{'config':14s} {'joint':26s} {'rmse°':>6s} {'lag ms':>7s} {'τrms':>6s} {'τpk':>6s} {'/cont':>6s} {'TVsat%':>7s}")
    for e, (dl, ks) in enumerate(cfgs):
        key = f"d{dl}_k{int(ks)}"
        per = {}
        for j in range(J):
            err = q[skip:, e, j] - ref[skip:, j]
            rmse = float(np.degrees(np.sqrt(np.mean(err ** 2))))
            # lag: shift of measured vs reference maximising correlation, +-150 ms
            a_ = ref[skip:, j] - ref[skip:, j].mean()
            b_ = q[skip:, e, j] - q[skip:, e, j].mean()
            best, best_lag = -1e9, 0
            for s in range(0, 8):
                c = float(np.dot(a_[: len(a_) - s], b_[s:])) if s else float(np.dot(a_, b_))
                if c > best:
                    best, best_lag = c, s
            lag_ms = best_lag * 1000.0 / fps
            trms = float(np.sqrt(np.mean(tau[skip:, e, j] ** 2)))
            tpk = float(np.abs(tau[skip:, e, j]).max())
            cont = CONTINUOUS[jn[j][-3:]]
            motoring = np.sign(tau[skip:, e, j]) == np.sign(qd[skip:, e, j])
            clipf = float(np.mean(motoring & (np.abs(tau[skip:, e, j]) >= 0.98 * lim[skip:, e, j])))
            per[jn[j]] = {"rmse_deg": rmse, "lag_ms": lag_ms, "tau_rms": trms, "tau_peak": tpk,
                          "rms_over_continuous": trms / cont, "tv_saturation_frac": clipf}
            print(f"{key:14s} {jn[j]:26s} {rmse:6.2f} {lag_ms:7.0f} {trms:6.2f} {tpk:6.2f} {trms / cont:6.2f} {100 * clipf:7.1f}")
        d = np.degrees(defl[skip:, e])
        per["ankle_series_deflection_deg"] = {"rms": float(np.sqrt(np.mean(d ** 2))), "peak": float(np.abs(d).max())}
        print(f"{key:14s} ankle series deflection: rms {per['ankle_series_deflection_deg']['rms']:.2f}° peak {per['ankle_series_deflection_deg']['peak']:.2f}°")
        results[key] = per
    with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
        json.dump({"clip": os.path.basename(args.clip), "window_s": [lo, hi], "configs": cfgs, "results": results}, f, indent=1)
    print(f"[out] {os.path.join(OUT, args.tag + '.json')}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
