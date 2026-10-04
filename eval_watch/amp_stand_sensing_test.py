"""Stand-stability test under the robot's SENSING PATH (rig RIG_HW_STAND_ADDENDUM_SENSING.md, 2026-10-04).

On the robot the policy did not see what it sees in training: the IMU delivered 20 new samples a second (a
repeated sample on 57 % of ticks) 20-70 ms old, the joint readings were ~14 ms late, and the policy server froze
for ~116 ms at tick 37 of every episode. walker_v5_3200 engaged cleanly, stood 1.9 s, then a ~2 Hz pitch rocking
grew until the watchdog cut it. This test asks whether the same happens in Isaac.

Every robot: at rest at the zero pose, stand command and hard phase pin from tick 0, authority 1.0, the plant AS
TRAINED (randomization and sensor noise on; --ankle rig pins the ankle to the rig's K_s 52 / play 2 deg). The
policy's 10-frame input is rebuilt here from the env's fresh frames through a sensing model per group:
  imu_hold_ms   the IMU yields a new sample every N ms (0 = every tick), random phase per robot
  imu_delay     IMU terms (projected gravity, gyro) are this many ticks old
  jnt_delay     joint position / velocity are this many ticks old
  stall         the previous action is held for 6 ticks from tick 37 (no inference, no frames pushed)
  rotor         ankle rotor stiction, Nm
A 5 Nm pitch moment for 0.25 s at --kick_at seconds probes the damping.
Reported per group: watchdog trips (tilt > 12 deg or joint speed > 8 rad/s) before the kick, and for the robots
still up the fast (> 1 Hz) torso pitch rms / peak-to-peak and dominant frequency over 2 s .. kick, ankle encoder
rms, ankle torque peak-to-peak, and the fast pitch rms in three windows after the kick (decays or sustains).
READ THE CONTROL ROW FIRST ("stock sensing"): it must stand quietly, or nothing else in the table means anything.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/amp_stand_sensing_test.py \
      --checkpoint <ckpt> --tag stand_sensing_v5 --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--per_group", type=int, default=32)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--kick_at", type=float, default=10.0)
parser.add_argument("--kick_nm", type=float, default=5.0)
parser.add_argument("--ankle", default="rig", choices=("rig", "trained"), help="rig = ankle spring 52 Nm/rad and play 2 deg on every robot; trained = as drawn by the training randomization")
parser.add_argument("--max_speed", type=float, default=8.0)
parser.add_argument("--max_tilt", type=float, default=12.0)
parser.add_argument("--rotor_in_watchdog", type=int, default=0, help="1 = the ankle ROTOR speed also counts against --max_speed (the rig's watchdog reads motor-side encoders)")
parser.add_argument("--tag", default="stand_sensing")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs.amp import KbotAmpRunner  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
# name, imu_hold_ms, imu_delay ticks, jnt_delay ticks, stall, rotor stiction Nm
GROUPS = [
    ("stock sensing (control)", 0, 0, 0, False, 0.0),
    ("IMU 20 Hz only", 50, 0, 0, False, 0.0),
    ("robot as measured: IMU 20 Hz + 1 tick, joints + 1 tick", 50, 1, 1, False, 0.0),
    ("as measured + 0.12 s stall at tick 37", 50, 1, 1, True, 0.0),
    ("as measured + ankle rotor stiction 1 Nm", 50, 1, 1, False, 1.0),
    ("as measured + stall + stiction 1 Nm", 50, 1, 1, True, 1.0),
    ("IMU 20 Hz + 2 ticks, joints + 1 tick", 50, 2, 1, False, 0.0),
    ("rig after its fix: fresh IMU + 1 tick, joints + 1 tick", 0, 1, 1, False, 0.0),
    ("after fix + ankle rotor stiction 1 Nm", 0, 1, 1, False, 1.0),
    ("fresh IMU + 2 ticks, joints + 1 tick", 0, 2, 1, False, 0.0),
    ("IMU + 2 ticks, joints + 2 ticks (40 ms, constant)", 0, 2, 2, False, 0.0),
    ("stock sensing + ankle rotor stiction 1 Nm", 0, 0, 0, False, 1.0),
]
FRAME = [3, 3, 10, 10, 3, 10, 4]  # projgrav, velcmd, jointpos, jointvel, imu gyro, actions, gaitphase
IMU_TERMS, JNT_TERMS = (0, 4), (2, 3)
MAXD = 3


def main() -> int:
    n = args.per_group * len(GROUPS)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 2.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0
    c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    for name in ("walk_at_spawn", "stand_corridor", "push_robot", "amp_unanswered", "amp_axis_bias"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
    ev.sustained_push.interval_range_s = (0.02, 0.02)
    # terminations: only the time limit — the watchdog is evaluated here, robots are not reset mid-test
    for name in list(vars(env_cfg.terminations)):
        if name != "time_out" and getattr(env_cfg.terminations, name, None) is not None and not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = KbotAmpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    dt = float(uenv.step_dt)
    om = uenv.observation_manager
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    H = dims[0] // FRAME[0]
    assert all(d == H * f for d, f in zip(dims, FRAME)), (dims, FRAME)
    offs = np.cumsum([0] + dims)
    jn = list(robot.joint_names)
    aL, aR = jn.index("dof_left_ankle_02"), jn.index("dof_right_ankle_02")
    actL, actR = robot.actuators["dof_left_ankle_02"], robot.actuators["dof_right_ankle_02"]
    grp = torch.arange(n, device=dev) // args.per_group
    col = lambda i, dtype=torch.float32: torch.tensor([g[i] for g in GROUPS], device=dev, dtype=dtype)[grp]
    hold_ms, imu_d, jnt_d, stall, rotor = col(1), col(2, torch.long), col(3, torch.long), col(4, torch.bool), col(5)
    phase_ms = torch.rand(n, device=dev) * 50.0

    def newest(obs):  # the fresh 43-value frame, per term
        return [obs[:, offs[i] + dims[i] - FRAME[i]:offs[i] + dims[i]].clone() for i in range(7)]

    def set_plant():
        for act in robot.actuators.values():
            shape = (n, len(act.joint_names))
            if getattr(act, "_series_k", 0.0) > 0.0:
                if args.ankle == "rig":
                    act._series_k_env = torch.full(shape, 52.0, device=dev)
                    act._play = torch.full(shape, math.radians(2.0), device=dev)
                act._rotor_fc = rotor.unsqueeze(1).expand(shape).clone()

    T = int(args.seconds / dt)
    k0, k1 = int(args.kick_at / dt), int((args.kick_at + 0.25) / dt)
    pitch = torch.zeros(T, n, device=dev); enc = torch.zeros(T, n, 2, device=dev); tau = torch.zeros(T, n, 2, device=dev)
    trip_at = torch.full((n,), float("inf"), device=dev)
    why = torch.zeros(n, 3, dtype=torch.bool, device=dev)                    # what tripped first: tilt, link speed, ankle rotor speed
    pk = torch.zeros(n, 3, device=dev)                                      # peaks over the first 2 s: tilt deg, link speed, rotor speed
    stack_err = 0.0
    kick = torch.zeros(n, 2, device=dev)
    with torch.inference_mode():
        env.step(torch.zeros(n, env.num_actions, device=dev))
        set_plant()
        obs, _ = env.reset()
        set_plant()
        uenv._standing_moment_override = kick
        fr = newest(obs)
        raw = [[f.clone() for f in fr] for _ in range(MAXD + 1)]            # raw[k] = fresh frame k ticks ago
        held = [fr[0].clone(), fr[4].clone()]                              # the IMU sample currently held by the sensor
        held_hist = [[h.clone() for h in held] for _ in range(MAXD + 1)]   # what the sensor output k ticks ago
        last_idx = torch.zeros(n, device=dev)
        # the policy's own stack: per term, H frames, oldest first, filled with the first sensed frame
        stack = [f.unsqueeze(1).repeat(1, H, 1) for f in fr]
        a_prev = torch.zeros(n, env.num_actions, device=dev)
        ar = torch.arange(n, device=dev)
        for t in range(T):
            # ---- sensing model -> the frame the policy sees this tick
            idx = torch.floor((t * dt * 1000.0 - phase_ms) / hold_ms.clamp(min=1.0))
            new = (hold_ms <= 0) | (idx != last_idx) | (t == 0)
            last_idx = idx
            held[0][new] = raw[0][0][new]; held[1][new] = raw[0][4][new]
            held_hist = [[h.clone() for h in held]] + held_hist[:-1]
            seen = [raw[0][i].clone() for i in range(7)]
            for d in range(MAXD + 1):
                m = imu_d == d
                seen[0][m] = held_hist[d][0][m]; seen[4][m] = held_hist[d][1][m]
                m = jnt_d == d
                seen[2][m] = raw[d][2][m]; seen[3][m] = raw[d][3][m]
            stalled = stall & (t >= 37) & (t < 43)
            push = ~stalled                                                # a frozen server pushes no frames
            for i in range(7):
                rolled = torch.cat([stack[i][:, 1:], seen[i].unsqueeze(1)], dim=1)
                stack[i][push] = rolled[push]
            x = torch.cat([s.reshape(n, -1) for s in stack], dim=1)
            ctl = grp == 0                                                  # control group: the rebuilt input must equal the env's own
            stack_err = max(stack_err, float((x[ctl] - obs[ctl]).abs().max()))
            a = policy(x)
            a = torch.where(stalled.unsqueeze(1), a_prev, a)
            a_prev = a
            kick[:, 1] = args.kick_nm if k0 <= t < k1 else 0.0
            obs, _, _, _ = env.step(a)
            fr = newest(obs)
            raw = [[f.clone() for f in fr]] + raw[:-1]
            g = robot.data.projected_gravity_b
            pitch[t] = torch.asin(g[:, 0].clamp(-1.0, 1.0)) * 180.0 / math.pi
            tilt = torch.asin(g[:, :2].norm(dim=1).clamp(max=1.0)) * 180.0 / math.pi
            enc[t, :, 0] = actL.motor_pos[:, 0]; enc[t, :, 1] = actR.motor_pos[:, 0]
            tau[t, :, 0] = actL.applied_effort[:, 0]; tau[t, :, 1] = actR.applied_effort[:, 0]
            link = robot.data.joint_vel.abs().amax(dim=1)
            rot = torch.maximum(actL.motor_vel[:, 0].abs(), actR.motor_vel[:, 0].abs())
            spd = torch.maximum(link, rot) if args.rotor_in_watchdog else link
            if t < int(2.0 / dt):
                pk = torch.maximum(pk, torch.stack([tilt, link, rot], dim=1))
            bad = ((tilt > args.max_tilt) | (spd > args.max_speed)) & torch.isinf(trip_at)
            trip_at[bad] = (t + 1) * dt
            why[bad] = torch.stack([tilt > args.max_tilt, link > args.max_speed, rot > args.max_speed], dim=1)[bad]
    P, E, TQ, trip = pitch.cpu().numpy(), np.degrees(enc.cpu().numpy()), tau.cpu().numpy(), trip_at.cpu().numpy()

    def fast(x):  # remove everything slower than ~1 Hz: subtract a 1 s centred moving average
        w = int(1.0 / dt)
        ker = np.ones(w) / w
        return x - np.apply_along_axis(lambda v: np.convolve(np.pad(v, (w // 2, w - 1 - w // 2), mode="edge"), ker, mode="valid"), 0, x)

    s0, s1 = int(2.0 / dt), k0
    res = {"checkpoint": args.checkpoint, "ankle": args.ankle, "limits": [args.max_tilt, args.max_speed], "groups": {}}
    print(f"\n[sensing] {args.checkpoint} | plant as trained, ankle: {args.ankle} | {args.per_group} robots per group | watchdog {args.max_tilt} deg / {args.max_speed} rad/s | kick {args.kick_nm} Nm x 0.25 s at {args.kick_at} s")
    print(f"{'group':56s} {'tripped <kick':>13s} {'median t':>8s} | {'pitch fast rms':>14s} {'p2p':>6s} {'freq':>6s} | {'ankle enc rms L/R':>18s} | {'ankle torque p2p L/R':>21s} | after kick: fast pitch rms 0-1 s / 1-2 s / 2-4 s")
    print(f"[sensing] control check: rebuilt policy input vs the env's own, max |diff| over the run = {stack_err:.2e} (must be 0)")
    WHY, PK = why.cpu().numpy(), pk.cpu().numpy()
    for gi, (name, *_r) in enumerate(GROUPS):
        m = (grp == gi).cpu().numpy()
        print(f"   {name[:54]:54s} first 2 s peaks (mean per robot): tilt {PK[m, 0].mean():5.1f} deg, link speed {PK[m, 1].mean():5.1f}, ankle rotor speed {PK[m, 2].mean():5.1f} rad/s | tripped by tilt {WHY[m, 0].mean():.2f} link {WHY[m, 1].mean():.2f} rotor {WHY[m, 2].mean():.2f}")
    for gi, (name, *_r) in enumerate(GROUPS):
        m = (grp == gi).cpu().numpy()
        tr = trip[m]
        up = np.isinf(tr) | (tr > args.kick_at)
        r = {"tripped_before_kick": float(1.0 - up.mean()), "trip_time_median_s": float(np.median(tr[~up])) if (~up).any() else None}
        txt = f"{name:56s} {r['tripped_before_kick']:13.2f} {('%.1f s' % r['trip_time_median_s']) if r['trip_time_median_s'] is not None else '   -':>8s} | "
        if up.any():
            pf = fast(P[:, m][:, up])
            seg = pf[s0:s1]
            spec = np.abs(np.fft.rfft(seg - seg.mean(0), axis=0)).mean(1)
            fr_ = np.fft.rfftfreq(seg.shape[0], dt)
            band = (fr_ >= 0.7) & (fr_ <= 6.0)
            ef = fast(E[:, m][:, up].reshape(T, -1)).reshape(T, -1, 2)[s0:s1]
            tq = TQ[:, m][:, up][s0:s1]
            stays = np.isinf(tr[up]) | (tr[up] > args.seconds)              # for the ring-down: robots that also survive the kick
            aft = lambda a, b: float(np.sqrt((pf[k1 + int(a / dt):k1 + int(b / dt)][:, stays] ** 2).mean())) if stays.any() else float("nan")
            r.update({"pitch_fast_rms_deg": float(np.sqrt((seg ** 2).mean())), "pitch_fast_p2p_deg": float((seg.max(0) - seg.min(0)).mean()),
                      "dominant_hz": float(fr_[band][spec[band].argmax()]), "ankle_enc_fast_rms_deg": [float(np.sqrt((ef[..., 0] ** 2).mean())), float(np.sqrt((ef[..., 1] ** 2).mean()))],
                      "ankle_torque_p2p_nm": [float((tq[..., 0].max(0) - tq[..., 0].min(0)).mean()), float((tq[..., 1].max(0) - tq[..., 1].min(0)).mean())],
                      "after_kick_fast_rms_deg": [aft(0.0, 1.0), aft(1.0, 2.0), aft(2.0, 3.7)], "tripped_after_kick": float(1.0 - stays.mean())})
            txt += (f"{r['pitch_fast_rms_deg']:11.2f} deg {r['pitch_fast_p2p_deg']:5.2f} {r['dominant_hz']:5.1f}Hz | {r['ankle_enc_fast_rms_deg'][0]:7.2f} /{r['ankle_enc_fast_rms_deg'][1]:5.2f} deg | "
                    f"{r['ankle_torque_p2p_nm'][0]:8.1f} /{r['ankle_torque_p2p_nm'][1]:5.1f} Nm     | {r['after_kick_fast_rms_deg'][0]:.2f} / {r['after_kick_fast_rms_deg'][1]:.2f} / {r['after_kick_fast_rms_deg'][2]:.2f} deg (tripped after kick {r['tripped_after_kick']:.2f})")
        else:
            txt += "every robot tripped before the kick"
        res["groups"][name] = r
        print(txt)
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
