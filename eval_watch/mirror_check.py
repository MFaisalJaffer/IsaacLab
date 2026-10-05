"""Is symmetry.py's observation mirror the physical mirror?  A test in the simulator, no assumptions about frames.

Pairs of robots on one nominal plant (no randomization, no noise, equal actuator delays, zero command, zero start
pose). Robot A of a pair gets a smooth random action sequence; robot B gets the MIRRORED actions (swap left/right
and negate — the action mirror the training uses). If the robot model is mirror-symmetric, B's motion is the mirror
image of A's, so B's observation must equal mirror(A's observation). The test compares B's newest observation frame
with two candidate mirrors of A's:

  "symmetry.py"  what the mirror loss uses: gravity flips index 1, gyro flips indices 0 and 2
  "imu frame"    what the URDF's imu link says (imu x = down, y = backward, z = left): gravity flips index 2,
                 gyro flips indices 0 and 1

The joint terms (swap + negate in both candidates) are the control: they must match, otherwise the plant itself is
not mirror-symmetric and nothing can be concluded. Also prints the imu axes measured in the running simulator.

  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 <walker env vars> ./isaaclab.sh -p eval_watch/mirror_check.py --headless
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-AMP-v0")
parser.add_argument("--pairs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=1.2)
parser.add_argument("--amp", type=float, default=0.35, help="action amplitude (action units; x0.5 rad)")
parser.add_argument("--com", default="centre", choices=("centre", "zero"), help="torso CoM: centre of the training range, or no correction at all")
parser.add_argument("--tag", default="mirror_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot_legs import symmetry  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
TERMS = (("grav", 3), ("cmd", 3), ("jp", 10), ("jv", 10), ("imu", 3), ("act", 10), ("phase", 4))


def mirror_candidate(frame: torch.Tensor, which: str) -> torch.Tensor:
    """Mirror of one 43-value frame. 'module' = symmetry.py as it is now; 'legacy' = gravity flips y, gyro flips
    x and z (the map used until 2026-10-04); 'imu' = gravity flips z, gyro flips x and y."""
    m = symmetry._mirror_policy_obs(frame)          # joints, command, phase, last action: same in every candidate
    if which == "module":
        return m
    gs, ws = {"legacy": ([1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]), "imu": ([1.0, 1.0, -1.0], [-1.0, -1.0, 1.0])}[which]
    m[..., 0:3] = frame[..., 0:3] * torch.tensor(gs, device=frame.device)
    m[..., 26:29] = frame[..., 26:29] * torch.tensor(ws, device=frame.device)
    return m


def main() -> int:
    P = args.pairs
    n = 2 * P
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=n)
    env_cfg.episode_length_s = args.seconds + 5.0
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (0.0, 0.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 1.0; c.resampling_time_range = (1000.0, 1000.0)
    ev = env_cfg.events
    for name in ("walk_at_spawn", "stand_corridor", "push_robot", "amp_unanswered", "amp_axis_bias", "randomize_actuator_gains",
                 "randomize_gains_small_joints", "randomize_gains_04_joints", "add_limb_masses", "randomize_joint_properties",
                 "randomize_imu_mount", "randomize_joint_play", "randomize_joint_friction_ankles", "physics_material"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    for cu in ("sustained_push_level", "velocity_push_curriculum", "series_k_band", "plant_friction_level", "ankle_play_level", "series_stiffness_level"):
        if getattr(env_cfg.curriculum, cu, None) is not None:
            setattr(env_cfg.curriculum, cu, None)
    env_cfg.observations.policy.enable_corruption = False
    com_centre = None
    if getattr(ev, "correct_torso_com", None) is not None:
        cr = ev.correct_torso_com.params["com_range"]
        com_centre = {k: 0.5 * (v[0] + v[1]) for k, v in cr.items()}
        if args.com == "zero":
            ev.correct_torso_com = None
        else:
            ev.correct_torso_com.params["com_range"] = {k: (v, v) for k, v in com_centre.items()}
    ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    sp = ev.sustained_push.params
    sp["force_range"] = (0.0, 0.0); sp["hold_torque_range"] = (0.0, 0.0); sp["standing_moment"] = 0.0
    for a in env_cfg.scene.robot.actuators.values():                # one command latency for every joint of every robot
        if hasattr(a, "min_delay"):
            a.min_delay = 3; a.max_delay = 3
        if hasattr(a, "tv_randomization"):
            a.tv_randomization = 0.0
    for name in list(vars(env_cfg.terminations)):                   # no resets inside the window
        if name != "time_out" and getattr(env_cfg.terminations, name, None) is not None and not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    imu = uenv.scene["imu"]
    dev = uenv.device
    dt = float(uenv.step_dt)
    om = uenv.observation_manager
    dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    H = int(getattr(env_cfg.observations.policy, "history_length", 0) or 0) or 1
    fdims = [d // H for d in dims]
    assert fdims == [d for _, d in TERMS], fdims

    def newest(x: torch.Tensor) -> torch.Tensor:
        out, off = [], 0
        for d, f in zip(dims, fdims):
            out.append(x[:, off + d - f:off + d]); off += d
        return torch.cat(out, dim=1)

    T = int(args.seconds / dt)
    nact = uenv.action_manager.total_action_dim
    g = torch.Generator(device="cpu").manual_seed(7)
    raw = torch.randn(T + 40, P, nact, generator=g)
    k = torch.hann_window(21, periodic=False); k = k / k.sum()      # smooth: about 0.2 s
    sm = torch.nn.functional.conv1d(raw.permute(1, 2, 0).reshape(P * nact, 1, -1), k.view(1, 1, -1)).reshape(P, nact, -1).permute(2, 0, 1)[:T]
    sm = sm / sm.std() * args.amp
    ramp = torch.linspace(0.0, 1.0, T).clamp(max=0.3) / 0.3          # start from zero action
    aA = (sm * ramp.view(T, 1, 1)).to(dev)
    aB = symmetry._mirror_joints(aA)
    frames = torch.zeros(T, n, 43, device=dev)
    with torch.inference_mode():
        env.reset()
        env.step(torch.zeros(n, nact, device=dev))
        # the actuator model draws per-robot play and ankle stiffness on its first step: make every robot the same
        plays = []
        for act in robot.actuators.values():
            if getattr(act, "_play", None) is not None:
                plays.append(float(act._play.max())); act._play.zero_()
            if getattr(act, "_series_k_env", None) is not None:
                act._series_k_env.fill_(float(act._series_k))
        print(f"[mirror] per-robot play drawn by the actuator model (max over joints, deg): {np.degrees(max(plays)) if plays else 0.0:.2f} -> set to 0 for all")
        obs, _ = env.reset()
        # ---- the imu axes, measured in the running simulator at the start pose ----
        names = list(robot.body_names)
        pos = robot.data.body_pos_w[0]
        hipL = pos[names.index("KC_D_102L_L_Hip_Yoke_Drive")]; hipR = pos[names.index("KC_D_102R_R_Hip_Yoke_Drive")]
        left = (hipL - hipR); left[2] = 0.0; left = left / left.norm()
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        fwd = torch.linalg.cross(left, up)
        Rimu = math_utils.matrix_from_quat(imu.data.quat_w[0:1])[0]      # columns: imu axes in the world
        Rroot = math_utils.matrix_from_quat(robot.data.root_quat_w[0:1])[0]
        lab = ["forward", "left", "up"]
        def describe(Rm):
            out = []
            for kx, nx in enumerate("xyz"):
                v = torch.stack([Rm[:, kx] @ fwd, Rm[:, kx] @ left, Rm[:, kx] @ up])
                i = int(v.abs().argmax()); out.append(f"{nx} = {'+' if v[i] > 0 else '-'}{lab[i]} ({float(v[i]):+.3f})")
            return ", ".join(out)
        print(f"\n[mirror] root body '{names[0]}': {describe(Rroot)}")
        print(f"[mirror] imu sensor axes:        {describe(Rimu)}")
        print(f"[mirror] gravity seen by the policy at the start pose: {[round(float(v), 3) for v in newest(obs['policy'] if isinstance(obs, dict) else obs)[0, 0:3]]}")
        coms = robot.root_physx_view.get_coms()[0, names.index('Torso_Side_Right'), :3]
        print(f"[mirror] torso CoM in the torso body frame (m): {[round(float(v), 4) for v in coms]}  | training range centre {com_centre} | --com {args.com}")
        for t in range(T):
            a = torch.cat([aA[t], aB[t]], dim=0)
            o, _, _, _, _ = env.step(a)
            frames[t] = newest(o["policy"] if isinstance(o, dict) else o)
    A = frames[:, :P]; B = frames[:, P:]
    res = {"pairs": P, "seconds": args.seconds, "amp": args.amp, "com": args.com, "terms": {}}
    print(f"\n[mirror] {P} pairs, {args.seconds} s, action amplitude {args.amp} (rms, action units), torso CoM {args.com}")
    tilt = torch.rad2deg(torch.asin(A[:, :, 1:3].norm(dim=-1).clamp(max=1.0)))
    print(f"[mirror] how far the robots moved: tilt at the end {float(tilt[-1].mean()):.1f} deg mean, {float(tilt[-1].max()):.1f} max; gyro rms {float(A[:, :, 26:29].pow(2).mean().sqrt()):.2f} rad/s; joint angle rms {float(torch.rad2deg(A[:, :, 6:16].pow(2).mean().sqrt())):.1f} deg")
    print(f"[mirror] symmetry.py is set to the '{symmetry.IMU_MIRROR}' map")
    print(f"{'term':8s} | {'rms of the change':>18s} | rms of B - mirror(A):  {'legacy map':>12s} {'imu-frame map':>14s} {'symmetry.py now':>16s}")
    off = 0
    for name, d in TERMS:
        sl = slice(off, off + d); off += d
        sig = float((A[:, :, sl] - A[0:1, :, sl]).pow(2).mean().sqrt())          # how much the term moved from its start value
        e = {w: float((B - mirror_candidate(A, w))[:, :, sl].pow(2).mean().sqrt()) for w in ("legacy", "imu", "module")}
        res["terms"][name] = {"change_rms": sig, "err_legacy": e["legacy"], "err_imu_frame": e["imu"], "err_symmetry_py_now": e["module"]}
        print(f"{name:8s} | {sig:18.4f} | {'':22s} {e['legacy']:12.4f} {e['imu']:14.4f} {e['module']:16.4f}")
    print("\n[mirror] per component: correlation of B's value with A's value over time and pairs (the physical mirror gives +1 or -1)")
    comp = {}
    for name, base, labels in (("gravity", 0, ("x", "y", "z")), ("gyro", 26, ("x", "y", "z"))):
        row = []
        for kx in range(3):
            a_ = A[:, :, base + kx].flatten(); b_ = B[:, :, base + kx].flatten()
            a0 = a_ - a_.mean(); b0 = b_ - b_.mean()
            r = float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-12))
            row.append(r)
        comp[name] = row
        s1 = {"gravity": ("+", "-", "+"), "gyro": ("-", "+", "-")}[name]; s2 = {"gravity": ("+", "+", "-"), "gyro": ("-", "-", "+")}[name]
        print(f"  {name:8s} " + "  ".join(f"{labels[kx]}: {row[kx]:+.3f}" for kx in range(3)) + f"   | legacy map expects {' '.join(s1)} ; imu-frame map expects {' '.join(s2)}")
    res["component_correlation"] = comp
    json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
