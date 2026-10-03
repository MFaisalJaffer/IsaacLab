"""Replay another robot's walking clip on the K-Bot legs as a kinematic puppet, render it.

Stage-1 tool of the AMP plan (eval_watch/AMP_PLAN.md §5): before a clip is ever shown to a
discriminator we (a) map its joints onto ours with signs derived from GEOMETRY, not guessed,
(b) play it through our kinematic chain so width / clearance / touchdown pitch / limits are
measured on OUR body, (c) render it so a human can judge the style.

The base is moved by stance-foot integration (the lower foot is assumed planted, the base
moves by minus that foot's displacement), so nothing slides; height follows the lowest
foot so the sole sits on the ground. No dynamics are involved: joint and root state are
written every frame and the sim is stepped once so the renderer sees them.

Sign mapping: the clip's own body quaternions tell us which base-frame axis each of its
joints rotates about and with what sign (done offline, passed in as --src_conv). The same
test is run on our robot here (+15 deg on each joint at the zero pose, read child-vs-parent
link rotation in the base frame). A joint maps with sign +1 if both agree, -1 otherwise, and
the script refuses if the axes differ.

Usage (service env vars for the flat world):
  KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 ./isaaclab.sh -p eval_watch/amp_replay_clip.py \
      --clip /path/clip.npz --headless --rendering_mode preview
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", required=True, help="npz with joint_pos, joint_vel, joint_names, body_quat_w, body_names, fps")
parser.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
parser.add_argument("--tag", default="amp_ref_asimov", help="output file prefix inside eval_watch/")
parser.add_argument("--segments", default="4:14,33:41", help="seconds to render, comma-separated a:b")
parser.add_argument("--strip_t0", type=float, default=6.0, help="filmstrip start (s), one stride from here")
parser.add_argument("--stride_s", type=float, default=1.0, help="stride period of the clip (s) for the filmstrip")
parser.add_argument("--render_every", type=int, default=2, help="render every Nth clip frame")
parser.add_argument("--time_scale", type=float, default=1.0, help="playback speed factor (1 = native)")
parser.add_argument("--no_render", action="store_true", help="metrics only")
parser.add_argument("--kbot_npz", action="store_true", help="clip is already in OUR joint order/signs (amp_lafan_to_kbot.py); has base_yaw")
parser.add_argument("--t_range", default="", help="seconds a:b of the clip to use (metrics + render), e.g. 26:31")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_render
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
FF = os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.11", "site-packages",
                  "imageio_ffmpeg", "binaries", "ffmpeg-linux-x86_64-v7.0.2")

# ---- source-clip conventions (Menlo Asimov-1 clip, derived from its body quaternions) ----
# joint -> (axis in pelvis frame, sign)   x = forward/roll axis, y = left/pitch axis, z = up/yaw axis
SRC_CONV = {
    "left":  {"hip_pitch": ("y", +1), "hip_roll": ("x", +1), "hip_yaw": ("z", -1), "knee": ("y", +1), "ankle": ("y", +1)},
    "right": {"hip_pitch": ("y", -1), "hip_roll": ("x", +1), "hip_yaw": ("z", -1), "knee": ("y", -1), "ankle": ("y", -1)},
}
SRC_JOINT = {"hip_pitch": "hip_pitch_joint", "hip_roll": "hip_roll_joint", "hip_yaw": "hip_yaw_joint",
             "knee": "knee_joint", "ankle": "ankle_pitch_joint"}
OUR_JOINT = {"hip_pitch": "hip_pitch_04", "hip_roll": "hip_roll_04", "hip_yaw": "hip_yaw_03",
             "knee": "knee_04", "ankle": "ankle_02"}
# our link chain, substrings unique per side (R/L): parent -> child of each joint
OUR_LINKS = {"hip_pitch": ("ROOT", "102{S}"), "hip_roll": ("102{S}", "RS03_{N}"), "hip_yaw": ("RS03_{N}", "301{S}"),
             "knee": ("301{S}", "401{S}"), "ankle": ("401{S}", "501{S}")}
SIDE_CODE = {"right": {"S": "R", "N": "4"}, "left": {"S": "L", "N": "5"}}
URDF_VEL_LIMIT = {"hip_pitch": 17.488, "hip_roll": 17.488, "hip_yaw": 18.849, "knee": 17.488, "ankle": 37.699}


def quat_to_R(q: torch.Tensor) -> torch.Tensor:  # wxyz (N,4) -> (N,3,3)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    return R


def rotvec(R: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)


def yaw_quat(psi: float) -> list[float]:
    return [math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]


def main() -> int:
    clip = np.load(args.clip, allow_pickle=True)
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    src_names = [str(s) for s in clip["joint_names"]]
    src_q = clip["joint_pos"].astype(np.float32)
    if args.kbot_npz:
        if "base_yaw" in clip:
            src_yaw = np.unwrap(clip["base_yaw"].astype(np.float64))
        elif "base_yaw_rate" in clip:  # cut clips (amp_build_clips) carry the rate only
            src_yaw = np.cumsum(clip["base_yaw_rate"].astype(np.float64)) / fps
        else:
            src_yaw = np.zeros(src_q.shape[0])
    else:
        src_bodies = [str(s) for s in clip["body_names"]]
        pelvis_quat = clip["body_quat_w"][:, src_bodies.index("pelvis_link")]  # wxyz
        src_yaw = np.unwrap(np.arctan2(2 * (pelvis_quat[:, 0] * pelvis_quat[:, 3] + pelvis_quat[:, 1] * pelvis_quat[:, 2]),
                                       1 - 2 * (pelvis_quat[:, 2] ** 2 + pelvis_quat[:, 3] ** 2)))
    if args.t_range:
        a_, b_ = (float(x) for x in args.t_range.split(":"))
        sl = slice(int(a_ * fps), int(b_ * fps))
        src_q, src_yaw = src_q[sl], src_yaw[sl]
    src_yaw = src_yaw - src_yaw[0]
    n_frames = src_q.shape[0]
    print(f"[clip] {os.path.basename(args.clip)}: {n_frames} frames @ {fps:.0f} Hz = {n_frames / fps:.1f} s")

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.resolution = (960, 540)
    env_cfg.viewer.eye = (0.15, -1.9, 0.25)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None if args.no_render else "rgb_array")
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.physics_dt
    dev = uenv.device
    with torch.inference_mode():
        env.reset()
    jn = robot.joint_names
    bn = robot.body_names
    J = {n: i for i, n in enumerate(jn)}
    print(f"[ours] joints: {jn}")
    print(f"[ours] bodies: {bn}")

    def body_idx(pat: str) -> int:
        if pat == "ROOT":
            return 0
        m = [i for i, n in enumerate(bn) if pat in n]
        assert len(m) == 1, f"link pattern {pat!r} matched {m} in {bn}"
        return m[0]

    zero = torch.zeros(1, len(jn), device=dev)
    root_hi = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=dev)
    zero6 = torch.zeros(1, 6, device=dev)

    def puppet(q: torch.Tensor, root: torch.Tensor):
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(q, zero)
        uenv.sim.step(render=False)
        uenv.scene.update(dt)

    # ---- 1. our sign conventions, from geometry ----
    # Rotate ONE joint by a small angle inside its limits and read how the child link's
    # orientation CHANGES from the zero pose (R_c(q) R_c(0)^T = rotation about the joint's
    # world axis). Comparing child to parent directly would include the fixed mounting
    # rotation between the two link frames and swamp the signal.
    conv_ours = {}
    lim_lo_t = robot.data.joint_pos_limits[0, :, 0]
    lim_hi_t = robot.data.joint_pos_limits[0, :, 1]
    with torch.inference_mode():
        puppet(zero, root_hi)
        Rc0_all = quat_to_R(robot.data.body_quat_w[0])
        for side in ("right", "left"):
            conv_ours[side] = {}
            for jt, (_par_pat, ch_pat) in OUR_LINKS.items():
                ci = body_idx(ch_pat.format(**SIDE_CODE[side]))
                ji = J[f"dof_{side}_{OUR_JOINT[jt]}"]
                delta = math.radians(10.0)
                if lim_hi_t[ji] < delta:  # positive direction not allowed -> test negative, undo below
                    delta = -delta
                q = zero.clone()
                q[0, ji] = delta
                puppet(q, root_hi)
                Rc = quat_to_R(robot.data.body_quat_w[0, ci:ci + 1])
                Rroot = quat_to_R(robot.data.root_quat_w[0:1])
                rv = rotvec(Rc @ Rc0_all[ci:ci + 1].transpose(1, 2))
                rv = (Rroot.transpose(1, 2) @ rv.unsqueeze(-1)).squeeze(-1)[0] / (delta / abs(delta))
                ax = int(torch.argmax(rv.abs()))
                mag = float(rv.norm())
                assert abs(mag - abs(delta)) < 0.02, f"{side} {jt}: rotvec magnitude {mag:.3f} != {abs(delta):.3f} (joint clamped?)"
                conv_ours[side][jt] = ("xyz"[ax], +1 if rv[ax] > 0 else -1)
                print(f"[ours] {side:5s} {jt:9s} +q -> rotvec(base)/|q| {(rv / mag).cpu().numpy().round(3)}  => {'xyz'[ax]} {'+' if rv[ax] > 0 else '-'}")
    sign = {}
    q_all = torch.zeros(n_frames, len(jn), device=dev)
    if args.kbot_npz:
        assert src_names == jn, f"kbot npz joint order {src_names} != robot {jn}"
        q_all[:] = torch.tensor(src_q, device=dev)
        sign = {("n/a", "already mapped"): 1}
        print("[map] clip already in our joint order/signs (--kbot_npz)")
    else:
        for side in ("right", "left"):
            for jt in OUR_JOINT:
                (ax_s, sg_s), (ax_o, sg_o) = SRC_CONV[side][jt], conv_ours[side][jt]
                if ax_s != ax_o:
                    raise SystemExit(f"axis mismatch on {side} {jt}: clip rotates about {ax_s}, ours about {ax_o}")
                sign[(side, jt)] = sg_s * sg_o
        print("[map] sign multipliers (clip -> ours):", {f"{s[0]}_{s[1]}": v for s, v in sign.items()})

        # ---- 2. build our joint trajectory ----
        for side in ("right", "left"):
            for jt in OUR_JOINT:
                src = src_q[:, src_names.index(f"{side}_{SRC_JOINT[jt]}")]
                q_all[:, J[f"dof_{side}_{OUR_JOINT[jt]}"]] = torch.tensor(src * sign[(side, jt)], device=dev)
    qd_all = torch.gradient(q_all, spacing=1.0 / fps, dim=0)[0]

    # ---- 3. standing reference: zero pose, lowered (root written each step) until the
    # feet touch -> ground height from the contact sensor, no free-fall, no balance needed ----
    sensor = uenv.scene.sensors["contact_forces"]
    s_feet = [i for i, n in enumerate(sensor.body_names) if n.endswith("FOOT")]
    assert len(s_feet) == 2, sensor.body_names
    with torch.inference_mode():
        fi = [body_idx("501R"), body_idx("501L")]
        puppet(zero, root_hi)
        foot_rel_z0 = float((robot.data.body_pos_w[0, fi, 2] - robot.data.root_pos_w[0, 2]).mean())
        R_f0 = quat_to_R(robot.data.body_quat_w[0, fi])  # (2,3,3) foot orientation at zero pose (base upright)
        fwd_local = (R_f0.transpose(1, 2) @ torch.tensor([1.0, 0.0, 0.0], device=dev).expand(2, 3).unsqueeze(-1)).squeeze(-1)
        h_stand = None
        z_try = -foot_rel_z0 + 0.06
        while z_try > -foot_rel_z0 - 0.06:
            root = torch.tensor([[0.0, 0.0, z_try, 1.0, 0.0, 0.0, 0.0]], device=dev)
            puppet(zero, root)
            f = sensor.data.net_forces_w[0, s_feet].norm(dim=-1)
            if float(f.max()) > 5.0:
                h_stand = z_try
                break
            z_try -= 0.001
        assert h_stand is not None, "feet never touched the ground while lowering the zero pose"
        print(f"[ours] standing base height {h_stand:.3f} m (contact), foot link {-foot_rel_z0:.3f} m below base, "
              f"sole {h_stand + foot_rel_z0:.3f} m below foot link")

    # ---- 4. pre-pass: feet relative to a fixed upright base ----
    p_rel = torch.zeros(n_frames, 2, 3, device=dev)
    pitch = torch.zeros(n_frames, 2, device=dev)
    with torch.inference_mode():
        for t in range(n_frames):
            puppet(q_all[t:t + 1], root_hi)
            p_rel[t] = robot.data.body_pos_w[0, fi] - robot.data.root_pos_w[0]
            Rf = quat_to_R(robot.data.body_quat_w[0, fi])
            fwd = (Rf @ fwd_local.unsqueeze(-1)).squeeze(-1)
            pitch[t] = torch.atan2(-fwd[:, 2], torch.hypot(fwd[:, 0], fwd[:, 1]))  # + = toe down
    p = p_rel.cpu().numpy()
    pitch = pitch.cpu().numpy()
    z = p[:, :, 2]
    stance = np.argmin(z, axis=1)
    base = np.zeros((n_frames, 3))
    base[:, 2] = h_stand + foot_rel_z0 - z.min(axis=1)
    for t in range(1, n_frames):
        s = stance[t]
        d = -(p[t, s, :2] - p[t - 1, s, :2])
        c, sn = math.cos(src_yaw[t]), math.sin(src_yaw[t])
        base[t, :2] = base[t - 1, :2] + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])

    # ---- metrics on our body ----
    ground = z.min(axis=1)
    both_down = (z[:, 0] - ground < 0.01) & (z[:, 1] - ground < 0.01)
    width = np.abs(p[:, 0, 1] - p[:, 1, 1])
    swing_h = np.maximum(z - ground[:, None], 0)
    v_base = np.linalg.norm(np.gradient(base[:, :2], 1.0 / fps, axis=0), axis=1)
    qn = q_all.cpu().numpy()
    qdn = qd_all.cpu().numpy()
    lim_lo = robot.data.joint_pos_limits[0, :, 0].cpu().numpy()
    lim_hi = robot.data.joint_pos_limits[0, :, 1].cpu().numpy()
    margin = np.minimum(qn - lim_lo, lim_hi - qn).min(axis=0)
    touchdown = []
    for f in range(2):
        down = z[:, f] - ground < 0.01
        td = np.where(down[1:] & ~down[:-1])[0] + 1
        touchdown += [float(np.degrees(pitch[t, f])) for t in td if t > 5]
    metrics = {
        "clip": os.path.basename(args.clip), "frames": int(n_frames), "fps": fps, "time_scale": args.time_scale,
        "sign_map": {f"{s[0]}_{s[1]}": int(v) for s, v in sign.items()}, "t_range": args.t_range,
        "stance_width_cm": {"median": float(np.median(width[both_down]) * 100), "p10": float(np.percentile(width[both_down], 10) * 100),
                            "p90": float(np.percentile(width[both_down], 90) * 100)},
        "swing_apex_clearance_cm": [float(swing_h[:, f].max() * 100) for f in range(2)],
        "implied_speed_mps": {"mean": float(v_base.mean()), "p10": float(np.percentile(v_base, 10)), "p90": float(np.percentile(v_base, 90))},
        "touchdown_foot_pitch_deg": {"median_abs": float(np.median(np.abs(touchdown))) if touchdown else None, "n": len(touchdown)},
        "joint_limit_margin_deg": {jn[i]: float(np.degrees(margin[i])) for i in range(len(jn))},
        "peak_joint_speed_frac_of_urdf_limit": {jn[i]: float(np.abs(qdn[:, i]).max() / URDF_VEL_LIMIT[next(k for k in URDF_VEL_LIMIT if OUR_JOINT[k] in jn[i])])
                                                for i in range(len(jn))},
        "hip_roll_range_deg": {jn[i]: [float(np.degrees(qn[:, i].min())), float(np.degrees(qn[:, i].max()))] for i in range(len(jn)) if "hip_roll" in jn[i]},
        "knee_range_deg": {jn[i]: [float(np.degrees(qn[:, i].min())), float(np.degrees(qn[:, i].max()))] for i in range(len(jn)) if "knee" in jn[i]},
        "ankle_range_deg": {jn[i]: [float(np.degrees(qn[:, i].min())), float(np.degrees(qn[:, i].max()))] for i in range(len(jn)) if "ankle" in jn[i]},
        "standing_base_height_m": h_stand,
    }
    with open(os.path.join(OUT, f"{args.tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    # per-frame data for downstream tools (stride segmentation, reference building)
    np.savez(os.path.join(OUT, f"{args.tag}_frames.npz"), joint_names=np.array(jn), fps=fps,
             joint_pos=qn, joint_vel=qdn, foot_rel=p, foot_pitch=pitch, base=base, base_yaw=src_yaw,
             stance=stance, h_stand=h_stand, foot_rel_z0=foot_rel_z0, sole_below_foot_link=h_stand + foot_rel_z0)
    print(json.dumps(metrics, indent=1))
    if args.no_render:
        env.close()
        return 0

    # ---- 5. render pass: two views per frame ----
    import imageio.v2 as imageio
    views = {"side": ((0.15, -2.15, 0.35), (0.0, 0.0, -0.3)), "rear34": ((-1.55, 1.25, 0.6), (0.0, 0.0, -0.35))}
    segs = []
    for s in args.segments.split(","):
        a, b = (float(x) for x in s.split(":"))
        segs.append((int(a * fps), min(int(b * fps), n_frames - 1)))
    frames = {k: [] for k in views}
    strip_frames = {k: [] for k in views}
    strip_idx = set(int(round(args.strip_t0 * fps + k * args.stride_s * fps / 8)) for k in range(8))
    t_render = time.time()

    def capture() -> np.ndarray:
        # env.render(recompute=True) deliberately SKIPS sim.render() (meant for RTX-sensor
        # setups), so drive the renderer ourselves: twice, so the camera move and the new
        # robot pose are both in the frame the annotator hands back.
        uenv.sim.render()
        uenv.sim.render()
        return uenv.render(recompute=True)

    with torch.inference_mode():
        # warm-up: the RTX pipeline returns empty buffers for its first few frames
        root = torch.tensor([[base[0, 0], base[0, 1], base[0, 2], *yaw_quat(src_yaw[0])]], device=dev, dtype=torch.float32)
        puppet(q_all[0:1], root)
        for _ in range(8):
            capture()
        for a, b in segs:
            for t in range(a, b):
                want_strip = t in strip_idx
                if (t - a) % args.render_every and not want_strip:
                    continue
                root = torch.tensor([[base[t, 0], base[t, 1], base[t, 2], *yaw_quat(src_yaw[t])]], device=dev, dtype=torch.float32)
                puppet(q_all[t:t + 1], root)
                for name, (eye, lookat) in views.items():
                    uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                    img = capture()
                    if (t - a) % args.render_every == 0:
                        frames[name].append(img)
                    if want_strip:
                        strip_frames[name].append(img)
    blank = [k for k, v in frames.items() if v and float(np.mean(v[len(v) // 2])) < 1.0]
    assert not blank, f"rendered frames are black for views {blank}"
    print(f"[render] {sum(len(v) for v in frames.values())} frames in {time.time() - t_render:.0f} s")
    out_fps = fps / args.render_every * args.time_scale
    for name in views:
        mp4 = os.path.join(OUT, f"{args.tag}_{name}.mp4")
        imageio.mimwrite(mp4, frames[name], fps=out_fps, codec="libx264", quality=7, macro_block_size=None)
        gif = os.path.join(OUT, f"{args.tag}_{name}.gif")
        subprocess.run([FF, "-y", "-i", mp4, "-vf", "fps=12,scale=620:-1", gif], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[out] {gif}  ({os.path.getsize(gif) / 1e6:.1f} MB)")
    rows = []
    for name in views:
        fr = strip_frames[name]
        if fr:
            h = 360
            tiles = []
            for img in fr:
                w = int(img.shape[1] * h / img.shape[0])
                ys = (np.arange(h) * img.shape[0] / h).astype(int)
                xs = (np.arange(w) * img.shape[1] / w).astype(int)
                tiles.append(img[ys][:, xs])
            rows.append(np.concatenate(tiles, axis=1))
    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
        strip = np.concatenate(rows, axis=0)
        png = os.path.join(OUT, f"{args.tag}_filmstrip.png")
        imageio.imwrite(png, strip)
        print(f"[out] {png}  one stride ({args.stride_s:.2f} s) from t={args.strip_t0:.1f} s, 8 tiles, rows = {list(views)}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
