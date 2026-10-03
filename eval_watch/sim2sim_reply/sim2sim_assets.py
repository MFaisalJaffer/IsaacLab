"""NOMINAL asset facts — every startup randomiser nulled.

Written after reading one env's DOMAIN-RANDOMISED draw as though it were the
asset: `add_base_mass` (+-5 kg) and `correct_torso_com` (+-15 mm fore-aft,
+-10 mm lateral) are startup events, so `get_masses()[0]` is a sample, not the
model. Reports BOTH: the nominal asset, and the randomisation band around it,
because the rig needs to know which one any given number came from.
"""
import argparse, functools, json, math, sys
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Rough-KbotLegs-v0")
p.add_argument("--com_center", action="store_true",
               help="keep correct_torso_com but pin it to the CENTRE of its band, i.e. the correction with the +-15 mm DR removed -- the number the rig should diff against.")
p.add_argument("--out", default="eval_watch/sim2sim_out/asset_facts_nominal.json")
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a); simulation_app = app.app
import gymnasium as gym, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
D2R = math.pi/180
_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
MJCF = {"dof_right_hip_pitch_04":(-127,60),"dof_right_hip_roll_04":(-130,12),
        "dof_right_hip_yaw_03":(-90,90),"dof_right_knee_04":(-155,0),"dof_right_ankle_02":(-13,72),
        "dof_left_hip_pitch_04":(-60,127),"dof_left_hip_roll_04":(-12,130),
        "dof_left_hip_yaw_03":(-90,90),"dof_left_knee_04":(0,155),"dof_left_ankle_02":(-72,13)}

cfg = parse_env_cfg(a.task, device=a.device, num_envs=2)
# NULL EVERY randomiser that perturbs the physical model. This is the control
# the first asset pass was missing.
killed, kept_ranges = [], {}
for name in list(vars(cfg.events)):
    term = getattr(cfg.events, name, None)
    if term is None or not hasattr(term, "func"): continue
    fn = getattr(term.func, "__name__", str(term.func))
    if name == "correct_torso_com" and a.com_center:
        c = term.params["com_range"]
        term.params["com_range"] = {"x": (0.0, 0.0),
                                    "y": (sum(c["y"])/2, sum(c["y"])/2), "z": (0.0, 0.0)}
        print(f"COM correction PINNED at band centre: y={sum(c['y'])/2:.5f} m")
        continue
    if any(k in fn or k in name for k in ("mass","com","material","gains","friction",
                                          "play","inertia","push","scale")):
        kept_ranges[name] = {k: str(v) for k, v in (term.params or {}).items()
                             if "range" in k or "params" in k}
        setattr(cfg.events, name, None); killed.append(f"{name} ({fn})")
for cu in ("sustained_push_level","velocity_push_curriculum","plant_friction_level",
           "ankle_play_level","series_stiffness_level"):
    if getattr(cfg.curriculum, cu, None) is not None: setattr(cfg.curriculum, cu, None)
cfg.episode_length_s = 1e4
print("NULLED:", *killed, sep="\n  ")

env = gym.make(a.task, cfg=cfg, render_mode=None); u = env.unwrapped; u.reset()
r = u.scene["robot"]; nm = list(r.data.joint_names); bn = list(r.data.body_names)
M = r.root_physx_view.get_masses()
f = {"randomisers_nulled": killed, "randomisation_bands_that_were_active": kept_ranges}
f["total_mass_kg"] = round(float(M[0].sum()), 4)
f["link_mass_kg"] = {b: round(float(m), 5) for b, m in zip(bn, M[0])}
f["mass_identical_across_envs"] = bool(torch.allclose(M[0], M[1]))
# L/R symmetry of the NOMINAL asset
pairs, asym = [], {}
for b in bn:
    if "_R_" in b or b.endswith("_R") or "R_R" in b:
        tw = b.replace("R_R","L_L").replace("_R_","_L_")
        if tw in bn: pairs.append((b, tw))
for rb, lb in pairs:
    mr, ml = float(M[0][bn.index(rb)]), float(M[0][bn.index(lb)])
    if abs(mr-ml) > 1e-4: asym[f"{rb} vs {lb}"] = [round(mr,5), round(ml,5), round(mr-ml,5)]
f["lr_mass_pairs_checked"] = len(pairs); f["lr_mass_asymmetries"] = asym
with torch.inference_mode():
    zq = torch.zeros(u.num_envs, len(nm), device=u.device)
    root = r.data.default_root_state.clone(); root[:, :3] += u.scene.env_origins
    for _ in range(300):
        r.write_root_pose_to_sim(root[:, :7]); r.write_root_velocity_to_sim(torch.zeros(u.num_envs,6,device=u.device))
        r.write_joint_state_to_sim(zq, zq)
        r.set_joint_position_target(zq); u.scene.write_data_to_sim(); u.sim.step(render=False)
        u.scene.update(u.sim.get_physics_dt())
    mm = M[0].to(u.device)
    # TRUE COM: each link's inertial COM sits at an OFFSET inside its frame, and
    # randomize_rigid_body_com moves exactly that offset. Averaging body_pos_w
    # (frame ORIGINS) ignores both -- it silently reported the same number with
    # the COM correction on and off, which is how the bug surfaced.
    coms = r.root_physx_view.get_coms().to(u.device)      # (envs, bodies, 7) pos+quat, body frame
    from isaaclab.utils.math import quat_apply
    off_w = quat_apply(r.data.body_quat_w[0], coms[0, :, :3])
    com_pos_w = r.data.body_pos_w[0] + off_w
    com_w = (com_pos_w * mm.unsqueeze(1)).sum(0) / mm.sum()
    f["link_com_offset_local_m"] = {b: [round(float(x), 5) for x in coms[0, i, :3]]
                                    for i, b in enumerate(bn)}
    ank = bn.index(_FEET[0]); rootp = r.data.root_pos_w[0]
    f["com_x_rel_root_m_zero_pose"] = round(float(com_w[0]-rootp[0]), 5)
    f["ankle_axis_x_rel_root_m_zero_pose"] = round(float(r.data.body_pos_w[0,ank,0]-rootp[0]), 5)
    f["com_minus_ankle_axis_x_mm_zero_pose"] = round(1000*float(com_w[0]-r.data.body_pos_w[0,ank,0]), 3)
lim = r.data.joint_limits[0]
f["joint_limits_deg"] = {j: [round(float(lim[nm.index(j),0])/D2R,2), round(float(lim[nm.index(j),1])/D2R,2)] for j in MJCF}
f["joint_limit_mismatch_vs_mjcf"] = {j: v for j, v in f["joint_limits_deg"].items()
                                     if abs(v[0]-MJCF[j][0])>1 or abs(v[1]-MJCF[j][1])>1}
print(json.dumps({k: v for k, v in f.items() if k not in ("randomisation_bands_that_were_active",)}, indent=1))
json.dump(f, open(a.out, "w"), indent=1)
print("wrote", a.out); env.close(); simulation_app.close()
