#!/usr/bin/env python3
"""Build a rig-side meta for a WALKER run from its params/env.yaml + agent.yaml (pulled from the
training box) and the handoff §6 contract. Used only until training ships its own export; the
`provenance` field says so and the server prints it.

usage: build_run_meta.py <env.yaml> <agent.yaml> <exp> <run> <out.json>
"""
import sys, json, re, yaml, math
class L(yaml.SafeLoader): pass
def tup(loader, node): return tuple(loader.construct_sequence(node))
def anyc(loader, suffix, node):
    if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
    return loader.construct_scalar(node)
L.add_constructor("tag:yaml.org,2002:python/tuple", tup)
L.add_multi_constructor("tag:yaml.org,2002:python/", anyc)
L.add_multi_constructor("!", anyc)

env_p, agent_p, exp, run, out = sys.argv[1:6]
e = yaml.load(open(env_p), Loader=L); a = yaml.load(open(agent_p), Loader=L)

# joint order: the rig's established Isaac order for this robot asset (meta_l8 lineage)
JN = ["dof_left_hip_pitch_04", "dof_right_hip_pitch_04", "dof_left_hip_roll_04", "dof_right_hip_roll_04",
      "dof_left_hip_yaw_03", "dof_right_hip_yaw_03", "dof_left_knee_04", "dof_right_knee_04",
      "dof_left_ankle_02", "dof_right_ankle_02"]
acts = e["scene"]["robot"]["actuators"]
kp = {}; kd = {}; delays = set()
for name, ac in acts.items():
    for j in ac["joint_names_expr"]:
        kp[j] = float(ac["stiffness"][j] if isinstance(ac["stiffness"], dict) else ac["stiffness"])
        kd[j] = float(ac["damping"][j] if isinstance(ac["damping"], dict) else ac["damping"])
    delays.add((ac.get("min_delay"), ac.get("max_delay")))
assert set(kp) == set(JN), (set(kp) ^ set(JN))
ap = e["actions"]["joint_pos"]
assert ap["use_default_offset"] is True
init = e["scene"]["robot"]["init_state"]["joint_pos"]
default = [float(init.get(j, 0.0)) for j in JN]
clip = {j: [round(float(ap["clip"][j][0]), 6), round(float(ap["clip"][j][1]), 6)] for j in JN}

pol = e["observations"]["policy"]
assert pol["history_length"] == 10 and pol["flatten_history_dim"] is True and pol["concatenate_terms"] is True
terms = [k for k, v in pol.items() if isinstance(v, dict) and "func" in v]
assert terms == ["projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "imu_ang_vel", "actions", "gait_phase"], terms
jv = pol["joint_vel"]; assert jv["func"].endswith("joint_vel_rel_motorside_filtered")
lpf = float(jv["params"]["cutoff_hz"])
gp = pol["gait_phase"]["params"]
assert gp["signed"] is True and gp.get("freq_map") is None
f = float(gp["gait_freq"]); thr = float(gp["stand_still_threshold"])
rg = e["commands"]["base_velocity"]["ranges"]
assert a["empirical_normalization"] is False and a["policy"]["activation"] == "elu"
H = 10
names = ["imu_projected_gravity", "velocity_commands", "joint_pos_rel", "joint_vel_rel", "imu_ang_vel", "last_action", "gait_phase"]
dims = [3, 3, 10, 10, 3, 10, 4]
slices = []; off = 0
for n, d in zip(names, dims):
    slices.append({"name": n, "offset": off, "length": H * d}); off += H * d

meta = {
    "ckpt": None,   # filled per checkpoint by ckpt_manager (meta serves every iteration of the run)
    "run": "%s/%s" % (exp, run),
    "task": "Isaac-Velocity-Rough-KbotLegs-AMP-v0",
    "provenance": "RIG-RECONSTRUCTED 2026-10-02 from the run's params/env.yaml + agent.yaml and "
                  "RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md section 6. Not training's export. Replace when it ships.",
    "joint_names": JN,
    "kp_array": [kp[j] for j in JN], "kd_array": [kd[j] for j in JN],
    "kp": {j: kp[j] for j in JN}, "kd": {j: kd[j] for j in JN},
    "action_scale": float(ap["scale"]), "use_default_offset": True, "default_joint_pos": default,
    "action_clip": clip,
    "decimation": int(e["decimation"]), "sim_dt": float(e["sim"]["dt"]),
    "actuator_delay_steps": sorted(delays)[0] if len(delays) == 1 else sorted(delays),
    "obs_dim": H * 43, "frame_dim": 43, "action_dim": 10,
    "obs_history": {"length": H, "layout": "per_term_contiguous_oldest_first",
                    "startup": "fill every slot with the first frame", "term_slices": slices},
    "jvel_lpf_hz": lpf,
    "gait": {
        "gait_freq": f, "gait_freq_map": None,
        "clock_direction": {"rule": "dir = -1 if cmd_vx < -back_threshold else +1", "back_threshold": 0.05},
        "phase_synthesis": "integrate: theta += dir * 2*pi*gait_freq*dt each tick; pin obs to stand_phase when |cmd| < stand_still_threshold",
        "stand_still_threshold": thr,
        "start_phase": [0.0, math.pi], "stand_phase": [math.pi, math.pi], "stand_pin_anneal_s": 1.0,
        "start_hold_ticks": 2,
    },
    "command_envelope": {"vx": [float(rg["lin_vel_x"][0]), float(rg["lin_vel_x"][1])],
                         "vy": [float(rg["lin_vel_y"][0]), float(rg["lin_vel_y"][1])],
                         "wz": [float(rg["ang_vel_z"][0]), float(rg["ang_vel_z"][1])],
                         "stop_via": [0.12, 0.0, 0.0], "stop_via_s": 1.5},
}
json.dump(meta, open(out, "w"), indent=1)
print("wrote", out)
print("  kp", meta["kp_array"]); print("  kd", meta["kd_array"])
print("  scale", meta["action_scale"], "lpf", lpf, "f", round(f, 4), "thr", thr, "delay", meta["actuator_delay_steps"])
print("  envelope", meta["command_envelope"])
