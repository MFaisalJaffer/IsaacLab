#!/usr/bin/env python3
"""The handoff's §6 block VERBATIM (RIG_HANDOFF_HISTORY_SIGNED_CLOCK.md), merged onto the l8
meta's joint/gain/clip fields. Plus bad_e: start_phase that the rig does not implement."""
import json, copy, sys
base = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "meta_l8_8000.json"))
s6 = json.loads('''{
"task": "Isaac-Velocity-Rough-KbotLegs-AMP-v0",
"obs_dim": 430,
"frame_dim": 43,
"obs_history": {
  "length": 10,
  "layout": "per_term_contiguous_oldest_first",
  "startup": "fill every slot with the first frame",
  "term_slices": [
    {"name": "imu_projected_gravity", "offset": 0,   "length": 30},
    {"name": "velocity_commands",     "offset": 30,  "length": 30},
    {"name": "joint_pos_rel",         "offset": 60,  "length": 100},
    {"name": "joint_vel_rel",         "offset": 160, "length": 100},
    {"name": "imu_ang_vel",           "offset": 260, "length": 30},
    {"name": "last_action",           "offset": 290, "length": 100},
    {"name": "gait_phase",            "offset": 390, "length": 40}
  ]
},
"gait": {
  "gait_freq": 0.9433962264150942,
  "gait_freq_map": null,
  "clock_direction": {"rule": "dir = -1 if cmd_vx < -back_threshold else +1", "back_threshold": 0.05},
  "phase_synthesis": "integrate: theta += dir * 2*pi*gait_freq*dt each tick; pin obs to stand_phase when |cmd| < stand_still_threshold",
  "stand_still_threshold": 0.1,
  "start_phase": [0.0, 3.141592653589793],
  "stand_phase": [3.141592653589793, 3.141592653589793],
  "stand_pin_anneal_s": 1.0
},
"command_envelope": {"vx": [-0.40, 0.45], "vy": [-0.13, 0.13], "wz": [-0.56, 0.56], "stop_via": [0.12, 0.0, 0.0], "stop_via_s": 1.5}
}''')
m = copy.deepcopy(base); m.pop("gait", None); m.update(s6); m["ckpt"] = "walker_v5_model_400.pt"
json.dump(m, open("s6_verbatim.meta.json", "w"), indent=1)
e = copy.deepcopy(m); e["gait"]["start_phase"] = [1.5707963, 4.7123889]
json.dump(e, open("bad_e.meta.json", "w"), indent=1)
print("wrote s6_verbatim.meta.json, bad_e.meta.json")
