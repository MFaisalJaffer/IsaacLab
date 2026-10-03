#!/usr/bin/env python3
"""Crafted metas for the walker contract tests. Starts from the current l8 meta so joint
order / gains / clip stay realistic, then edits only the contract fields.

  walker430.meta.json   valid 430 walker meta   -> golden smoke on walker_v5_model_400.pt
  bad_a.meta.json       same, but pointed at a 43-input ckpt -> FATAL in_dim
  bad_b.meta.json       bogus clock_direction rule           -> FATAL rule
  bad_c.meta.json       obs_dim 430 with no obs_history      -> FATAL history
  bad_d.meta.json       term_slices wrong order              -> FATAL slices
"""
import json, copy, sys
base = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "meta_l8_8000.json"))
H = 10
terms = [("imu_projected_gravity", 3), ("velocity_commands", 3), ("joint_pos_rel", 10),
         ("joint_vel_rel", 10), ("imu_ang_vel", 3), ("last_action", 10), ("gait_phase", 4)]
slices = []; off = 0
for n, d in terms:
    slices.append({"name": n, "offset": off, "length": H * d}); off += H * d

w = copy.deepcopy(base)
w.pop("gait", None)
w.update({
    "ckpt": "walker_v5_model_400.pt",
    "run": "kbot_legs_amp/2026-10-02_13-18-12_walker_v5",
    "obs_dim": 430, "frame_dim": 43,
    "obs_history": {"length": H, "layout": "per_term_contiguous_oldest_first",
                    "startup": "fill every slot with the first frame", "term_slices": slices},
    "gait": {"gait_freq": 0.9434, "phase_synthesis": "integrate theta += dir*2*pi*f*dt, no reset mid-walk",
             "clock_direction": {"rule": "dir = -1 if cmd_vx < -back_threshold else +1", "back_threshold": 0.05},
             "start_hold_ticks": 2, "stand_still_threshold": 0.1, "stand_phase": [3.141592653589793, 3.141592653589793],
             "stand_pin_anneal_s": 1.0},
    "command_envelope": {"vx": [-0.40, 0.45], "vy": [-0.13, 0.13], "wz": [-0.56, 0.56],
                         "stop_via": [0.12, 0.0, 0.0], "stop_via_s": 1.5},
    "jvel_lpf_hz": 4.0,
})
json.dump(w, open("walker430.meta.json", "w"), indent=1)

a = copy.deepcopy(w); a["ckpt"] = "kbot_legs_amp__2026-10-01_09-16-57__model_4200.pt"
json.dump(a, open("bad_a.meta.json", "w"), indent=1)

b = copy.deepcopy(w); b["gait"]["clock_direction"]["rule"] = "dir = sign(cmd_vx)"
json.dump(b, open("bad_b.meta.json", "w"), indent=1)

c = copy.deepcopy(w); c.pop("obs_history")
json.dump(c, open("bad_c.meta.json", "w"), indent=1)

d = copy.deepcopy(w); d["obs_history"]["term_slices"] = [slices[1], slices[0]] + slices[2:]
json.dump(d, open("bad_d.meta.json", "w"), indent=1)
print("wrote walker430 / bad_a / bad_b / bad_c / bad_d .meta.json")
