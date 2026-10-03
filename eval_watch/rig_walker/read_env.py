import yaml, re, sys, json
class L(yaml.SafeLoader): pass
def tup(loader, node): return tuple(loader.construct_sequence(node))
def anyc(loader, suffix, node):
    if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
    return loader.construct_scalar(node)
L.add_constructor("tag:yaml.org,2002:python/tuple", tup)
L.add_multi_constructor("tag:yaml.org,2002:python/", anyc)
L.add_multi_constructor("!", anyc)
p = sys.argv[1] if len(sys.argv) > 1 else "env.yaml"
e = yaml.load(open(p), Loader=L)
def g(d, *ks, default="<missing>"):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d
print("decimation", e.get("decimation"), "sim.dt", g(e, "sim", "dt"), "episode_length_s", e.get("episode_length_s"))
act = g(e, "actions", "joint_pos")
if isinstance(act, dict):
    print("actions.joint_pos:", {k: act.get(k) for k in ("scale", "use_default_offset", "clip", "joint_names", "preserve_order", "class_type")})
print("clip_actions:", e.get("clip_actions"))
rob = g(e, "scene", "robot")
print("init joint_pos:", g(rob, "init_state", "joint_pos"))
for name, a in (g(rob, "actuators") or {}).items():
    print("actuator", name, "| names", a.get("joint_names_expr"), "| kp", a.get("stiffness"), "| kd", a.get("damping"),
          "| delay", a.get("min_delay"), a.get("max_delay"), "| class", a.get("class_type"))
pol = g(e, "observations", "policy")
print("policy group: history_length", pol.get("history_length"), "flatten", pol.get("flatten_history_dim"),
      "concat", pol.get("concatenate_terms"), "corruption", pol.get("enable_corruption"))
for k, v in pol.items():
    if isinstance(v, dict) and "func" in v:
        nz = v.get("noise"); nz = (nz.get("func"), {kk: vv for kk, vv in nz.items() if kk != "func"}) if isinstance(nz, dict) else nz
        print("  term %-26s func=%s scale=%s clip=%s hist=%s params=%s noise=%s" % (
            k, str(v.get("func")).split(":")[-1], v.get("scale"), v.get("clip"), v.get("history_length"),
            {kk: vv for kk, vv in (v.get("params") or {}).items() if kk != "asset_cfg"}, nz))
cmd = g(e, "commands", "base_velocity")
print("commands.base_velocity ranges:", g(cmd, "ranges"), "| rel_standing", cmd.get("rel_standing_envs"),
      "| heading", cmd.get("heading_command"), "| resampling", cmd.get("resampling_time_range"))
txt = open(p).read()
for m in re.finditer(r".*(lpf|low_pass|lowpass|filter|cutoff|gait|phase|clock|back_thresh).*", txt, flags=re.I):
    print("KEYLINE:", m.group(0).strip()[:170])
