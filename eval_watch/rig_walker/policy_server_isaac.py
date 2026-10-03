#!/usr/bin/env python3
"""HIL policy server for the ISAAC LAB legs policy (rsl_rl MLP).

Drop-in alternative to policy_server_gpu.py — same wire plumbing, different policy.
  emulator -> :9999 : 27 f64 = jpos(10,HIL), jvel(10,HIL), quat_wxyz(4,imu), gyro(3,imu)
  :9999 -> emulator : 24 f64 = action20 + base_quat(4)   [emulator DRAINS the action]
  mirror   -> :9998 : 20 f64 = pos_deltas(10,HIL) + vel_deltas(10,HIL)  [the REAL action]
  joystick :9992 (vx,vy,wz) , status beacon :9991

Isaac FRAME (43): imu_proj_grav(3,unit,imu) · vel_cmd(3) · joint_pos_rel(10,interleaved) ·
  joint_vel(10,interleaved) · imu_ang_vel(3) · last_action(10,interleaved) · gait_phase(4).
Action: q_target = default(0) + 0.5*a (interleaved -> HIL order); vel_des = 0.
HIL joint order is R-leg then L-leg; Isaac order is interleaved L/R -> remap both ways.

2026-10-02 (RIG_HANDOFF_HISTORY_SIGNED_CLOCK): the network input may be the last H
frames stacked PER TERM, oldest first (H=10 -> 430), and the gait clock may run
backward under a backward command. Both are driven by legs_policy_meta.json and
validated against hard contracts at startup — a bundle whose meta the rig cannot
honour refuses to run instead of running wrong. Bundles without those fields are
unchanged (replay-regression bit-exact, see REPLAY=).

Offline modes (no robot, no emulator):
  GAIT_SELFTEST=1          clock + history + shaper unit tests, then exit
  GOLDEN_VECTORS=x.npz     push the bundle's frames through OUR stacker + model, diff
  REPLAY=ep_*.npz          re-run a logged episode through the pipeline, diff obs/action
"""
import json, os, socket, struct, threading, time
import numpy as np

GAIT_FREQ_HZ = 1.4
ACTION_SCALE = 0.5
# clip_actions: training clamps actions at +/-8 from run 2026-07-12_19:00 onward
# (declared interface field). Clip BEFORE last_action obs write and the x0.5 scale,
# matching rsl_rl. Also a safety rail vs feedback runaway. ACTION_CLIP=0 disables
# (only for pre-clip checkpoints <= 111500).
ACTION_CLIP = float(os.environ.get("ACTION_CLIP", "8") or 0)
HOST, PORT = "0.0.0.0", 9999
OBS_FLOATS = 27
OBS_BYTES = OBS_FLOATS * 8
UNIT_G = np.array([0.0, 0.0, -1.0])
ISAAC_CKPT = os.environ.get("ISAAC_CKPT", "/root/model_8200.pt")
CTRL_DT = 0.02

# joint-order remap: HIL (emulator/mit_bridge) <-> Isaac (policy)
HIL_JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03",
              "dof_right_knee_04", "dof_right_ankle_02",
              "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03",
              "dof_left_knee_04", "dof_left_ankle_02"]
ISAAC_JOINTS = ["dof_left_hip_pitch_04", "dof_right_hip_pitch_04",
                "dof_left_hip_roll_04", "dof_right_hip_roll_04",
                "dof_left_hip_yaw_03", "dof_right_hip_yaw_03",
                "dof_left_knee_04", "dof_right_knee_04",
                "dof_left_ankle_02", "dof_right_ankle_02"]
HIL_TO_ISAAC = np.array([HIL_JOINTS.index(j) for j in ISAAC_JOINTS])  # obs_isaac = obs_hil[HIL_TO_ISAAC]
ISAAC_TO_HIL = np.array([ISAAC_JOINTS.index(j) for j in HIL_JOINTS])  # act_hil  = act_isaac[ISAAC_TO_HIL]

# One observation FRAME, in network order. The per-term history layout (handoff 2026-10-02)
# is derived from this table and checked against the meta's term_slices at startup.
FRAME_TERMS = (("imu_projected_gravity", 3), ("velocity_commands", 3), ("joint_pos_rel", 10),
               ("joint_vel_rel", 10), ("imu_ang_vel", 3), ("last_action", 10), ("gait_phase", 4))
FRAME_DIM = sum(d for _, d in FRAME_TERMS)          # 43
HIST_LAYOUT = "per_term_contiguous_oldest_first"

_cmd = os.environ.get("POLICY_CMD", "0,0,0").split(",")
CMD = (float(_cmd[0]), float(_cmd[1]), float(_cmd[2]))
CMD_PORT = int(os.environ.get("CMD_PORT", "9992"))
STATUS_PORT = int(os.environ.get("STATUS_PORT", "9991"))
IMU_SOURCE = os.environ.get("IMU_SOURCE", "sim").lower()

# ---- chatter injection (sim2real test; see 2026-08-22 incident notes) -------
# BUZZ="f_hz,vel_rms,joint_substr" adds a 15 Hz-style oscillation to the OBS
# jpos/jvel only (plant untouched). HIL order; R/L pairs get opposite phase.
BUZZ_CFG = None
_bz = os.environ.get("BUZZ", "")
if _bz:
    _f, _v, _j = _bz.split(",")
    _HILN = ["r_hipp", "r_hipr", "r_yaw", "r_knee", "r_ankl",
             "l_hipp", "l_hipr", "l_yaw", "l_knee", "l_ankl"]
    if _j.lower().strip() == "hw":
        # per-joint RMS profile measured on the real robot at the 2026-08-22
        # failure peak (CAN fb, t=332.2, HIL order); _v scales it (1.0=measured)
        _prof = [1.9, 2.8, 3.7, 2.1, 5.5, 1.2, 2.4, 4.8, 1.1, 3.1]
        _idx = list(range(10))
        _amp = [float(_v) * a for a in _prof]
    else:
        _idx = [i for i, n in enumerate(_HILN) if _j.lower().strip() in ("all", n)
                or _j.lower().strip() in n]
        _amp = [float(_v)] * 10
    BUZZ_CFG = (float(_f), _amp, _idx)
    print("[BUZZ] %s Hz, %s rad/s RMS on HIL joints %s (OBS-ONLY; plant clean)"
          % (_f, _v, _idx), flush=True)

JVEL_LPF_HZ = float(os.environ.get("JVEL_LPF_HZ", "0") or 0)   # resolved against the meta below
_jf = os.environ.get("JOINT_FLIP", "").strip()
JOINT_FLIP = [int(x) for x in _jf.split(",") if x.strip() != ""] if _jf else []
if JOINT_FLIP:
    print("[flip] negating HIL joint indices %s in obs AND action "
          "(axis-convention test)" % JOINT_FLIP, flush=True)
PSTAT = {"connected": False, "rate_hz": 0.0, "tick": 0, "imu_source": None, "cmd": [0., 0., 0.],
         "imu_off": [0., 0., 0.], "obs_dim": FRAME_DIM, "hist": 1, "clock": "legacy", "clock_dir": 1,
         "cmd_eff": [0., 0., 0.]}

# ---- IMU mounting-offset injection (sim2real test) -------------------------
# Models a physical IMU bolted on at a DIFFERENT angle than the policy assumes.
# The IMU reports vectors in its own (rotated) axes, so both the projected
# gravity and the gyro the policy sees are the true values expressed in the
# rotated frame: v_reported = R(rpy)^T @ v_true.  Physics, the emulator and the
# preflight fall-guard are untouched on purpose -- the ROBOT is not tilted, only
# its sense of "down" is wrong, which is exactly the mis-mount failure mode.
# ---- flight recorder --------------------------------------------------------
# One compressed npz per episode (a TCP connection = one episode) + a JSONL
# event journal (dashboard commands, preflight transitions — sent here by
# dash_server over UDP :9990 so everything lives in ONE place).
# Debrief with: docker exec kbot-zed python3 /root/kbot_log.py show latest
LOG_DIR = os.environ.get("POLICY_LOG_DIR", "/root/policy_logs")
EVT_PORT = int(os.environ.get("EVT_PORT", "9990"))
LOG_KEEP = int(os.environ.get("POLICY_LOG_KEEP", "40"))
ROW_COLS = ("t,dt_inf_ms,obs27[27](jpos10,jvel10,quat4,gyro3),"
            "obs43[43](pgrav3,cmd3,jpos10,jvel10,gyro3,last_act10,phase4),araw[10],aclip[10]")
os.makedirs(LOG_DIR, exist_ok=True)
_EVT_LOCK = threading.Lock()

# ---- CAN wire tap -----------------------------------------------------------
# Records what is ACTUALLY on the motor bus (post limit-clamp, post ramp),
# alongside the policy-intent rows: cmd frames (pos,vel,kp,kd,tff) and fb
# frames (pos,vel,tau) per node. Same code on vcan (HIL) and can0/1 (metal).
WIRE_IFACES = os.environ.get("WIRE_IFACES", "auto")
WIRE_MAX_ROWS = 400_000
_WIRE = {"buf": [], "on": False, "drops": 0, "lock": threading.Lock()}
_P_MIN, _P_MAX, _V_MIN, _V_MAX = -12.5, 12.5, -45.0, 45.0
_KP_MAX, _KD_MAX, _T_MIN, _T_MAX = 500.0, 5.0, -18.0, 18.0
_FBP, _FBV, _FBT = 12.5, 65.0, 50.0


def _u2f(x, lo, hi, b):
    return x * (hi - lo) / ((1 << b) - 1) + lo


def _wire_thread(iface):
    try:
        sk = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sk.bind((iface,))
    except OSError as e:
        print(f"[isaac] wire tap: {iface} unavailable ({e})", flush=True)
        return
    print(f"[isaac] wire tap on {iface}", flush=True)
    while True:
        try:
            frame = sk.recv(16)
        except OSError:
            time.sleep(0.5); continue
        if not _WIRE["on"] or len(frame) < 16:
            continue
        cid = struct.unpack("<I", frame[0:4])[0] & 0x1FFFFFFF
        dlc = frame[4]; d = frame[8:8 + dlc]
        nid, cmd = cid >> 5, cid & 0x1F
        if cmd != 0x008:
            continue
        now = time.time()
        with _WIRE["lock"]:
            if len(_WIRE["buf"]) >= WIRE_MAX_ROWS:
                _WIRE["drops"] += 1; continue
            if dlc >= 8:      # host->motor command
                p = _u2f((d[0] << 8) | d[1], _P_MIN, _P_MAX, 16)
                v = _u2f((d[2] << 4) | (d[3] >> 4), _V_MIN, _V_MAX, 12)
                kp = _u2f(((d[3] & 0xF) << 8) | d[4], 0, _KP_MAX, 12)
                kd = _u2f((d[5] << 4) | (d[6] >> 4), 0, _KD_MAX, 12)
                tf = _u2f(((d[6] & 0xF) << 8) | d[7], _T_MIN, _T_MAX, 12)
                _WIRE["buf"].append((now, nid, 0.0, p, v, kp, kd, tf))
            elif dlc >= 6:    # motor->host feedback
                p = _u2f((d[1] << 8) | d[2], -_FBP, _FBP, 16)
                v = _u2f((d[3] << 4) | (d[4] >> 4), -_FBV, _FBV, 12)
                tq = _u2f(((d[4] & 0xF) << 8) | d[5], -_FBT, _FBT, 12)
                _WIRE["buf"].append((now, nid, 1.0, p, v, tq, 0.0, 0.0))


def _start_wire_taps():
    ifaces = ["vcan0", "vcan1", "can0", "can1"] if WIRE_IFACES == "auto" else WIRE_IFACES.split(",")
    for i in ifaces:
        threading.Thread(target=_wire_thread, args=(i,), daemon=True).start()


def log_event(kind, **kw):
    rec = {"t": time.time(), "kind": kind}
    rec.update(kw)
    try:
        with _EVT_LOCK, open(os.path.join(LOG_DIR, "events.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _event_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", EVT_PORT))
    while True:
        try:
            data, _ = s.recvfrom(1024)
            rec = json.loads(data.decode())
            with _EVT_LOCK, open(os.path.join(LOG_DIR, "events.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass


def _write_episode(rows, ep_start, end_reason):
    if len(rows) < 10:
        return None
    try:
        import datetime
        stamp = datetime.datetime.fromtimestamp(ep_start).strftime("%Y%m%d_%H%M%S")
        with _WIRE["lock"]:
            wire = np.asarray(_WIRE["buf"]); _WIRE["buf"] = []
            drops = _WIRE["drops"]; _WIRE["drops"] = 0
        meta = {"ckpt": os.path.basename(ISAAC_CKPT), "action_clip": ACTION_CLIP,
                "imu_source": IMU_SOURCE, "imu_off": IMU_OFF, "ticks": len(rows),
                "ep_start": ep_start, "end_reason": end_reason, "cols": ROW_COLS,
                "wire_cols": "t,node,kind(0=cmd:pos,vel,kp,kd,tff / 1=fb:pos,vel,tau)",
                "wire_drops": drops,
                "gait_mode": CLOCK.mode, "gait_map": CLOCK.map,
                "obs_dim": PIPE.obs_dim, "hist_len": PIPE.hist.H,
                "cmd_shaping": SHAPER.on}
        path = os.path.join(LOG_DIR, f"ep_{stamp}_{len(rows)}t.npz")
        np.savez_compressed(path, data=np.asarray(rows), wire=wire, meta=json.dumps(meta))
        eps = sorted(f for f in os.listdir(LOG_DIR) if f.startswith("ep_") and f.endswith(".npz"))
        for old in eps[:-LOG_KEEP]:
            os.remove(os.path.join(LOG_DIR, old))
        print(f"[isaac] episode log -> {path}", flush=True)
        log_event("episode_saved", file=os.path.basename(path), ticks=len(rows), reason=end_reason)
        return path
    except Exception as e:
        print(f"[isaac] episode log FAILED: {e}", flush=True)
        return None


IMU_OFF = [0.0, 0.0, 0.0]          # roll, pitch, yaw in DEGREES
IMU_OFF_RT = None                  # R^T, or None when the offset is zero


def _set_imu_offset(roll_deg, pitch_deg, yaw_deg):
    global IMU_OFF, IMU_OFF_RT
    IMU_OFF = [float(roll_deg), float(pitch_deg), float(yaw_deg)]
    if max(abs(v) for v in IMU_OFF) < 1e-9:
        IMU_OFF_RT = None
        return
    r, p, y = (np.radians(v) for v in IMU_OFF)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    IMU_OFF_RT = (Rz @ Ry @ Rx).T


def _joystick_listener():
    global CMD
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", CMD_PORT))
    print(f"[isaac] joystick cmd listener udp :{CMD_PORT}", flush=True)
    while True:
        try:
            data, _ = s.recvfrom(64)
            if data.startswith(b"IMUOFF:"):
                r, p, y = struct.unpack("<3d", data[7:31])
                _set_imu_offset(r, p, y)
                print(f"[isaac] IMU OFFSET <- roll{r:+.1f} pitch{p:+.1f} yaw{y:+.1f} deg", flush=True)
                continue
            if len(data) >= 24:
                vx, vy, wz = struct.unpack("<3d", data[:24]); CMD = (vx, vy, wz)
                print(f"[isaac] CMD <- ({vx:+.2f},{vy:+.2f},{wz:+.2f})", flush=True)
        except Exception:
            pass


def _status_beacon():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            PSTAT["imu_source"] = IMU_SOURCE
            PSTAT["cmd"] = [round(CMD[0], 3), round(CMD[1], 3), round(CMD[2], 3)]
            PSTAT["imu_off"] = [round(v, 2) for v in IMU_OFF]
            s.sendto(json.dumps(PSTAT).encode(), ("127.0.0.1", STATUS_PORT))
        except Exception:
            pass
        time.sleep(0.2)


def rotate_vector_by_quat(vector, quat_wxyz, inverse=False, eps=1e-6):
    q = np.asarray(quat_wxyz, float); q = q / (np.linalg.norm(q) + eps); w, x, y, z = q
    if inverse: x, y, z = -x, -y, -z
    vx, vy, vz = vector
    xx = (w*w*vx + 2*y*w*vz - 2*z*w*vy + x*x*vx + 2*y*x*vy + 2*z*x*vz - z*z*vx - y*y*vx)
    yy = (2*x*y*vx + y*y*vy + 2*z*y*vz + 2*w*z*vx - z*z*vy + w*w*vy - 2*w*x*vz - x*x*vy)
    zz = (2*x*z*vx + 2*y*z*vy + z*z*vz - 2*w*y*vx + w*w*vz + 2*w*x*vy - y*y*vz - x*x*vz)
    return np.array([xx, yy, zz])


# Stand-pin convention: 07-09+ lineage trains with BOTH feet planted (pi, pi)
# (mirror-invariant; the old MJX-derived (pi/2, pi) read as "right foot rising"
# and drove stand fidget — see mdp_gait.py docstring). STAND_PIN=old reproduces
# the pre-07-09 convention for old checkpoints (<= 07-05 runs).
STAND_PIN = ((np.pi / 2.0, np.pi) if os.environ.get("STAND_PIN", "new").lower() == "old"
             else (np.pi, np.pi))


def _wrap_pi(a):
    """wrap to [-pi, pi) — shortest signed arc"""
    return ((a + np.pi) % (2.0 * np.pi)) - np.pi


def _fatal(msg):
    raise SystemExit("[isaac] FATAL: " + msg)


class GaitClock:
    """Gait-phase synthesizer (spec: HIL handoff 2026-08-05, frozen; signed mode 2026-10-02).

    legacy      : no meta -> fixed GAIT_FREQ_HZ, multiply form, STAND_PIN env.
    fixed-meta  : meta gait_freq set -> fixed at that frequency, multiply form (shipped bundles).
    adaptive    : meta gait_freq null + gait_freq_map -> f(|cmd_vx|) linear ramp,
                  phase INTEGRATED per control tick (Rule 2 — multiply form would
                  jump on speed change); theta free-runs, reset only on episode
                  start; stand pin overrides the OBSERVATION only (Rule 3).
    signed      : meta gait_freq set + gait.clock_direction -> fixed f, INTEGRATED,
                  dir = -1 while cmd_vx < -back_threshold else +1 (walker v4+).
                  Integrate-then-emit with theta held at 0 for the first
                  `start_hold_ticks` ticks (default 2) = training's
                  `episode_length_buf <= 1` zeroing, so traces are bit-comparable.
    Contracts (hard startup errors, never a silent fallback):
      gait_freq null WITHOUT a map; clock_direction present with an unknown rule or
      no back_threshold; clock_direction together with a gait_freq_map; a
      phase_synthesis string that does not say integrate."""

    def __init__(self, meta_gait):
        self.mode = "legacy"; self.f_fixed = GAIT_FREQ_HZ
        self.map = None; self.thr = 0.1; self.pin = STAND_PIN
        self.theta = 0.0; self.tick = 0; self.dir = 1.0
        self.back_thr = 0.05; self.hold_ticks = 0
        self.anneal_s = 0.0      # 0 = hard snap (pre-lineage-3 behaviour)
        self.stand_s = None      # None = walking; else elapsed stand fraction
        self.entry_phi = (0.0, np.pi)   # phase latched at stand entry
        if meta_gait is not None:
            self.thr = float(meta_gait.get("stand_still_threshold", 0.1))
            self.anneal_s = float(meta_gait.get("stand_pin_anneal_s", 0.0) or 0.0)
            sp = meta_gait.get("stand_phase")
            if sp:
                self.pin = (float(sp[0]), float(sp[1]))
            gf = meta_gait.get("gait_freq", "ABSENT")
            m = meta_gait.get("gait_freq_map")
            cd = meta_gait.get("clock_direction")
            if gf is None:
                if not m:
                    _fatal("meta gait_freq is null and no gait_freq_map — adaptive contract violated")
                if cd:
                    _fatal("meta has BOTH gait_freq_map and clock_direction — ambiguous clock, refusing")
                self.mode = "adaptive"
                self.map = {k: float(m[k]) for k in ("f_min", "f_max", "v_lo", "v_hi")}
            elif gf != "ABSENT":
                self.f_fixed = float(gf)
                if cd:
                    if m:
                        _fatal("meta has gait_freq AND gait_freq_map AND clock_direction — refusing")
                    rule = str(cd.get("rule", "")).replace(" ", "").lower()
                    if rule != "dir=-1ifcmd_vx<-back_thresholdelse+1":
                        _fatal("gait.clock_direction.rule %r is not implemented on the rig "
                               "(known: 'dir = -1 if cmd_vx < -back_threshold else +1')" % cd.get("rule"))
                    if not isinstance(cd.get("back_threshold"), (int, float)):
                        _fatal("gait.clock_direction.back_threshold missing or not a number")
                    self.back_thr = float(cd["back_threshold"])
                    ps = str(meta_gait.get("phase_synthesis", "integrate")).strip().lower()
                    if not ps.startswith("integrate"):
                        _fatal("gait.phase_synthesis %r but clock_direction requires integration" % ps)
                    self.hold_ticks = int(meta_gait.get("start_hold_ticks", 2))
                    # the rig integrates theta from 0 with right = theta + pi; a bundle that
                    # starts elsewhere is a contract the rig does not have
                    sp0 = meta_gait.get("start_phase")
                    if sp0 is not None:
                        d0 = (float(sp0[0]) - 0.0 + np.pi) % (2 * np.pi) - np.pi
                        d1 = (float(sp0[1]) - np.pi + np.pi) % (2 * np.pi) - np.pi
                        if abs(d0) > 1e-9 or abs(d1) > 1e-9:
                            _fatal("gait.start_phase %r not implemented on the rig (rig starts theta=0: "
                                   "left 0, right pi)" % (sp0,))
                    self.mode = "signed"
                else:
                    if str(meta_gait.get("phase_synthesis", "")).strip().lower().startswith("integrate"):
                        _fatal("gait.phase_synthesis says integrate but no clock_direction — "
                               "the rig's fixed clock is multiply-form; contract unclear, refusing")
                    self.mode = "fixed-meta"
            elif cd:
                _fatal("gait.clock_direction present but gait_freq absent")

    def reset(self):
        self.theta = 0.0
        self.tick = 0
        self.dir = 1.0
        self.stand_s = None
        self.entry_phi = (0.0, np.pi)
        # training (gait_phase_obs + stand_transition_corridor): the anneal is gated by a
        # stand ONSET that only a walk->stand corridor publishes; an env that is standing
        # with no onset (spawn, probes) gets s=1 = the HARD pin. And walk_at_spawn means
        # "stand is never assigned at spawn" — so a stand at episode start must be the hard
        # pin, never a sweep from start_phase. 2026-10-02: the sweep put a moving walking
        # clock in all 10 history slots under a zero command and walker_v5 ran away in 0.6 s.
        self.walked = False

    def freq(self, vx_cmd):
        """Rule 1: frequency from the COMMANDED forward speed (|vx| only)."""
        m = self.map
        s = (abs(float(vx_cmd)) - m["v_lo"]) / (m["v_hi"] - m["v_lo"])
        return min(max(m["f_min"] + (m["f_max"] - m["f_min"]) * s, m["f_min"]), m["f_max"])

    def obs(self, t, cmd, dt=CTRL_DT):
        if self.mode == "adaptive":
            x = self.theta                                       # emit at the CURRENT phase...
            self.theta += 2.0 * np.pi * self.freq(cmd[0]) * dt   # ...then advance for the next tick
            # (emit-then-advance so tick 0 is exactly start_phase [0, pi]; the
            #  integrator runs every tick, including while the stand pin is on)
        elif self.mode == "signed":
            # training (_phase_signed): integrate once per step from the COMMANDED vx sign,
            # then force phi=0 while episode_length_buf <= 1 -> emitted 0, 0, inc, 2inc, ...
            self.dir = -1.0 if float(cmd[0]) < -self.back_thr else 1.0
            self.theta += self.dir * 2.0 * np.pi * self.f_fixed * dt
            self.tick += 1
            if self.tick <= self.hold_ticks:
                self.theta = 0.0
            x = self.theta
        else:
            x = 2.0 * np.pi * self.f_fixed * t
        phi_l = ((x + np.pi) % (2 * np.pi)) - np.pi
        phi_r = (x % (2 * np.pi)) - np.pi
        if float(np.linalg.norm(cmd)) < self.thr:
            # Rule 3: observation-only pin. Entry from a WALK is continuous: blend the LIVE
            # phase to pi along the shortest arc over anneal_s, exactly gait_phase_obs
            # (pin = phi + s*wrap(pi - phi), phi re-read every tick; the integrator keeps
            # free-running underneath). A stand with no walk before it = hard pin (s=1).
            if self.anneal_s > 0.0 and self.walked:
                if self.stand_s is None:           # stand ENTRY (= corridor onset)
                    self.stand_s = 0.0
                else:
                    self.stand_s = min(1.0, self.stand_s + dt / self.anneal_s)
                sfrac = self.stand_s
                phi_l = _wrap_pi(phi_l + sfrac * _wrap_pi(self.pin[0] - phi_l))
                phi_r = _wrap_pi(phi_r + sfrac * _wrap_pi(self.pin[1] - phi_r))
            else:
                phi_l, phi_r = self.pin
        else:
            self.stand_s = None                    # walking: entry timer disarmed
            self.walked = True
        return np.array([np.cos(phi_l), np.sin(phi_l), np.cos(phi_r), np.sin(phi_r)])

    def force_established_stand(self):
        """Jump to s=1 (fully pinned). The training-side static-tilt probe runs at
        an established stand, so offline comparisons must too."""
        self.walked = True
        self.stand_s = 1.0


# ---- observation history (handoff 2026-10-02 §2) ----------------------------
def rig_term_slices(H):
    out, off = [], 0
    for name, d in FRAME_TERMS:
        out.append({"name": name, "offset": off, "length": H * d}); off += H * d
    return out, off


class ObsHistory:
    """Per-term ring of the last H frames, flattened PER TERM, oldest first:
         [grav(t-H+1..t) | cmd(..) | q(..) | qd(..) | gyro(..) | act(..) | phase(..)]
    Inside a block the newest frame is LAST. H=1 is exactly the single 43-frame
    (legacy path — verified bit-exact by REPLAY=). Startup / reset: every slot of
    every term holds the FIRST frame (IsaacLab CircularBuffer semantics), not zeros."""

    def __init__(self, H):
        self.H = int(H)
        if self.H < 1:
            _fatal("obs_history.length must be >= 1 (got %r)" % H)
        self.buf = None

    def reset(self):
        self.buf = None

    def push(self, parts):
        parts = [np.asarray(p, dtype="float32").reshape(-1) for p in parts]
        for (name, d), p in zip(FRAME_TERMS, parts):
            if p.shape[0] != d:
                _fatal("frame term %s has %d values, expected %d" % (name, p.shape[0], d))
        if self.buf is None:
            self.buf = [[p.copy() for _ in range(self.H)] for p in parts]
        else:
            for i, p in enumerate(parts):
                self.buf[i] = self.buf[i][1:] + [p.copy()]

    def vector(self):
        return np.concatenate([np.concatenate(b) for b in self.buf]).astype("float32")

    @property
    def dim(self):
        return self.H * FRAME_DIM


def _validate_history_meta(mj):
    """Return H (1 for legacy) after checking the obs_history contract. Hard errors
    for anything the rig cannot honour; nothing here is a soft fallback."""
    if not isinstance(mj, dict):
        return 1
    oh = mj.get("obs_history")
    od = mj.get("obs_dim")
    fd = mj.get("frame_dim")
    if fd is not None and int(fd) != FRAME_DIM:
        _fatal("meta frame_dim %r != rig frame %d" % (fd, FRAME_DIM))
    if oh is None:
        if od is not None and int(od) != FRAME_DIM:
            _fatal("meta obs_dim %r but no obs_history section — the rig feeds %d; refusing"
                   % (od, FRAME_DIM))
        return 1
    H = int(oh.get("length", 0))
    if H < 1:
        _fatal("obs_history.length %r invalid" % oh.get("length"))
    layout = str(oh.get("layout", "")).strip()
    if layout != HIST_LAYOUT:
        _fatal("obs_history.layout %r not implemented on the rig (known: %r)" % (layout, HIST_LAYOUT))
    su = str(oh.get("startup", "fill every slot with the first frame")).strip().lower()
    if "first frame" not in su:
        _fatal("obs_history.startup %r not implemented (rig fills every slot with the first frame)" % su)
    want, total = rig_term_slices(H)
    got = oh.get("term_slices")
    if got is not None:
        gl = [(str(s.get("name")), int(s.get("offset")), int(s.get("length"))) for s in got]
        wl = [(s["name"], s["offset"], s["length"]) for s in want]
        if gl != wl:
            _fatal("obs_history.term_slices disagree with the rig's frame layout:\n  meta: %s\n  rig : %s"
                   % (gl, wl))
    if od is not None and int(od) != total:
        _fatal("meta obs_dim %r != length %d x frame %d = %d" % (od, H, FRAME_DIM, total))
    return H


# ---- command envelope + stop-via shaping (handoff 2026-10-02 §4) --------------
class CommandShaper:
    """Turns the operator command into the command the policy sees.
      * clamp to the trained envelope (meta command_envelope), log when it bites
      * stop-via: training only ever entered a stand through ~1.5 s of slow forward
        walking (0.12,0,0). A jump from a real walk straight to zero was never seen.
        So when the operator drops to stand from a walk, hold `stop_via` for
        `stop_via_s`, THEN zero. Not applied when the previous command already was
        the via for long enough, nor when coming from a stand.
    Disabled when the meta has no command_envelope, or CMD_SHAPING=0."""

    def __init__(self, env_meta, thr):
        self.on = isinstance(env_meta, dict) and os.environ.get("CMD_SHAPING", "1") != "0"
        self.thr = float(thr)
        self.rng = None; self.via = None; self.via_s = 0.0
        self.via_until = None; self.prev_walk = False; self.via_since = None
        self._last_clip_msg = 0.0
        if self.on:
            try:
                self.rng = {k: (float(env_meta[k][0]), float(env_meta[k][1])) for k in ("vx", "vy", "wz")}
            except Exception:
                _fatal("command_envelope must give [lo, hi] for vx, vy, wz")
            sv = env_meta.get("stop_via")
            if sv is not None:
                self.via = (float(sv[0]), float(sv[1]), float(sv[2]))
                self.via_s = float(env_meta.get("stop_via_s", 1.5))

    def reset(self):
        self.via_until = None; self.prev_walk = False; self.via_since = None

    def step(self, raw, now):
        if not self.on:
            return tuple(float(v) for v in raw)
        vx = min(max(float(raw[0]), self.rng["vx"][0]), self.rng["vx"][1])
        vy = min(max(float(raw[1]), self.rng["vy"][0]), self.rng["vy"][1])
        wz = min(max(float(raw[2]), self.rng["wz"][0]), self.rng["wz"][1])
        if (vx, vy, wz) != tuple(float(v) for v in raw) and now - self._last_clip_msg > 1.0:
            self._last_clip_msg = now
            print("[cmd] clamped (%+.2f,%+.2f,%+.2f) -> (%+.2f,%+.2f,%+.2f) [trained envelope]"
                  % (raw[0], raw[1], raw[2], vx, vy, wz), flush=True)
        eff = (vx, vy, wz)
        standing = (vx * vx + vy * vy + wz * wz) ** 0.5 < self.thr
        if standing and self.via is not None:
            if self.via_until is None and self.prev_walk:
                long_enough_via = (self.via_since is not None and now - self.via_since >= self.via_s)
                if not long_enough_via:
                    self.via_until = now + self.via_s
                    print("[cmd] stop requested from a walk -> via %s for %.1f s, then zero"
                          % (self.via, self.via_s), flush=True)
            if self.via_until is not None and now < self.via_until:
                eff = self.via
            else:
                eff = (0.0, 0.0, 0.0)
        else:
            self.via_until = None
        # bookkeeping on the EFFECTIVE command
        eff_walk = (eff[0] ** 2 + eff[1] ** 2 + eff[2] ** 2) ** 0.5 >= self.thr
        is_via = (self.via is not None and max(abs(eff[i] - self.via[i]) for i in range(3)) < 1e-6)
        if is_via:
            if self.via_since is None:
                self.via_since = now
        else:
            self.via_since = None
        self.prev_walk = eff_walk and not (self.via_until is not None and now < self.via_until)
        return eff


# ---- meta -------------------------------------------------------------------
def _meta_candidates():
    cands = [os.environ.get("LEGS_META")]
    if ISAAC_CKPT.endswith(".pt") or ISAAC_CKPT.endswith(".onnx"):
        cands.append(os.path.splitext(ISAAC_CKPT)[0] + ".meta.json")
    cands.append(os.path.join(os.path.dirname(ISAAC_CKPT) or ".", "legs_policy_meta.json"))
    return [c for c in cands if c and os.path.exists(c)]


def _load_meta():
    cands = _meta_candidates()
    for cand in cands:
        with open(cand) as fh:
            mj = json.load(fh)
        if isinstance(mj, dict):
            print("[meta] %s" % cand, flush=True)
            if mj.get("provenance"):
                # rig-reconstructed metas (ckpt_manager ~/rig_metas) say so; training's export has no such field
                print("[meta] provenance: %s" % mj["provenance"], flush=True)
                PSTAT["meta_provenance"] = str(mj["provenance"])[:80]
            own = (cand == os.environ.get("LEGS_META")
                   or cand == os.path.splitext(ISAAC_CKPT)[0] + ".meta.json")
            if not own:
                # dirname fallback: /root/legs_policy_meta.json is whatever bundle last
                # dropped it there. Found 2026-10-02: an AMP checkpoint from 10-01 ran
                # for a day on a meta written 08-05. Loud, and refuse if the meta names
                # a different checkpoint.
                mck = mj.get("ckpt")
                if mck and os.path.basename(ISAAC_CKPT) not in (mck, mck + ".pt", os.path.splitext(mck)[0] + ".pt"):
                    _fatal("fallback meta %s is for checkpoint %r, not %s" % (cand, mck, os.path.basename(ISAAC_CKPT)))
                msg = ("FALLBACK META: %s has no %s — using %s (ckpt field: %r). Ship the bundle's "
                       "legs_policy_meta.json as <ckpt>.meta.json." % (os.path.basename(ISAAC_CKPT),
                       os.path.basename(os.path.splitext(ISAAC_CKPT)[0] + ".meta.json"), cand, mck))
                print("[meta] *** WARNING *** " + msg, flush=True)
                PSTAT["meta_warning"] = msg
            return mj
    print("[meta] none found (legacy bundle)", flush=True)
    return None


def _action_meta(mj):
    """Per-joint action clip + offsets from the meta, in HIL order.
    Returns (lo_hil, hi_hil, default_hil) or (None, None, None)."""
    ac = mj.get("action_clip") if isinstance(mj, dict) else None
    if not isinstance(ac, dict):
        print("[clip] no per-joint action_clip in meta -- blanket ACTION_CLIP only", flush=True)
        return None, None, None
    missing = [j for j in HIL_JOINTS if j not in ac]
    if missing:
        print("[clip] meta has action_clip but is MISSING %d joint(s): %s -- IGNORING"
              % (len(missing), missing[:3]), flush=True)
        return None, None, None
    lo = np.array([float(ac[j][0]) for j in HIL_JOINTS])
    hi = np.array([float(ac[j][1]) for j in HIL_JOINTS])
    dj = mj.get("default_joint_pos"); names = mj.get("joint_names")
    if isinstance(dj, list) and len(dj) == 10 and isinstance(names, list) and len(names) == 10:
        dmap = {n: float(v) for n, v in zip(names, dj)}
        dfl = np.array([dmap[j] for j in HIL_JOINTS])
    else:
        dfl = np.zeros(10)
    print("[clip] PER-JOINT action clip from meta (rad, after scale+offset); default_joint_pos %s"
          % ("nonzero" if np.any(dfl) else "all zero"), flush=True)
    return lo, hi, dfl


# ---- the per-tick pipeline ---------------------------------------------------
class Pipeline:
    """obs27 (HIL wire) + operator command -> network input -> action20 (HIL wire).
    Pure function of its inputs and `now` (the jvel LPF and the legacy clock use
    time), so a logged episode can be replayed bit-exactly (REPLAY=)."""

    def __init__(self, forward, clock, hist, shaper, pos_lo, pos_hi, pos_def):
        self.forward = forward; self.clock = clock; self.hist = hist; self.shaper = shaper
        self.pos_lo, self.pos_hi, self.pos_def = pos_lo, pos_hi, pos_def
        self.obs_dim = hist.dim
        self.reset(0.0)

    def reset(self, ep_start):
        self.ep_start = ep_start
        self.last_action = np.zeros(10, dtype="float32")
        self._lpf_v = None; self._lpf_t = None
        self.clock.reset(); self.hist.reset(); self.shaper.reset()

    def frame_parts(self, obs27, cmd_raw, now):
        obs = np.asarray(obs27, dtype="<f8")
        jpos = obs[0:10]; jvel = obs[10:20]; quat = obs[20:24]; gyro = obs[24:27]
        if JOINT_FLIP:
            jpos = jpos.copy(); jvel = jvel.copy()
            for _fi in JOINT_FLIP:
                jpos[_fi] = -jpos[_fi]; jvel[_fi] = -jvel[_fi]
        if BUZZ_CFG is not None:
            _bf, _bamp, _bidx = BUZZ_CFG
            _bw = 2.0 * np.pi * _bf
            _bt = now - self.ep_start
            jpos = jpos.copy(); jvel = jvel.copy()
            for _bi in _bidx:
                _bph = np.pi if _bi >= 5 else 0.0     # R/L anti-phase
                _ba = _bamp[_bi]
                jvel[_bi] += np.sqrt(2.0) * _ba * np.sin(_bw * _bt + _bph)
                jpos[_bi] += -(np.sqrt(2.0) * _ba / _bw) * np.cos(_bw * _bt + _bph)
        if JVEL_LPF_HZ > 0:
            if self._lpf_v is None:
                self._lpf_v = np.asarray(jvel, dtype=float).copy()
            else:
                _dt = max(1e-4, min(0.1, now - self._lpf_t))
                _al = _dt / (_dt + 1.0 / (2.0 * np.pi * JVEL_LPF_HZ))
                self._lpf_v = self._lpf_v + _al * (np.asarray(jvel, dtype=float) - self._lpf_v)
            self._lpf_t = now
            jvel = self._lpf_v
        pg = rotate_vector_by_quat(UNIT_G, quat, inverse=True)   # unit gravity in imu frame
        if IMU_OFF_RT is not None:          # mis-mounted IMU: policy's view only
            pg = IMU_OFF_RT @ pg
            gyro = IMU_OFF_RT @ np.asarray(gyro, dtype=float)
        cmd = self.shaper.step(cmd_raw, now)
        gp = self.clock.obs(now - self.ep_start, cmd)
        parts = [pg, np.asarray(cmd, dtype=float), jpos[HIL_TO_ISAAC], jvel[HIL_TO_ISAAC],
                 np.asarray(gyro, dtype=float), self.last_action, gp]
        return parts, quat, cmd, pg

    def step(self, obs27, cmd_raw, now):
        parts, quat, cmd, pg = self.frame_parts(obs27, cmd_raw, now)
        frame = np.concatenate(parts).astype("float32")
        self.hist.push(parts)
        x = self.hist.vector()
        t0 = time.time()
        a_raw = self.forward(x)                               # (10,) interleaved
        dt_inf_ms = (time.time() - t0) * 1000.0
        a = np.clip(a_raw, -ACTION_CLIP, ACTION_CLIP) if ACTION_CLIP > 0 else a_raw
        self.last_action = a.astype("float32")
        pos_hil = (ACTION_SCALE * a)[ISAAC_TO_HIL]            # -> HIL order
        if JOINT_FLIP:
            pos_hil = pos_hil.copy()
            for _fi in JOINT_FLIP:
                pos_hil[_fi] = -pos_hil[_fi]
        if self.pos_lo is not None:
            # training semantics: clamp(a*scale + default, lo, hi) per joint, rad
            pos_hil = np.clip(pos_hil + self.pos_def, self.pos_lo, self.pos_hi) - self.pos_def
        action20 = np.concatenate([pos_hil, np.zeros(10)]).astype("<f8")
        return {"x": x, "frame": frame, "a_raw": a_raw, "a": a, "action20": action20,
                "quat": quat, "cmd": cmd, "pg": pg, "dt_inf_ms": dt_inf_ms}


# ---- model ------------------------------------------------------------------
def load_model(expected_in):
    """rsl_rl actor MLP from a .pt state dict. Layer widths are read from the
    checkpoint (walker bundles have a 430-wide first layer); the input width must
    equal what the meta says the rig will feed — a mismatch is a HARD error, not a
    reshape. An observation normalizer in the checkpoint is also a hard error until
    the rig implements one (silently skipping it would be the worst outcome)."""
    if ISAAC_CKPT.endswith(".onnx"):
        try:
            import onnxruntime as ort
        except ImportError:
            _fatal("checkpoint is ONNX but onnxruntime is not installed in this container; "
                   "ship the .pt (model_state_dict) instead")
        sess = ort.InferenceSession(ISAAC_CKPT)
        inp = sess.get_inputs()[0]
        in_dim = int(inp.shape[-1])
        if in_dim != expected_in:
            _fatal("ONNX input width %d != rig obs_dim %d" % (in_dim, expected_in))

        def forward(x):
            return sess.run(None, {inp.name: x.reshape(1, -1).astype("float32")})[0].reshape(-1)
        print("[isaac] ONNX loaded (in=%d)" % in_dim, flush=True)
        return forward, "onnx"
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ISAAC_CKPT, map_location="cpu"); msd = sd.get("model_state_dict", sd)
    # rsl_rl's actor has `log_std` (policy noise) — that is NOT a normalizer. Flag only
    # EmpiricalNormalization-style buffers (obs_normalizer.*, actor_obs_normalizer.*, ...).
    norm_keys = [k for k in msd if "normaliz" in k.lower()
                 or ("obs" in k.lower() and (k.endswith("mean") or k.endswith("var") or k.endswith("std")))]
    if norm_keys:
        _fatal("checkpoint carries an observation normalizer (%s) which the rig does not apply; "
               "refusing to run un-normalized" % norm_keys[:4])
    layers = sorted(int(k.split(".")[1]) for k in msd if k.startswith("actor.") and k.endswith(".weight"))
    if not layers:
        _fatal("no actor.N.weight tensors in %s" % ISAAC_CKPT)
    mods = []
    for i, li in enumerate(layers):
        w = msd["actor.%d.weight" % li]
        mods.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(layers) - 1:
            mods.append(nn.ELU())
    net = nn.Sequential(*mods)
    net.load_state_dict({"%d.%s" % (2 * i, p): msd["actor.%d.%s" % (li, p)]
                         for i, li in enumerate(layers) for p in ("weight", "bias")})
    in_dim = int(msd["actor.%d.weight" % layers[0]].shape[1])
    out_dim = int(msd["actor.%d.weight" % layers[-1]].shape[0])
    if in_dim != expected_in:
        _fatal("checkpoint expects %d inputs but the rig will feed %d (meta obs_history length x %d). "
               "Wrong meta, wrong bundle, or a history contract the rig does not have"
               % (in_dim, expected_in, FRAME_DIM))
    if out_dim != 10:
        _fatal("actor output width %d != 10" % out_dim)
    net.eval().to(dev)

    @torch.no_grad()
    def forward(x):
        return net(torch.from_numpy(np.ascontiguousarray(x, dtype="float32")).to(dev)
                   .unsqueeze(0)).squeeze(0).cpu().numpy()

    for _ in range(5):
        forward(np.zeros(in_dim, dtype="float32"))
    widths = [int(msd["actor.%d.weight" % li].shape[0]) for li in layers]
    print("[isaac] MLP loaded on %s (ckpt=%s, in=%d, widths=%s)" % (dev, ISAAC_CKPT, in_dim, widths), flush=True)
    return forward, dev


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


# ---- self-test (no robot) ----------------------------------------------------
if os.environ.get("GAIT_SELFTEST"):
    _SPEC = {"gait_freq": None,
             "gait_freq_map": {"f_min": 0.9, "f_max": 1.4, "v_lo": 0.15, "v_hi": 0.45},
             "stand_still_threshold": 0.1, "stand_phase": [np.pi, np.pi]}
    c = GaitClock(_SPEC)
    for vx, want in ((0.0, 0.9), (0.15, 0.9), (0.30, 1.15), (0.45, 1.4), (0.60, 1.4), (-0.30, 1.15)):
        assert abs(c.freq(vx) - want) < 1e-9, ("freq", vx, c.freq(vx), want)
    c.reset(); prev = None; maxstep = 0.0
    for i in range(1500):                          # 0.15 -> 0.45 -> 0.15 ramp
        vx = 0.15 if i < 500 else (0.45 if i < 1000 else 0.15)
        c.obs(i * 0.02, np.array([vx, 0.0, 0.0]))
        if prev is not None:
            step = c.theta - prev
            assert step > 0.0, "theta must be monotone"
            maxstep = max(maxstep, step)
        prev = c.theta
    assert maxstep <= 2 * np.pi * 1.4 * 0.02 + 1e-9, ("phase jump", maxstep)
    c.reset(); o = c.obs(0.0, np.array([0.30, 0.0, 0.0]))
    assert np.allclose(o, [1.0, 0.0, -1.0, 0.0], atol=1e-12), ("start phase [0,pi]", o)
    c.reset(); o = c.obs(0.0, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(o, [-1.0, 0.0, -1.0, 0.0], atol=1e-12), ("stand pin", o)
    th0 = c.theta
    for _ in range(10):
        c.obs(0.0, np.array([0.0, 0.0, 0.0]))
    assert c.theta > th0, "integrator must free-run under the pin"
    try:
        GaitClock({"gait_freq": None}); raise AssertionError("null freq w/o map must hard-fail")
    except SystemExit:
        pass
    c = GaitClock(None)
    for t in (0.0, 0.123, 1.0, 7.77):
        base = 2.0 * np.pi * GAIT_FREQ_HZ * t
        pl = ((base + np.pi) % (2 * np.pi)) - np.pi
        pr = (base % (2 * np.pi)) - np.pi
        want = np.array([np.cos(pl), np.sin(pl), np.cos(pr), np.sin(pr)])
        assert np.allclose(c.obs(t, np.array([0.5, 0.0, 0.0])), want, atol=1e-15), ("legacy drift", t)
        assert np.allclose(c.obs(t, np.array([0.05, 0.0, 0.0])),
                           np.array([np.cos(STAND_PIN[0]), np.sin(STAND_PIN[0]),
                                     np.cos(STAND_PIN[1]), np.sin(STAND_PIN[1])]), atol=1e-15)
    PIN = np.array([-1.0, 0.0, -1.0, 0.0])
    c = GaitClock(dict(_SPEC, stand_pin_anneal_s=1.0))
    # (a) stand at EPISODE START = hard pin from tick 0 (training: no corridor onset -> s=1;
    #     walk_at_spawn never spawns a stand). The 2026-10-02 runaway was a sweep here.
    c.reset()
    for k in range(60):
        o = c.obs(0.0, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(o, PIN, atol=1e-9), ("stand at spawn must be the hard pin, tick %d: %s" % (k, o))
    # (b) walk -> stand: tick-by-tick equality with training's gait_phase_obs,
    #     pin = wrap(phi + s*wrap(pi - phi)) on the LIVE phi, s = (now - onset)/anneal_s clamped,
    #     onset = first stand tick. NOTE this is NOT smooth: the shortest arc to pi flips sign
    #     when the live phi crosses 0 mid-anneal (a ~pi jump in the pinned phase). Training has
    #     the same jump; the policy learned with it, so the rig reproduces it rather than smoothing.
    def _ref_pin(theta, s):
        pl = ((theta + np.pi) % (2 * np.pi)) - np.pi; pr = (theta % (2 * np.pi)) - np.pi
        pl = _wrap_pi(pl + s * _wrap_pi(np.pi - pl)); pr = _wrap_pi(pr + s * _wrap_pi(np.pi - pr))
        return np.array([np.cos(pl), np.sin(pl), np.cos(pr), np.sin(pr)])
    c.reset()
    walk = np.array([0.5, 0.0, 0.0])
    for _ in range(37):
        c.obs(0.0, walk)
    saw_jump = 0.0; prev = None
    for k in range(60):                                  # s = 0, 0.02, ... 1, 1, ...
        th_live = c.theta                                # adaptive clock emits the CURRENT theta
        o = c.obs(0.0, np.array([0.0, 0.0, 0.0]))
        s = min(1.0, k * CTRL_DT / 1.0)
        assert np.allclose(o, _ref_pin(th_live, s), atol=1e-9), ("anneal differs from gait_phase_obs at tick %d" % k)
        if k == 0: assert not np.allclose(o, PIN, atol=1e-3), "test must enter the stand away from the pin"
        if prev is not None: saw_jump = max(saw_jump, float(np.abs(o - prev).max()))
        prev = o
    assert np.allclose(o, PIN, atol=1e-9), ("anneal end must be the pin", o)
    assert saw_jump > 1.0, "this scenario crosses the antipode; the training-faithful jump must be present"
    # (c) stand -> walk -> stand again anneals again (onset re-published); established-stand shortcut
    c.obs(0.0, walk); c.obs(0.0, walk)
    o = c.obs(0.0, np.array([0.0, 0.0, 0.0]))
    assert not np.allclose(o, PIN, atol=1e-3), "re-entry from a walk must anneal, not snap"
    c.force_established_stand()
    assert np.allclose(c.obs(0.0, np.array([0.0, 0.0, 0.0])), PIN, atol=1e-9)
    # (d) anneal_s = 0 (pre-lineage-3 metas): hard pin always, unchanged
    c = GaitClock(dict(_SPEC)); c.reset(); c.obs(0.0, walk)
    assert np.allclose(c.obs(0.0, np.array([0.0, 0.0, 0.0])), PIN, atol=1e-9)
    print("GAIT CLOCK SELFTEST OK (spec table, integrator continuity, free-run pin, "
          "hard-fail contract, legacy bit-identical, stand-at-spawn hard pin, walk->stand live-phase anneal)", flush=True)

    # ---------------- signed clock (handoff 2026-10-02 §3) ----------------
    F = 0.9433962264150942
    _SG = {"gait_freq": F, "gait_freq_map": None,
           "clock_direction": {"rule": "dir = -1 if cmd_vx < -back_threshold else +1", "back_threshold": 0.05},
           "phase_synthesis": "integrate: theta += dir * 2*pi*gait_freq*dt each tick; pin obs to stand_phase when |cmd| < stand_still_threshold",
           "stand_still_threshold": 0.1, "start_phase": [0.0, np.pi], "stand_phase": [np.pi, np.pi],
           "stand_pin_anneal_s": 1.0}
    inc = 2 * np.pi * F * 0.02
    c = GaitClock(_SG); assert c.mode == "signed" and c.hold_ticks == 2, (c.mode, c.hold_ticks)
    c.reset()
    fwd = np.array([0.3, 0.0, 0.0])
    th = [c.obs(0, fwd) is not None and c.theta for _ in range(5)]
    assert np.allclose(th, [0.0, 0.0, inc, 2 * inc, 3 * inc], atol=1e-12), ("training trace 0,0,inc,2inc", th)
    # start phase at tick 0 is exactly [0, pi]
    c.reset(); o = c.obs(0, fwd)
    assert np.allclose(o, [1.0, 0.0, -1.0, 0.0], atol=1e-12), ("start phase", o)
    # backward: theta decreases; sideways / turn / mixed-forward: forward
    for cmdv, want in (([-0.3, 0, 0], -1), ([0.0, 0.13, 0], +1), ([0, 0, 0.5], +1),
                       ([-0.3, 0.13, 0.5], -1), ([-0.05, 0, 0], +1), ([-0.0501, 0, 0], -1), ([0.0, 0, 0], +1)):
        c.reset(); [c.obs(0, np.array(cmdv, float)) for _ in range(3)]
        assert c.dir == want, ("direction", cmdv, c.dir, want)
    c.reset(); [c.obs(0, np.array([-0.3, 0.0, 0.0])) for _ in range(5)]
    assert c.theta < 0.0 and abs(c.theta + 3 * inc) < 1e-12, ("backward theta", c.theta)
    # flip mid-walk: increment magnitude never exceeds one tick (no jump), and the
    # observed phase is continuous across the switch
    c.reset(); prev_th = None; prev_o = None; worst_o = 0.0
    for i in range(300):
        v = 0.3 if (i < 100 or i >= 200) else -0.3
        o = c.obs(0, np.array([v, 0.0, 0.0]))
        if prev_th is not None and i > 2:
            assert abs(abs(c.theta - prev_th) - inc) < 1e-12, ("flip jump", i, c.theta - prev_th)
            worst_o = max(worst_o, float(np.abs(o - prev_o).max()))
        prev_th = c.theta; prev_o = o
    assert worst_o <= inc + 1e-9, ("phase obs discontinuity at flip", worst_o)
    # cadence is the fixed f at every speed (no adaptive map)
    for v in (0.1, 0.3, 0.45, -0.4):
        c.reset(); [c.obs(0, np.array([v, 0.0, 0.0])) for _ in range(3)]
        assert abs(abs(c.theta) - inc) < 1e-12, ("cadence must be fixed", v, c.theta)
    # stand at spawn under the signed clock = hard pin from tick 0 (this is the GO case);
    # the integrator free-runs underneath (held at 0 for the first 2 ticks, then advancing)
    c.reset(); th0 = c.theta
    for k in range(60):
        o = c.obs(0, np.zeros(3))
        assert np.allclose(o, [-1.0, 0.0, -1.0, 0.0], atol=1e-9), ("GO stand must be the hard pin, tick %d" % k)
    assert c.theta > th0 + 50 * inc, "integrator must free-run under the pin"
    # walk -> stop: the shaper's 1.5 s corridor at 0.12 keeps the clock walking (0.12 > thr),
    # then the stand entry anneals the live phase to pi over 1 s
    c.reset()
    for _ in range(40): c.obs(0, np.array([0.3, 0.0, 0.0]))
    for _ in range(75): o = c.obs(0, np.array([0.12, 0.0, 0.0]))
    assert not np.allclose(o, [-1.0, 0.0, -1.0, 0.0], atol=1e-3), "corridor speed 0.12 must keep the clock live"
    o0 = c.obs(0, np.zeros(3))
    assert not np.allclose(o0, [-1.0, 0.0, -1.0, 0.0], atol=1e-3), "stand entry from a walk must start at the live phase"
    for _ in range(60): o = c.obs(0, np.zeros(3))
    assert np.allclose(o, [-1.0, 0.0, -1.0, 0.0], atol=1e-9), "anneal must end at the pin"
    # hard contracts
    for bad in (dict(_SG, gait_freq_map={"f_min": 0.9, "f_max": 1.4, "v_lo": 0.15, "v_hi": 0.45}),
                dict(_SG, clock_direction={"rule": "dir = sign(measured_vx)", "back_threshold": 0.05}),
                dict(_SG, clock_direction={"rule": "dir = -1 if cmd_vx < -back_threshold else +1"}),
                dict(_SG, phase_synthesis="multiply"),
                {"gait_freq": None, "gait_freq_map": _SPEC["gait_freq_map"], "clock_direction": _SG["clock_direction"]},
                {"gait_freq": 1.4, "phase_synthesis": "integrate ..."}):
        try:
            GaitClock(bad); raise AssertionError(("contract must hard-fail", bad))
        except SystemExit:
            pass
    # a plain fixed-meta bundle (gait_freq number, no clock_direction) is UNCHANGED multiply form
    c = GaitClock({"gait_freq": 1.25, "stand_still_threshold": 0.1, "stand_phase": [np.pi, np.pi]})
    assert c.mode == "fixed-meta"
    for t in (0.0, 0.37, 2.2):
        base = 2 * np.pi * 1.25 * t
        want = np.array([np.cos(((base + np.pi) % (2 * np.pi)) - np.pi), np.sin(((base + np.pi) % (2 * np.pi)) - np.pi),
                         np.cos((base % (2 * np.pi)) - np.pi), np.sin((base % (2 * np.pi)) - np.pi)])
        assert np.allclose(c.obs(t, np.array([0.3, 0, 0])), want, atol=1e-15)
    print("SIGNED CLOCK SELFTEST OK (training trace 0,0,inc,2inc; direction table incl. threshold edge; "
          "no jump on flip; fixed cadence; pin/anneal; 6 hard contracts; fixed-meta untouched)", flush=True)

    # ---------------- observation history (handoff 2026-10-02 §2) ----------------
    H = 10
    want, total = rig_term_slices(H)
    assert total == 430 and [(s["name"], s["offset"], s["length"]) for s in want] == [
        ("imu_projected_gravity", 0, 30), ("velocity_commands", 30, 30), ("joint_pos_rel", 60, 100),
        ("joint_vel_rel", 160, 100), ("imu_ang_vel", 260, 30), ("last_action", 290, 100), ("gait_phase", 390, 40)]
    h = ObsHistory(H)
    rng = np.random.RandomState(0)

    def mkframe(k):
        return [rng.randn(d).astype("float32") + k for _, d in FRAME_TERMS]
    f0 = mkframe(0.0); h.push(f0); x = h.vector()
    assert x.shape == (430,)
    for (name, d), s, p in zip(FRAME_TERMS, want, f0):
        blk = x[s["offset"]:s["offset"] + s["length"]].reshape(H, d)
        assert np.array_equal(blk, np.tile(p, (H, 1))), ("startup fill", name)
    frames = [f0] + [mkframe(float(k)) for k in range(1, 15)]
    for f in frames[1:]:
        h.push(f)
    x = h.vector()
    for (name, d), s in zip(FRAME_TERMS, want):
        blk = x[s["offset"]:s["offset"] + s["length"]].reshape(H, d)
        ti = FRAME_TERMS.index((name, d))
        exp = np.stack([frames[k][ti] for k in range(len(frames) - H, len(frames))])   # oldest first
        assert np.array_equal(blk, exp), ("order within block", name)
    # the CURRENT joint positions are input[150:160]
    assert np.array_equal(x[150:160], frames[-1][2]), "newest frame must be LAST in its block"
    # and this is NOT the frame-major layout
    fm = np.concatenate([np.concatenate(frames[k]) for k in range(len(frames) - H, len(frames))])
    assert not np.array_equal(x, fm), "per-term layout must differ from frame-major"
    # H=1 is exactly one frame in network order
    h1 = ObsHistory(1); h1.push(frames[3]); assert np.array_equal(h1.vector(), np.concatenate(frames[3]))
    # reset -> next push refills every slot
    h.reset(); h.push(frames[5]); x = h.vector()
    assert np.array_equal(x[60:160].reshape(H, 10), np.tile(frames[5][2], (H, 1)))
    # meta contract
    good = {"obs_dim": 430, "frame_dim": 43, "obs_history": {"length": 10, "layout": HIST_LAYOUT,
            "startup": "fill every slot with the first frame", "term_slices": want}}
    assert _validate_history_meta(good) == 10
    assert _validate_history_meta({"obs_dim": 43}) == 1 and _validate_history_meta(None) == 1
    for bad in (dict(good, obs_dim=43), dict(good, frame_dim=44),
                {"obs_dim": 430},                                                   # obs_dim without history
                dict(good, obs_history=dict(good["obs_history"], layout="frame_major")),
                dict(good, obs_history=dict(good["obs_history"], startup="zeros")),
                dict(good, obs_history=dict(good["obs_history"], term_slices=want[::-1])),
                dict(good, obs_history=dict(good["obs_history"], length=5))):
        try:
            _validate_history_meta(bad); raise AssertionError(("history contract must hard-fail", bad))
        except SystemExit:
            pass
    print("OBS HISTORY SELFTEST OK (slices 0/30/60/160/260/290/390 = 430, startup fill, oldest-first, "
          "newest last, not frame-major, H=1 == frame, reset refill, 7 hard contracts)", flush=True)

    # ---------------- command shaper (handoff 2026-10-02 §4) ----------------
    env = {"vx": [-0.40, 0.45], "vy": [-0.13, 0.13], "wz": [-0.56, 0.56], "stop_via": [0.12, 0.0, 0.0], "stop_via_s": 1.5}
    sh = CommandShaper(env, 0.1)
    assert sh.step((0.9, -0.5, 1.0), 0.0) == (0.45, -0.13, 0.56), "clamp to envelope"
    T = lambda k: k * 0.02                    # integer ticks -> no float accumulation at the boundary
    sh.reset()
    for k in range(50):                       # walk 1 s
        assert sh.step((0.3, 0, 0), T(k)) == (0.3, 0.0, 0.0)
    e = sh.step((0, 0, 0), T(50)); assert e == (0.12, 0.0, 0.0), ("stop from walk -> via", e)
    for k in range(51, 124):                  # t = 1.02 .. 2.46 < 1.0 + 1.5
        e = sh.step((0, 0, 0), T(k))
    assert e == (0.12, 0.0, 0.0), "via held for 1.5 s"
    e = sh.step((0, 0, 0), T(126)); assert e == (0.0, 0.0, 0.0), ("then zero (t=2.52)", e)
    sh.reset()                                # stand -> stand: no via
    assert sh.step((0, 0, 0), 0.0) == (0.0, 0.0, 0.0)
    sh.reset()                                # operator already did the via for 1.5 s: no extra via
    for k in range(80):
        sh.step((0.12, 0, 0), T(k))
    assert sh.step((0, 0, 0), T(80)) == (0.0, 0.0, 0.0), "operator-provided via honoured"
    sh.reset()                                # walk -> via (own) -> walk again: via dropped immediately
    for k in range(10):
        sh.step((0.3, 0, 0), T(k))
    sh.step((0, 0, 0), T(10))
    assert sh.step((0.3, 0, 0), T(11)) == (0.3, 0.0, 0.0)
    off = CommandShaper(None, 0.1); assert off.step((9, 9, 9), 0.0) == (9.0, 9.0, 9.0), "no meta -> passthrough"
    print("COMMAND SHAPER SELFTEST OK (envelope clamp, stop-via 1.5 s then zero, stand->stand, "
          "operator via honoured, walk resumes, passthrough without meta)", flush=True)
    raise SystemExit(0)


# ---- startup wiring -------------------------------------------------------------
META = _load_meta()
# jvel low-pass: the bundle's meta declares it (jvel_lpf_hz); the env var is an explicit
# override only. Found 2026-10-02: run_isaac_policy.sh never set JVEL_LPF_HZ, so a meta
# declaring 4 Hz ran UNFILTERED for a day — the 15 Hz chatter protection was off.
if os.environ.get("JVEL_LPF_HZ", "") != "":
    _lpf_src = "env override"
elif isinstance(META, dict) and META.get("jvel_lpf_hz") is not None:
    JVEL_LPF_HZ = float(META["jvel_lpf_hz"]); _lpf_src = "meta"
else:
    _lpf_src = "default: meta declares no jvel_lpf_hz"
print("[LPF] jvel low-pass %s (from %s)" % (("@ %.1f Hz" % JVEL_LPF_HZ) if JVEL_LPF_HZ > 0 else "OFF", _lpf_src), flush=True)
PSTAT["jvel_lpf_hz"] = JVEL_LPF_HZ
CLOCK = GaitClock(META.get("gait") if isinstance(META, dict) else None)
HIST = ObsHistory(_validate_history_meta(META))
SHAPER = CommandShaper(META.get("command_envelope") if isinstance(META, dict) else None, CLOCK.thr)
POS_LO, POS_HI, POS_DEF = _action_meta(META)
PSTAT["obs_dim"] = HIST.dim; PSTAT["hist"] = HIST.H; PSTAT["clock"] = CLOCK.mode
print("[gait] clock mode=%s %s%s" % (CLOCK.mode,
      CLOCK.map if CLOCK.map else "f=%.4f Hz" % CLOCK.f_fixed,
      "  back_threshold=%.3f hold_ticks=%d" % (CLOCK.back_thr, CLOCK.hold_ticks) if CLOCK.mode == "signed" else ""),
      flush=True)
print("[obs] input = %d (%d frame x %d history, layout %s)" % (HIST.dim, FRAME_DIM, HIST.H, HIST_LAYOUT), flush=True)
print("[cmd] shaping %s" % ("ON: envelope %s, stop_via %s for %.1f s" % (SHAPER.rng, SHAPER.via, SHAPER.via_s)
                           if SHAPER.on else "off (no command_envelope in meta)"), flush=True)
PIPE = None   # set in main()/offline modes once the model is loaded


# ---- offline mode: golden vectors (handoff §7.1) ------------------------------------
def _golden(path, forward):
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    arr2 = {k: z[k] for k in keys if hasattr(z[k], "ndim") and z[k].ndim == 2}
    frames = next((v for k, v in arr2.items() if v.shape[1] == FRAME_DIM), None)
    inputs = next((v for k, v in arr2.items() if v.shape[1] == HIST.dim and HIST.dim != FRAME_DIM), None)
    acts = next((v for k, v in arr2.items() if v.shape[1] == 10 and "act" in k.lower()), None)
    if acts is None:
        acts = next((v for k, v in arr2.items() if v.shape[1] == 10), None)
    print("[golden] %s keys=%s" % (path, {k: tuple(z[k].shape) for k in keys}), flush=True)
    if frames is None:
        _fatal("golden file has no (N,%d) frame array" % FRAME_DIM)
    h = ObsHistory(HIST.H); h.reset()
    worst_x = 0.0; worst_a_ours = 0.0; worst_a_theirs = 0.0; n_exact = 0
    for i in range(frames.shape[0]):
        f = frames[i].astype("float32")
        parts = []; off = 0
        for _, d in FRAME_TERMS:
            parts.append(f[off:off + d]); off += d
        h.push(parts); x = h.vector()
        if inputs is not None:
            dx = float(np.max(np.abs(x - inputs[i].astype("float32"))))
            worst_x = max(worst_x, dx); n_exact += int(np.array_equal(x, inputs[i].astype("float32")))
        if acts is not None:
            worst_a_ours = max(worst_a_ours, float(np.max(np.abs(forward(x) - acts[i]))))
            if inputs is not None:
                worst_a_theirs = max(worst_a_theirs, float(np.max(np.abs(forward(inputs[i].astype("float32")) - acts[i]))))
    print("[golden] stacker: %d/%d vectors bit-exact, worst |dx| = %.3e" % (n_exact, frames.shape[0], worst_x), flush=True)
    if acts is not None:
        print("[golden] model on OUR vectors  : worst |da| = %.3e  (%s 1e-4)" % (worst_a_ours, "PASS" if worst_a_ours <= 1e-4 else "FAIL"), flush=True)
        if inputs is not None:
            print("[golden] model on THEIR vectors: worst |da| = %.3e" % worst_a_theirs, flush=True)
    ok = (inputs is None or n_exact == frames.shape[0]) and (acts is None or worst_a_ours <= 1e-4)
    print("[golden] %s" % ("PASS" if ok else "FAIL"), flush=True)
    raise SystemExit(0 if ok else 1)


# ---- offline mode: replay a logged episode (regression) ----------------------------
def _replay(path, forward, pos):
    z = np.load(path, allow_pickle=True)
    rows = z["data"]; em = json.loads(str(z["meta"]))
    if em.get("ckpt") and os.path.basename(ISAAC_CKPT) != em["ckpt"]:
        print("[replay] WARNING: episode was recorded with %s, server has %s — actions will differ"
              % (em["ckpt"], os.path.basename(ISAAC_CKPT)), flush=True)
    t = rows[:, 0]; obs27 = rows[:, 2:29]; obs43 = rows[:, 29:72]; araw = rows[:, 72:82]
    pipe = Pipeline(forward, CLOCK, HIST, SHAPER, *pos)
    # the logged episode's own ep_start (connection time) — the legacy/fixed clocks are
    # wall-clock, so the phase at tick 0 depends on it, not on the first row's time
    pipe.reset(float(em.get("ep_start", t[0])))
    SHAPER.on = False     # the logged cmd is already the effective command; no re-shaping
    worst = {n: 0.0 for n, _ in FRAME_TERMS}; worst_a = 0.0
    for i in range(rows.shape[0]):
        cmd = tuple(float(v) for v in obs43[i, 3:6])
        r = pipe.step(obs27[i], cmd, float(t[i]))
        fr = r["frame"]; off = 0
        for n, d in FRAME_TERMS:
            worst[n] = max(worst[n], float(np.max(np.abs(fr[off:off + d] - obs43[i, off:off + d].astype("float32"))))); off += d
        worst_a = max(worst_a, float(np.max(np.abs(r["a_raw"] - araw[i]))))
        pipe.last_action = (np.clip(araw[i], -ACTION_CLIP, ACTION_CLIP) if ACTION_CLIP > 0 else araw[i]).astype("float32")
    # What CAN be bit-exact: gravity, cmd, jpos, gyro, last_action. The original server
    # stamped the jvel LPF and a wall-clock gait phase a few hundred us BEFORE the row's
    # timestamp (inference sat in between), so those two terms carry a timing residual
    # that is not a code difference: jvel ~1e-4 rad/s, phase up to 2pi*f*dt_inf ~ 1e-2.
    # Integrating clocks (adaptive/signed) have no wall-clock term. Their only residual is
    # that the episode logs cmd as float32 while the live server integrated the float64
    # joystick value: freq(cmd) differs by ~1e-8 Hz, which over a 4000-tick episode is
    # ~1e-6 rad of phase (measured 1.6e-6 on ep_20261002_042109). 1e-5 is 10x that and
    # still 3 orders below anything the policy could see.
    tol = {"joint_vel_rel": 1e-3 if JVEL_LPF_HZ > 0 else 0.0,
           "gait_phase": 2e-2 if CLOCK.mode in ("legacy", "fixed-meta") else 1e-5}
    print("[replay] %s: %d ticks (ckpt %s, clock %s, lpf %.1f Hz)"
          % (os.path.basename(path), rows.shape[0], em.get("ckpt"), CLOCK.mode, JVEL_LPF_HZ), flush=True)
    ok = True
    for n, _ in FRAME_TERMS:
        lim = tol.get(n, 0.0); good = worst[n] <= lim
        ok = ok and good
        print("[replay]   %-22s worst |d| = %.3e   %s%s" % (n, worst[n], "ok" if good else "DIFF",
              ("  (tolerance %.0e: %s)" % (lim, "float32-logged cmd" if n == "gait_phase" and lim < 1e-3
                                               else "timing residual, see note")) if lim > 0 else "  (must be exact)"), flush=True)
    print("[replay]   a_raw                  worst |d| = %.3e" % worst_a, flush=True)
    print("[replay] %s" % ("PASS" if ok else "FAIL"), flush=True)
    raise SystemExit(0 if ok else 1)


def main():
    global PIPE
    print(f"[isaac] loading model... cmd={CMD} IMU_SOURCE={IMU_SOURCE}", flush=True)
    forward, dev = load_model(HIST.dim)
    pos = (POS_LO, POS_HI, POS_DEF)
    if os.environ.get("GOLDEN_VECTORS"):
        _golden(os.environ["GOLDEN_VECTORS"], forward)
    if os.environ.get("REPLAY"):
        _replay(os.environ["REPLAY"], forward, pos)
    PIPE = Pipeline(forward, CLOCK, HIST, SHAPER, *pos)
    threading.Thread(target=_joystick_listener, daemon=True).start()
    threading.Thread(target=_status_beacon, daemon=True).start()
    threading.Thread(target=_event_listener, daemon=True).start()
    _start_wire_taps()
    mirror = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _mp = int(os.environ.get("MIRROR_PORT", "9998"))
    MIRROR = ("127.0.0.1", _mp)
    if _mp != 9998:
        print("[isaac] *** PREVIEW MODE *** actions -> :%d (mit_bridge is NOT listening; "
              "robot will not move)" % _mp, flush=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind((HOST, PORT)); srv.listen(1)
    print(f"[isaac] listening on {HOST}:{PORT} (action mirror -> {MIRROR})", flush=True)
    while True:
        conn, addr = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[isaac] client connected {addr}", flush=True); PSTAT["connected"] = True
        n = 0; rate_t = time.time(); ep_start = time.time()
        PIPE.reset(ep_start)
        ep_rows = []; end_reason = "disconnect"
        with _WIRE["lock"]: _WIRE["buf"] = []
        _WIRE["on"] = True
        log_event("episode_start", ckpt=os.path.basename(ISAAC_CKPT), imu_off=list(IMU_OFF),
                  obs_dim=HIST.dim, clock=CLOCK.mode)
        try:
            while True:
                raw = recv_exact(conn, OBS_BYTES)
                if raw is None:
                    break
                obs = np.frombuffer(raw, dtype="<f8")
                now = time.time()
                r = PIPE.step(obs, CMD, now)
                resp = np.concatenate([r["action20"], np.asarray(r["quat"], dtype="<f8")]).astype("<f8")
                conn.sendall(resp.tobytes())
                try:
                    mirror.sendto(r["action20"].tobytes(), MIRROR)
                except OSError:
                    pass
                ep_rows.append(np.concatenate([[now, r["dt_inf_ms"]], obs, r["frame"],
                                               np.asarray(r["a_raw"], float), np.asarray(r["a"], float)]))
                n += 1; PSTAT["tick"] = n; PSTAT["clock_dir"] = int(CLOCK.dir)
                PSTAT["cmd_eff"] = [round(v, 3) for v in r["cmd"]]
                if n % 50 == 0:
                    pg = r["pg"]; a = r["a"]
                    PSTAT["rate_hz"] = 50.0 / max(1e-3, now - rate_t); rate_t = now
                    print(f"[isaac] tick {n} pg=[{pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}] "
                          f"a[:5]=[{','.join(f'{v:+.2f}' for v in a[:5])}]"
                          + (f" dir={int(CLOCK.dir):+d} cmd_eff={PSTAT['cmd_eff']}" if CLOCK.mode == "signed" else ""),
                          flush=True)
        except (ConnectionResetError, BrokenPipeError):
            end_reason = "conn_reset"
        finally:
            _WIRE["on"] = False
            _write_episode(ep_rows, ep_start, end_reason)
            conn.close(); PSTAT["connected"] = False; PSTAT["rate_hz"] = 0.0
            print(f"[isaac] disconnected after {n} ticks", flush=True)


if __name__ == "__main__":
    main()
