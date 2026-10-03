"""AMP task for the legs-only K-Bot (AMP_PLAN.md §1.1 variant P, §2.2).

On top of the live lineage package (env vars KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0):

  * observation group ``amp``: joint_pos_rel + joint_vel_rel, each with a 2-deep history
    (oldest first) -> the discriminator's transition pair; no noise, no deployment impact
    (the policy's own observation vector is unchanged);
  * observation group ``amp_gate``: commanded-to-move x body-velocity-tracking gate (pilot 0
    gated on the command alone and learned to stand still under a walk command);
  * the hand-made gait-SHAPE terms are retired (the style reward replaces them); task,
    safety, regularisers, the whole standing package and the push machinery stay;
  * the policy's gait clock runs at the dataset's stride frequency (env var KBOT_AMP_CLOCK);
  * commands can be narrowed to the dataset's speed band (KBOT_AMP_VX="lo:hi") for pilots
    on a single-speed dataset.

The dataset itself and the style weight are algorithm config (agents/rsl_rl_ppo_cfg.py).
"""
from __future__ import annotations

import os

import numpy as np

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import mdp_amp
from .rough_env_cfg import KBotLegsRoughEnvCfg

_IL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
AMP_REFS = os.path.join(_IL_ROOT, "eval_watch", "amp_refs")
DEFAULT_CYCLE_META = os.path.join(AMP_REFS, "asimov_walk_cycle.npz")

# Variant H (AMP_PLAN §1.1): feet_alternation is KEPT — pilots 0/0b showed that with every stepping
# carrot gone nothing pays for taking a step; alternation rewards contacts alternating, not shape.
_RETIRE_REWARDS = ("knee_swing", "feet_phase", "flight_phase", "feet_air_time",
                   "walk_hip_abduction", "joint_deviation_hip", "joint_deviation_ankles", "joint_deviation_hip_pitch_knee")
_RETIRE_CURRICULA = ("walk_hip_abduction_ramp",)


@configclass
class AmpObsCfg(ObsGroup):
    """[q_t, q_t+1] and [qd_t, qd_t+1] over the robot's joints (term-major, oldest first)."""
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=2, flatten_history_dim=True)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=2, flatten_history_dim=True)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AmpGateCfg(ObsGroup):
    gate = ObsTerm(func=mdp_amp.walk_gate, params={"command_name": "base_velocity", "stand_still_threshold": 0.1,
                                                   "track_gate_lin": float(os.environ.get("KBOT_AMP_GATE_LIN", "0.08")),  # 0.3 let a still robot keep 74% of style (pilot 0b)
                                                   "track_gate_ang": float(os.environ.get("KBOT_AMP_GATE_ANG", "0.25")),
                                                   "mode": os.environ.get("KBOT_AMP_GATE_MODE", "speed_fraction"),
                                                   "yaw_cmd_max": float(os.environ.get("KBOT_AMP_GATE_YAW_MAX", "0.0"))})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class KBotLegsAmpEnvCfg(KBotLegsRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ---- AMP observation groups ----
        self.observations.amp = AmpObsCfg()
        self.observations.amp_gate = AmpGateCfg()

        # ---- retire the gait-shape carrots; keep task/safety/standing/push ----
        retired = []
        for name in _RETIRE_REWARDS:
            if getattr(self.rewards, name, None) is not None:
                setattr(self.rewards, name, None)
                retired.append(name)
        for name in _RETIRE_CURRICULA:
            if hasattr(self.curriculum, name):
                setattr(self.curriculum, name, None)
        if getattr(self.rewards, "hip_yaw_deviation", None) is not None:
            self.rewards.hip_yaw_deviation.weight = -0.5  # heading/safety only; style owns yaw posture now
        self.rewards.feet_slide.weight = -0.3  # skating is invisible to a joint-space discriminator

        # ---- gait clock at the dataset's stride frequency ----
        clock = os.environ.get("KBOT_AMP_CLOCK")
        if clock is None and os.path.isfile(DEFAULT_CYCLE_META):
            clock = 1.0 / float(np.load(DEFAULT_CYCLE_META, allow_pickle=True)["period_s"])
        if clock is not None:
            gp = self.observations.policy.gait_phase
            gp.params["gait_freq"] = float(clock)
            gp.params["freq_map"] = None

        # ---- optional: freeze the sustained pushes at the curriculum's START band ----
        # Skills first, hardening later (lineage-10 sequencing). Measured on pilot 1b's final
        # policy: pushes cost standing survival 98% -> 85% even at the lowest band, and the
        # thermostat ramps them the moment episodes pass 12 s, so an AMP pilot spends its
        # capacity on push recovery instead of turning/standing. The event's own params are the
        # CEILING (10-30 N, 0.5-2.5 s) that the curriculum ramps toward, so freezing means
        # pinning the params to the start band AND removing the ramp. Velocity kicks stay.
        if os.environ.get("KBOT_AMP_PUSH_FREEZE", "0") == "1" and getattr(self.events, "sustained_push", None) is not None:
            ramp = getattr(self.curriculum, "sustained_push_level", None)
            start_force = tuple(ramp.params.get("start_force", (3.0, 8.0))) if ramp is not None else (3.0, 8.0)
            start_dur = tuple(ramp.params.get("start_dur", (0.3, 0.8))) if ramp is not None else (0.3, 0.8)
            self.events.sustained_push.params["force_range"] = start_force
            self.events.sustained_push.params["duration_range"] = start_dur
            self.curriculum.sustained_push_level = None
            print(f"[amp-env] sustained pushes FROZEN at {start_force} N / {start_dur} s (no ramp)")

        # ---- optional: stronger standing-stance term ----
        # Pilot 2 (warm start from the turning tracker) stands 40-44 cm wide and the lineage's
        # stand_stance_geometry at -6 costs a 44 cm stance ~0.004/step — invisible. The lineage never
        # needed more because it never stood wide. KBOT_AMP_STANCE_W multiplies that weight.
        sw = float(os.environ.get("KBOT_AMP_STANCE_W", "1.0"))
        if sw != 1.0 and getattr(self.rewards, "stand_stance_geometry", None) is not None:
            self.rewards.stand_stance_geometry.weight *= sw
            print(f"[amp-env] stand_stance_geometry weight x{sw} = {self.rewards.stand_stance_geometry.weight}")

        # ---- optional command band for single-speed datasets ----
        vx = os.environ.get("KBOT_AMP_VX")
        if vx:
            lo, hi = (float(x) for x in vx.split(":"))
            self.commands.base_velocity.ranges.lin_vel_x = (lo, hi)
        # dataset v3 (LAFAN1 clips: backward, side-steps, pivots): lateral and yaw bands too
        vy = os.environ.get("KBOT_AMP_VY")
        if vy:
            lo, hi = (float(x) for x in vy.split(":"))
            self.commands.base_velocity.ranges.lin_vel_y = (lo, hi)
        wz = os.environ.get("KBOT_AMP_WZ")
        if wz:
            lo, hi = (float(x) for x in wz.split(":"))
            self.commands.base_velocity.ranges.ang_vel_z = (lo, hi)
        if os.environ.get("KBOT_AMP_ALT", "1") != "1":
            self.rewards.feet_alternation = None

        # ---- walker v4 (2026-10-02): inputs and pressures matched to the multi-cycle tracker ----
        # The tracker (and every other run of this lineage) uses KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0. The
        # 2026-10-01 "AMP v3" stage was launched WITHOUT them (different plant model and observations than the
        # policy it started from), which invalidated its result. Refuse to build silently in that state.
        if os.environ.get("KBOT_ADAPT") != "1" or os.environ.get("KBOT_FLAT") != "1":
            msg = "[amp-env] KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0 are NOT set: this is not the lineage environment"
            if os.environ.get("KBOT_AMP_ALLOW_MISMATCH", "0") != "1":
                raise RuntimeError(msg + " (set KBOT_AMP_ALLOW_MISMATCH=1 to build anyway)")
            print(msg)
        hist = int(os.environ.get("KBOT_AMP_HIST", "1"))
        if hist > 1:  # same layout as the tracker: every term's frames contiguous, oldest first
            self.observations.policy.history_length = hist
            self.observations.policy.flatten_history_dim = True
        signed = os.environ.get("KBOT_AMP_SIGNED_CLOCK", "0") == "1"
        if signed:  # clock runs backward while cmd_vx < 0 (mdp_gait._phase_signed)
            self.observations.policy.gait_phase.params["signed"] = True
        vstd = os.environ.get("KBOT_AMP_VEL_STD")
        if vstd:  # tight velocity kernels: only really matching the command pays
            self.rewards.track_lin_vel_xy_exp.params["std"] = float(vstd)
            self.rewards.track_lin_vel_xy_exp.weight = float(os.environ.get("KBOT_AMP_VEL_W", self.rewards.track_lin_vel_xy_exp.weight))
            self.rewards.track_ang_vel_z_exp.params["std"] = float(os.environ.get("KBOT_AMP_YAW_STD", "0.3"))
            self.rewards.track_ang_vel_z_exp.weight = float(os.environ.get("KBOT_AMP_YAW_W", self.rewards.track_ang_vel_z_exp.weight))
        p_axis = float(os.environ.get("KBOT_AMP_AXIS_P", "0"))
        if p_axis > 0:
            self.events.amp_axis_bias = EventTerm(func=mdp_amp.axis_bias, mode="interval", interval_range_s=(0.02, 0.02),
                                                  params={"p_axis": p_axis, "min_cmd": float(os.environ.get("KBOT_AMP_AXIS_MIN", "0.1"))})
        drift = float(os.environ.get("KBOT_AMP_DRIFT", "0"))
        if drift > 0:
            self.terminations.root_drift = DoneTerm(func=mdp_amp.root_drift,
                                                    params={"max_dist": drift, "max_yaw": float(os.environ.get("KBOT_AMP_DRIFT_YAW", "1.5"))})
        # ---- walker v5 (prepared 2026-10-02, OFF unless KBOT_AMP_LIFT_W > 0): stepping anchor ----
        # Walker v4 kept every direction and learned to stand but slid its feet (see mdp_amp.ref_foot_lift).
        lift_w = float(os.environ.get("KBOT_AMP_LIFT_W", "0"))
        if lift_w > 0:
            assert clock is not None, "ref_foot_lift needs the gait clock frequency"
            self.rewards.ref_foot_lift = RewTerm(func=mdp_amp.ref_foot_lift, weight=lift_w,
                                                 params={"lib_file": os.path.join(AMP_REFS, "multicycle_v1.npz"), "gait_freq": float(clock),
                                                         "rel_sigma": float(os.environ.get("KBOT_AMP_LIFT_REL", "0.5"))})
            print(f"[amp-env] v5: reference foot-lift anchor, weight {lift_w}, sigma {self.rewards.ref_foot_lift.params['rel_sigma']} x each cycle's peak lift")
        # ---- walker v6 (prepared 2026-10-03, every part OFF unless its env var is set): the first hardware engage ----
        # eval_watch/RIG_HW_ENGAGE_FINDINGS.md: the real robot carries a standing load at the zero pose and its
        # ankles do not answer small commands at once; the policy integrated on its own last_action.
        v6 = []
        stand_m = float(os.environ.get("KBOT_AMP_STAND_MOMENT", "0"))
        if stand_m > 0 and getattr(self.events, "sustained_push", None) is not None:
            self.events.sustained_push.params["standing_moment"] = stand_m
            v6.append(f"standing load +-{stand_m} Nm (pitch and roll, constant per episode)")
        dead = float(os.environ.get("KBOT_AMP_DEADBAND", "0"))
        rotor = float(os.environ.get("KBOT_AMP_ROTOR_FC", "0"))
        eng_p = float(os.environ.get("KBOT_AMP_ENGAGE_P", "0"))
        if dead > 0 or rotor > 0 or eng_p > 0:
            self.events.amp_unanswered = EventTerm(func=mdp_amp.randomize_unanswered, mode="reset",
                                                   params={"deadband_max": dead, "rotor_fc_max": rotor, "engage_p": eng_p,
                                                           "engage_gain_min": float(os.environ.get("KBOT_AMP_ENGAGE_GAIN", "0.3")),
                                                           "engage_ramp_max": float(os.environ.get("KBOT_AMP_ENGAGE_RAMP", "1.0"))})
            v6.append(f"dead band U(0,{dead}) Nm, ankle rotor stiction U(0,{rotor}) Nm")
            if eng_p > 0:
                self.events.amp_engage_ramp = EventTerm(func=mdp_amp.engage_gain_ramp, mode="interval", interval_range_s=(0.02, 0.02))
                v6.append(f"weak start p={eng_p}: gains x U({self.events.amp_unanswered.params['engage_gain_min']},1) ramping over U(0,{self.events.amp_unanswered.params['engage_ramp_max']}) s")
        ks = os.environ.get("KBOT_AMP_SERIES_K")          # "nominal:lo:hi", e.g. 52:30:120 (rig: loaded ankle 52 Nm/rad)
        if ks:
            nom, lo, hi = (float(x) for x in ks.split(":"))
            for name, act in self.scene.robot.actuators.items():
                if "ankle" in name and getattr(act, "series_k", 0.0) > 0.0:
                    act.series_k = nom
            if getattr(self.events, "randomize_joint_play", None) is not None:
                self.events.randomize_joint_play.params["series_k_range"] = (lo, hi)
            band = getattr(self.curriculum, "series_k_band", None)
            if band is not None:
                band.params["lo_end"] = lo
                band.params["lo_start"] = max(band.params.get("lo_start", lo), lo)
            v6.append(f"ankle spring nominal {nom}, band {lo}-{hi} Nm/rad")
        keep_p = float(os.environ.get("KBOT_AMP_SPAWN_STAND_P", "0"))
        if keep_p > 0 and getattr(self.events, "walk_at_spawn", None) is not None:
            self.events.walk_at_spawn.params["keep_stand_p"] = keep_p
            v6.append(f"spawn stands kept with p={keep_p}")
        if v6:
            print("[amp-env] v6: " + "; ".join(v6))
        print(f"[amp-env] v4: history {hist}, signed clock {signed}, lin vel w {self.rewards.track_lin_vel_xy_exp.weight} std "
              f"{self.rewards.track_lin_vel_xy_exp.params.get('std')}, yaw w {self.rewards.track_ang_vel_z_exp.weight} std "
              f"{self.rewards.track_ang_vel_z_exp.params.get('std')}, axis bias {p_axis} (single-direction commands >= {os.environ.get('KBOT_AMP_AXIS_MIN', '0.1')}), "
              f"drift {drift} m, standing envs {self.commands.base_velocity.rel_standing_envs}")
        print(f"[amp-env] clock {clock} Hz, lin_vel_x {self.commands.base_velocity.ranges.lin_vel_x}, lin_vel_y {self.commands.base_velocity.ranges.lin_vel_y}, "
              f"ang_vel_z {self.commands.base_velocity.ranges.ang_vel_z}, feet_alternation "
              f"{getattr(self.rewards.feet_alternation, 'weight', None)}, gate {self.observations.amp_gate.gate.params['mode']} (lin {self.observations.amp_gate.gate.params['track_gate_lin']}, yaw_cmd_max {self.observations.amp_gate.gate.params['yaw_cmd_max']}), "
              f"retired {retired}")
