"""Multi-cycle TRACKING task (option 2). See mdp_trackmulti.py.

The stride tracker (track_env_cfg) with a library of clean one-period cycles instead of one: forward,
backward, side-steps, pivots. The command selects the cycle; observations are the lineage's 43-D
(command + gait clock at the shared period), so the weights warm-start the AMP walker directly.
Needs the lineage env vars: KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0.
"""
from __future__ import annotations

import os

import numpy as np

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp_track, mdp_trackmulti
from .rough_env_cfg import KBotLegsRoughEnvCfg
from .track_env_cfg import _RETIRE_CURRICULA, _RETIRE_EVENTS, _RETIRE_REWARDS

_IL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
LIB_FILE = os.environ.get("KBOT_TM_LIB", os.path.join(_IL_ROOT, "eval_watch", "amp_refs", "multicycle_v1.npz"))


@configclass
class KBotLegsTrackMultiEnvCfg(KBotLegsRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        assert os.path.isfile(LIB_FILE), f"cycle library not found: {LIB_FILE}"
        d = np.load(LIB_FILE, allow_pickle=True)
        period = float(d["period_s"])
        freq = 1.0 / period
        # commands are written every step from each env's cycle; the term itself only has to exist
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1000.0, 1000.0)
        # phase clock = the shared cycle clock (same 4-vector layout as the lineage's gait phase)
        gp = self.observations.policy.gait_phase
        gp.func = mdp_trackmulti.tm_phase_obs   # signed clock: runs backward for the backward cycle
        gp.params = {"lib_file": LIB_FILE, "gait_freq": freq}
        # v2 (user go 2026-10-02): the actor gets H frames of history. Every deployed tracker/walker uses memory
        # (5-25 frames or an LSTM); ours had one frame and no body-velocity sensor, so it could not tell how
        # fast it was moving. Layout: each term's H frames contiguous, oldest first (symmetry.py handles it).
        hist = int(os.environ.get("KBOT_TM_HIST", "10"))
        if hist > 1:
            self.observations.policy.history_length = hist
            self.observations.policy.flatten_history_dim = True
        for name in _RETIRE_REWARDS:
            if hasattr(self.rewards, name):
                setattr(self.rewards, name, None)
        for name in _RETIRE_EVENTS:
            if hasattr(self.events, name):
                setattr(self.events, name, None)
        for name in _RETIRE_CURRICULA:
            if hasattr(self.curriculum, name):
                setattr(self.curriculum, name, None)
        self.rewards.feet_slide.weight = float(os.environ.get("KBOT_TM_SLIDE_W", "-0.3"))
        task_w = float(os.environ.get("KBOT_TM_TASK_W", "1.0"))
        self.rewards.track_lin_vel_xy_exp.weight *= task_w
        self.rewards.track_ang_vel_z_exp.weight *= task_w
        # v2: TIGHT velocity kernels. The lineage's std 0.5 pays 85% for a 0.2 m/s error, i.e. for creeping at half
        # speed. With a tight kernel the term only pays for really matching the command, so it can weigh more.
        vstd = float(os.environ.get("KBOT_TM_VEL_STD", "0.2"))
        wstd = float(os.environ.get("KBOT_TM_YAW_STD", "0.3"))
        if vstd > 0:
            self.rewards.track_lin_vel_xy_exp.params["std"] = vstd
            self.rewards.track_lin_vel_xy_exp.weight = float(os.environ.get("KBOT_TM_VEL_W", "4.0"))
            self.rewards.track_ang_vel_z_exp.params["std"] = wstd
            self.rewards.track_ang_vel_z_exp.weight = float(os.environ.get("KBOT_TM_YAW_W", "3.0"))
        joints = SceneEntityCfg("robot", joint_names=[".*"])
        common = {"asset_cfg": joints, "lib_file": LIB_FILE, "gait_freq": freq}
        self.rewards.track_ref_pose = RewTerm(func=mdp_trackmulti.tm_pose, weight=6.0, params={**common, "sigma_deg": 5.0})
        self.rewards.track_ref_vel = RewTerm(func=mdp_trackmulti.tm_vel, weight=1.0, params={**common, "sigma": 1.5})
        # v2: feet where the cycle puts them (pays for step length) + end episodes that fall behind the command
        feet_w = float(os.environ.get("KBOT_TM_FEET_W", "3.0"))
        if feet_w > 0 and "feet" in d:
            self.rewards.track_ref_feet = RewTerm(func=mdp_trackmulti.tm_feet, weight=feet_w,
                                                  params={**common, "sigma": float(os.environ.get("KBOT_TM_FEET_STD", "0.06"))})
        drift = float(os.environ.get("KBOT_TM_DRIFT", "1.0"))
        if drift > 0:
            self.terminations.root_drift = DoneTerm(func=mdp_trackmulti.tm_root_drift,
                                                    params={"lib_file": LIB_FILE, "max_dist": drift, "max_yaw": float(os.environ.get("KBOT_TM_DRIFT_YAW", "1.0"))})
        self.curriculum.track_report = CurrTerm(func=mdp_trackmulti.tm_report, params=dict(common))
        # declared last so they run after reset_base / reset_robot_joints
        self.events.track_rsi = EventTerm(func=mdp_trackmulti.tm_rsi, mode="reset", params=dict(common))
        self.events.tm_command = EventTerm(func=mdp_trackmulti.tm_command, mode="interval", interval_range_s=(0.02, 0.02), params={"lib_file": LIB_FILE})
        self.episode_length_s = 20.0
        print(f"[multi-env] library {os.path.basename(LIB_FILE)}: {[str(s) for s in d['names']]}, period {period:.3f} s ({freq:.3f} Hz), "
              f"task weights x{task_w}, feet_slide {self.rewards.feet_slide.weight}, history {hist}, "
              f"lin vel w {self.rewards.track_lin_vel_xy_exp.weight} std {self.rewards.track_lin_vel_xy_exp.params.get('std')}, "
              f"yaw w {self.rewards.track_ang_vel_z_exp.weight} std {self.rewards.track_ang_vel_z_exp.params.get('std')}, feet w {feet_w}, drift {drift} m")
