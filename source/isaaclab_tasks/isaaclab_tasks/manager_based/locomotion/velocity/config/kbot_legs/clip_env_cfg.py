"""Time-indexed CLIP tracking task (dataset expansion). See mdp_clip.py.

Built on the lineage package (KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0), like the stride tracker:
gait-shape / standing / push terms retired, tracking terms added — but the reference is a labelled
clip followed frame by frame, the command is the clip's own body motion (written every step), the
episode starts mid-clip (RSI) and ends when the clip does.
"""
from __future__ import annotations

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp_clip
from .rough_env_cfg import KBotLegsRoughEnvCfg

_IL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
CLIP_DIR = os.environ.get("KBOT_CLIP_DIR", os.path.join(_IL_ROOT, "eval_watch", "amp_refs", "lafan1", "clips"))

_RETIRE_REWARDS = (
    "knee_swing", "feet_phase", "flight_phase", "feet_alternation", "feet_air_time",
    "walk_hip_abduction", "hip_yaw_deviation", "joint_deviation_hip", "joint_deviation_ankles", "joint_deviation_hip_pitch_knee",
    "stand_upright", "stand_pose", "stand_stance_geometry", "stand_foot_scuff", "stand_calm", "stand_feet_planted", "stand_gyro",
    "stand_hip_roll_brace_ema", "stand_action_magnitude", "stand_action_rate", "stand_tilt_excursion", "stand_still", "push_brace", "push_step",
)
_RETIRE_EVENTS = ("push_robot", "sustained_push", "stand_corridor", "walk_at_spawn")
_RETIRE_CURRICULA = ("velocity_push_curriculum", "sustained_push_level", "walk_hip_abduction_ramp")


@configclass
class KBotLegsClipEnvCfg(KBotLegsRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        assert os.path.isdir(CLIP_DIR), f"clip dir not found: {CLIP_DIR}"
        joints = SceneEntityCfg("robot", joint_names=[".*"])
        # commands come from the clip; the term still exists (obs + tracking rewards read it)
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1000.0, 1000.0)
        # phase obs = clip progress (same 4-vector layout)
        gp = self.observations.policy.gait_phase
        gp.func = mdp_clip.clip_phase_obs
        gp.params = {"clip_dir": CLIP_DIR}
        # upcoming reference poses (+1, +10 frames) — the policy must know which clip it follows
        self.observations.policy.clip_ref = ObsTerm(func=mdp_clip.clip_ref_obs, params={"asset_cfg": joints, "clip_dir": CLIP_DIR, "offsets": (1, 10)})
        self.observations.critic.clip_ref = ObsTerm(func=mdp_clip.clip_ref_obs, params={"asset_cfg": joints, "clip_dir": CLIP_DIR, "offsets": (1, 10)})
        for name in _RETIRE_REWARDS:
            if getattr(self.rewards, name, None) is not None:
                setattr(self.rewards, name, None)
        for name in _RETIRE_EVENTS:
            if getattr(self.events, name, None) is not None:
                setattr(self.events, name, None)
        for name in _RETIRE_CURRICULA:
            if getattr(self.curriculum, name, None) is not None:
                setattr(self.curriculum, name, None)
        self.rewards.feet_slide.weight = -0.3
        task_w = float(os.environ.get("KBOT_CLIP_TASK_W", "2.0"))
        self.rewards.track_lin_vel_xy_exp.weight *= task_w
        self.rewards.track_ang_vel_z_exp.weight *= task_w
        gate_lin = float(os.environ.get("KBOT_CLIP_GATE", "0.1"))
        gate_ang = float(os.environ.get("KBOT_CLIP_GATE_ANG", "0.05"))
        self.rewards.track_clip_pose = RewTerm(func=mdp_clip.track_clip_pose, weight=6.0,
                                               params={"asset_cfg": joints, "clip_dir": CLIP_DIR, "sigma_deg": 5.0, "cmd_gate_lin": gate_lin, "cmd_gate_ang": gate_ang})
        self.rewards.track_clip_vel = RewTerm(func=mdp_clip.track_clip_vel, weight=1.0, params={"asset_cfg": joints, "clip_dir": CLIP_DIR, "sigma": 1.5})
        self.curriculum.clip_report = CurrTerm(func=mdp_clip.clip_report, params={"asset_cfg": joints, "clip_dir": CLIP_DIR})
        self.events.clip_rsi = EventTerm(func=mdp_clip.clip_rsi, mode="reset", params={"asset_cfg": joints, "clip_dir": CLIP_DIR})
        self.events.clip_command = EventTerm(func=mdp_clip.clip_command, mode="interval", interval_range_s=(0.02, 0.02), params={"clip_dir": CLIP_DIR})
        self.terminations.clip_end = DoneTerm(func=mdp_clip.clip_end, params={"clip_dir": CLIP_DIR}, time_out=True)
        self.episode_length_s = 20.0
        print(f"[clip-env] clips from {CLIP_DIR}; task weights x{task_w}; gate lin {gate_lin} ang {gate_ang}")
