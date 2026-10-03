"""Reference-cycle TRACKING task for the legs-only K-Bot (AMP_PLAN §5, stage 2b).

Built on top of the live lineage package (needs the same env vars the training
service uses: KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0), then:

  * one fixed forward command = the cycle's own speed, no standers, no turns;
  * the policy's gait-phase observation runs at the cycle's frequency (same clock the
    tracking reward uses), so the deployment interface is unchanged;
  * every hand-made gait-SHAPE carrot/penalty, the standing package and the push
    machinery are retired (walking-only task, nothing to stand for);
  * two tracking terms are added (mdp_track), a self-pacing/logging curriculum, and
    reference-state initialisation at reset (random phase, joints on the reference).

Everything physical (plant, delays 2-5 steps, K_s band, masses, friction) is inherited
untouched: the question this task answers is "can THIS plant walk THAT gait".

Separate task id + experiment name so the watcher/watchdog of the main run never see it.
"""
from __future__ import annotations

import os

import numpy as np

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp_track
from .rough_env_cfg import KBotLegsRoughEnvCfg

_IL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
CYCLE_FILE = os.environ.get("KBOT_TRACK_CYCLE", os.path.join(_IL_ROOT, "eval_watch", "amp_refs", "asimov_walk_cycle.npz"))

_RETIRE_REWARDS = (
    # gait shape (what the reference now provides)
    "knee_swing", "feet_phase", "flight_phase", "feet_alternation", "feet_air_time",
    "walk_hip_abduction", "hip_yaw_deviation", "joint_deviation_hip", "joint_deviation_ankles",
    "joint_deviation_hip_pitch_knee",
    # standing package + push shaping (no standers in this task)
    "stand_upright", "stand_pose", "stand_stance_geometry", "stand_foot_scuff", "stand_calm",
    "stand_feet_planted", "stand_gyro", "stand_hip_roll_brace_ema", "stand_action_magnitude",
    "stand_action_rate", "stand_tilt_excursion", "stand_still", "push_brace", "push_step",
)
_RETIRE_EVENTS = ("push_robot", "sustained_push", "stand_corridor", "walk_at_spawn")
_RETIRE_CURRICULA = ("velocity_push_curriculum", "sustained_push_level", "walk_hip_abduction_ramp")


@configclass
class KBotLegsTrackEnvCfg(KBotLegsRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        assert os.path.isfile(CYCLE_FILE), f"cycle file not found: {CYCLE_FILE}"
        d = np.load(CYCLE_FILE, allow_pickle=True)
        period = float(d["period_s"])
        speed = float(d["speed_mps"])
        # derived cycles (amp_design_cycles.py) carry vel_b = (vx, vy): backward walking has vx < 0,
        # side-steps have vy != 0. Older cycle files: (speed, 0).
        vel_b = [float(x) for x in d["vel_b"]] if "vel_b" in d else [speed, 0.0]
        freq = 1.0 / period

        # ---- one command: the cycle's own body velocity ----
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (vel_b[0], vel_b[0])
        cmd.ranges.lin_vel_y = (vel_b[1], vel_b[1])
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1000.0, 1000.0)
        # TURNING / MULTI-SPEED TRACKER (2026-09-30): the AMP dataset recorded from the straight,
        # single-speed tracker holds only symmetric strides, so the judge blocks turning. With
        # KBOT_TRACK_YAW=w the tracker is commanded yaw rates in [-w, w] (resampled every 10 s) and
        # must find the stride asymmetry that turns the body while staying near the cycle; with
        # KBOT_TRACK_VX="lo:hi" it is commanded a speed band around the cycle's own speed. Its
        # rollouts then carry turning (and speed variation) in the SAME style.
        yaw_w = float(os.environ.get("KBOT_TRACK_YAW", "0"))
        vx = os.environ.get("KBOT_TRACK_VX", "")
        if yaw_w > 0.0 or vx:
            if yaw_w > 0.0:
                cmd.ranges.ang_vel_z = (-yaw_w, yaw_w)
            if vx:
                lo, hi = (float(x) for x in vx.split(":"))
                cmd.ranges.lin_vel_x = (lo, hi)
            cmd.resampling_time_range = (10.0, 10.0)
            print(f"[track] commands: lin_vel_x {cmd.ranges.lin_vel_x}, ang_vel_z {cmd.ranges.ang_vel_z}, resample 10 s")

        # ---- phase clock = the reference's clock (same 4-vector layout, plus the RSI offset) ----
        gp = self.observations.policy.gait_phase
        gp.func = mdp_track.ref_phase_obs
        gp.params = {"gait_freq": freq}

        # ---- retire shape / standing / push terms ----
        for name in _RETIRE_REWARDS:
            if hasattr(self.rewards, name):
                setattr(self.rewards, name, None)
        for name in _RETIRE_EVENTS:
            if hasattr(self.events, name):
                setattr(self.events, name, None)
        for name in _RETIRE_CURRICULA:
            if hasattr(self.curriculum, name):
                setattr(self.curriculum, name, None)
        # skating is invisible to a joint-space reference: price it a little higher
        self.rewards.feet_slide.weight = -0.3

        # ---- tracking terms ----
        joints = SceneEntityCfg("robot", joint_names=[".*"])
        common = {"asset_cfg": joints, "cycle_file": CYCLE_FILE, "gait_freq": freq}
        # gated on body-velocity tracking (same gate the lineage's gait carrots use; 0 = off)
        gate_lin = float(os.environ.get("KBOT_TRACK_GATE", "0.3"))
        gate_ang = float(os.environ.get("KBOT_TRACK_GATE_ANG", "0.25"))
        self.rewards.track_ref_pose = RewTerm(func=mdp_track.track_ref_pose, weight=6.0,
                                              params={**common, "sigma_deg": 5.0, "cmd_gate_lin": gate_lin, "cmd_gate_ang": gate_ang})
        # Tracker v2 (turn + speed commands), 300 iters: stride held at 2.3 deg but commands ignored
        # (yaw 0.04 vs 0.30, speed 0.43 at both 0.3 and 0.5) — with lin 0.3 / ang 0.25 a robot that
        # ignores a 0.3 rad/s turn keeps exp(-0.09/0.25) = 70% of the stride carrot. KBOT_TRACK_TASK_W
        # scales the velocity/yaw tracking weights (default 1 = lineage values 2/2).
        task_w = float(os.environ.get("KBOT_TRACK_TASK_W", "1.0"))
        if task_w != 1.0:
            self.rewards.track_lin_vel_xy_exp.weight *= task_w
            self.rewards.track_ang_vel_z_exp.weight *= task_w
            print(f"[track] task weights x{task_w}: lin {self.rewards.track_lin_vel_xy_exp.weight}, ang {self.rewards.track_ang_vel_z_exp.weight}")
        print(f"[track] pose carrot gate: lin {gate_lin}, ang {gate_ang}")
        self.rewards.track_ref_vel = RewTerm(func=mdp_track.track_ref_vel, weight=1.0, params={**common, "sigma": 1.5})
        self.curriculum.track_report = CurrTerm(func=mdp_track.track_report, params=dict(common))
        # reference-state initialisation: declared last so it runs after reset_base / reset_robot_joints
        self.events.track_rsi = EventTerm(func=mdp_track.track_rsi, mode="reset", params=dict(common))

        self.episode_length_s = 20.0
        print(f"[track] cycle {os.path.basename(CYCLE_FILE)}: period {period:.3f} s ({freq:.3f} Hz), speed {speed:.3f} m/s")
