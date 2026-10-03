"""Obstacle course task for the legs walker — stage 1 of eval_watch/OBSTACLE_PERCEPTION_PROPOSAL.md.

The walker's AMP task (amp_env_cfg, same env vars as walker v5) plus, and only plus:
  1. terrain: obstacle tiles (mdp_obstacle.obstacle_terrain_cfg) instead of the plane, with a height curriculum;
  2. commands: forward-only on obstacle tiles (mdp_obstacle.ObstacleVelocityCommand);
  3. policy input: the walker's 430 values (every term 10 frames, oldest first — unchanged) followed by the
     height map of the current tick (187 values) = 617;
  4. rewards: foot-lift anchor measured above the local ground, toe/leg contact penalties;
  5. style gate closed next to an obstacle.

Needs the walker's env vars (KBOT_FLAT=1 KBOT_ADAPT=1 KBOT_MAXINIT=0, KBOT_AMP_HIST=10, KBOT_AMP_SIGNED_CLOCK=1,
KBOT_AMP_LIFT_W=5, ...). KBOT_FLAT=1 stays: it selects the lineage's plant and events; the plane it installs is
replaced here.
"""
from __future__ import annotations

import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from . import mdp_obstacle
from .amp_env_cfg import KBotLegsAmpEnvCfg

MAP_SENSOR = "height_scanner"   # 17 x 11 cells of 10 cm on the base (1.6 m long, 1.0 m wide), yaw-aligned
# base height of the reference cycles above the floor (0.999 m forward/backward, 1.006 m side-step/pivot)
MAP_Z_NOMINAL = float(os.environ.get("KBOT_OBST_Z_NOMINAL", "1.0"))


@configclass
class KBotLegsObstacleEnvCfg(KBotLegsAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        e = os.environ.get

        # ---- 1. terrain + height curriculum ----
        rows = int(e("KBOT_OBST_LEVELS", "10"))
        hmax = float(e("KBOT_OBST_HMAX", "0.20"))
        hmin = float(e("KBOT_OBST_HMIN", "0.02"))
        self.scene.terrain.terrain_type = "generator"
        flat_only = e("KBOT_OBST_FLAT_ONLY", "0") == "1"   # all-flat course: the walker's own tests on this policy
        self.scene.terrain.terrain_generator = mdp_obstacle.obstacle_terrain_cfg(
            num_rows=rows, num_cols=int(e("KBOT_OBST_COLS", "20")), height_range=(hmin, hmax),
            **({"p_flat": 1.0, "p_platform": 0.0, "p_beam": 0.0} if flat_only else {}))
        self.scene.terrain.max_init_terrain_level = int(e("KBOT_OBST_INIT_LEVEL", "0"))
        self.curriculum.obstacle_levels = CurrTerm(func=mdp_obstacle.obstacle_levels)

        # ---- ground under each foot: a 0.6 x 0.6 m patch of rays per foot (world-aligned) ----
        for name, body in zip(mdp_obstacle.FOOT_SCANNERS, mdp_obstacle.tm.FEET):
            setattr(self.scene, name, RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body, offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="world", pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.6]),
                debug_vis=False, mesh_prim_paths=["/World/ground"], update_period=self.decimation * self.sim.dt))

        # ---- 2. commands: forward-only on obstacle tiles ----
        old = self.commands.base_velocity
        cmd = mdp_obstacle.ObstacleVelocityCommandCfg()
        for key, val in old.__dict__.items():
            if key != "class_type":
                setattr(cmd, key, val)
        # 0.30 and up: at the walker's ~85% speed tracking a slower robot does not get past the platform in an episode
        lo, hi = (float(x) for x in e("KBOT_OBST_VX", "0.30:0.45").split(":"))
        cmd.obstacle_vx = (lo, hi)
        self.commands.base_velocity = cmd

        # ---- 3. policy input: walker terms keep their 10 frames, the height map is the current tick only ----
        pol = self.observations.policy
        hist = pol.history_length
        if hist:
            for term in pol.__dict__.values():
                if isinstance(term, ObsTerm):
                    term.history_length = hist
                    term.flatten_history_dim = True
            pol.history_length = None
        pol.height_map = ObsTerm(func=mdp_obstacle.height_map, history_length=0,
                                 params={"sensor_cfg": SceneEntityCfg(MAP_SENSOR), "z_nominal": MAP_Z_NOMINAL, "scale": 5.0, "clip": 0.5})

        # ---- 4. rewards ----
        lift = getattr(self.rewards, "ref_foot_lift", None)
        if lift is not None:  # the walker's anchor, measured above the ground under each foot
            lift.func = mdp_obstacle.ref_foot_lift_terrain
            lift.params = {"lib_file": lift.params["lib_file"], "gait_freq": lift.params["gait_freq"],
                           "rel_sigma": lift.params["rel_sigma"], "margin": float(e("KBOT_OBST_LIFT_MARGIN", "0.05"))}
        w = float(e("KBOT_OBST_CONTACT_W", "5.0"))
        if w > 0:
            self.rewards.feet_stumble = RewTerm(func=mdp_obstacle.feet_stumble, weight=-w,
                                                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_LEG_FOOT")})
            self.rewards.leg_contact = RewTerm(func=mdp_obstacle.leg_contact, weight=-w,
                                               params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*Shin_Drive", ".*Femur_Lower_Drive"])})

        # ---- 5. style gate closed next to an obstacle ----
        self.observations.amp_gate.gate.func = mdp_obstacle.walk_gate_obstacle

        print(f"[obstacle-env] {rows} heights {hmin * 100:.0f}-{hmax * 100:.0f} cm, kinds per column "
              f"{mdp_obstacle.column_kinds(self.scene.terrain.terrain_generator)}, start level {self.scene.terrain.max_init_terrain_level}, "
              f"obstacle-tile vx {cmd.obstacle_vx}, policy history {hist} + height map ({'BLIND: zeroed' if mdp_obstacle.BLIND else 'on'}, "
              f"z_nominal {MAP_Z_NOMINAL}), lift anchor {'terrain-aware' if lift is not None else 'OFF'}, contact penalties -{w}")
