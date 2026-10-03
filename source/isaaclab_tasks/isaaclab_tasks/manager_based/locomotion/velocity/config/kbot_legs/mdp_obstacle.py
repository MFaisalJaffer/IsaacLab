"""Obstacle course for the walker — stage 1 of eval_watch/OBSTACLE_PERCEPTION_PROPOSAL.md.

"Can it climb with a perfect map?" The walker (v5: 10-frame history, signed clock, foot-lift anchor) gets a
robot-centric height map and trains on a flat floor with single obstacles of 2-20 cm:

  * terrain: a grid of 8 x 8 m tiles; every column is one kind of tile, every row one obstacle height. Kinds:
    ``flat`` (stays in at every level so flat walking does not decay), ``platform`` (a square ring 1.0 m wide
    around the spawn area: step up, a few steps on top, step down) and ``beam`` (the same ring 0.3 m wide: step
    onto or over it). The rings are concentric, so a robot walking straight out of the centre in any direction
    meets the obstacle after 0.8-2 m, at an angle between head-on and 45 degrees;
  * commands: robots on obstacle tiles walk forward only (vx in a band, no lateral, no yaw, no standing) —
    forward first, the other directions later. Robots on flat tiles keep the walker's full command mix;
  * curriculum: a robot that gets past the obstacle moves one height up, one that falls (or falls behind its
    command) before getting past moves one down;
  * policy input: the 430 walker values, unchanged, followed by the height map of the current tick (17 x 11
    cells of 10 cm, 0.8 m ahead/behind and 0.5 m to each side);
  * rewards: the foot-lift anchor measures height above the ground under the foot and lets the foot go higher
    next to an obstacle; a penalty for hitting an obstacle with the toe or the leg; no style reward next to an
    obstacle (the judge only knows flat-ground steps).

KBOT_OBST_BLIND=1 zeroes the map (same network, same training) — the control that shows what the map buys.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

from . import mdp_amp
from . import mdp_trackmulti as tm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BLIND = os.environ.get("KBOT_OBST_BLIND", "0") == "1"
FOOT_SCANNERS = ("foot_scan_r", "foot_scan_l")  # same [R, L] order as mdp_trackmulti.FEET


# ---------------------------------------------------------------- terrain
def level_height(level: int, height_range: tuple[float, float], num_levels: int) -> float:
    lo, hi = height_range
    return lo + level * (hi - lo) / max(num_levels - 1, 1)


def ring_obstacle_terrain(difficulty: float, cfg: "RingObstacleTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat tile with one square ring of height h around a flat spawn area. The generator hands every row a
    difficulty in [row, row + 1) / num_rows; it is quantised here so that a row is exactly one height."""
    level = min(int(difficulty * cfg.num_levels), cfg.num_levels - 1)
    h = level_height(level, cfg.height_range, cfg.num_levels)
    inner = 2.0 * cfg.inner_half_width
    outer = inner + 2.0 * cfg.ring_width
    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * h)
    meshes = list(make_border((outer, outer), (inner, inner), h, center))
    thick = 1.0
    ground = trimesh.creation.box((cfg.size[0], cfg.size[1], thick),
                                  trimesh.transformations.translation_matrix((0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * thick)))
    meshes.append(ground)
    return meshes, np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])


@configclass
class RingObstacleTerrainCfg(SubTerrainBaseCfg):
    function = ring_obstacle_terrain
    inner_half_width: float = 1.3
    """Half width of the flat spawn area (m). Robots spawn within +-0.5 m of the centre."""
    ring_width: float = 1.0
    height_range: tuple[float, float] = (0.02, 0.20)
    num_levels: int = 10


def obstacle_terrain_cfg(num_rows: int = 10, num_cols: int = 20, height_range: tuple[float, float] = (0.02, 0.20),
                         p_flat: float = 0.25, p_platform: float = 0.45, p_beam: float = 0.30) -> TerrainGeneratorCfg:
    """p_platform = p_beam = 0 gives an all-flat course: every env then keeps the walker's commands, which is
    how the walker's own six-direction tests are run on an obstacle-course policy."""
    ring = dict(height_range=height_range, num_levels=num_rows)
    return TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=20.0, num_rows=num_rows, num_cols=num_cols, horizontal_scale=0.1,
        vertical_scale=0.005, slope_threshold=0.75, use_cache=False, curriculum=True,
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(proportion=p_flat),
            "platform": RingObstacleTerrainCfg(proportion=p_platform, ring_width=1.0, **ring),
            "beam": RingObstacleTerrainCfg(proportion=p_beam, ring_width=0.3, **ring),
        },
    )


def column_kinds(gen_cfg: TerrainGeneratorCfg) -> list[str]:
    """Kind of tile in every column — the generator's own rule (TerrainGenerator._generate_curriculum_terrains)."""
    names = list(gen_cfg.sub_terrains.keys())
    p = np.array([s.proportion for s in gen_cfg.sub_terrains.values()], dtype=float)
    p /= p.sum()
    return [names[int(np.min(np.where(i / gen_cfg.num_cols + 0.001 < np.cumsum(p))[0]))] for i in range(gen_cfg.num_cols)]


def course(env: "ManagerBasedRLEnv") -> dict:
    """Static description of the course (cached): kind per column, outer edge of each kind's ring, height per row."""
    st = getattr(env, "_obst_course", None)
    if st is None:
        gen = env.scene.terrain.cfg.terrain_generator
        names = list(gen.sub_terrains.keys())
        subs = list(gen.sub_terrains.values())
        rings = [s for s in subs if isinstance(s, RingObstacleTerrainCfg)]
        dev = env.device
        st = {
            "names": names,
            "col_kind": torch.tensor([names.index(k) for k in column_kinds(gen)], device=dev),
            "kind_inner": torch.tensor([s.inner_half_width if isinstance(s, RingObstacleTerrainCfg) else 0.0 for s in subs], device=dev),
            "kind_outer": torch.tensor([s.inner_half_width + s.ring_width if isinstance(s, RingObstacleTerrainCfg) else 0.0 for s in subs], device=dev),
            "kind_is_obstacle": torch.tensor([isinstance(s, RingObstacleTerrainCfg) for s in subs], device=dev),
            "heights": torch.tensor([level_height(r, rings[0].height_range, rings[0].num_levels) for r in range(gen.num_rows)], device=dev),
        }
        env._obst_course = st
    return st


def env_kind(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,) kind index of every env's home tile (an env keeps its column for life; only its row changes)."""
    return course(env)["col_kind"][env.scene.terrain.terrain_types]


def obstacle_env_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return course(env)["kind_is_obstacle"][env_kind(env)]


# ---------------------------------------------------------------- commands
class ObstacleVelocityCommand(UniformVelocityCommand):
    """The walker's velocity command, except that robots on obstacle tiles walk straight ahead: vx from
    ``obstacle_vx``, no lateral, no yaw, never standing. Stage 1 asks one question — can it climb what it
    sees when walking forward — and sideways/backward/stopping next to obstacles would blur the answer.
    Publishes env._cmd_protect_mask so mdp_amp.axis_bias leaves these commands alone."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        mask = getattr(self, "_obst_mask", None)
        if mask is None:
            mask = obstacle_env_mask(self._env)
            self._obst_mask = mask
            self._env._cmd_protect_mask = mask
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device)
        ids = ids[mask[ids]]
        if len(ids) > 0:
            self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.obstacle_vx)
            self.vel_command_b[ids, 1:] = 0.0
            self.is_standing_env[ids] = False


@configclass
class ObstacleVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ObstacleVelocityCommand
    obstacle_vx: tuple[float, float] = (0.25, 0.45)


# ---------------------------------------------------------------- observation
def height_map(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), z_nominal: float = 0.0,
               scale: float = 5.0, clip: float = 0.5) -> torch.Tensor:
    """Ground height in every cell of the scanner grid, measured from the robot's base: 0 = the floor a robot
    walking on flat ground stands on (base z_nominal above it), +0.10 = ground 10 cm higher. Clipped to
    +-clip m and multiplied by ``scale`` (the usual 5, so 20 cm reads 1.0). Cell order: y rows from the
    robot's right (-0.5 m) to its left, x from behind (-0.8 m) to ahead inside each row."""
    sensor = env.scene.sensors[sensor_cfg.name]
    h = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2:3] + z_nominal
    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0).clamp(-clip, clip) * scale
    return torch.zeros_like(h) if BLIND else h


# ---------------------------------------------------------------- ground under the feet
def foot_ground(env: "ManagerBasedRLEnv") -> dict:
    """World height of the ground around each foot, from the two foot scanners (a 0.6 x 0.6 m patch of rays
    centred on the foot): ``under`` the foot's own position, and the ``max`` / ``min`` of the patch. (N, 2) [R, L]."""
    z = []
    for name in FOOT_SCANNERS:
        z.append(torch.nan_to_num(env.scene.sensors[name].data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0))
    z = torch.stack(z, dim=1)                       # (N, 2, rays)
    return {"under": z[..., z.shape[-1] // 2], "max": z.amax(dim=-1), "min": z.amin(dim=-1)}


def near_obstacle(env: "ManagerBasedRLEnv", tol: float = 0.01) -> torch.Tensor:
    """(N,) bool: the ground within 0.3 m of either foot is not level (an edge is within reach)."""
    g = foot_ground(env)
    return ((g["max"] - g["min"]) > tol).any(dim=1)


# ---------------------------------------------------------------- rewards
def ref_foot_lift_terrain(env: "ManagerBasedRLEnv", lib_file: str, gait_freq: float, rel_sigma: float = 0.5, margin: float = 0.05,
                          command_name: str = "base_velocity", stand_still_threshold: float = 0.1,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The walker's foot-lift anchor (mdp_amp.ref_foot_lift) on uneven ground.

    Each foot's height is measured above the ground UNDER THAT FOOT and must follow the reference cycle's
    lift at the gait clock's phase — but next to an obstacle the foot may be higher than the reference, by up
    to the height of the highest ground within 0.3 m of it plus ``margin``. So lifting too little is
    penalised exactly as on flat ground, and the extra lift needed to get onto (or down from) an obstacle is
    free. On level ground the allowance is zero and the term equals the walker's."""
    asset = env.scene[asset_cfg.name]
    lib, prof = mdp_amp._lift_profile(lib_file, env.device)
    cmd = env.command_manager.get_command(command_name)
    k = mdp_amp.cmd_cycle(cmd, lib)
    ref = tm._ref(prof, k, mdp_amp._clock_phase_now(env, gait_freq, command_name))  # (N, 2) [R, L]
    ids = getattr(env, "_amp_feet_ids", None)
    if ids is None:
        ids = asset.find_bodies(tm.FEET, preserve_order=True)[0]
        env._amp_feet_ids = ids
    g = foot_ground(env)
    z = asset.data.body_pos_w[:, ids, 2] - g["under"]
    step = (g["max"] - g["under"]).clamp(min=0.0)
    allow = step + margin * (step > 0.01).float()
    err = (ref - z).clamp(min=0.0) + (z - ref - allow).clamp(min=0.0)
    sigma = rel_sigma * (prof.amax(dim=1) - prof.amin(dim=1)).mean(dim=1)[k]
    r = torch.exp(-(err ** 2).sum(dim=1) / sigma ** 2)
    return r * (torch.norm(cmd[:, :3], dim=1) >= stand_still_threshold).float()


def feet_stumble(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, ratio: float = 3.0, min_force: float = 15.0) -> torch.Tensor:
    """Number of feet pushing against something sideways: horizontal contact force above ``ratio`` x the
    vertical one (friction alone cannot do that) and above ``min_force`` N — a toe driven into a step's face."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    fxy = f[..., :2].norm(dim=-1)
    return ((fxy > ratio * f[..., 2].abs()) & (fxy > min_force)).float().sum(dim=1)


def leg_contact(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """Number of leg links (shins, thighs) touching the ground or an obstacle."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (f.norm(dim=-1).amax(dim=1) > threshold).float().sum(dim=1)


def walk_gate_obstacle(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", stand_still_threshold: float = 0.1,
                       track_gate_lin: float = 0.3, track_gate_ang: float = 0.25, mode: str = "speed_fraction",
                       yaw_cmd_max: float = 0.0, near_tol: float = 0.01,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The walker's style gate (mdp_amp.walk_gate, same parameters), closed while an obstacle's edge is within
    reach of a foot: the judge's dataset is flat-ground walking, so it would mark every step-up as wrong —
    and it must not be trained on them."""
    gate = mdp_amp.walk_gate(env, command_name, stand_still_threshold, track_gate_lin, track_gate_ang, mode, yaw_cmd_max, asset_cfg)
    return gate * (~near_obstacle(env, near_tol)).float().unsqueeze(1)


# ---------------------------------------------------------------- curriculum
def obstacle_levels(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, cross_margin: float = 0.5,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict:
    """Runs at every reset. A robot on an obstacle tile that ended its episode beyond the obstacle (further
    than the ring's outer edge + cross_margin from the tile centre) moves one height up; one that fell, or
    fell behind its command, before that moves one down. Robots that clear the top height are sent to a random
    one. Flat tiles have no levels."""
    c = course(env)
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    kind = c["col_kind"][terrain.terrain_types[env_ids]]
    is_obst = c["kind_is_obstacle"][kind]
    d = (asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]).abs().amax(dim=1)
    crossed = d > c["kind_outer"][kind] + cross_margin
    failed = env.termination_manager.terminated[env_ids] & ~crossed
    st = getattr(env, "_obst_stats", None)
    if st is None:
        st = {"cross": torch.zeros(len(c["names"]), device=env.device), "seen": torch.zeros(len(c["names"]), device=env.device)}
        env._obst_stats = st
    # The reset at start-up is not an episode: the robots are not on their tiles yet, so their distance from
    # the tile centre means nothing (the first dry run promoted every robot one level right there).
    real = env.episode_length_buf[env_ids] > 10
    for i in range(len(c["names"])):
        m = real & (kind == i)
        if m.any():
            a = 0.02 if st["seen"][i] > 0 else 1.0
            st["cross"][i] = (1 - a) * st["cross"][i] + a * crossed[m].float().mean()
            st["seen"][i] = 1.0
    terrain.update_env_origins(env_ids, crossed & is_obst & real, failed & is_obst & real)
    kinds_all = env_kind(env)
    lv = terrain.terrain_levels.float()
    out = {}
    for i, name in enumerate(c["names"]):
        if c["kind_is_obstacle"][i]:
            out[f"level_{name}"] = lv[kinds_all == i].mean()
            out[f"height_cm_{name}"] = c["heights"][terrain.terrain_levels[kinds_all == i]].mean() * 100.0
            out[f"cross_{name}"] = st["cross"][i]
    return out
