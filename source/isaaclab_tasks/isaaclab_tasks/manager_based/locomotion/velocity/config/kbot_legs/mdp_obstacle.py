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

Stage 1b (2026-10-03) adds a fourth kind, ``stairs`` (a square staircase around the spawn area: 3 risers up, a
landing, 3 down; the row's height is the riser), a curriculum that replays lower heights and does not demote
a robot for merely falling behind its command, and a fall check measured above the local ground.
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


def stairs_obstacle_terrain(difficulty: float, cfg: "StairsObstacleTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat tile with a square staircase around the spawn area: ``num_steps`` risers up, a landing, the same
    number down. Each level is a square ring one tread narrower on both sides than the one it sits on."""
    level = min(int(difficulty * cfg.num_levels), cfg.num_levels - 1)
    h = level_height(level, cfg.height_range, cfg.num_levels)
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    r_out = cfg.inner_half_width + 2.0 * (cfg.num_steps - 1) * cfg.tread + cfg.landing
    meshes = []
    for i in range(cfg.num_steps):
        a = 2.0 * (cfg.inner_half_width + i * cfg.tread)
        b = 2.0 * (r_out - i * cfg.tread)
        meshes += list(make_border((b, b), (a, a), h, (cx, cy, (i + 0.5) * h)))
    thick = 1.0
    meshes.append(trimesh.creation.box((cfg.size[0], cfg.size[1], thick), trimesh.transformations.translation_matrix((cx, cy, -0.5 * thick))))
    return meshes, np.array([cx, cy, 0.0])


@configclass
class StairsObstacleTerrainCfg(SubTerrainBaseCfg):
    function = stairs_obstacle_terrain
    inner_half_width: float = 1.2
    num_steps: int = 3
    """Risers on the way up (and the same number on the way down)."""
    tread: float = 0.30
    landing: float = 0.60
    height_range: tuple[float, float] = (0.02, 0.20)
    """Riser height at the lowest and the highest level."""
    num_levels: int = 10


def kind_extent(sub) -> tuple[float, float] | None:
    """(inner, outer) half widths of a tile kind's obstacle; None for a kind without one."""
    if isinstance(sub, RingObstacleTerrainCfg):
        return sub.inner_half_width, sub.inner_half_width + sub.ring_width
    if isinstance(sub, StairsObstacleTerrainCfg):
        return sub.inner_half_width, sub.inner_half_width + 2.0 * (sub.num_steps - 1) * sub.tread + sub.landing
    return None


def tile_height(sub, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Ground height of a tile of this kind at Chebyshev distance r from its centre, for step/riser height h
    (probe and test use; the simulation uses the mesh)."""
    out = torch.zeros_like(r)
    ext = kind_extent(sub)
    if ext is None:
        return out
    if isinstance(sub, RingObstacleTerrainCfg):
        return torch.where((r > ext[0]) & (r < ext[1]), h + out, out)
    for i in range(sub.num_steps):
        out = torch.where((r > ext[0] + i * sub.tread) & (r < ext[1] - i * sub.tread), (i + 1) * h + torch.zeros_like(r), out)
    return out


STAGE1_MIX = {"flat": 0.25, "platform": 0.45, "beam": 0.30, "stairs": 0.0}


def obstacle_terrain_cfg(num_rows: int = 10, num_cols: int = 20, height_range: tuple[float, float] = (0.02, 0.20),
                         mix: dict | None = None) -> TerrainGeneratorCfg:
    """``mix`` = share of the columns (and so of the robots) per kind; a kind with share 0 exists but has no
    tiles. {"flat": 1} gives an all-flat course: every env then keeps the walker's commands, which is how the
    walker's own six-direction tests are run on an obstacle-course policy."""
    m = {**{k: 0.0 for k in STAGE1_MIX}, **(STAGE1_MIX if mix is None else mix)}
    lv = dict(height_range=height_range, num_levels=num_rows)
    return TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=20.0, num_rows=num_rows, num_cols=num_cols, horizontal_scale=0.1,
        vertical_scale=0.005, slope_threshold=0.75, use_cache=False, curriculum=True,
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(proportion=m["flat"]),
            "platform": RingObstacleTerrainCfg(proportion=m["platform"], ring_width=1.0, **lv),
            "beam": RingObstacleTerrainCfg(proportion=m["beam"], ring_width=0.3, **lv),
            "stairs": StairsObstacleTerrainCfg(proportion=m["stairs"], **lv),
        },
    )


def column_kinds(gen_cfg: TerrainGeneratorCfg) -> list[str]:
    """Kind of tile in every column — the generator's own rule (TerrainGenerator._generate_curriculum_terrains)."""
    names = list(gen_cfg.sub_terrains.keys())
    p = np.array([s.proportion for s in gen_cfg.sub_terrains.values()], dtype=float)
    p /= p.sum()
    return [names[int(np.min(np.where(i / gen_cfg.num_cols + 0.001 < np.cumsum(p))[0]))] for i in range(gen_cfg.num_cols)]


def parse_mix(text: str) -> dict:
    """"flat:0.4,platform:0.2,beam:0.15,stairs:0.25" -> {"flat": 0.4, ...}"""
    return {k.strip(): float(v) for k, v in (item.split(":") for item in text.split(",") if item.strip())}


def course(env: "ManagerBasedRLEnv") -> dict:
    """Static description of the course (cached): kind per column, outer edge of each kind's ring, height per row."""
    st = getattr(env, "_obst_course", None)
    if st is None:
        gen = env.scene.terrain.cfg.terrain_generator
        names = list(gen.sub_terrains.keys())
        subs = list(gen.sub_terrains.values())
        ext = [kind_extent(s) for s in subs]
        first = next(s for s, e in zip(subs, ext) if e is not None)
        dev = env.device
        st = {
            "names": names,
            "subs": subs,
            "col_kind": torch.tensor([names.index(k) for k in column_kinds(gen)], device=dev),
            "kind_inner": torch.tensor([e[0] if e else 0.0 for e in ext], device=dev),
            "kind_outer": torch.tensor([e[1] if e else 0.0 for e in ext], device=dev),
            "kind_is_obstacle": torch.tensor([e is not None for e in ext], device=dev),
            "heights": torch.tensor([level_height(r, first.height_range, first.num_levels) for r in range(gen.num_rows)], device=dev),
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


# ---------------------------------------------------------------- termination
def base_height_terrain(env: "ManagerBasedRLEnv", minimum_height: float, grace_steps: int = 30, stand_grace_s: float = 0.0,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")) -> torch.Tensor:
    """The lineage's fall check (mdp_gait.base_height_after_grace, same grace and stand exemption) with the base
    height measured above the ground under the base instead of above z = 0: on top of a 0.6 m staircase a
    robot lying down is still 0.8 m above z = 0 and would never be caught. On flat ground it is the same test."""
    from . import mdp_gait

    asset = env.scene[asset_cfg.name]
    scan = env.scene.sensors[sensor_cfg.name]
    centre = getattr(env, "_obst_scan_centre", None)
    if centre is None:
        centre = int((scan.ray_starts[0, :, :2] ** 2).sum(dim=1).argmin())
        env._obst_scan_centre = centre
    ground = torch.nan_to_num(scan.data.ray_hits_w[:, centre, 2], nan=0.0, posinf=0.0, neginf=0.0)
    below = (asset.data.root_pos_w[:, 2] - ground) < minimum_height
    alive = below & (env.episode_length_buf > grace_steps) & mdp_gait._outside_stand_grace(env, stand_grace_s)
    cmd = env.command_manager.get_command("base_velocity")
    return alive & ~(torch.norm(cmd[:, :3], dim=1) < 0.1)


# ---------------------------------------------------------------- curriculum
def obstacle_levels(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, cross_margin: float = 0.5, behind_fails: bool = True,
                    replay_p: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict:
    """Runs at every reset. A robot on an obstacle tile that ended its episode beyond the obstacle (further
    than the obstacle's outer edge + cross_margin from the tile centre) moves one height up; one that failed
    before that moves one down. Robots that clear the top height are sent to a random one. Flat tiles have no
    levels.

    Failed = fell before getting past. ``behind_fails`` (stage 1): falling behind the command before getting
    past also counts. With it off (stage 1b) falling behind only counts for a robot that never got onto the
    obstacle (it refused): in stage 1 the walker's heading drift ended about half of all episodes that way and
    each one demoted a robot that was in fact climbing — training heights sat at 9 / 13 cm while the test
    passed 18-20 cm.

    ``replay_p``: share of episodes played at a random LOWER height than the robot's own (its own height does
    not change in such an episode). Stage 1's policy ended up worse at 4 cm than at 8-18 cm: once robots
    moved up nothing sent them back to the low obstacles."""
    c = course(env)
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    dev = env.device
    kind = c["col_kind"][terrain.terrain_types[env_ids]]
    is_obst = c["kind_is_obstacle"][kind]
    d = (asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]).abs().amax(dim=1)
    crossed = d > c["kind_outer"][kind] + cross_margin
    tm_ = env.termination_manager
    terminated = tm_.terminated[env_ids]
    if behind_fails:
        failed = terminated & ~crossed
    else:
        fell = torch.zeros_like(terminated)
        for name in ("base_contact", "bad_orientation", "base_height"):
            if name in tm_.active_terms:
                fell |= tm_.get_term(name)[env_ids]
        refused = terminated & ~fell & (d <= c["kind_inner"][kind])
        failed = ~crossed & (fell | refused)
    st = getattr(env, "_obst_stats", None)
    if st is None:
        st = {"cross": torch.zeros(len(c["names"]), device=dev), "seen": torch.zeros(len(c["names"]), device=dev),
              "home": terrain.terrain_levels.clone(), "replay": torch.zeros(env.num_envs, dtype=torch.bool, device=dev)}
        env._obst_stats = st
    # The reset at start-up is not an episode: the robots are not on their tiles yet, so their distance from
    # the tile centre means nothing (the first dry run promoted every robot one level right there).
    real = env.episode_length_buf[env_ids] > 10
    own = real & is_obst & ~st["replay"][env_ids]     # an episode at the robot's own height
    for i in range(len(c["names"])):
        m = own & (kind == i)
        if m.any():
            a = 0.02 if st["seen"][i] > 0 else 1.0
            st["cross"][i] = (1 - a) * st["cross"][i] + a * crossed[m].float().mean()
            st["seen"][i] = 1.0
    home = st["home"][env_ids] + (crossed & own).long() - (failed & own).long()
    home = torch.where(home >= terrain.max_terrain_level, torch.randint_like(home, terrain.max_terrain_level), home.clamp(min=0))
    st["home"][env_ids] = home
    replay = is_obst & (home > 0) & (torch.rand(len(env_ids), device=dev) < replay_p)
    play = torch.where(replay, (torch.rand(len(env_ids), device=dev) * home.float()).long().clamp(max=terrain.max_terrain_level - 1), home)
    st["replay"][env_ids] = replay
    terrain.terrain_levels[env_ids] = play
    terrain.env_origins[env_ids] = terrain.terrain_origins[play, terrain.terrain_types[env_ids]]
    kinds_all = env_kind(env)
    out = {}
    for i, name in enumerate(c["names"]):
        m = kinds_all == i
        if c["kind_is_obstacle"][i] and m.any():
            out[f"level_{name}"] = st["home"][m].float().mean()
            out[f"height_cm_{name}"] = c["heights"][st["home"][m]].mean() * 100.0
            out[f"cross_{name}"] = st["cross"][i]
    return out
