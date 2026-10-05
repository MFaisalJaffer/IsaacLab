"""Left/right mirror-symmetry augmentation for the legs walker (rsl_rl mirror loss).

Fixes the persistent CIRCLING + LIMP — both are learned left/right gait
asymmetries. The robot itself is balanced (measured COM only ~3 mm off the foot
center), so nothing physical forces the drift; the policy just invents an
asymmetric gait because nothing constrains symmetry. This makes the policy
EQUIVARIANT under a sagittal mirror: a mirrored observation must produce the
mirrored action, so a left-drifting gait is no longer free.

Mirror = reflection across the sagittal plane (left <-> right). Verified from the
contact Jacobian that ALL 10 leg joints mirror as swap-L/R + NEGATE. The velocity
command (base frame: x forward, y left) flips vy and wz; the gait-phase swaps its
L/R pair. Observations are NOT normalized (empirical_normalization=False), so
mirroring the raw obs is exact.

IMU TERMS — FIXED 2026-10-04. projected_gravity and imu_ang_vel are read in the
URDF's `imu` link frame, which is NOT the base frame: imu x = DOWN, y = BACKWARD,
z = LEFT (upright gravity is (+1, 0, 0); measured in the running simulator by
eval_watch/mirror_check.py). The lateral axis is z, so the mirror flips gravity z
(a vector loses its lateral component's sign) and gyro x and y (a pseudovector
keeps only its lateral component). Until this fix the file flipped gravity y and
gyro x, z — the map of a standard frame — so the mirror loss called a forward
lean the mirror image of a backward lean and never mirrored a sideways lean; it
has not enforced the physical symmetry in any run before this date (mirrored
rollouts: gravity y / z correlate +0.98 / -0.98, gyro y / z -0.97 / +0.95).
KBOT_MIRROR_IMU=legacy restores the old map, only to reproduce those runs.

Policy-obs layout (43): projgrav(3) velcmd(3) jointpos(10) jointvel(10)
                        imu_angvel(3) actions(10) gaitphase(4)
Joint order in the 10-blocks: [Lhp,Rhp, Lhr,Rhr, Lhy,Rhy, Lk,Rk, La,Ra].
"""

import os

import torch

# swap each L/R joint pair (and negate — see module docstring)
_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]

# which components of the imu-frame gravity / angular velocity change sign under the mirror (see docstring)
IMU_MIRROR = os.environ.get("KBOT_MIRROR_IMU", "imu")
assert IMU_MIRROR in ("imu", "legacy"), f"KBOT_MIRROR_IMU={IMU_MIRROR!r}: expected 'imu' or 'legacy'"
_GRAV_FLIP = (2,) if IMU_MIRROR == "imu" else (1,)
_GYRO_FLIP = (0, 1) if IMU_MIRROR == "imu" else (0, 2)
print(f"[symmetry] imu mirror '{IMU_MIRROR}': gravity flips {_GRAV_FLIP}, gyro flips {_GYRO_FLIP}"
      + ("" if IMU_MIRROR == "imu" else "  — LEGACY MAP, not the physical mirror"))


def _mirror_joints(x: torch.Tensor) -> torch.Tensor:
    """(..., 10) leg-joint vector -> swap L/R + negate."""
    return -x[..., _PERM]


def _mirror_policy_obs(o: torch.Tensor) -> torch.Tensor:
    m = o.clone()
    for k in _GRAV_FLIP:
        m[..., k] = -o[..., k]                      # projected gravity (imu frame): flip the lateral component
    m[..., 4] = -o[..., 4]                          # velocity command: flip vy
    m[..., 5] = -o[..., 5]                          # velocity command: flip wz (yaw)
    m[..., 6:16] = _mirror_joints(o[..., 6:16])     # joint_pos
    m[..., 16:26] = _mirror_joints(o[..., 16:26])   # joint_vel
    for k in _GYRO_FLIP:
        m[..., 26 + k] = -o[..., 26 + k]            # imu ang-vel (imu frame): flip all but the lateral component
    m[..., 29:39] = _mirror_joints(o[..., 29:39])   # previous actions
    m[..., 39:41] = o[..., 41:43]                    # gait phase: swap L<->R pair
    m[..., 41:43] = o[..., 39:41]
    return m


_TERMS = (("grav", 3), ("cmd", 3), ("jp", 10), ("jv", 10), ("imu", 3), ("act", 10), ("phase", 4))


def _mirror_term(name: str, x: torch.Tensor) -> torch.Tensor:
    if name in ("jp", "jv", "act"):
        return _mirror_joints(x)
    if name == "phase":
        return torch.cat([x[..., 2:4], x[..., 0:2]], dim=-1)
    m = x.clone()
    if name == "grav":
        for k in _GRAV_FLIP:
            m[..., k] = -x[..., k]
    elif name == "cmd":
        m[..., 1] = -x[..., 1]; m[..., 2] = -x[..., 2]
    elif name == "imu":
        for k in _GYRO_FLIP:
            m[..., k] = -x[..., k]
    return m


def _mirror_policy_obs_hist(o: torch.Tensor) -> torch.Tensor:
    """Policy obs with H frames of history (group history_length=H, flattened): every term's H frames are
    contiguous, oldest first -> mirror each frame of each term."""
    h = o.shape[-1] // 43
    out, off = [], 0
    for name, d in _TERMS:
        blk = o[..., off:off + h * d].reshape(*o.shape[:-1], h, d)
        out.append(_mirror_term(name, blk).reshape(*o.shape[:-1], h * d))
        off += h * d
    return torch.cat(out, dim=-1)


def data_augmentation_func(env=None, obs=None, actions=None, obs_type="policy", **kwargs):
    """rsl_rl symmetry hook: returns [original ; mirrored] for obs and actions."""
    if actions is None:
        actions = kwargs.get("action")
    aug_obs = None
    if obs is not None:
        if obs_type == "policy":
            assert obs.shape[-1] % 43 == 0, f"policy obs dim {obs.shape[-1]} is not a multiple of 43"
            mir = _mirror_policy_obs(obs) if obs.shape[-1] == 43 else _mirror_policy_obs_hist(obs)
            aug_obs = torch.cat([obs, mir], dim=0)
        else:
            # critic obs is not mirrored (mirror-loss path only augments 'policy');
            # duplicate so shapes stay consistent if ever called for logging.
            aug_obs = torch.cat([obs, obs], dim=0)
    aug_actions = None
    if actions is not None:
        aug_actions = torch.cat([actions, _mirror_joints(actions)], dim=0)
    return aug_obs, aug_actions
