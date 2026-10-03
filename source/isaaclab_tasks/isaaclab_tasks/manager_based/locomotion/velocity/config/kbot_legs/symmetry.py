"""Left/right mirror-symmetry augmentation for the legs walker (rsl_rl mirror loss).

Fixes the persistent CIRCLING + LIMP — both are learned left/right gait
asymmetries. The robot itself is balanced (measured COM only ~3 mm off the foot
center), so nothing physical forces the drift; the policy just invents an
asymmetric gait because nothing constrains symmetry. This makes the policy
EQUIVARIANT under a sagittal mirror: a mirrored observation must produce the
mirrored action, so a left-drifting gait is no longer free.

Mirror = reflection across the sagittal (x-z) plane, y -> -y. Verified from the
contact Jacobian that ALL 10 leg joints mirror as swap-L/R + NEGATE. Non-joint
obs: projected-gravity flips its y; the velocity command flips vy and wz; imu
angular-velocity (a pseudovector) flips wx and wz; the gait-phase swaps its L/R
pair. Observations are NOT normalized (empirical_normalization=False), so
mirroring the raw obs is exact.

Policy-obs layout (43): projgrav(3) velcmd(3) jointpos(10) jointvel(10)
                        imu_angvel(3) actions(10) gaitphase(4)
Joint order in the 10-blocks: [Lhp,Rhp, Lhr,Rhr, Lhy,Rhy, Lk,Rk, La,Ra].
"""

import torch

# swap each L/R joint pair (and negate — see module docstring)
_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]


def _mirror_joints(x: torch.Tensor) -> torch.Tensor:
    """(..., 10) leg-joint vector -> swap L/R + negate."""
    return -x[..., _PERM]


def _mirror_policy_obs(o: torch.Tensor) -> torch.Tensor:
    m = o.clone()
    m[..., 1] = -o[..., 1]                          # projected gravity: flip y
    m[..., 4] = -o[..., 4]                          # velocity command: flip vy
    m[..., 5] = -o[..., 5]                          # velocity command: flip wz (yaw)
    m[..., 6:16] = _mirror_joints(o[..., 6:16])     # joint_pos
    m[..., 16:26] = _mirror_joints(o[..., 16:26])   # joint_vel
    m[..., 26] = -o[..., 26]                         # imu ang-vel: flip wx
    m[..., 28] = -o[..., 28]                         # imu ang-vel: flip wz
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
        m[..., 1] = -x[..., 1]
    elif name == "cmd":
        m[..., 1] = -x[..., 1]; m[..., 2] = -x[..., 2]
    elif name == "imu":
        m[..., 0] = -x[..., 0]; m[..., 2] = -x[..., 2]
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
