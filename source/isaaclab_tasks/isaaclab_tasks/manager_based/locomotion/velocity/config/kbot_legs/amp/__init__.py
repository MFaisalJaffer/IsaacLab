"""Adversarial Motion Priors for the legs-only K-Bot (AMP_PLAN.md §2).

Pure-torch package: nothing here imports Isaac Sim, so the discriminator, dataset and
algorithm can be unit-tested on a CPU. Env-side pieces (observation groups, the walk gate)
live in ``..mdp_amp`` / ``..amp_env_cfg``.

Structure follows menloresearch/isaac_asimov (BSD-3-Clause, Menlo Research), adapted to
rsl_rl 2.3.3 and to this project's guards and conventions.
"""

from .amp_ppo import AMPPPO
from .discriminator import AMPDiscriminator
from .motion_dataset import MotionDataset, mirror_transitions
from .replay_buffer import ReplayBuffer
from .runner import KbotAmpRunner

__all__ = ["AMPPPO", "AMPDiscriminator", "MotionDataset", "ReplayBuffer", "KbotAmpRunner", "mirror_transitions"]
