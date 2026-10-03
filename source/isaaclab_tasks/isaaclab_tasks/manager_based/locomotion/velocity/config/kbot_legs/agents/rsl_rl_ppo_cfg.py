"""RSL-RL PPO config for the legs-only K-Bot (reuses the full-kbot runner)."""

import os

from isaaclab.utils import configclass

from ...kbot.agents.rsl_rl_ppo_cfg import KBotRoughPPORunnerCfg


@configclass
class KBotLegsRoughPPORunnerCfg(KBotRoughPPORunnerCfg):
    # KBOT_EXP lets a parallel experiment (e.g. bent-knee bisect) log to a
    # separate dir instead of colliding with the main kbot_legs_rough run.
    experiment_name = os.environ.get("KBOT_EXP", "kbot_legs_rough")

    def __post_init__(self):
        super().__post_init__()
        # Save every 200 iters (was 50): at ~15-20 GB/day of ckpts the 50-interval
        # cadence filled the 458 GB root fs and killed training silently for ~6 h
        # (2026-07-14). We only ever keep milestone ckpts, so the 4x wider interval
        # cuts accumulation ~4x with negligible loss (worst case a rollback loses
        # up to 200 iters vs 50 — cheap vs a disk-full outage). NB the curriculum
        # fast-forward and watchdog anchoring both key off model_<iter>.pt names,
        # so the coarser grid just means coarser rollback granularity.
        self.save_interval = 200
        # Clip raw policy actions at the wrapper (was None = UNBOUNDED). Collapse
        # forensics @119.6k: one bad update made the net emit actions of 100s+
        # (action_rate_l2 -3 -> -2e6 in 60 iters), robots got launched, every env
        # died in ~37 steps, value targets hit 1e6 -> death spiral. Healthy actions
        # are |a| ~ 1-3, so 8.0 (joint-target offset ±4 rad at scale 0.5) never
        # binds in normal operation — it only amputates the blowup mode. Eval,
        # export and HIL paths all build RslRlVecEnvWrapper with
        # agent_cfg.clip_actions, so every consumer sees the same clamp.
        self.clip_actions = 8.0
        # Log-parameterized action std: std = exp(log_std), so it can NEVER go
        # negative. The default "scalar" std is a direct nn.Parameter that
        # gradients pushed below zero after ~18k iters -> crashed with
        # "normal expects all elements of std >= 0.0" and the Restart=always
        # service then looped 683x. (log mode stores 'log_std' not 'std', so it
        # is NOT load-compatible with old scalar checkpoints -> requires a fresh run.)
        self.policy.noise_std_type = "log"
        # Lower base LR 1e-3 -> 3e-4. Under our DR the value-function loss diverged
        # to inf around iter ~10k -> inf actions -> NaN std -> crash. A gentler LR
        # (with the adaptive schedule on top) keeps the value function stable.
        self.algorithm.learning_rate = 3.0e-4
        # CONSOLIDATION LR (2026-08-08): 3 abrupt VF explosions in 36 h at the
        # 25 N plateau (14.8k, 15.7k, then only 3.6k iters after anchor; VF
        # 2.5 -> >5 sustained within ~250 iters each time; noise_std flat, LR
        # flat — measured, watchdog_log + autoroll TBs). The adaptive KL
        # schedule sat at its ~3e-5 equilibrium throughout and did not prevent
        # any of them; it also RAISES LR whenever KL is small, i.e. exactly at
        # a plateau, so a lower base alone would drift back up. Hard-pin small
        # steps for the at-plateau consolidation phase: 3x smaller updates,
        # schedule fixed. (A 4th explosion at 16.2k iters later showed the pin
        # EXTENDS time-to-explosion but does not eliminate it.)
        #
        # LINEAGE 2 FROM-SCRATCH (2026-08-10, user call): the pin above is a
        # PLATEAU-phase measure and would cripple a fresh network. Six
        # falsified interventions proved the warm lineage cannot learn the
        # sustained-push reflex in situ (outcome shaping never gets sampled by
        # local exploration from the statue prior), so lineage 2 trains from
        # scratch on the fully-audited stack — reliefs, push shaping, ramped
        # bursts, curricula from day 0 — with the ORIGINAL fresh-net LR.
        # RE-PIN 1e-5/"fixed" when this run converges to its plateau.
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        # ENTROPY RESCALE (2026-08-12, lineage-2 bootstrap): with the
        # straight-through std ceiling fixed, noise still parked at 0.97-0.99
        # for 9k iters and ep_len capped ~510 — because at std~1.0 the entropy
        # bonus pays 1.41 nats/dim x 10 x 0.008 = 0.113/step while the
        # newborn's ENTIRE net reward rate is 0.01-0.02/step: staying maximally
        # noisy out-bids the task 5-10x. (Lineage 1 ran 0.008 fine at std 0.33
        # because its reward rate ~0.08-0.12/step matched the bonus — the coef
        # must scale with reward rate.) 0.003 rebalances for this stack's
        # heavily-priced (low-net-reward) regime while keeping real exploration
        # pressure. Watch: success = noise finally descending; overshoot =
        # noise crashing <0.15 fast (premature collapse) -> nudge back up.
        self.algorithm.entropy_coef = 0.003
        # L/R mirror-symmetry loss to kill the learned CIRCLING + LIMP (both are
        # gait asymmetries; the robot is physically balanced, COM ~3mm off center).
        # Makes the policy equivariant under a sagittal mirror. Mirror map verified
        # via the contact Jacobian (all joints swap+negate). See kbot_legs/symmetry.py.
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
        from .. import symmetry

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=symmetry.data_augmentation_func,
            mirror_loss_coeff=1.0,
        )


@configclass
class KBotLegsTrackPPORunnerCfg(KBotLegsRoughPPORunnerCfg):
    """Reference-cycle tracking pilot (AMP_PLAN stage 2b). Own experiment dir so the
    main run's watcher/watchdog never pick its checkpoints up."""
    experiment_name = "kbot_legs_track"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.save_interval = 100


@configclass
class KBotLegsTrackMultiPPORunnerCfg(KBotLegsTrackPPORunnerCfg):
    """Multi-cycle tracker (option 2): clean forward / backward / side-step / pivot cycles, command-selected."""
    experiment_name = "kbot_legs_trackmulti"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = int(os.environ.get("KBOT_TM_ITERS", "10000"))
        self.save_interval = 200
        # v2: published trackers keep exploring (entropy 0.005-0.01) and train 10-30k iterations; at 0.003 our
        # action noise collapsed to 0.1 by 600 iterations and learning stalled at 7-8 deg.
        self.algorithm.entropy_coef = float(os.environ.get("KBOT_TM_ENTROPY", "0.005"))


# ---------------------------------------------------------------------------- AMP (AMP_PLAN.md)
import dataclasses  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg  # noqa: E402

_IL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 9)))


@configclass
class KBotAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO settings + the AMP block consumed by amp.runner.KbotAmpRunner / amp.AMPPPO."""
    class_name: str = "AMPPPO"
    amp_cfg: dict = {
        "motion_files": [os.path.join(_IL_ROOT, "eval_watch", "amp_refs", "asimov_tracked_kbot.npz")],
        "style_weight": 6.0,   # RewTerm units (scaled by step dt inside AMPPPO)
        "disc_hidden_dims": (256, 256),
        "disc_activation": "relu",
        "disc_learning_rate": 5.0e-5,
        "disc_weight_decay": 1.0e-4,
        "logit_reg": 0.3,
        "reward_map": "log",
        "grad_pen_lambda": 10.0,
        "replay_size": 200000,
        "replay_frac": 0.5,
        "disc_steps": 8,
        "disc_batch": 2048,
        "update_interval": 1,
        "mirror_augment": True,
    }


@configclass
class KBotLegsAmpPPORunnerCfg(KBotLegsRoughPPORunnerCfg):
    """AMP lineage runner: same PPO/guard/symmetry settings as the lineage, AMPPPO on top.
    Own experiment dir so the main run's watcher/watchdog never see it."""
    experiment_name = "kbot_legs_amp"

    def __post_init__(self):
        super().__post_init__()
        base = self.algorithm
        alg = KBotAmpAlgorithmCfg()
        for f in dataclasses.fields(base):
            if f.name not in ("class_name",):
                setattr(alg, f.name, getattr(base, f.name))
        alg.class_name = "AMPPPO"
        alg.amp_cfg = dict(alg.amp_cfg)
        w = os.environ.get("KBOT_AMP_STYLE_W")
        if w:
            alg.amp_cfg["style_weight"] = float(w)
        ent = os.environ.get("KBOT_AMP_ENTROPY")
        if ent:  # walker v4: keep exploring (the tracker needed 0.005 not to stall)
            alg.entropy_coef = float(ent)
        files = os.environ.get("KBOT_AMP_MOTION_FILES")
        if files:
            alg.amp_cfg["motion_files"] = [p if os.path.isabs(p) else os.path.join(_IL_ROOT, p) for p in files.split(",")]
        # KBOT_AMP_MIRROR=0 disables the rsl_rl mirror-symmetry loss for the AMP lineage. The torso COM
        # sits 2.4 cm to the robot's right (hardware-measured correction); a policy forced to be L/R
        # symmetric cannot brace correctly on both sides. Measured 2026-10-01: hardened pilot 2 survives
        # 20 N pushes 256/256 from -y and 1/256 from +y; with the COM centred, 256/256 both ways.
        # The judge's dataset is mirror-augmented, so walking style stays symmetric without the loss.
        if os.environ.get("KBOT_AMP_MIRROR", "1") == "0" and alg.symmetry_cfg is not None:
            alg.symmetry_cfg.use_mirror_loss = False
            alg.symmetry_cfg.use_data_augmentation = False
            print("[amp-agent] mirror-symmetry loss OFF (KBOT_AMP_MIRROR=0)")
        self.algorithm = alg
        self.save_interval = 200


@configclass
class KBotLegsClipPPORunnerCfg(KBotLegsRoughPPORunnerCfg):
    """Time-indexed clip tracker (dataset expansion: backward / sideways / pivots / stops)."""
    experiment_name = "kbot_legs_trackclip"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 200
        # policy obs has an extra clip_ref block that symmetry.py's mirror does not know about
        if self.algorithm.symmetry_cfg is not None:
            self.algorithm.symmetry_cfg.use_mirror_loss = False
            self.algorithm.symmetry_cfg.use_data_augmentation = False
