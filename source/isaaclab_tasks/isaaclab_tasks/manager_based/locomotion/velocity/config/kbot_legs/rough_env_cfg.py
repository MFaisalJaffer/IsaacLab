"""Rough-terrain locomotion env for the LEGS-ONLY K-Bot (10 DOF).

Subclasses the full-kbot rough env and swaps in the legs-only robot
(`KBOT_LEGS_CFG`, with the ported T-V curve actuators) plus the body-name
fixes the legs morphology requires:
  - root body is `floating_base_link` (full kbot used a massless `base`),
  - torso (for fall-contact) is `Torso_Side_Right`,
  - no arms → remove arm reward terms and arm body references.
"""

import os

from isaaclab.utils import configclass

from isaaclab_assets import KBOT_LEGS_CFG  # noqa: F401  (registered via robots __init__)

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..kbot.rough_env_cfg import KBotRoughEnvCfg
from . import mdp_gait

# Fixed gait clock frequency (Hz); MJX used a 1.25-1.5 commanded range.
# CALM-WALK lever #1 (1.4 -> 1.15) FAILED its gate @216.6k (2026-07-28,
# calm_walk_cadence_gate.txt) and is REVERTED: the f^2 swing-savings premise
# missed the STRIDE-LENGTH coupling (stride = v/f, so -22% cadence = +22%
# stride at fixed command) -> hip_pitch rms 9.8 -> 15.1 Nm (2.0x continuous!),
# sat 7 -> 57%, tilt 10.5, tracking 79%, shuffle (apex 1.7-2.0 cm). The f^2
# dividend never appeared (knee/ankle FLAT) because this robot's walk torque
# is STANCE/SUPPORT-dominated, not swing-dominated (consistent with the
# speed-independence finding). Slower clock = longer support = hotter hips.
# DO NOT lower the clock at fixed commanded speed. (It also retro-explains
# part of the full-dose run's hip_pitch shunt.) Next-cycle candidate that
# SHORTENS strides at low speed: speed-scaled feet_alternation
# step_separation (reserved, needs own STEP-0). Keep exporter at 1.4.
# KBOT_CALM=1 (2026-07-30, calm-from-scratch experiment, user-approved): grow
# the gait FROM SCRATCH in a slow regime — 1.0 Hz clock + commands capped 0.30
# m/s (stride = v/f stays SHORT, the inverse of the warm-cadence failure) +
# thermal pricing from iter 0. Full design: eval_watch/calm_scratch_plan.md.
# Deploy config is untouched when the env var is unset.
_GAIT_FREQ = 1.0 if os.environ.get("KBOT_CALM") == "1" else 1.4

_FEET = ["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]

# Legs joint names. NOTE: hip_roll is "_04" on the legs robot (full kbot uses
# "_03"), so the inherited reward joint lists must be retargeted.
_HIP_YAW_ROLL = [
    "dof_left_hip_yaw_03", "dof_right_hip_yaw_03",
    "dof_left_hip_roll_04", "dof_right_hip_roll_04",
]
_HIPS_KNEES = [
    "dof_left_hip_pitch_04", "dof_left_hip_roll_04", "dof_left_hip_yaw_03", "dof_left_knee_04",
    "dof_right_hip_pitch_04", "dof_right_hip_roll_04", "dof_right_hip_yaw_03", "dof_right_knee_04",
]

# Rigid-body link names that actually exist on the legs robot (no arm/forearm/
# bayonet bodies). Used to retarget per-link mass randomization.
_LEGS_LINKS = [
    "Torso_Side_Right",
    "KC_D_102L_L_Hip_Yoke_Drive", "KC_D_102R_R_Hip_Yoke_Drive",
    "RS03_5", "RS03_4",
    "KC_D_301L_L_Femur_Lower_Drive", "KC_D_301R_R_Femur_Lower_Drive",
    "KC_D_401L_L_Shin_Drive", "KC_D_401R_R_Shin_Drive",
    "KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT",
]


@configclass
class KBotLegsRoughEnvCfg(KBotRoughEnvCfg):
    def __post_init__(self):
        # build the full-kbot env first, then retarget everything to legs
        super().__post_init__()

        # ---- robot: legs-only with T-V actuators ----
        self.scene.robot = KBOT_LEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # SPAWN-EXECUTION FIX (2026-08-12, death_forensics.txt): the stock
        # reset_base z range (-0.1, 0.1) spawns bases as low as 0.62 m; with
        # touchdown compression (~5-10 cm) the bad draws pass through the
        # 0.55 m base_height kill line BY CONSTRUCTION — measured: 75% of all
        # base_height deaths at median step 15 (0.3 s), ~1/3 of ALL episodes
        # executed on arrival, policy irrelevant. Burned a third of training
        # experience for the whole lineage and arithmetically capped mean
        # ep_len at ~620-660 (both lineages' mystery ceiling). Spawning INSIDE
        # the death margin is not robustness DR. Keep every other spawn draw
        # (tilt, joints, yaw, velocity, above-nominal drops) as-is.
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.1)

        # ---- body-name fixes (legs root is floating_base_link, not base) ----
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/floating_base_link"
        # fall termination via contact on the real torso body (has a collider)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Torso_Side_Right"
        # Height floor: catch a "sitting"/collapsed pose that dodges the other two
        # terminations — torso upright (no bad_orientation) and held off the ground
        # by folded legs (no torso contact), but the base sunk low. Straight-leg
        # standing base is ~1.0; a bent-knee gait stays >~0.7; a sit/collapse drops
        # well below. World-frame height (valid while on flat terrain level ~0).
        # SPAWN-SETTLE GRACE (2026-08-12, death-forensics verdict): see
        # mdp_gait.base_height_after_grace — stand-commanded spawn settles
        # were a 100% deterministic death loop at step ~16 (31238/31253 with
        # DR nulled + all-stand); the floor now ignores the first 30 steps
        # (0.6 s) so the settle can complete and the skill become learnable.
        # Mid-episode standard unchanged. Wean grace_steps once established.
        self.terminations.base_height = DoneTerm(
            func=mdp_gait.base_height_after_grace,
            params={"minimum_height": 0.55, "grace_steps": 30,
                    "asset_cfg": SceneEntityCfg("robot")},
        )
        # KBOT_NOHEIGHT=1 (2026-08-13, user experiment — "lineage-3 overnight
        # hypothesis test"): remove the height floor ENTIRELY, replicating the
        # exact termination rules lineage 1 learned standing under (the floor
        # was a late-era anti-exploit; the from-scratch forensics chain showed
        # it forbids the stand-learning path, and post-exemption the collapse
        # just died by bad_orientation instead on the ENTRENCHED lineage-2
        # policy — a FRESH policy under no-floor rules is the clean test:
        # if standing forms overnight like lineage 1's did, hypothesis
        # confirmed). stand_height_slope stays (the known sit-exploit of the
        # no-floor era, priced as a slope). Toggle via the service file; drop
        # the env var to restore the floor (e.g., when resuming lineage 2).
        # (the actual None-ing happens AFTER the last base_height param edit
        # below — setting it None here crashed the later stand_grace_s line)
        # STAND-FROM-MOTION ONLY (2026-08-12, forensics part 2): spawn-drawn
        # stand commands are converted to walking within the first ~10 steps;
        # standing enters only via the 10 s mid-episode resample (decelerate-
        # to-stand from a walk — the learnable path). See walk_at_spawn.
        self.events.walk_at_spawn = EventTerm(
            func=mdp_gait.walk_at_spawn,
            mode="interval",
            interval_range_s=(0.1, 0.1),
            params={"command_name": "base_velocity", "spawn_window_steps": 10},
        )
        # STAND LEARNABILITY PACKAGE (2026-08-13, user-approved; see
        # stand_transition_corridor docstring — the stand command was a binary
        # cliff, collapse to 0.39 m within 0.5 s, stand_pose paid 0.000 for
        # the lineage's entire life): (1) every drawn stand becomes a 1.5 s
        # slow-walk deceleration corridor, then the true stand; (2) height +
        # orientation terminations grace 1.5 s after the stand onset; (3) the
        # burst engine never STARTS a push on a corridor or a <3 s-old stand.
        # Wean all three once stand_pose collects and stands hold.
        self.events.stand_corridor = EventTerm(
            func=mdp_gait.stand_transition_corridor,
            mode="interval",
            interval_range_s=(0.1, 0.1),
            params={"command_name": "base_velocity",
                    "decel_speed": 0.12, "decel_s": 1.5,
                    "spawn_window_steps": 12},
        )
        self.terminations.base_height.params["stand_grace_s"] = 1.5
        if os.environ.get("KBOT_NOHEIGHT") == "1":
            self.terminations.base_height = None   # lineage-3 hypothesis run
        self.terminations.bad_orientation.func = mdp_gait.bad_orientation_stand_grace
        self.terminations.bad_orientation.params["stand_grace_s"] = 1.5
        # STAND EXEMPTION companion (2026-08-13, user proposal — see
        # base_height_after_grace + stand_height_slope docstrings): sinking at
        # stand no longer terminates; it costs a per-step SLOPE instead, so a
        # crumpled stander has a continuous gradient back up and sitting is
        # never free. Wean: restore the stander floor at stand_pose ~0.3+.
        self.rewards.stand_height_slope = RewTerm(
            func=mdp_gait.stand_height_slope,
            weight=-3.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "base_velocity",
                    "target_height": 0.85},
        )
        # base_external_force_torque is active in the training cfg (only nulled in
        # _PLAY) and references the full-kbot "base" frame, which doesn't exist on
        # the legs robot -> retarget to the real root body.
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = (
                "floating_base_link"
            )
        # add_limb_masses randomizes per-link mass over a full-kbot body list
        # (incl. arm/forearm/bayonet bodies) -> retarget to the legs links.
        if getattr(self.events, "add_limb_masses", None) is not None:
            self.events.add_limb_masses.params["asset_cfg"].body_names = list(_LEGS_LINKS)

        # ---- Domain randomization transferred from the MJX deploy-fidelity sim ----
        # Isaac equivalents of the DR we layered into MJX (RANDOMIZATION_HANDOFF).
        # NOTE: action-latency has NO built-in Isaac mechanism — omitted here
        # (would need a custom action-delay buffer); everything else maps.
        #
        # #2 kd-ONLY, DOWNWARD-ONLY gain randomization. The base kd=5.0 (kscale_legs)
        # is the CEILING: scale by [0.75, 1.0] -> kd in [3.75, 5.0], never above 5.0.
        # kp is NOT randomized (omitting stiffness_distribution_params leaves it fixed
        # at 150). This is the safe subset of the gain DR we removed during the crash
        # fix: the full version also pushed kp +/-25%, and an underdamped low-kd +
        # high-kp draw blew up the unbounded L2 reward -> value-loss inf -> crash.
        # Now safe to reintroduce kd-down because (a) no kp variation, and (b) the
        # 3-layer numeric containment (raw-obs clamp + reward clamp + nan/loss guards
        # in rsl_rl) bounds any residual underdamping spike. Per-episode (reset).
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "damping_distribution_params": (0.75, 1.0),  # kd scaled DOWN only
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # kp-DR on the DE-CHATTERED joints only (02 ankle / 03 yaw), added
        # 2026-07-17 after the de-chatter rig gate PASS (HIL-sanctioned step 2):
        # scale kp 0.8-1.2 around the new nominal 60 -> 48-72 so the policy
        # cannot overfit one chatter-averaged impedance (real HW matches
        # neither sim's sub-tick behavior). Safety vs the crash-era gain DR
        # (kp +/-25% at kp150 + kd-down -> underdamped blowup): worst corner
        # here is kp72/kd2.25 ~ 9.5 Hz natural freq, inside the 15 ms delay
        # phase margin, plus the 3-layer numeric containment + watchdog now
        # exist. kd variation stays covered by the kd-down event above (the
        # two events compose: kd from default x [0.75,1.0], kp x [0.8,1.2]).
        # Hips/knees kp stays FIXED at 150 (high-inertia, no chatter, and the
        # crash history lives there).
        self.events.randomize_gains_small_joints = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*ankle.*"]),
                # (L6 cap check: 60 x 1.2 = 72 < 96 hardware ceiling — legal)
                "stiffness_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # kp-DR on the 04 joints (hip pitch/roll, knee), added same phase
        # (2026-07-17, user call): real firmware PD at kHz vs sim PD at 200 Hz
        # means effective impedance differs on EVERY joint — this completes
        # actuator-parameter DR coverage (T-V, delay, stiction, mass, friction,
        # kd-down, 02/03 kp already randomized). Band deliberately NARROWER
        # (0.9-1.1 -> kp 135-165): the crash-era blowup was kp +/-25% + kd-down
        # exactly here (underdamped high-kp/low-kd draws on the loaded joints);
        # containment + watchdog make a repeat survivable, but start narrow and
        # widen only if the gate is boring. No chatter risk on 04 (inertia puts
        # kp150 at ~2-3 Hz, far below the delay phase margin). NB hip roll is
        # T-V-saturated 80-95% of stance where kp doesn't bind — this DR mostly
        # exercises swing + unsaturated phases.
        self.events.randomize_gains_04_joints = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch.*", ".*hip_roll.*", ".*knee.*"]),
                "stiffness_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # PER-JOINT action clips = PHYSICAL joint limits (2026-07-20 thermal
        # fix; replaces the ±100 blowup guard with something strictly tighter).
        # Audit: at stand the policy railed knee targets 4 rad PAST the
        # hyperextension stop (±8 clip / 0.5 scale) and hip-roll targets 0.7
        # rad past the adduction stop -> ~20 Nm continuous stall on 4 motors
        # while "standing still" (sim doesn't model heat; hardware does).
        # A target that cannot cross the stop generates no stall torque at it,
        # by construction. Values = 2x joint_pos_limits (action = target/0.5;
        # all-zero defaults), queried from the live robot 2026-07-20.
        # ⚠ DEPLOY INTERFACE: the rig/port must adopt these per-joint clamps
        # (meta must carry them) — blanket ACTION_CLIP=8 alone re-enables
        # stop-pressing on hardware.
        self.actions.joint_pos.clip = {
            "dof_left_hip_pitch_04": (-2.094, 4.434),
            "dof_right_hip_pitch_04": (-4.434, 2.094),
            "dof_left_hip_roll_04": (-0.418, 4.538),
            "dof_right_hip_roll_04": (-4.538, 0.418),
            "dof_left_hip_yaw_03": (-3.142, 3.142),
            "dof_right_hip_yaw_03": (-3.142, 3.142),
            "dof_left_knee_04": (0.0, 5.410),
            "dof_right_knee_04": (-5.410, 0.0),
            "dof_left_ankle_02": (-2.514, 0.454),
            "dof_right_ankle_02": (-0.454, 2.514),
        }
        # Friction: base config left it FIXED (0.8/0.6). Enable real ranges
        # (this physics_material event is mode="startup", bucketed per-env).
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.1)
        # #3 contact-compliance ANALOG. PhysX has no MJX-style solref (contact
        # stiffness/damping). Approximate "the ground feels different" via
        # restitution. (Collider-offset randomization was tried and dropped —
        # PhysX rejects restOffset >= contactOffset; friction + restitution already
        # cover the accessible PhysX contact-behavior DR.)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.2)

        # Cap the push curriculum at 0.7 m/s (the robot's ~walk speed). The kbot
        # base ramps pushes to 2.0 m/s (~3x walk speed), which beat the policy
        # down (reward 42 -> 8 as push climbed past ~1 m/s) — same fix as the MJX
        # 0.7 m/s push cap. min/start/stop unchanged, so it still ramps in
        # gradually, just to a recoverable ceiling.
        if getattr(self.curriculum, "velocity_push_curriculum", None) is not None:
            self.curriculum.velocity_push_curriculum.params["max_push"] = 0.7
        # Command-aware push split (2026-07-13): STANDING envs get pushes scaled to
        # 35% (0.7 -> ~0.25 m/s = a realistic bump) while MOVING envs keep the full
        # curriculum push. Full-strength shoves at stand were fighting the stillness
        # objective (lifts tripled once the curriculum fast-forward removed the
        # accidental gentle-push regime). See mdp_gait.push_by_setting_velocity_cmd_scaled.
        if getattr(self.events, "push_robot", None) is not None:
            self.events.push_robot.func = mdp_gait.push_by_setting_velocity_cmd_scaled
            self.events.push_robot.params["standing_scale"] = 0.35
            self.events.push_robot.params["command_name"] = "base_velocity"
            self.events.push_robot.params["stand_still_threshold"] = 0.1
        # ---- JOINT-FRICTION DR Stage A (2026-07-13). The sim was stiction-free
        # (verified: PhysX DOF friction 0.0 everywhere) while the rig/hardware have
        # real dry friction — rig MJCF: ankles 0.1 Nm (67x the other joints). The
        # 120k rig test proved the cost: our stillest-ever sim policy (34 lifts/min)
        # fell 3/3 at a deterministic ~3.1 s on the rig — its fine ankle corrections
        # are swallowed by the stiction dead-band. Brackets per rig team: bracket
        # the known value with margin, don't chase it. abs-mode, per-episode draws
        # (per-joint independent -> L/R asymmetry covered). Armature DR unchanged
        # (kbot-level event); the old no-op friction scale event is harmless.
        self.events.randomize_joint_friction_ankles = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
                # 2026-08-20 handoff: fitted ankle Fc = 0.1 Nm — range confirmed
                # (brackets 0.1 with margin both ways).
                "friction_distribution_params": (0.03, 0.18),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        # RIG HANDOFF §3 friction now lives in TVCurveActuator as TORQUES
        # (coulomb_fc / viscous_b, Nm) — PhysX joint_friction is a UNITLESS
        # load multiplier and feeding Fc into it welded the hips (ablation
        # 2026-08-21: rig-validated ckpt 100% -> 0% alive on that term alone).
        # PhysX dof friction for the driven joints stays at the legacy ~0.
        # (old fitted-friction event block removed here)
        # RIG HANDOFF 2026-08-20 §3 (historical note): Coulomb friction FITTED
        # step responses (deploy gains, 5.7-deg steps, all ten joints):
        # hip pitch/roll Fc=4.0 Nm EFFECTIVE (bench stall 1.5-2 + lumped
        # drive-filter lag + rig sway — rig: "use as-is, it reproduces what
        # the policy will feel"), knee 0.6, yaw 0.4, ankle 0.1. Draws bracket
        # each fit with margin. (Viscous b lives in TVCurveActuator; the
        # 10-20 ms command latencies are inside the existing 0-40 ms delay DR.)
        # friction thermostat (L7R): ramps the three fitted-Coulomb events
        # 40% -> 100% on the coping gate; probes null it -> evals at 100%.
                        
        # COM CORRECTION (rig handoff #3 §3, measured x2: moment sweep and mass
        # table both put the model COM 11.6 mm BEHIND the ankle axis; hardware
        # is near-centred). Rig validated the fix: passive upright releases went
        # from "fell backward 4/4" to "stood >20 s, 6/6" on the rigid plant.
        # NB this retro-explains our own passive-stand check falling on
        # 2026-08-21 — that was the COM bias, not a broken probe.
        # ONE startup draw = correction (+11.6 mm) PLUS the §3 randomization
        # (x +-15 mm, y +-10 mm) around it. Startup, not reset: Isaac's
        # randomize_rigid_body_com writes all envs while sampling only the
        # resetting ones (shape crash in reset mode), and a robot's COM does
        # not change between episodes anyway — per-env fixed is the physical
        # truth, and 12288 envs cover the band every iteration.
        # FRAME FIX (rig reply to 3B, 2026-08-24): randomize_rigid_body_com
        # applies its offset in the BODY frame, and Torso_Side_Right is quat-
        # rotated +90 deg about z: local +x -> world +y (LATERAL), local -y ->
        # world +x (FORWARD). The previous "+11.6 mm on x" therefore moved the
        # COM SIDEWAYS (and was undersized anyway: the offset applies to the
        # 6.24 kg torso, so moving the 13.06 kg whole robot +11.6 mm needs
        # 11.6*13.06/6.24 ~ 24.3 mm of torso shift — exactly the rig's MJCF
        # edit, y: -0.000957 -> -0.025257). Correct mapping:
        #   world-forward correction +24.3 mm torso  => local y = -0.0243
        #   world fore-aft DR +-15 mm                => local y +-0.015
        #   world lateral  DR +-10 mm                => local x +-0.010
        self.events.correct_torso_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="Torso_Side_Right"),
                    "com_range": {"x": (-0.010, 0.010),
                                  "y": (-0.0243 - 0.015, -0.0243 + 0.015),
                                  "z": (0.0, 0.0)}},
        )

        # ---- remove arm-specific terms (no arms) ----
        self.rewards.joint_deviation_arms = None
        # foot_impact_penalty also lists forearm bodies -> keep feet only
        self.rewards.foot_impact_penalty.params["sensor_cfg"].body_names = list(_FEET)

        # ---- critic obs body_poses references "base" -> floating_base_link ----
        self.observations.critic.body_poses.params["asset_cfg"].body_names = [
            "floating_base_link",
            *_FEET,
        ]

        # ---- retarget reward joint lists to legs naming (hip_roll_04) ----
        self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=_HIP_YAW_ROLL
        )
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=_HIPS_KNEES
        )
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*"]
        )

        # ---- reward rebalance (option 1): keep balance, FORCE locomotion.
        # Option A (soft termination -75 + low deviation) destabilized the stander
        # WITHOUT inducing stepping (feet_air_time can't bootstrap a gait — it only
        # pays for steps already taken cleanly). Reward crashed 11->0.1, ep len
        # 987->~400, feet_air_time stayed ~0. So instead: restore stability, keep
        # the stepping carrot, and make standing actively unrewarding by commanding
        # movement almost always + raising the velocity-tracking weight.
        self.rewards.termination_penalty.weight = -150.0      # was -200; -75 (A) fell too much
        self.rewards.joint_deviation_hip.weight = -0.5        # restored (A had -0.25)
        self.rewards.joint_deviation_ankles.weight = -0.5     # restored
        self.rewards.feet_air_time.weight = 1.0               # keep boosted stepping carrot
        self.rewards.track_lin_vel_xy_exp.weight = 3.0        # was 2.0 — reward moving more
        self.rewards.track_ang_vel_z_exp.weight = 2.0         # was 1.0 — punish uncommanded yaw
        # Straighten the torso. Measured a +29 deg forward lean (falling-forward
        # walk); the base -1.0 was too weak (a 29 deg lean only cost ~-0.23/step).
        # -3.0 makes leaning expensive so it learns upright push-off; near-zero cost
        # when upright.
        self.rewards.flat_orientation_l2.weight = -3.0
                                                              # (attacks the residual circling/lean)
        # 0.15 (was 0.05): with explicit stand-still PENALTIES now in place (feet-
        # planted + joint-vel, gated to cmd~0), give standing 3x more training signal
        # so a quiet planted stand actually develops. Safe from the old stand-still
        # attractor because these are penalties (not a stand bonus) and the velocity-
        # tracking weights (3.0 lin / 2.0 ang) keep walking strongly preferred on move.
        self.commands.base_velocity.rel_standing_envs = 0.20  # more standing practice
        # ---- direct angular-velocity commands (NOT heading tracking) ----
        # heading_command=True trains the robot to chase a target heading; an
        # immature biped never reaches it, so the derived wz stays nonzero all
        # episode and it never learns to go straight (wz=0). Diagnosed via a
        # straight-command eval: 16/16 envs circled at ~1.1 rad/s. Direct wz
        # commands (incl. frequent ~0 = "go straight") teach straight-line
        # control — and match the MJX setup that walked.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.heading = None  # required when heading_command=False
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ---- flat-terrain mode: KBOT_FLAT=1 ----
        # Replaces the curriculum terrain with a flat plane and disables the
        # terrain_levels curriculum. HISTORY (corrected 2026-07-16): this began
        # as an experiment testing whether the curriculum's survive-by-not-
        # advancing incentive caused the circling — it did NOT; the circling was
        # a policy-learned L/R asymmetry, fixed by the rsl_rl mirror-symmetry
        # loss (see kbot_legs/symmetry.py), and it stayed fixed. Flat mode was
        # simply retained while the campaign focused on flat-floor stand/HIL
        # work. Re-enabling terrain: drop the env var (service-side), consider
        # KBOT_MAXINIT=0 and a blind-feasible sub-terrain mix — this policy has
        # NO height scan (43-d proprioceptive obs).
        if os.environ.get("KBOT_FLAT") == "1":
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.curriculum.terrain_levels = None
        else:
            # BLIND-FEASIBLE terrain mix (prepared 2026-07-17, activates when
            # KBOT_FLAT is dropped from the service): this policy has NO height
            # scan (43-d proprioceptive obs), so stairs (5-23 cm) and boxes are
            # untraversable — those envs would just demote and idle. Keep the
            # blind-walkable 40% of the default mix at full proportion:
            # random roughness + pyramid slopes. Pair with KBOT_MAXINIT=0 in
            # the service so every robot bootstraps on the easiest tiles.
            if getattr(self.scene.terrain, "terrain_generator", None) is not None:
                from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

                tg = self.scene.terrain.terrain_generator
                keep = {}
                for name, sub in tg.sub_terrains.items():
                    if "rough" in name or "slope" in name:
                        keep[name] = sub
                # EXPLICIT FLAT tiles (25%), added @230.8k terrain-gate FAIL
                # (2026-07-17): the rough+slopes-only trim removed flat ground
                # from the training distribution entirely — the curriculum's job
                # is to push robots OFF the easy (quasi-flat) tiles, so flat
                # stand/walk decayed as levels climbed (flat gyro 0.11 -> 0.28,
                # walk 130% -> 84% in 10k iters) while the DEPLOY surface is a
                # flat floor. A flat sub-terrain column stays flat at EVERY
                # curriculum level -> flat skills stay in-distribution no matter
                # how high the rough curriculum climbs (terrain becomes additive,
                # not substitutive). Re-gate @~240k: flat metrics recovering
                # toward the 220k records AND terrain_levels still climbing.
                keep["flat"] = MeshPlaneTerrainCfg(proportion=0.25)
                for name, sub in keep.items():
                    sub.proportion = 1.0 / len(keep)
                tg.sub_terrains = keep

        # ---- (EXPERIMENT) curriculum start level: KBOT_MAXINIT=<n> ----
        # The velocity-env default max_init_terrain_level=5 scatters FRESH robots
        # onto level-~3 terrain (mean 2.86, measured), which a can't-walk policy
        # fails → demoted → it learns NOT to advance (circles). KBOT_MAXINIT=0
        # starts every robot on the EASIEST tile so it bootstraps a walk first,
        # then climbs as it improves (the curriculum the way it's meant to work).
        _maxinit = os.environ.get("KBOT_MAXINIT")
        if _maxinit is not None and getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.max_init_terrain_level = int(_maxinit)

        # ---- reward rebalance (option 2): explicit gait clock to MAKE it walk.
        # Velocity-tracking alone -> stand/sway local optimum (A and option 1 both
        # confirmed it). This ports the MJX FeetPhaseReward: an anti-phase gait
        # clock gives each foot a target lift/plant trajectory and rewards the foot
        # for tracking it, which directly bootstraps an alternating gait. The phase
        # is also added to the policy obs so it can time its steps (obs 39 -> 43,
        # hence a FRESH run — a 39-d checkpoint cannot resume).
        self.observations.policy.gait_phase = ObsTerm(
            func=mdp_gait.gait_phase_obs,
            params={"gait_freq": _GAIT_FREQ, "command_name": "base_velocity",
                    "stand_still_threshold": 0.1},
        )
        self.rewards.feet_phase = RewTerm(
            func=mdp_gait.feet_phase_reward,
            weight=3.5,  # was 2.0 — boosted so clean alternating stepping out-rewards
                         # the lurch-forward optimum (velocity-track weight is 3.0).
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(_FEET)),
                "gait_freq": _GAIT_FREQ,
                # GENTLE-WALK (@101k, 2026-07-24). Baseline probe: leg torque is a
                # flat 8-10 Nm rms REGARDLESS of speed (0.14-0.48 m/s) -> heat is a
                # gait-STYLE cost, not propulsion; all _04+_02 joints run 1.25-1.62x
                # REAL continuous (hip_pitch 10.5/7.5=1.40, ankle 8.1/5.0=1.62).
                # Phase 1a (0.12->0.08 + knee 0.55->0.35, ONE-SHOT): COLLAPSED @110k
                # (terrain 2.2->0.32, ep_len 810->250, bad_orientation falls ~22).
                # LESSON: the 12 cm lift is largely LOAD-BEARING for terrain
                # clearance (cf. tv_headroom -5.0 vigor collapse), not luxury.
                # Phase 1b HELD @141k (terrain 2.19, 0 falls, knee 1.25->1.00x, ankle
                # 1.62->1.48x, NO knee->hip-pitch shift). Terrain build frozen as the
                # robust fallback (eval_watch/terrain_fallback.txt).
                # Phase 2 · FLAT-ONLY gentle branch (@142k, 2026-07-24, user call).
                # 0.06 @146k probe: cooled ankle to ~1.3x BUT the feet SHUFFLE — swing
                # apex only ~1.2 cm, foot in ground contact ~85-91% of the time (vs the
                # original ~2.7 cm / 79% stance). Part of the "gentleness" was the policy
                # deciding to stop picking up its feet. Bumped back to 0.12 (@146k,
                # 2026-07-24, user). 0.12 alone did NOT fix the shuffle (1.7 cm) —
                # the knee (restored 0.55 @177k) is the actual clearance mechanism.
                # On flat there's no collapse risk. Re-probe clearance + torque: expect
                # the ankle/knee to settle warmer than the shuffle but with real steps.
                # NB feet_phase historically PLATEAUS (~0.4 of max) so actual lift <<
                # target; if 0.12 still under-lifts, the plateau (sensitivity/weight)
                # needs fixing so the target actually controls the lift.
                "max_foot_height": 0.12,
                "foot_offset": 0.05,        # measured planted foot-body z
                "sensitivity": 0.01,
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
                # gate the gait reward by actual translation in the commanded
                # direction -> no credit for marching/circling in place.
                "translation_gated": True,
                # was 0.25 — too tight: at the policy's ~0.4 vel error the gate
                # multiplied the gait reward by ~0.5, choking it (can't earn gait
                # reward without good velocity, can't get velocity without a gait).
                # 0.6 loosens that chicken-and-egg so the gait can bootstrap.
                "translation_gate_sensitivity": 0.6,
            },
        )
        # ---- anti-hop: penalize a flight phase (both feet off the ground at once).
        # The diagnosis showed feet_phase plateauing at ~0.4 of max in BOTH the old
        # and new runs -> a hopping/poorly-phased gait. feet_phase rewards LIFTING;
        # this penalizes BOTH feet leaving -> together they shape one-foot-swings-
        # while-other-plants (a real walk). 0 for a clean gait; tunable weight.
        self.rewards.flight_phase = RewTerm(
            func=mdp_gait.flight_phase_penalty,
            # -2.5 (was -1.0): the warm-start showed -1.0 was too weak (only ~0.1 of
            # reward) to break the hop. Stronger so walking strictly beats hopping at
            # the same forward speed, while feet_phase (+3.5) still rewards lifting
            # ONE foot -> lift-one-plant-other (a walk), not bounce-both (a hop).
            weight=-2.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
                "force_threshold": 1.0,
            },
        )
        # ---- fore-aft alternation: forces the legs to SWAP front/back, not just
        # bob in place. feet_phase (height-only) is blind to which foot leads, so a
        # one-leg-always-front shuffle can satisfy it. This rewards the fore-aft foot
        # separation tracking the same anti-phase clock -> the lifting foot must also
        # swing forward, the planted one sweeps back = a real alternating step.
        self.rewards.feet_alternation = RewTerm(
            func=mdp_gait.feet_alternation_reward,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(_FEET)),
                "gait_freq": _GAIT_FREQ,
                "step_separation": 0.30,   # peak fore-aft foot separation target (m)
                # STRIDE-SHORTENING lever (@~205k, 2026-07-29, user-approved; the
                # last calm-walk lane after cadence + headroom both gate-failed).
                # STEP-0 (stride_step0.txt) REVISED the mechanism: the policy never
                # overreaches — realized stride ~= natural v/f at every speed
                # (9.7 vs 10.6 cm @0.15) while the FIXED 0.30 target demands 2.8x
                # natural at low cmd. The unsatisfiable reward PULLS the legs apart
                # against a policy that refuses = commanded-position pressure
                # without displacement = isometric PD torque on hip_pitch (the
                # fore-aft sibling of the A-frame brace; hip_pitch rms 10.5 @0.15
                # vs 8.6 @0.50). Speed-scaling makes the demand satisfiable at the
                # natural stride -> the pull disappears; high-speed unchanged
                # (cap 0.30). GATE @+8-10k vs calm_walk_reference.txt: hip_pitch
                # rms @0.15 <= 10.0 (ref 10.5) trending down, apex >= 2.5 cm,
                # off-ground >= 7%, tracking >= 85%, falls 0, ankle not worse.
                # Watch-item from STEP-0: realized cadence 1.0-1.1 Hz < the 1.4
                # clock + p10 envs with ~0 amp — recheck at the gate.
                # GATE FAILED @208.6k (2026-07-29, stride_gate.txt) — REVERTED
                # (lever #3 falsified): the lever worked mechanically (realized
                # stride 7.8 cm ~= the new satisfiable target) but hip_pitch got
                # HOTTER, not cooler (rms 10.5 -> 12.0 @0.15, sat 9-12 -> 20%),
                # ankle sat 37-42 -> 54%, tilt +3 deg. The unsatisfiable 0.30
                # demand was a POSTURE PRIOR: constant fore-aft splay pressure =
                # wider effective support base = passive pitch stability. Relaxing
                # it bought nothing and cost active hip/ankle balancing. Gait
                # stayed healthy (apex 2.6-2.7, tracking 101-105%, 0 falls) — a
                # clean falsification, not a degeneration. The fixed 0.30 stays.
                "speed_scale_sep": False,
                "sep_min": 0.08,
                "sep_max": 0.30,
                "sensitivity": 0.03,
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
            },
        )
        # ---- #1 KILL hip-yaw circumduction. Gait analysis showed the locked-knee
        # walker clears its straight legs by swinging them out/around with the hip-yaw
        # joint (53-86 deg!). The shared joint_deviation_hip (-0.5 on yaw+roll) was
        # far too weak. A strong dedicated L1 penalty on hip-yaw (~0 deg is correct
        # for straight walking; bipeds turn by stepping, not yaw-twist) removes that
        # foot-clearance shortcut -> the robot must clear via the knee instead.
        self.rewards.hip_yaw_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw.*"])},
        )
        # (knee_straight penalty removed — it was for the straight-leg variant, which
        # fell over. Bent knees are back on via knee_swing +4.0 above.)
        # ---- #2 DEMAND knee flexion in swing. Pairs with #1: the only foot-clearance
        # path left is bending the knee, and this rewards it on the gait clock
        # (bend at mid-swing, straight at stance). Without it the knees stay locked.
        self.rewards.knee_swing = RewTerm(
            func=mdp_gait.knee_swing_reward,
            # 4.0 = BENT-KNEE (54k build). The straight-leg variant (0.0) quieted the
            # ankle but then COULDN'T BALANCE -> fell within ~1s. The active ankle IS
            # the balance mechanism; the bent-knee build stands 60s+ robustly.
            weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "gait_freq": _GAIT_FREQ,
                # GENTLE-WALK POSTMORTEM (2026-07-26): the "knee is a thermal lever,
                # clearance-independent" hypothesis was DISPROVED by direct probe.
                # Same 0.12 foot-lift target: knee 0.55 -> 2.7 cm apex / 8% airborne
                # (real steps); knee 0.35 -> 1.7 cm / 2% (shuffle) AND hip_pitch rms
                # rose 1.17->1.39x (policy burns hip flexion failing to lift a
                # too-straight leg). The knee bend IS the clearance mechanism — which
                # was this term's original design purpose (anti-circumduction). The
                # "knee thermal win" at 0.35 was mechanically the robot ceasing to
                # lift its feet. RESTORED to 0.55 (user, 2026-07-26). Calm-walk
                # gentleness must come from levers that do NOT trade against
                # clearance (cadence, headroom pricing, impact softening — see
                # eval_watch/gentle_walk_review.md lever hunt).
                "flex": 0.55,        # ~32 deg — restored; knee bend IS foot clearance
                # widened 0.1 -> 0.3: the tight kernel gave ~exp(-6)=0 reward AND ~0
                # gradient at locked knees, so nothing pulled them to start bending.
                # wider kernel gives a real gradient from straight toward the target.
                "sensitivity": 0.3,
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
            },
        )
        # ---- free the knee + hip-pitch for stepping. joint_deviation_hip_pitch_knee
        # (-0.1) penalized the knee & hip-pitch for leaving the straight zero-default
        # — one reason the knees barely flex (a straight hip-hike satisfies the
        # height clock for free). Now that feet_alternation DEMANDS a real forward
        # swing (which needs knee flexion), drop this to a light regularizer so it
        # doesn't fight it. hip roll/yaw stay pinned by joint_deviation_hip (-0.5).
        self.rewards.joint_deviation_hip_pitch_knee.weight = -0.02
        # ---- stand-still: stop the ankle fidget when commanded to stand. Measured
        # ~15 deg ankle jitter at ~380 deg/s + 11% foot-lifting while standing.
        # Penalize joint velocity ONLY when cmd~0 (a penalty, not a stand bonus, so
        # it can't recreate the stand-still attractor or discourage walking).
        self.rewards.stand_still = RewTerm(
            func=mdp_gait.stand_still_joint_motion,
            # -0.03: REVERTED from -0.15. -0.15 damped ankle velocity (400->193 deg/s)
            # but BROKE the planted stand (both-down 84%->40%, hip swing back to 43deg):
            # the fast ankle micro-adjustments are load-bearing for STATIC balance on
            # near-straight legs, so penalizing them forces stepping. The 84%-planted
            # stand with a little ankle jitter is the best result; keep -0.03 and accept it.
            weight=-0.03,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
            },
        )
        # ---- keep both feet planted when commanded to stand. The walking gait leaked
        # into standing (feet planted only ~30% of the time, 8% airborne = step in
        # place). track_lin_vel doesn't catch it (in-place stepping doesn't translate),
        # so penalize feet-off-ground directly, gated to cmd~0.
        # ---- POSITIVE stand-pose reward (the target-based fix). Penalizing motion/
        # foot-lift plateaued standing at ~50% planted regardless of strength. This
        # rewards HOLDING the straight-leg default pose when commanded to stand, gated
        # to cmd~0 so it can't create a stand-still attractor. Standing matters, so
        # this is a strong (+2.0) positive driver toward a quiet planted stand.
        self.rewards.stand_pose = RewTerm(
            func=mdp_gait.stand_pose_reward,
            # +15 (was +2.0): +2 plateaued standing at ~50% both-down, same as the
            # penalties — far too weak (MJX's working StandStillReward used +50). At
            # +15 with the wider 0.5 kernel the stand reward is ~2.5 while stepping,
            # ~8 at a good stand, 15 at a perfect one -> DOMINATES when standing, so
            # holding the pose beats stepping in place. Still gated to cmd~0 (no attractor).
            weight=15.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
                # 0.5 (was 0.25): 0.25 gave ~0 reward at the robot's actual standing
                # pose-error (~0.9); wider kernel makes it meaningful there with gradient.
                "sensitivity": 0.5,
                # 0.0 = POSITION-ONLY (the 54k build). The velocity term (0.01) tried
                # to quiet the ankle but that IS the balance -> the robot fell in ~1s.
                # DO NOT penalize ankle/joint velocity at stand. "Minimal movement" is
                # pursued only via keeping the feet PLANTED (stand_feet_planted below).
                "vel_coeff": 0.0,
            },
        )
        self.rewards.stand_feet_planted = RewTerm(
            func=mdp_gait.stand_still_feet_lift,
            # -3.0: measured on FLAT (the training terrain): -3.0 (54k build) plants
            # 89% both-down vs -6.0 (69k build) only 75% — the stronger penalty did
            # NOT help (penalty-strength plateaus; confirmed twice now). The REAL
            # fidget lever is the gait_phase stand pin -> (pi, pi) in mdp_gait
            # (mirror-invariant "both planted" signal). Penalizes foot-LIFTS only,
            # NEVER ankle velocity (that's the balance mechanism).
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
                "force_threshold": 1.0,
            },
        )
        self.rewards.stand_gyro = RewTerm(
            func=mdp_gait.stand_body_gyro,
            # HIL stillness gap (2026-07-11, instrumented): rig stand ran |gyro|
            # mean 1.07 / p95 2.1 rad/s + 6.8° tilt vs calm in Isaac. Penalize BODY
            # motion at stand (torso gyro + sway), NOT joint/ankle velocity (that
            # is the balance — penalizing it made the robot fall; see stand_pose
            # vel_coeff note above). Pairs with max_delay 8 (0-40 ms): the delay
            # recreates rig-busy standing in sim; this demands calm under it.
            # -1.0 -> -2.5 @105k: gyro stalled at ~1.03 in sim for 24k iters at
            # -1.0 (policy happily paid the tax); escalated per the lever ladder.
            weight=-2.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
                "lin_coeff": 0.5,
            },
        )
        # [GRAVEYARD] stand_foot_slide / stand_foot_anchor v1-v3 / stand_torque
        # — five stand-thermal/drift reward attempts, all plateaued or backfired
        # (Goodhart re-routing, tap-ratchet, correction-suppression, catch-
        # torque suppression). Removed for the 2026-07-21 FRESH RUN. Full
        # autopsies: memory isaac-legs-skating + archived BUILD_NOTES. LAW:
        # outcome channels at stand (slide/displacement/torque) are shared
        # between brace and balance — price CAUSES (action signatures), never
        # outcomes.
        # BENT-START RESET (@297k, 2026-07-19, user catch): the inherited
        # reset_joints_by_scale is a SILENT NO-OP on this robot — multiplicative
        # on the ALL-ZERO default pose, so every episode ever trained started at
        # exactly 0 deg. Measured hardware risk on 289400 (bent-start ladder):
        # power-on falls 3%/13%/19%/28% at 0/±6/±11/±20 deg set-down error
        # (survivors settle cleanly — the risk is falls, not thrashing).
        # Additive ±0.15 rad (±8.6 deg) makes power-on settling a trained
        # skill. GATE @~305k: bent20 probe falls <2/32, stand battery holds,
        # transient ep_len dip recovered. Rig-side: add bent-start protocol to
        # the next HIL bundle (all rig gates share the zero-pose blind spot).
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        }
        self.rewards.stand_hip_roll_brace = RewTerm(
            func=mdp_gait.stand_hip_roll_offset,
            # THE anti-A-frame term (@326.7k restart, 2026-07-21, user-mandated
            # with the key spec: 9 Nm continuous is hardware-FINE, so the
            # narrow stance's geometric load is acceptable — only the BRACE
            # excess must go). Targets the CAUSE: the constant hip-roll action
            # offset (brace ~1.4 vs gravity-honest ~0.12 — 10x separation;
            # deadband 0.3 sits between with huge margin). Hip-roll-scoped
            # (sagittal balance untouched), stand-gated (walking free),
            # calm-gated (push recovery free).
            # ESCALATED -0.5 -> -1.5 @~45k (2026-07-22, user call): at -0.5 the
            # brace paid ~0.55/step but LOST to stand_pose's rigidity advantage
            # (the brace pins q=0 rock-solid -> ~+0.8 higher stand_pose than a
            # wobblier relaxed stand), so the brace persisted unchanged (l2 @40k:
            # |action| 1.4, 20 Nm — identical to lineage-1). -1.5 -> brace pays
            # ~1.65/step: braced net 2.8-1.65=1.15 vs relaxed ~2.0 -> relaxed
            # WINS, flipping the arithmetic. Deadband 0.3 (unchanged) protects
            # legitimate hip-roll modulation (~0.12 gravity action); calm-gate
            # protects push recovery. RISK: if a quiet-stand micro-correction
            # needs |action|>0.3 while base is calm, the higher weight suppresses
            # it -> falls (correction-suppression, cf. stand_torque). WATCH
            # falls at the 50k gate; if they spike, raise deadband 0.3->0.4
            # before backing off weight. NB stacks with the @40.8k calm-gate ->
            # 50k gate attributes the COMBINATION (both target the brace, so
            # acceptable).
            # ZEROED @48.2k (2026-07-23): -1.5 was SAFE (upright 61/64) but
            # INEFFECTIVE (brace 20 Nm unchanged, policy paid & kept it).
            # Replaced by the TIME-WINDOWED stand_hip_roll_brace_ema (H3) which
            # can crank harder without clipping catches. See brace_hypotheses.md.
            weight=0.0,
            params={"command_name": "base_velocity", "stand_still_threshold": 0.1,
                    "vel_release_threshold": 0.15, "deadband": 0.3},
        )
        self.rewards.stand_hip_roll_brace_ema = RewTerm(
            func=mdp_gait.stand_hip_roll_brace_ema,
            # H3 (@48.2k, 2026-07-23, user): time-windowed replacement for the
            # instantaneous brace penalty. EMA (~0.5 s) of the commanded hip-roll
            # offset -> penalizes only the SUSTAINED brace, exempts transient
            # balance catches -> safe to crank. -2.5 (vs the -1.5 instantaneous
            # that did nothing): sustained brace EMA ~1.1-1.4 -> excess ~0.8-1.1
            # -> ~2.2/step; a brief catch barely moves the EMA -> ~0.
            weight=-2.5,
            params={"command_name": "base_velocity", "stand_still_threshold": 0.1,
                    "vel_release_threshold": 0.15, "deadband": 0.3, "ema_alpha": 0.04},
        )
        self.rewards.stand_action_magnitude = RewTerm(
            func=mdp_gait.stand_action_magnitude,
            # THERMAL FIX part 2 (@~325k, 2026-07-20): price |raw action| above
            # a 1.0 deadband at commanded stand only. Audit: stand actions
            # railed at ±8 (knees) / ~±2 (hip rolls) = 4 motors at continuous
            # stall; calm stand-holding needs |a| ~ 0.1-0.3, so the deadband
            # exempts everything sane and walking is cmd-gated out entirely.
            # -1.0: current railing costs ~0.8/step (strong), -> ~0 once sane.
            # Pairs with the per-joint action clips (structural stop-pressing
            # kill); together: expect stand knee torque ~20 -> <5 Nm, hip roll
            # ~19 -> toward the ~9 Nm geometric floor. See
            # mdp_gait.stand_action_magnitude for the full audit trail.
            weight=-1.0,
            params={"command_name": "base_velocity", "stand_still_threshold": 0.1,
                    "deadband": 1.0},
        )
        self.rewards.tv_headroom = RewTerm(
            func=mdp_gait.tv_headroom_penalty,
            # ANTI-RAIL-RIDING (@290.6k, 2026-07-18, HIL rec after the 289400
            # speed-gap autopsy): knees rode the T-V clamp 86-87% of the walk
            # -> engine-sensitive propulsion (sims agreed on 210000's 25%-sat
            # gait, split 106%-vs-73% on this one) AND a metal thermal hazard.
            # The 400/500Hz decisive test was NEGATIVE (flicker is solver-
            # intrinsic, rate doesn't fix it) so an under-limit gait is the
            # ONLY engine-robust fix. Prices motoring torque above 90% of the
            # live T-V limit; below = free. -5.0: pinned knees cost ~0.1-0.15/
            # step vs track_lin ~1.4. GATE @+10k (co-registered with anti-veer):
            # knee walk-sat < 50%, sim vx >= ~95%, stance-duty RISING toward
            # MuJoCo's 78% (flicker co-gate), veer < 0.02, stand battery holds.
            # -5.0 COLLAPSED THE FLEET in 5k iters (ep_len 712 -> 117, reward
            # 46 -> -6): on level-3 rough tiles knee push-off vigor is LOAD-
            # BEARING, not rail-riding luxury — the tax made the policy too
            # timid to catch itself. Rolled back to 292600, redeployed at -2.5
            # per the pre-registered soften branch; threshold raised to 0.93 so
            # only the deepest rail-riding pays. If ep_len dips again -> this
            # term needs terrain-conditional gating or flat-only pricing.
            # ESCALATED -2.5 -> -4.0 @319.9k (2026-07-20, pre-authorized): at
            # -2.5 knee-sat stayed 89% for 14k iters (penalty plateau) while
            # costing 9% walk speed. -4 is the time-boxed attempt to break the
            # rail-riding basin (210000 proved ~25% sat reachable); if knee-sat
            # not < 65% by ~330k, declare plateau + accept rail-riding (210000
            # stays deploy walker). Veer+bent PASSED first (veer 0.0156<0.02
            # @32env, bent 1/32) so this is the isolated lever. threshold 0.93
            # unchanged. NB the -5 collapse was the substep/window BUG (frac in
            # hundreds), now clamped — -4 clamped is untested but guarded.
            # Collapse guards: VF>10 or ep_len<300 -> roll back to pre-esc ckpt.
            # FRESH-RUN dose (2026-07-21): -2.5 (prevention, not cure — a
            # fresh policy never entrenches rail-riding, so the -4 escalation
            # pressure is unnecessary and risks suppressing push-off during
            # gait FORMATION, cf. the rough-tile vigor lesson).
            # CALM-WALK lever #2 FULL DOSE (-4.0/0.88) FAILED @208.8k (2026-07-28):
            # STEP-0 justified it (ankle at-clamp ~50-60% motoring-visible) and the
            # first ~5k were healthy, but the policy then restructured into the
            # PRE-REGISTERED failure trio (calm_walk_midrecovery.txt): gate-flip
            # dithering (tick-flips knee 28->40%, hip_roll 24->40%), hip_pitch SHUNT
            # (sat 7->27%, rms 12.3 > the 10.4 cap), shuffle relapse (apex 1.1-1.6cm,
            # off-ground 1-2%) + ASYMMETRIC limp (L knee rms 4.4 vs R 7.5) + tracking
            # 31% + tilt 12.2 + 2/32 falls. The falling penalty-bite that read as
            # "adapting" was the degenerate escape (stop walking = stop paying).
            # ROLLED BACK per ladder step 2: -2.5/0.93 restored, cadence 1.15 KEPT,
            # resumed from the healthy pre-deploy basin (198600, fallen run archived
            # at kbot_legs_rough_archive/fallen_calmwalk_fulldose). On FLAT the -4.0
            # dose is confirmed toxic even without terrain vigor — the escalation
            # lane is closed; ankle time-at-limit needs a different class of lever
            # (next-cycle candidate: speed-scaled stride / reserved withhold).
            weight=-2.5,
            params={"threshold": 0.93},
        )
        # W2 · torque_thermal_ema — NOT DEPLOYED (designed then held @95k,
        # 2026-07-24). Adversarial review (3 lenses) + inline re-derivation found
        # a BLOCKER: the _02 ankle rating (4.5 Nm, a 48%-of-peak GUESS) sits BELOW
        # the honest walking ankle torque (~7.8 rms) -> the term prices the gait
        # FLOOR at 3.0x, not the excess (inverse of the stand 12-Nm-deadband fix
        # that WORKED because 12 > the 9 Nm honest floor). ~69% of a bill ~48% the
        # size of track_lin lands on the ANKLE = the sole sagittal balance actuator
        # (no ankle-roll), whose load the gait clock (feet_phase 12cm lift +3.5,
        # knee_swing 32deg +4.0, feet_alternation 30cm +2.0) FREEZES -> the
        # sanctioned "walk slower" escape can't pay it (ankle load ~speed-indep),
        # leaving only "balance worse" = the stand_torque falls-3x failure with a
        # 1.5 s delay line. Calibration also rests on a guessed rating + a fuzzy
        # cross-probe torque number. The function stays in mdp_gait.py for a
        # corrected design (threshold ABOVE a demonstrated gentle-gait floor,
        # ankle spared/capped, real ratings) once the gait-clock lever is chosen.
        # See eval_watch/brace_hypotheses.md (W-series) + gentle_walk_review.md.
        self.rewards.ang_vel_error_l1 = RewTerm(
            func=mdp_gait.ang_vel_error_l1,
            # ANTI-VEER (@289.6k, 2026-07-18): rig caught systematic LEFT veer
            # +0.031 rad/s (~7 m circles, all eps) on the terrain build; sim
            # confirmed policy-learned (+0.08, 7/8 envs; seed visible at 190600
            # already). It was FREE under track_ang_vel_z_exp (exp kernel std
            # 0.5 pays 97.5% at 0.08 err — no small-error gradient) and the
            # mirror loss only constrains L/R gait, not heading. L1 gives
            # constant gradient to zero. -1.0: 0.08 veer costs 0.08/step —
            # real; a commanded turn tracked within 0.05 costs 0.05 (noise).
            # GATE @+10k: sim straight-walk veer -> |v| < 0.02 with tracking/
            # gait intact.
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.stand_action_rate = RewTerm(
            func=mdp_gait.stand_action_rate,
            # Lever 2 of the statue campaign (@152k): freeze the COMMANDS at stand,
            # not the joints — see mdp_gait.stand_action_rate. Global action_rate_l2
            # (-0.05) is too weak to matter at stand; this stand-gated term at -1.0
            # prices per-step command jitter (typical ||da||^2 ~ 0.4-2.5) without
            # crushing push recovery.
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
            },
        )
        self.rewards.stand_upright = RewTerm(
            func=mdp_gait.stand_upright,
            # Companion to stand_gyro, added @105k: gyro-only penalty bred a rigid
            # STATIC LEAN (9-10° tilt in sim AND rig — a lean is gyro-free). This
            # prices the lean itself, at stand only. At 10° tilt: pgrav_xy^2=0.030
            # -> 0.45/step at -15 (vs global flat_orientation_l2 -3.0 = 0.09/step).
            # L5 (2026-08-18, hardware deadband-cliff report): THE smooth servo
            # teacher — zero-deadband quadratic, -15 -> -80 so the gradient
            # exists at EVERY angle and beats the pose-kernel economics that
            # preserved the L4 lean (arithmetic: hip-fix posture costs ~0.85/
            # step in stand_pose; lean must cost more): 1 deg = 0.024/step
            # (no jitter-chasing), 3 deg = 0.22, 7 deg = 1.19. Replaces the
            # deadbanded stand_tilt_wall entirely — the wall's free zone +
            # cliff is exactly what the rig measured (no correction below
            # ~3 deg, jerky all-at-once past 4).
            weight=-80.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "stand_still_threshold": 0.1,
            },
        )

        # ---- preview render resolution for the eval-watcher's play.py.
        # The earlier VRAM failure was the *balanced* RTX mode (too many
        # pipelines), NOT the resolution — framebuffer memory is cheap relative
        # to the RT pipeline, so we can keep 1280x720 as long as play.py uses
        # --rendering_mode performance. No-op during headless training. ----
        self.viewer.resolution = (1280, 720)

        # ---- CALM-FROM-SCRATCH regime: KBOT_CALM=1 (2026-07-30) ----
        # Placed LAST so it overrides everything above. Design + pre-registered
        # gates: eval_watch/calm_scratch_plan.md. Science run (fresh, iter 0):
        # can a thermally sustainable walk (every joint <= 1.0x continuous;
        # ratings _04 7.5 / _02_03 5.0 Nm) exist for this morphology? Speed is
        # explicitly NOT a goal. _GAIT_FREQ resolves to 1.0 Hz under this var
        # (module-level gate) -> all clock consumers follow.
        if os.environ.get("KBOT_CALM") == "1":
            # slow-regime commands: stride = v/f stays short (0.30/1.0 = 0.30 m
            # worst case, 0.15-0.25 m typical). vy/wz shrunk proportionally.
            # BUG FIX @32k (2026-08-01): the original (-0.15, 0.30) range made the
            # stand/walk DEAD ZONE (|cmd|<0.1, where every gait reward gates OFF
            # and stand rewards gate ON) ~44% of walker envs — vs ~10% in the
            # fast build's +/-1.0 range. Half the fleet trained on contradictory
            # orders (track 0.05 m/s while being reward-classified as standing)
            # -> the creeping wide-lunge attractor seen in the 32k render; 0/32
            # could stand, stand_pose flat 0.00. Walk commands now START ABOVE
            # the 0.1 threshold: every walker is unambiguously a walker; the 20%
            # rel_standing envs (pinned exactly 0) own the stand skill. Reverse
            # walking dropped (never a calm-gait goal).
            self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.30)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.30, 0.30)
            # stride target sized to the regime (natural at ~0.22-0.25 m/s):
            # kills the fixed-0.30 isometric-pull mismatch AT the design point
            # (the speed-scaled version was falsified on warm-start; a fresh
            # policy FORMS around a correctly-sized fixed target instead).
            self.rewards.feet_alternation.params["step_separation"] = 0.22
            # THERMAL-PRICE ISOLATION TEST (attempt 3, 2026-08-02, user call).
            # Attempts 1-2 ran this at -1.5e-4 (1000x the inherited near-zero)
            # and NEITHER could walk: probes at 29.8k AND 56.4k both showed all
            # 32 envs falling under a sustained walk command (apex 33-42 cm,
            # max 82 cm, stance-duty 42-51% = tumbling, not stepping), while
            # the fast fresh control was walking well before that age. Two
            # candidate blockers were confounded: the 1.0 Hz clock and this
            # tau^2 price. This run isolates the PRICE — reverted to the base
            # -1.5e-7 with the clock still at 1.0 Hz and every other calm
            # element unchanged (cmds <=0.3, stride 0.22, track 2.0).
            #   WALKS  -> the effort penalty was suppressing the vigor needed to
            #             learn balance; re-introduce pricing AFTER the gait forms.
            #   FALLS  -> the 1.0 Hz cadence is the blocker (4th independent
            #             low-cadence failure) -> this morphology needs ~1.4 Hz.
            # GATE @~30-40k: alive-masked walk probe at cmd 0.15/0.30 — envs must
            # STAY UP (alive > 0/32), stance-duty toward ~75%, apex ~2-3 cm.
            self.rewards.dof_torques_l2.weight = -1.5e-7   # was -1.5e-4 (isolation test)

            # ---- ANTI-LEAN @21.2k (2026-08-03, user-approved) ----
            # PARTIAL RESULT of the isolation test: removing the tau^2 price WORKED
            # for locomotion — the policy now genuinely WALKS (32/32 upright,
            # 112% of commanded speed at cmd 0.15, roll ~0, base height constant).
            # (My earlier "ALL FELL" verdicts were a PROBE ARTIFACT — the sticky
            # 25-deg total-tilt mask; the user caught it from the render. Mask now
            # uses the env's own fall criteria; see walk_gap_probe.py.)
            # BUT it walks with a steady **+36 deg forward lean**, and that lean is
            # a continuous gravitational load on the hips -> torque roughly DOUBLE
            # the deploy build: hip_pitch 18.2 rms (2.4x continuous), hip_roll 18.8
            # (2.5x), knee 12.7 (1.7x), ankle 9.4 (1.9x); T-V sat 87-95% on three
            # joint pairs vs the deploy build's 8-47%. i.e. currently the OPPOSITE
            # of a calm gait, and the lean is the single cause.
            # flat_orientation_l2 is already at the historical fix value (-3.0, which
            # once cut a 29-deg lean to 1.4) and costs ~1.04/step at 36 deg — the
            # policy pays it, so the lean is buying propulsion. Escalate hard and
            # warm-continue (gait is formed; warm-start refines posture).
            #   UPRIGHT + torque drops -> calm regime back in play.
            #   LEAN PERSISTS or it stops moving -> 1.0 Hz forces falling-forward
            #   propulsion => cadence confirmed as the blocker, stop the experiment.
            # GATE @~+10k: pitch < ~15 deg AND still translating (>=60% of cmd),
            # then re-probe torque.
            self.rewards.flat_orientation_l2.weight = -10.0   # was -3.0

        # ---- SPEED-ADAPTIVE GAIT CLOCK: KBOT_ADAPT=1 (2026-08-04, user design,
        # Option A approved) ----
        # f = g(|cmd_vx|), linear between anchors; stride target = v/f(v) (the
        # human walk-ratio behavior). Deterministic in the observed command ->
        # no new obs dims, no Goodhart surface; phase INTEGRATES per env
        # (mdp_gait._phase_adaptive) because a stateless clock can't vary f.
        # ANCHORS CALIBRATED by the Part-A study + anti-lean gate (2026-08-04):
        # transfer grid proved 0.8-1.4 Hz all upright (0.8 marginal at 0.30);
        # cadence check proved the legs phase-lock to the clock (0.69/0.96/1.16/
        # 1.36 realized at 0.8/1.0/1.2/1.4); torque surface mildly favors slow;
        # user visually signed off 0.8@0.15. f_min=0.9 (cleanest slow cell) at
        # v_lo=0.15; the linear map gives ~1.15 at 0.30 — clear of the 0.8@0.30
        # marginal zone. STRATEGY (user-approved after the anti-lean gate):
        # WARM-START FROM THE DEPLOY BUILD (198600) — its gait transfers across
        # the whole band upright at ~35 Nm sum vs ~57 for every from-scratch
        # calm attempt (3/3 converged hot, duty ~59%); training refines the
        # proven-cheap gait per frequency instead of reinventing. Genuine
        # objective change -> plain-resume law satisfied.
        if os.environ.get("KBOT_ADAPT") == "1":
            _FREQ_MAP = (0.9, 1.4, 0.15, 0.45)   # (f_min, f_max, v_lo, v_hi)
            # SUSTAINED-PUSH bursts (@200k, 2026-08-04). ROUND-1 RESULT @216.2k
            # (sustained_gate_clean.txt): impulse reflex fully preserved (knee
            # 80 ms, peak tilt 5.1 deg, anti-brace 0.00) and TYPICAL bursts
            # survived in training — but the severe-standing case (20 N x 1.5 s)
            # still fells it, latencies unchanged. Cause: the hard tail
            # (>=20 N AND >=1.5 s) was ~7% of bursts x 20% standers — no
            # gradient mass; RL mastered the easy average it was given.
            # ROUND 2 (user-approved 2026-08-05): shift the distribution ONTO
            # the failure — severe cases become the common case.
            # RE-GATE @+10k: 20 N x 1.5 s standing lean — respond < 200 ms at
            # < 3 deg roll, survive the full lean; impulse reflex + stand
            # battery + anti-brace (brace_ema = 0.00) must hold.
            self.events.sustained_push = EventTerm(
                func=mdp_gait.sustained_push_bursts,
                mode="interval",
                interval_range_s=(0.1, 0.1),     # 10 Hz state-machine tick
                # NB force_range/duration_range below are the CEILING values; the
                # sustained_push_curriculum term (added 2026-08-05) overwrites
                # them live, ramping from (3,8)N/(0.3,0.8)s up to these as the
                # fleet proves it can cope. Full difficulty from step 0 stalled
                # twice at ep_len ~548 with noise_std running away.
                params={"force_range": (10.0, 30.0),   # ceiling; curriculum ramps to it
                        "duration_range": (0.5, 2.5),  # ceiling; curriculum ramps to it
                        "rest_range": (3.0, 8.0),
                        # ROUND 3 (2026-08-05, user-approved after burst_audit):
                        # the audit PROVED the event fires (19.6 N mean, 21% of
                        # env-steps, correct base link) — so round 1's "rare hard
                        # tail" diagnosis was wrong on the axis. The rarity is
                        # DIRECTION x STANCE: uniform heading x 20% standers put
                        # near-lateral-while-standing at only ~6.6% of bursts, and
                        # force/duration escalation cannot change that. Bias half
                        # the bursts onto +/-y (worst case: no ankle-roll DOF, so
                        # lateral resistance = hip-roll or side-step only).
                        "lateral_bias": 0.5,
                        "lateral_spread_deg": 20.0,
                        # RAMPED ONSET (2026-08-09 phase 2): force builds over
                        # 0.3-0.8 s instead of instant-on. The instant cliff
                        # kills every partial response, so the skill was never
                        # practicable (surgical program: 0/64 survivors).
                        # Mid-ramp, partial leans/steps SURVIVE and reinforce.
                        # Tighten toward (0,0) once the skill exists.
                        "ramp_range": (0.3, 0.8),
                        # STAND PACKAGE part 3: no burst starts on a corridor
                        # or a <3 s-old stand (see stand_transition_corridor)
                        "stand_shield_s": 3.0,
                        # TILT-SERVO HOLDS (2026-08-16, rig report): ramped
                        # base MOMENTS on established stands, 70% roll-biased
                        # (the no-ankle-roll axis) — creates the tilted-but-
                        # stable state the rig proved missing. Motion taxes
                        # release during holds; tilt terms stay LIVE.
                        "hold_torque_range": (1.0, 12.0),   # L5: floor 3->1 Nm — constant 1-3 deg error diet
                        "hold_duration_range": (1.0, 3.0),
                        "hold_rest_range": (4.0, 10.0),
                        "hold_ramp_range": (0.3, 0.8),
                        # 0.7 -> 0.5 (lineage 4): pitch is the axis the loose
                        # ankles break (correction routes through the slack
                        # joint) — equal roll/pitch hold practice.
                        "hold_roll_bias": 0.5,
                        # POSE RE-HOMING part 2 (lineage 4): 2.5 s post-burst/
                        # post-hold window where the standing travel taxes
                        # soften x0.25 (see mdp_gait._rehome_scale) so the
                        # corrective step home is cheap. stance_geometry and
                        # stand_pose stay live — they are the gradient home.
                        "rehome_grace_s": 2.5},
            )
            # ALWAYS-ON stand attitude wall (rig report fix; see
            # stand_tilt_wall docstring): NOT push/hold-released, deadband
            # 3 deg protects the push counter-lean; firm slope beyond kills
            # the settled-lean tolerance and teaches the static servo.
            # stand_tilt_wall REMOVED (L5): the deadbanded wall taught the
            # measured hardware cliff (free zone below deadband -> jerky
            # escape past it). Smooth servo duty moved to stand_upright -80.
            # PLANT-STABILITY RAMP RETIRED (2026-08-28, lineage 9). It annealed
            # K_s 150 -> a fixed 23, which the rig has now asked us to stop
            # doing: hardware went 23 -> 35 -> 52 Nm/rad in three days and a
            # PA6-CF20 reprint will move it again, so a single target is a
            # moving one (RIG_ANKLE_POSTFIX §4). K_s is now drawn per env,
            # log-uniform 20-120, inside randomize_joint_play below.
            # The ramp existed because a NEWBORN policy topples at K_s=23
            # (passively unstable: 33 Nm/rad at the body vs gravity's ~86).
            # Lineage 9 warm-starts from l8_gatefail_model_72400, which already
            # stands at 23 — the hardest sample in the new band — so widening
            # upward adds mostly EASIER plant and needs no ramp.
            self.curriculum.series_stiffness_level = None
            # EXCURSION pricing (lineage 8) — see stand_tilt_excursion.
            # Weight sized against the quadratic it complements: at 9 deg it
            # costs ~2.1/step (comparable to stand_upright's 2.0 there), at
            # 5 deg 0.2, at 2 deg 0.005 — i.e. it is nearly free for a calm
            # stand and expensive only for the sway we measured.
            self.rewards.stand_tilt_excursion = RewTerm(
                func=mdp_gait.stand_tilt_excursion,
                weight=-0.2,
                params={"asset_cfg": SceneEntityCfg("robot"),
                        "command_name": "base_velocity",
                        "tilt_ref_deg": 5.0},
            )
            # POSE RE-HOMING part 1 (lineage 4, 2026-08-16 — user/rig
            # observation, pose_rehome_probe verdict on model_55200: foot
            # geometry RATCHETS after pushes, width 36->46->51 cm, parked
            # flat 10 s; falls climb 0->3->14% across pushes). Deadbanded
            # width/stagger penalty at stand, push/hold-released so
            # protective stepping stays free; live during the re-home grace
            # (it IS the gradient home). nominal_width verified against the
            # default-pose stance at smoke test.
            self.rewards.stand_stance_geometry = RewTerm(
                func=mdp_gait.stand_stance_geometry,
                # TIGHTENED @53.2k (2026-08-22, user call after the hardware
                # stand readout): the quiet stand settles at 36 cm — legal
                # under the old 6 cm deadband (free to 32) and nearly free
                # afterwards, and stand_pose barely sees it (9 deg of hip roll
                # costs ~5% of a +15 kernel = 0.05/step). Deadband 6->3 cm and
                # weight -2 -> -4 makes 36 cm cost ~0.28/step, ~5x the pose
                # term's opinion, so the feet have a reason to come home.
                # Still push/hold-released: widening under a shove stays free.
                # GATE: stance <= ~30 cm at the stand battery AND 20N push
                # survival >= 97% (from 99.8%) — a narrower base is less
                # laterally stable, so if pushes regress this reverts.
                weight=-4.0,
                params={"asset_cfg": SceneEntityCfg(
                            "robot", body_names=".*_LEG_FOOT"),
                        "command_name": "base_velocity",
                        # measured default-pose width 25.6 cm (l4 mechanics
                        # check) — nominal must AGREE with stand_pose's joint
                        # target or the two rewards fight. 0.26 + 0.06 dead-
                        # band: free to ~32 cm (covers the lineage-3 comfort
                        # zone 33-36 at pennies), the post-push 46-51 cm
                        # ratchet pays 0.28-0.38/step.
                        "nominal_width": 0.26,
                        "width_deadband": 0.03,
                        "stagger_deadband": 0.03,
                        "stagger_coeff": 1.0},
            )
            # SCUFF PENALTY (2026-08-22, user/hardware: feet slide home rather
            # than step). Weight sized against the lift tax it competes with:
            # a 5 cm correction by sliding (~2.5 s at 2 cm/s) costs w*0.05;
            # one discounted step (lift tax x0.25 while off-nominal) costs
            # ~0.22 -> w=5 makes STEPPING the cheaper path, with a 1 cm/s
            # deadband protecting balance micro-motion.
            self.rewards.stand_foot_scuff = RewTerm(
                func=mdp_gait.stand_foot_scuff,
                weight=-5.0,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_LEG_FOOT"),
                        "asset_cfg": SceneEntityCfg("robot", body_names=".*_LEG_FOOT"),
                        "command_name": "base_velocity",
                        # DEADBAND FIX @65.4k (2026-08-22): 0.01 m/s was set
                        # from a WRONG read of the scuff rate ("2-4 cm/s per
                        # foot"); the measured rate is ~0.9 cm/s per foot, i.e.
                        # BELOW the deadband — the term only clipped peaks, so
                        # travel barely moved (10.3 -> 9.0 cm/5 s) while all
                        # foot repositioning got quietly discouraged (stance
                        # parked at 35.7 cm, real steps ~0.03/robot). 0.003 m/s
                        # puts the actual scuffing inside the priced region.
                        # RISK (the graveyard lesson): at this threshold the
                        # term can start touching genuine ankle micro-balance
                        # -> gate on falls/gyro/drift, not just on travel.
                        "speed_deadband": 0.003},
            )
            # FREE-PLAY DR (lineage 4): per-reset backlash draws; ankle band
            # cap is thermostat-driven 2 -> 16 deg (hardware truth: 15-deg
            # total ankle band, other joints ~0). See randomize_joint_play +
            # ankle_play_curriculum docstrings.
            # RIG HANDOFF 2026-08-20 §3: ankle free-play mechanically tightened
            # to 0.3 deg (was 15 -> 4 -> now 0.3). Draw U(0, 1) deg brackets it
            # with wear margin; near-rigid included by construction. NB §2's kd
            # ceiling is backlash-driven — keeping play modeled lets the policy
            # feel the mechanism behind the low-kd world. (§4 two-mass rotor/
            # load ankle model queued as next actuator work — single-mass
            # deadband cannot reproduce the measured ~17% step bounce.)
            # LINEAGE 9 (2026-08-28, rig RIG_ANKLE_POSTFIX §4): play widened
            # 0-1 -> 0-3 deg, and ankle K_s now drawn here too, log-uniform
            # 20-120 Nm/rad. Their ask in one line: "do not treat 52 as the
            # target, treat it as one sample from a distribution the policy
            # should be robust across." 20-120 spans every value they have
            # measured (23 / 35 / 52) plus headroom for the carbon-fill
            # reprint. Log-uniform because stiffness acts multiplicatively —
            # 20->40 is the same felt change as 60->120.
            # Play 0-3 deg brackets both the post-fix hardware (0.99 deg
            # hysteresis) and the ~2 deg of UNLOADED free play they measured
            # when the linkage slack sits open. NB their load-dependent band
            # (opens in swing, closes in stance) is NOT modelled yet — that
            # needs foot normal force piped into the actuator and is queued
            # as the next actuator change.
            self.events.randomize_joint_play = EventTerm(
                func=mdp_gait.randomize_joint_play,
                mode="reset",
                params={"ankle_play_range": (0.0, 0.0524),    # 0 - 3 deg
                        "other_play_range": (0.0, 0.0087),    # 0-0.5 deg
                        "series_k_range": (20.0, 120.0)},     # Nm/rad, log-uniform
            )
            # MOTOR-SIDE ENCODER obs (lineage 4): the POLICY sees the virtual
            # motor angle/velocity (blind inside the play band, like the real
            # motor-side encoder); the critic keeps privileged truth. Exact-
            # legacy at play=0, so old checkpoints probe unchanged.
            self.observations.policy.joint_pos.func = mdp_gait.joint_pos_rel_motorside
            self.observations.policy.joint_vel.func = mdp_gait.joint_vel_rel_motorside_filtered
            self.observations.policy.joint_vel.params = {"cutoff_hz": 4.0}
            # light buzz DR on top of the filter (user call, handoff #3 §2
            # "both"): the policy should not be brittle if the deployed filter
            # is ever mistuned. Uniform 0-0.5x of the measured failure RMS.
            self.observations.policy.joint_vel.noise = Unoise(n_min=-1.5, n_max=1.5)
            # ADAPTIVE RAMP (2026-08-05, user idea — mirrors velocity_push_curriculum
            # for the impulse pushes, but gated on PERFORMANCE not step count):
            # start at 3-8 N / 0.3-0.8 s and climb toward the ceiling above only
            # while the fleet is coping (mean ep_len > 620), backing off below 480.
            # Rationale: full difficulty from step 0 put the policy permanently at
            # its failure boundary — it stalled at ep_len ~548 TWICE with noise_std
            # running away (0.32 -> 0.45). Level is logged as
            # Curriculum/sustained_push_level so the ramp is visible in TB.
            # end_force capped 30 -> 25 N (2026-08-06, user call): the gate is
            # 20 N x 1.5 s, so 25 N keeps a 25% train>test margin, and the
            # thermostat design means ep_len can never rise above ~620 while
            # level headroom exists — capping is what lets the run CONVERGE
            # (level pins at 1.0, then ep_len climbs freely). The 25->30 N
            # stretch was the highest-risk/lowest-value part of the curve (both
            # pre-curriculum collapses happened there). To reopen later: raise
            # end_force on a converged build; the curriculum re-ramps gently.
            self.curriculum.sustained_push_level = CurrTerm(
                func=mdp_gait.sustained_push_curriculum,
                params={"start_force": (3.0, 8.0), "end_force": (10.0, 25.0),
                        "start_dur": (0.3, 0.8), "end_dur": (0.5, 2.5),
                        "ep_up": 620.0, "ep_down": 480.0,
                        "rate": 0.02, "review_every": 200},
            )
            # EARLY-STEP SHAPING (2026-08-08): pays the fitness valley between
            # brace-lunge and early side-step directly — relief-alone was
            # falsified by a full 15.7k-iter run (survivors 1-3/64 at 20 N
            # lateral stand, step still at ~600 ms / ~26 deg). Quadruple-gated
            # (burst-on + near-lateral + tilt<15 deg + stand) and pays
            # foot-to-foot spread along the push axis, so neither quiet-stand
            # dancing, the counter-lean brace, nor the late lunge can collect.
            # SMALL weight: a nudge for exploration to find, not an attractor.
            # REACHABILITY TUNE (2026-08-08, ~2h in): as first shipped
            # (spread_lo 0.45, tilt_cap 15) the pay region overlapped NOTHING
            # the policy does — statue spread is 0.38-0.40, the brace never
            # moves the feet, and the lunge widens only after ~26 deg tilt. TB
            # showed push_step stuck at ~1e-4 (jitter clipping the ramp edge).
            # Backward-chain instead: start the ramp 1 cm above statue stance
            # (sway collects immediately; "wider stance when pushed" IS the
            # proto-skill, so this exploit is the desired gradient) and let the
            # first phase of the existing lunge (crosses ~18 deg) collect a
            # sliver, pulling the widening earlier. Liftoff = push_step ~1e-3
            # within hours; still ~1e-4 next day = term is wrong deeper.
            self.rewards.push_step = RewTerm(
                func=mdp_gait.push_step_shaping,
                weight=0.5,
                params={"asset_cfg": SceneEntityCfg(
                            "robot", body_names=".*_LEG_FOOT"),
                        "command_name": "base_velocity",
                        "spread_lo": 0.41, "spread_hi": 0.65,
                        "tilt_cap_deg": 18.0},
            )
            # BRACE SHAPING (2026-08-08 deep analysis, user goal: "compensate
            # right away"): statics show 20-25 N needs only a 7-11 cm CoP
            # shift on the 0.40 m stance — a WEIGHT-SHIFT, not a step. H4's
            # torque-gate spent weeks teaching the opposite (hip-roll >12 Nm
            # at stand forfeits stand_pose -> yield-then-lunge became optimal).
            # Pays load asymmetry toward the downwind foot, same quadruple
            # gating as push_step; collectible from the CURRENT repertoire on
            # step one, so no reachability gap this time.
            # weight 0.4 -> 3.0 (2026-08-08 evening, user call): the statue
            # prior was installed by stand_pose at weight FIFTEEN over ~100k
            # iters; a 0.4 counter-signal is ~40x weaker than the habit's
            # builder, at a 3x-slowed LR. 3.0 matches track_lin_vel (the
            # config's strongest active driver), stays 5x under stand_pose so
            # the quiet-stand hierarchy is untouched (mask already zeroes brace
            # there), and the burst window is ~4% of env-steps so the return
            # shift stays VF-benign. Goodhart re-audit at higher stakes: bursts
            # are event-driven (can't be induced), pay clamps at 8 cm lean,
            # wrong-way and late (>18 deg) still pay zero.
            # RELIEF COMPLETION (2026-08-09 audit): the two standard-mdp statue
            # terms also release during bursts (the 7-term conflict; see
            # _push_release in mdp_gait). ADAPT-scoped so other configs keep
            # stock behavior. The kbot stand_* terms release via their own
            # bodies; these two need func repoints.
            self.rewards.flat_orientation_l2.func = mdp_gait.flat_orientation_l2_push_released
            self.rewards.joint_deviation_hip.func = mdp_gait.joint_deviation_hip_push_released
            # WALK-UPRIGHT SPEC (2026-08-10, user call): pitch<=2deg /
            # roll<=3deg while walking, deadbanded wall (see walk_upright_wall
            # docstring for the measured 7-deg lean habit + roll-floor calib).
            self.rewards.walk_upright_wall = RewTerm(
                func=mdp_gait.walk_upright_wall,
                weight=-4.0,
                params={"asset_cfg": SceneEntityCfg("robot"),
                        "command_name": "base_velocity",
                        "pitch_deadband_deg": 2.0,
                        "roll_deadband_deg": 3.0},
            )
            self.rewards.push_brace = RewTerm(
                func=mdp_gait.push_brace_shaping,
                weight=3.0,
                params={"asset_cfg": SceneEntityCfg(
                            "robot", body_names=".*_LEG_FOOT"),
                        "sensor_cfg": SceneEntityCfg(
                            "contact_forces", body_names=".*_LEG_FOOT"),
                        "command_name": "base_velocity",
                        "tilt_cap_deg": 18.0},
            )
            # *** WRENCH-CLOBBER FIX (2026-08-05, measured: wrench_flag_test.txt) ***
            # Articulation.set_external_force_and_torque sets ONE GLOBAL flag:
            #   if forces.any() or torques.any(): has_external_wrench = True
            #   else:                             has_external_wrench = False
            # base_external_force_torque is mode="reset" with force_range (0,0) —
            # a pure no-op that exists only to clear forces at reset — so EVERY
            # reset call zeroes that flag for the WHOLE articulation, and
            # write_data_to_sim then skips applying OUR burst forces to every env
            # that step. More envs -> a reset almost every step -> near-permanent
            # clobber. MEASURED: force present in the buffer on 83% of steps but
            # actually ENABLED on only 68% (64 envs) / 37.7% (4096 envs) — i.e.
            # training at 8192 envs was receiving roughly a third of the pushes it
            # was configured to deliver, which is why ep_len never dipped (user
            # spotted the anomaly and would not let it go — correctly).
            # Safe to remove: it applies zero force, and sustained_push_bursts
            # already zeroes its own wrench per-env in reset().
            self.events.base_external_force_torque = None
            # more standing exposure so the biased bursts land on standers:
            # 6.6% -> ~20% of bursts hit the failure mode (3x gradient mass).
            # 0.30 not 0.35 — 0.35 measurably diluted gait rewards in the
            # from-scratch calm run; this policy is already competent, but the
            # re-gate checks walk quality for dilution damage regardless.
            self.commands.base_velocity.rel_standing_envs = 0.30
            for _t in (self.rewards.feet_phase, self.rewards.feet_alternation,
                       self.rewards.knee_swing, self.observations.policy.gait_phase):
                _t.params["freq_map"] = _FREQ_MAP
            # stride follows the live frequency: natural walk-ratio at every speed
            self.rewards.feet_alternation.params["speed_scale_sep"] = True
            self.rewards.feet_alternation.params["sep_min"] = 0.10
            self.rewards.feet_alternation.params["sep_max"] = 0.32
            # command regime: span both anchors (slow AND fast), walkers above
            # the 0.1 stand threshold (dead-zone lesson)
            self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.50)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.30, 0.30)
            # speed devalued vs effort (calm goal); tau^2 price stays BASE
            # (-1.5e-7) from birth — the isolation test proved pricing from
            # iter 0 blocks locomotion learning; re-price after formation.
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            # posture: value TBD at launch from the anti-lean gate result
            # (deploy -3.0 vs escalated -10.0).

        # =====================================================================
        # LINEAGE 10 (2026-09-23, user GO) — reward-economy fix. Full audit in
        # eval_watch/LINEAGE10_PROPOSAL.md. Between 177k and 382k of lineage 9
        # the optimizer sold ~0.10 units of standing for ~0.90 of gait and the
        # worst-tilt gate went 4.3 -> 7.1 deg while total reward ROSE. The
        # standing pot was 2.9% of the gait carrots, stand_pose was saturated,
        # and ~40% of standing time was push-released.
        # =====================================================================
        if os.environ.get("KBOT_ADAPT") == "1":
            # Change 1: QUIET / DISTURBED split of standing time (50/50, per
            # episode, unobserved). Bursts, holds and shoves skip quiet
            # standing envs (mdp_gait.sustained_push_bursts /
            # push_by_setting_velocity_cmd_scaled). Walking pushed regardless.
            # quiet_fraction 0.5 -> 0.35 (2026-09-26, warm continue from 64.8k):
            # at 0.5 the standing half of the fleet saw half the push exposure
            # and the lean-and-brace response never formed (20 N survival
            # 44-67% vs 95-100% for L8-same-age/L9; 100% of survivors stepped).
            # 0.35 keeps a real quiet regime while standing time is disturbed
            # 65% of the time. Paired with stand_calm's full push release.
            self.events.draw_quiet_stand = EventTerm(
                func=mdp_gait.draw_quiet_stand, mode="reset",
                params={"quiet_fraction": 0.35},
            )
            # Change 2: a POSITIVE calm-stand kernel on the gait-carrot scale.
            self.rewards.stand_calm = RewTerm(
                func=mdp_gait.stand_calm, weight=4.0,
                params={"asset_cfg": SceneEntityCfg("robot"),
                        "command_name": "base_velocity",
                        "stand_still_threshold": 0.1,
                        "tilt_sigma_deg": 3.0, "gyro_sigma": 0.3},
            )
            # Change 3: modest re-weights (NOT 20x — the graveyard says large
            # standing taxes breed freezing/bracing).
            self.rewards.stand_upright.weight = -120.0          # was -80
            self.rewards.stand_tilt_excursion.weight = -0.5     # was -0.2 (tail guard)
            self.rewards.stand_stance_geometry.weight = -6.0    # was -4; nominal 0.26 kept
            # Change 5: splayed-walk tax, cause-priced (hip roll beyond 3 deg),
            # weight RAMPED 0 -> -2 on the ep-len thermostat.
            self.rewards.walk_hip_abduction = RewTerm(
                func=mdp_gait.walk_hip_abduction, weight=0.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*hip_roll.*"),
                        "command_name": "base_velocity",
                        "stand_still_threshold": 0.1,
                        "deadband_deg": 3.0},
            )
            self.curriculum.walk_hip_abduction_ramp = CurrTerm(
                func=mdp_gait.reward_weight_thermostat,
                params={"term_name": "walk_hip_abduction", "w_start": 0.0, "w_end": -2.0,
                        "ep_up": 620.0, "ep_down": 480.0, "rate": 0.02, "review_every": 200},
            )
            # From-scratch requirement: K_s draw starts stiff (60-120) and the
            # lower bound anneals to the rig's 20 on the same thermostat.
            self.curriculum.series_k_band = CurrTerm(
                func=mdp_gait.series_k_band_curriculum,
                params={"lo_start": 60.0, "lo_end": 20.0,
                        "ep_up": 620.0, "ep_down": 480.0, "rate": 0.02, "review_every": 200},
            )
            # Per-regime visibility: tilt + stance width for quiet / disturbed / walk.
            self.curriculum.regime_report = CurrTerm(
                func=mdp_gait.regime_report,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_LEG_FOOT"),
                        "command_name": "base_velocity", "stand_still_threshold": 0.1},
            )
            # Change 4: dead terms (measured <= 1e-4 or weight 0). Zero effect.
            self.rewards.stand_hip_roll_brace = None
            self.rewards.stand_height_slope = None
            self.rewards.dof_torques_l2 = None
            self.rewards.lin_vel_z_l2 = None

        # ---- EVAL-ONLY knob: KBOT_NOPLAY=1 (lineage 4) — null the free-play
        # DR event + its curriculum so probes measure the RIGID configuration,
        # comparable to every lineage-3 baseline. Never set on the training
        # service. Fixed nonzero bands in probes: set _play directly
        # (--ankle_play_deg pattern) with this var also set.
        if os.environ.get("KBOT_NOPLAY") == "1":
            if getattr(self.events, "randomize_joint_play", None) is not None:
                self.events.randomize_joint_play = None
            if getattr(self.curriculum, "ankle_play_level", None) is not None:
                self.curriculum.ankle_play_level = None
            if getattr(self.curriculum, "series_k_band", None) is not None:
                self.curriculum.series_k_band = None

        # ---- DEMO/STUDY knobs: KBOT_CLOCK=<Hz> [+ KBOT_DEMO_VX=<m/s>] ----
        # Render/eval any checkpoint at an off-nominal gait clock (the freq-map
        # study's transfer trick: the policy feels the clock via the phase obs).
        # KBOT_DEMO_VX also pins a fixed straight-walk command for clean videos.
        # EVAL-ONLY knobs — never set on the training service.
        _clk = os.environ.get("KBOT_CLOCK")
        if _clk:
            for _t in (self.rewards.feet_phase, self.rewards.feet_alternation,
                       self.rewards.knee_swing, self.observations.policy.gait_phase):
                _t.params["gait_freq"] = float(_clk)
                _t.params.pop("freq_map", None)
        _dvx = os.environ.get("KBOT_DEMO_VX")
        if _dvx:
            _v = float(_dvx)
            self.commands.base_velocity.ranges.lin_vel_x = (_v, _v)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.rel_standing_envs = 0.0
            self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
            # speed devalued vs effort (3.0 was the anti-lurch bump; lurch is
            # a pre-registered watch item at the 15k probe).
            self.rewards.track_lin_vel_xy_exp.weight = 2.0
            # tv_headroom stays -2.5/0.93 (proven fresh dose); gait geometry
            # (lift 0.12, knee 0.55) kept — at 1.0 Hz the same swing costs
            # ~half the torque (f^2); knee=clearance law holds.

            # ---- STAND-BOOTSTRAP FIX @30k (2026-08-02, user-approved) ----
            # Diagnosis (stand_gate_diag.py + stand_sampling_test.py): standing
            # could not bootstrap — a closed loop. Sampling was FINE (forced
            # resample gives ~20%), but standing-commanded envs COLLAPSE in
            # ~0.5 s while walkers now survive ~645 steps, so standing is only
            # ~0.9% of env-steps; and when it does occur the policy sits at
            # pose-err ~2 rad^2 where exp(-err/0.5)=0.02 pays ~nothing AND has
            # ~no gradient. Result: Episode_Reward/stand_pose exactly 0.00 for
            # 30k iters (control: the fast fresh lineage paid 0.67 from iter 0
            # — partly via dead-zone walkers, which the command-range fix
            # correctly removed, taking that accidental bootstrap with it).
            # Fix mirrors the knee_swing precedent (kernel 0.1->0.3 when it gave
            # 0 reward AND 0 gradient at the visited pose):
            # >>> BOTH REVERTED @56.4k (2026-08-02): the fix FAILED its gate and
            # REGRESSED walking. stand_pose stayed 0.001 flat for 26.7k iters;
            # re-diagnosis showed the pose kernel was NOT the binding factor
            # (it improved 0.033->0.079 as designed) — the killers are the OTHER
            # two conjunctive gates: standing&calm together = 0.3% of env-steps
            # and torque_gate 0.028 (H4 was built for an ALREADY-standing policy
            # to break its brace; from scratch it withholds the very reward that
            # would teach standing). Meanwhile rel_standing 0.35 flooded the
            # batch with instantly-toppling standers: ep_len PLATEAUED ~440 for
            # 26k iters vs 634-and-climbing before the change (walker survival
            # itself 786 -> 663 steps, so real regression, not just dilution).
            #   self.rewards.stand_pose.params["sensitivity"] = 2.0   # reverted (was 0.5)
            #   self.commands.base_velocity.rel_standing_envs = 0.35  # reverted (was 0.2)
            # Standing needs a different class of fix (relax H4's torque gate
            # during bootstrap, or warm-start the stand skill from the deploy
            # build which already solved it) — NOT another kernel tweak.


@configclass
class KBotLegsRoughEnvCfg_PLAY(KBotLegsRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # smaller, kinder terrain for visualization
        if getattr(self.scene, "terrain", None) is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
