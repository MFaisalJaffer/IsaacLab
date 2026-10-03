"""K-Bot v2 LEGS-ONLY (10 DOF) articulation config for Isaac Lab.

Deployment target: torso + both legs, no arms. Mirrors the MJX
``kbot-v2-legs`` asset used in ksim training:
  - mass/inertia already hardware-corrected in the URDF (total 13.06 kg),
  - velocity-dependent (T-V curve) actuators ported via TVCurveActuator,
  - per-joint kp/kd/armature from the legs metadata + MJCF.

Joint -> motor-type mapping (NOTE: legs use hip_roll_04, unlike the full
kbot which uses hip_roll_03):
  hip_pitch_04, hip_roll_04, knee_04 -> robstride_04 (peak 18.70 / brake 22 Nm)
  hip_yaw_03                         -> robstride_03 (peak  9.35 / brake 11 Nm)
  ankle_02                           -> robstride_02 (peak  9.35 / brake 11 Nm)
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets.robots.kbot_tv_actuator import TVCurveActuatorCfg

KBOT_LEGS_USD = os.path.join(os.path.dirname(__file__), "temp_kbot_legs_usd", "robot.usd")

# Reset / target pose: ALL JOINTS ZERO — straight-leg flat-foot stance.
# This matches the MJX legs asset's JOINT_TARGETS (all 0.0, see
# ksim_kbot/walking/walking_legs.py). Because the Isaac action term uses
# use_default_offset=True and every joint_deviation_l1 reward measures against
# the default joint pos, this single pose is simultaneously the reset pose, the
# action offset (neutral action -> straight legs), and the deviation-reward
# target — exactly how MJX uses JOINT_TARGETS.
if os.environ.get("KBOT_BENTKNEE") == "1":
    # (EXPERIMENT) original bent-knee stance — tests whether the zero default
    # pose is what makes the legs robot circle instead of walk forward.
    _LEGS_INIT_JOINT_POS = {
        "dof_right_hip_pitch_04": math.radians(-20.0),
        "dof_right_hip_roll_04": 0.0,
        "dof_right_hip_yaw_03": 0.0,
        "dof_right_knee_04": math.radians(-50.0),
        "dof_right_ankle_02": math.radians(30.0),
        "dof_left_hip_pitch_04": math.radians(20.0),
        "dof_left_hip_roll_04": 0.0,
        "dof_left_hip_yaw_03": 0.0,
        "dof_left_knee_04": math.radians(50.0),
        "dof_left_ankle_02": math.radians(-30.0),
    }
else:
    _LEGS_INIT_JOINT_POS = {
        "dof_right_hip_pitch_04": 0.0,
        "dof_right_hip_roll_04": 0.0,
        "dof_right_hip_yaw_03": 0.0,
        "dof_right_knee_04": 0.0,
        "dof_right_ankle_02": 0.0,
        "dof_left_hip_pitch_04": 0.0,
        "dof_left_hip_roll_04": 0.0,
        "dof_left_hip_yaw_03": 0.0,
        "dof_left_knee_04": 0.0,
        "dof_left_ankle_02": 0.0,
    }

# Per-joint-type spec: (motor_type, kp, kd, armature).
#   kp/kd: UNIFORM kp=150, kd=5.0 across all 10 leg joints (manual retune; was
#   per-joint 150/200/100/150/40 kp & 8/8/4/8/8 kd from kbot-v2-legs metadata).
#   The kd here is the CEILING — randomize_actuator_gains scales it downward-only
#   (0.75-1.0x -> kd in [3.75, 5.0], never above 5.0). See rough_env_cfg.py.
#   armature from the legs MJCF default classes (motor_04=0.007, _03=0.005, _02=0.0015).
_LEG_JOINT_SPEC = {
    # DE-CHATTER @194.4k (2026-07-17): kp 150 / kd 5 on the low-inertia joints
    # (yaw 03, ankle 02) produced a ~15-25 Hz servo limit cycle in BOTH sims
    # (HIL trace-diff: yaw law-corr -0.66, qd tick-flips 50%/36%, |qd| p95
    # 13.5 rad/s; kd*qd damping demand ~67 Nm vs ~8 Nm T-V available = clamp-
    # killed damping; each sim averages the chatter into different effective
    # impedance -> the PhysX-walks/MuJoCo-limps gap; real HW matches neither).
    # kp 60 puts the servo's natural freq ~10 Hz, inside the 15 ms delay's
    # phase margin (150 put it at ~14-15 Hz = the limit-cycle band); kd 3 stays
    # near critical and inside the T-V envelope at sane speeds. Hips/knees (04,
    # high inertia, no chatter: 17-22% tick-flips) unchanged. Gains export via
    # legs_policy_meta.json so the rig follows automatically; REAL-ROBOT servo
    # firmware must accept these values (hardware team). Probe gates @+6-8k:
    # tick-flip <25% on 02/03, |qd| p95 <6, stand/walk batteries hold.
    # RIG HANDOFF 2026-08-20 ("hardware fixed, final gains"): torque-constant
    # bug fixed — commanded kp now REAL (126-150 Nm/rad measured; kp unchanged
    # here). kd set to the rig's measured HARD MECHANICAL CEILING: above these
    # the real robot limit-cycles at 30-45 Hz (backlash-driven, ~33 Hz coupled
    # frame resonance at ~half the single-joint thresholds). Per-joint now —
    # the 04 family is no longer uniform. DR keeps kd DOWN-ONLY from these
    # ceilings. kp/kd are a coupled pair: changing either -> tell the rig
    # (chatter threshold moves). Deploy zetas: hp 0.15 / hr 0.10 / yaw 0.67 /
    # knee 0.19 / ankle 0.66 — the policy IS the damper now.
    "hip_pitch_04": ("04", 150.0, 2.5, 0.007),
    "hip_roll_04": ("04", 150.0, 1.5, 0.007),
    "hip_yaw_03": ("03", 60.0, 1.0, 0.005),
    "knee_04": ("04", 150.0, 1.0, 0.007),
    "ankle_02": ("02", 60.0, 0.5, 0.0015),
}


# rig-fitted friction (2026-08-20 handoff §3), per joint type — TORQUES (Nm)
_VISCOUS_B = {"hip_pitch_04": 1.0, "hip_roll_04": 1.0, "knee_04": 0.2}
_COULOMB_FC = {"hip_pitch_04": 4.0, "hip_roll_04": 4.0, "knee_04": 0.6,
               "hip_yaw_03": 0.4, "ankle_02": 0.1}
# SERIES ELASTICITY (rig handoff #3 §1) — ANKLES ONLY: K_s = kp/(ratio-1)
# = 60/2.6 ~ 23 Nm/rad, damping 0.07. Tuned against the observable both worlds
# share, d(body)/d(ankle_encoder) ~ 3.6 (§6.5), NOT the unmeasurable true joint.
_SERIES_K = {"ankle_02": 23.0}
_SERIES_B = {"ankle_02": 0.07}


# COMMAND LATENCY — 10-25 ms on EVERY joint (lineage 11, 2026-09-29).
# Hardware evidence (rig REPLY4, archived 2026-08-20 Step Atlas traces, 100 Hz,
# suspended robot, 5.73 deg step via LegCmd): ten joints out of ten are still
# at baseline at +10 ms and clearly moving at +20 ms. No per-family structure.
# Cross-checked three ways: rig step onset 19-21 ms on all ten joints; rig yaw
# friction slope 20.2 ms (yaw has zero configured viscous, so slope = kp*tau);
# and our own calibration of that method reads a pinned delay back exactly
# (0.0 / 10.0 / 20.0 ms at 0 / 2 / 4 steps). N steps here = N x 5 ms real.
#
# WHAT THIS REPLACES, and why it matters: the previous table (hips 0-1, knee
# and yaw 1-3, ankle 3-5 steps) came from the rig's 2026-08-20 handoff, which
# called its figures "MEASURED per-family command latency". They were not
# measured. They were the fitting script's `lag_ms` SHAPE PARAMETER -- the time
# shift that minimised RMSE against the hardware trace, fitted jointly with Fc,
# b and play. The hips' "0 ms" was lag that Fc = 4.0 had absorbed; the ankles'
# "20 ms" matched reality by coincidence. Retracted by the rig 2026-09-29.
# Lineages 7R through 10 trained hips on 0-5 ms against a robot that has ~20.
#
# Do NOT widen the ankles toward 40 ms: the latencies do not add (the hardware
# traces show the ankles at the same 10-20 ms as everything else), and 40 ms at
# ankle kd 0.5 topples a passive stand in sim.
_DELAY_STEPS = {  # (min, max) physics steps @5 ms  ->  10-25 ms
    "hip_pitch_04": (2, 5), "hip_roll_04": (2, 5),
    "knee_04": (2, 5), "hip_yaw_03": (2, 5), "ankle_02": (2, 5),
}


def _build_leg_actuators() -> dict:
    actuators = {}
    for side in ("right", "left"):
        for jt, (motor, kp, kd, arm) in _LEG_JOINT_SPEC.items():
            jn = f"dof_{side}_{jt}"
            dmin, dmax = _DELAY_STEPS[jt]
            actuators[jn] = TVCurveActuatorCfg(
                joint_names_expr=[jn],
                motor_type=motor,
                stiffness={jn: kp},
                damping={jn: kd},
                viscous_b=_VISCOUS_B.get(jt, 0.0),
                coulomb_fc=_COULOMB_FC.get(jt, 0.0),
                series_k=_SERIES_K.get(jt, 0.0),
                series_b=_SERIES_B.get(jt, 0.0),
                armature=arm,
                tv_randomization=0.15,
                min_delay=dmin,
                max_delay=dmax,
            )
    return actuators


_LEGS_ACTUATORS = _build_leg_actuators()


KBOT_LEGS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KBOT_LEGS_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1_000.0,
            max_angular_velocity=1_000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Straight-leg standing height. MJX resets the base to z=1.01 with all
        # joints at 0; spawn a touch above (1.05) so the feet clear the ground
        # and the robot settles instead of penetrating on reset.
        pos=(0.0, 0.0, 1.05),
        joint_pos=_LEGS_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators=_LEGS_ACTUATORS,
)
