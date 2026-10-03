"""Velocity-dependent (torque-speed / T-V curve) actuator for K-Bot.

Ports the MJX ``TVCurveMITActuators`` model (ksim_kbot/common.py) into Isaac
Lab. A real BLDC motor cannot deliver peak torque at high speed — back-EMF
reduces the available *motoring* torque as the motor spins in the same
direction the torque is applied. *Braking* torque (opposite to motion) is not
back-EMF limited and gets the full thermal/current ceiling.

Per physics step, per joint:
    max_tau_motoring = interp(|joint_vel|, omega_curve, tau_curve)   # T-V curve
    max_tau_braking  = braking_torque                                # constant
    limit = where(sign(effort) == sign(joint_vel), motoring, braking)
    effort = clip(effort, -limit, +limit)

Plus per-step T-V randomization: the motoring limit is scaled by a random
factor in ``[1 - tv_randomization, 1.0]`` (only ever weaker — models a hot/worn
motor). Matches MJX's ``tv_curve_randomization=0.15``.

The T-V curves are the SAME arrays defined in ksim_kbot/common.py:TV_CURVES,
copied verbatim (already ×0.85 derated for sim-to-real margin).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_cfg import DelayedPDActuatorCfg
from isaaclab.utils import LinearInterpolation, configclass
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    pass

# ── T-V curves, verbatim from ksim_kbot/common.py:TV_CURVES (×0.85 derated) ──
# omega in rad/s, tau in Nm. Motoring torque ceiling vs |joint speed|.
TV_CURVES: dict[str, dict[str, tuple[float, ...]]] = {
    "04": {  # GIM_8108 — hip_pitch, hip_roll, knee. Peak 18.70 Nm.
        "omega": (0.0, 7.9, 9.9, 10.5, 11.5, 12.0, 13.1, 14.1, 15.2, 15.7,
                  16.8, 17.8, 18.3, 18.8, 19.4, 19.9, 20.9, 21.5),
        "tau": (18.70, 17.85, 17.00, 16.15, 14.45, 12.75, 11.90, 10.20, 8.50,
                7.65, 6.375, 5.10, 4.25, 3.40, 2.55, 1.70, 0.85, 0.0),
    },
    "03": {  # GIM_6010 — hip_yaw. Peak 9.35 Nm.
        "omega": (0.0, 3.1, 4.7, 12.0, 14.1, 16.2, 18.3, 19.9, 21.5, 23.0,
                  24.6, 25.7, 27.2, 29.8),
        "tau": (9.35, 8.925, 8.50, 8.245, 7.65, 6.80, 5.95, 5.10, 4.25, 3.40,
                2.55, 1.70, 0.935, 0.0),
    },
    "02": {  # robstride_02 — ankle (same motor as 03). Peak 9.35 Nm.
        "omega": (0.0, 3.1, 4.7, 12.0, 14.1, 16.2, 18.3, 19.9, 21.5, 23.0,
                  24.6, 25.7, 27.2, 29.8),
        "tau": (9.35, 8.925, 8.50, 8.245, 7.65, 6.80, 5.95, 5.10, 4.25, 3.40,
                2.55, 1.70, 0.935, 0.0),
    },
    "00": {  # wrist (no curve available) — flat 4.25 Nm.
        "omega": (0.0, 100.0),
        "tau": (4.25, 4.25),
    },
}

# Braking ceiling per motor type (MJX MAX_TORQUE / ctrl_clip, full stall torque).
BRAKING_TORQUE: dict[str, float] = {"04": 22.0, "03": 11.0, "02": 11.0, "00": 5.0}


class TVCurveActuator(DelayedPDActuator):
    """Delayed PD actuator with a velocity-dependent (T-V curve) torque limit."""

    cfg: "TVCurveActuatorCfg"

    def __init__(self, cfg: "TVCurveActuatorCfg", *args, **kwargs):
        # Remove the box effort/velocity constraints from the base PD model;
        # the T-V curve replaces the effort ceiling. (Same approach as
        # RemotizedPDActuator.)
        cfg.effort_limit = torch.inf
        cfg.velocity_limit = torch.inf
        super().__init__(cfg, *args, **kwargs)

        # FREE-PLAY / BACKLASH model (2026-08-16, user sim2real hypothesis):
        # inside a +-play/2 band around the target, the joint transmits ~no
        # corrective torque — the link wanders under load like a loose real
        # ankle, while a bolted sim joint would hold. Modeled as a deadband on
        # the PD position error (target shrunk toward measured pos before the
        # delayed-PD stage). Approximations, documented: (a) the shrink is
        # computed against CURRENT pos but the delay buffer applies it up to
        # 40 ms later (second-order at <=2 deg bands); (b) kd damping stays
        # active inside the band (real play also frees damping; grease/friction
        # on hardware partially damps anyway). Randomize via the
        # randomize_joint_play reset event or set `_play` directly in probes.
        # Default (0,0) = exact legacy behavior for existing checkpoints/evals.
        self._play = None  # lazily sized on first compute (num_envs known there)
        self._play_range = tuple(cfg.play_range)
        # MOTOR-SIDE ENCODER state (2026-08-16, lineage 4): the hardware ankle
        # encoder sits on the MOTOR side of the loose linkage and cannot see
        # where the link actually is inside the free band. Virtual motor angle
        # model: the servo tracks its commanded target well, but the linkage
        # constrains it to within +-play/2 of the true link angle ->
        # m = clamp(target, q - play/2, q + play/2). Memoryless, and exactly
        # the link angle when play=0 (legacy checkpoints unaffected).
        # motor_vel: engaged (error at/beyond the band edge) -> link and motor
        # move together, report true joint_vel; slack -> finite-difference of
        # m (near zero while the link wanders — the blindness is the point).
        # Read by mdp_gait.joint_pos_rel_motorside / joint_vel_rel_motorside.
        self.motor_pos: torch.Tensor | None = None
        self.motor_vel: torch.Tensor | None = None
        self._motor_dt = float(cfg.sim_dt)
        self._viscous_b = float(cfg.viscous_b)
        self._fc_nominal = float(cfg.coulomb_fc)
        self._fc = None   # lazily sized (num_envs known at first compute)
        # SERIES ELASTICITY (rig handoff #3, 2026-08-23): the real ankle is a
        # SPRING between actuator and foot, not a rigid link. Measured by dual-
        # stream lean test: d(body)/d(ankle_encoder) = 3.6 => ~72% of body lean
        # is INVISIBLE to the motor-side encoder, and effective ankle stiffness
        # at the body is kp/3.6 ~ 33 Nm/rad vs gravity's ~86 -> the hardware
        # CANNOT stand passively. Our rigid chain gave 120 Nm/rad of stability
        # the robot does not have, which is why checkpoints that pass every sim
        # battery fall in ~4 s on metal.
        #   rotor:  J_m*qdd_m = tau_pd(on ROTOR state) - tau_spring
        #   spring: tau_s = K_s*deadband(q_m - q_j, play) + b_s*(qd_m - qd_j)
        #   joint receives +tau_s ; obs reads the ROTOR (motor_pos/motor_vel)
        # J_m = K_s/(2*pi*15)^2 puts the motor-vs-body mode at the measured
        # 15 Hz (the resonance that destroyed the first live engage).
        self._series_k = float(cfg.series_k)
        # PER-ENV K_s (rig RIG_ANKLE_POSTFIX §4, 2026-08-26). The hardware
        # number went 23 -> 35 -> 52 Nm/rad in three days of mechanical work and
        # a PA6-CF20 reprint is queued, so the rig asked us to RANDOMISE this
        # rather than anneal to any one measurement: "a policy tuned to 52 needs
        # retraining the day the parts change, and the parts are changing."
        # `_series_k` stays the scalar nominal (branch guard, logging, and the
        # fill value used when nothing randomises it, so probes that null the
        # DR event keep evaluating at a single known stiffness); `_series_k_env`
        # is the per-env tensor the physics actually uses.
        self._series_k_env = None   # lazily sized on first compute
        self._series_b = float(cfg.series_b)
        self._rotor_pos = None      # lazily initialised to joint_pos
        self._rotor_vel = None
        self._series_substeps = 8   # 15 Hz mode at 5 ms dt needs substepping
        self._series_dbg = 0        # §6.1: prove the path executes

        omega = torch.tensor(TV_CURVES[cfg.motor_type]["omega"], device=self._device)
        tau = torch.tensor(TV_CURVES[cfg.motor_type]["tau"], device=self._device)
        # Interpolates motoring torque ceiling from |joint speed|.
        self._tv = LinearInterpolation(omega, tau, device=self._device)
        self._braking_torque = float(BRAKING_TORQUE[cfg.motor_type])
        self._tv_rand = float(cfg.tv_randomization)
        # exposed for logging / verification
        self.tv_motoring_limit: torch.Tensor | None = None
        # one-time confirmation that the T-V actuator is actually wired in
        import omni.log

        omni.log.info(
            f"[TVCurveActuator] ACTIVE motor_type={cfg.motor_type} "
            f"joints={self.joint_names} peak={float(tau.max()):.2f}Nm "
            f"brake={self._braking_torque:.1f}Nm tv_rand={self._tv_rand}"
        )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # FREE-PLAY deadband (see __init__ comment): shrink the position error
        # by +-play/2 before the delayed-PD stage. Zero-cost when play is 0.
        if self._play is None:
            self._play = torch.full_like(joint_pos, 0.0)
            if self._play_range[1] > 0.0:
                self._play.uniform_(*self._play_range)
        if self._series_k > 0.0 and control_action.joint_positions is not None:
            # ---- SERIES-ELASTIC PATH (rig handoff #3) --------------------
            # Firmware PD runs on the ROTOR encoder (that is what the real
            # drive closes on), the spring carries torque to the link, and the
            # 0.3 deg play lives INSIDE the spring (§6.4 — do NOT also shrink
            # the PD error, that would put two bands in series).
            if self._rotor_pos is None:
                self._rotor_pos = joint_pos.clone()
                self._rotor_vel = torch.zeros_like(joint_vel)
            if self._series_k_env is None:
                self._series_k_env = torch.full_like(joint_pos, self._series_k)
            # BUGFIX (rig handoff #3B §3.3, found 2026-08-24): this branch
            # RETURNS before super().compute(), which is where DelayedPDActuator
            # applies the actuation delay — so the ankles were running with ZERO
            # latency while every other joint had theirs. The rig's ankle is the
            # SLOWEST path (15-25 ms measured); removing that delay on the one
            # joint that carries the compliance makes our plant strictly easier
            # to stabilise than the robot, in exactly the direction of the
            # sim-to-sim gap they reported (we stand 45 s, their rig falls 2.5 s).
            # Pull the delayed command out of the buffer explicitly.
            target = self.positions_delay_buffer.compute(control_action.joint_positions)
            kp, kd = self.stiffness, self.damping
            k_s = self._series_k_env                    # per-env, see __init__
            j_m = k_s / (2.0 * 3.14159265 * 15.0) ** 2
            half_play = self._play * 0.5
            sub_dt = self._motor_dt / self._series_substeps
            tau_s = torch.zeros_like(joint_pos)
            for _ in range(self._series_substeps):
                defl = self._rotor_pos - joint_pos
                defl_eff = torch.sign(defl) * (defl.abs() - half_play).clamp(min=0.0)
                tau_s = k_s * defl_eff + self._series_b * (self._rotor_vel - joint_vel)
                tau_pd = kp * (target - self._rotor_pos) - kd * self._rotor_vel
                # the MOTOR is T-V limited (the spring only transmits what the
                # rotor can push); previously this branch skipped the clamp too
                mot = self._tv.compute(self._rotor_vel.abs())
                same = torch.sign(tau_pd) == torch.sign(self._rotor_vel)
                lim = torch.where(same, mot, torch.full_like(mot, self._braking_torque))
                tau_pd = tau_pd.clamp(-lim, lim)
                # semi-implicit (symplectic) Euler: velocity first, then position
                self._rotor_vel = self._rotor_vel + sub_dt * (tau_pd - tau_s) / j_m
                self._rotor_pos = self._rotor_pos + sub_dt * self._rotor_vel
            # the OBSERVATION is the rotor — this is what makes ~72% of body
            # lean invisible to the policy, exactly as on hardware (§6.3)
            self.motor_pos = self._rotor_pos.clone()
            self.motor_vel = self._rotor_vel.clone()
            # BUGFIX 2 (2026-08-24, found reading the rig's `return ts +
            # friction_terms`): this branch also returned before the Coulomb /
            # viscous stage, so the ankles ran FRICTIONLESS. Same mistake shape
            # as the delay bypass. The joint sees spring + friction, both acting
            # on the LINK side.
            if self._fc is None:
                self._fc = torch.full_like(joint_pos, self._fc_nominal)
            tau_s = tau_s - self._fc * torch.tanh(joint_vel / 0.02)
            if self._viscous_b > 0.0:
                tau_s = tau_s - self._viscous_b * joint_vel
            self.applied_effort = tau_s
            control_action.joint_efforts = tau_s
            control_action.joint_positions = None
            control_action.joint_velocities = None
            if self._series_dbg < 3:      # §6.1: prove this path executes
                import omni.log
                omni.log.info(
                    f"[series] ACTIVE K_s nominal={self._series_k} "
                    f"env mean={k_s.mean().item():.1f} "
                    f"[{k_s.min().item():.1f}, {k_s.max().item():.1f}] "
                    f"b={self._series_b} "
                    f"J_m={j_m.mean().item():.5f} |tau_s|mean={tau_s.abs().mean().item():.3f} "
                    f"|defl|mean={(self._rotor_pos-joint_pos).abs().mean().item():.5f}")
                self._series_dbg += 1
            return control_action
        if control_action.joint_positions is not None:
            # virtual motor angle from the RAW (pre-deadband) target — the
            # motor chases the command; the linkage edge constrains it to the
            # band around the true link angle. Runs for every group (play=0
            # -> m == joint_pos exactly, vel == true joint_vel: legacy obs).
            half = self._play * 0.5
            m = torch.clamp(control_action.joint_positions,
                            joint_pos - half, joint_pos + half)
            if self.motor_pos is None:
                self.motor_vel = joint_vel.clone()
            else:
                engaged = (control_action.joint_positions - joint_pos).abs() >= half
                fd = ((m - self.motor_pos) / self._motor_dt).clamp(-50.0, 50.0)
                self.motor_vel = torch.where(engaged, joint_vel, fd)
            self.motor_pos = m
        if (self._play_range[1] > 0.0 or bool(self._play.any())) \
                and control_action.joint_positions is not None:
            err = control_action.joint_positions - joint_pos
            half = self._play * 0.5
            err_eff = torch.sign(err) * (err.abs() - half).clamp(min=0.0)
            control_action.joint_positions = joint_pos + err_eff
        # PD torque (with action delay) from the base class.
        control_action = super().compute(control_action, joint_pos, joint_vel)
        effort = control_action.joint_efforts
        # RIG-FITTED JOINT FRICTION (handoff 2026-08-20 §3), their exact law:
        #     tau += -Fc*tanh(qd/0.02) - b*qd
        # BUGFIX 2026-08-21 (ablation on model_39000): Fc was previously fed
        # into PhysX's `joint_friction`, which Isaac documents as "a UNITLESS
        # quantity [relating] the magnitude of the spatial force transmitted
        # from the parent body to the child" — i.e. a multiplier on the joint's
        # load, NOT a torque in Nm. Fc=4.0 there ~ four times the weight-borne
        # spatial force = a welded hip: the rig-validated checkpoint went from
        # 100% alive (old plant, and with deploy-kd / viscous / measured delays
        # applied) to 0% with that one term. Implemented HERE in Nm, where the
        # rig's number means what the rig measured. PhysX dof friction stays
        # ~0 for the driven joints. `_fc` is per-env-per-joint so a DR event
        # can redraw it (see randomize_joint_coulomb).
        if self._fc is None:
            self._fc = torch.full_like(joint_pos, self._fc_nominal)
        if self._fc_nominal > 0.0 or bool(self._fc.any()):
            effort = effort - self._fc * torch.tanh(joint_vel / 0.02)
        if self._viscous_b > 0.0:
            effort = effort - self._viscous_b * joint_vel

        # Motoring ceiling from the T-V curve at the current |speed|.
        speed = torch.abs(joint_vel)
        motoring = self._tv.compute(speed)
        # Per-step randomization: scale motoring limit down by up to tv_rand.
        if self._tv_rand > 0.0:
            scale = 1.0 - self._tv_rand * torch.rand_like(motoring)
            motoring = motoring * scale
        self.tv_motoring_limit = motoring

        # Braking (effort opposes motion) gets the full constant ceiling.
        # MJX convention: same-sign(effort, vel) => motoring; else braking.
        same_sign = torch.sign(effort) == torch.sign(joint_vel)
        limit = torch.where(same_sign, motoring, torch.full_like(motoring, self._braking_torque))

        effort = torch.clamp(effort, min=-limit, max=limit)
        self.applied_effort = effort
        control_action.joint_efforts = effort
        return control_action


@configclass
class TVCurveActuatorCfg(DelayedPDActuatorCfg):
    play_range: tuple = (0.0, 0.0)
    """Free-play (backlash) band in rad, drawn U(range) per env at first
    compute; redrawable per reset via the randomize_joint_play event.
    (0,0) = rigid joint = legacy behavior."""

    series_k: float = 0.0
    """Series-elastic stiffness between actuator and load (Nm/rad). Rig-measured
    ankle value 23. 0 = rigid (legacy)."""

    series_b: float = 0.0
    """Damping across the series spring (Nm.s/rad). Rig-fitted ankle 0.07."""

    sim_dt: float = 0.005
    """Physics step (s) — used only for the slack-side virtual-motor velocity
    estimate (finite difference of motor_pos). Must match the sim dt."""

    coulomb_fc: float = 0.0
    """Coulomb friction torque (Nm), rig-fitted: tau -= Fc*tanh(qd/0.02).
    hip pitch/roll 4.0 (EFFECTIVE — lumps drive-filter lag + rig sway per the
    handoff), knee 0.6, yaw 0.4, ankle 0.1. NOT PhysX joint_friction (that is
    a unitless load multiplier — see compute())."""

    viscous_b: float = 0.0
    """Viscous joint-friction coefficient (Nm·s/rad): tau -= b*qd, applied
    after the PD stage, before the T-V clamp. Rig-fitted 2026-08-20:
    hip pitch/roll 1.0, knee 0.2, yaw/ankle 0."""
    """Configuration for a T-V curve actuator (one per Robstride motor type)."""

    class_type: type = TVCurveActuator

    motor_type: str = "04"
    """Robstride motor type key into TV_CURVES / BRAKING_TORQUE: '04','03','02','00'."""

    tv_randomization: float = 0.15
    """Per-step downward scaling of the motoring torque limit, in [1-x, 1]."""
