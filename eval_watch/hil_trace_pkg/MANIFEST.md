# Isaac walk trace for HIL trace-diff — model_190600, vx=0.3

## Purpose
Both sims enforce identical T-V limits (ksim TV_CURVES x0.85, verified verbatim)
and both saturate comparably (Isaac hip roll ~95% @stand, ankle ~83% @walk ≈
your measured numbers), yet Isaac walks this policy at 126% tracking while your
emulator gets 43% + 22 deg wobble. The sims disagree about dynamics AROUND
saturation. Replay this trace at the three levels below and report the first
joint + gait phase where trajectories diverge.

## Recording conditions (all deterministic)
- checkpoint: model_190600 (= the bundle you already gated; interface
  STAND_PIN=new, ACTION_CLIP=8)
- command: vx=0.3, vy=0, wz=0, fixed; 600 policy steps @ 50 Hz = 12 s; 2 envs
  (env 0 and env 1, independent starts; use env 0 as primary)
- contact friction pinned: static 1.0 / dynamic 0.8 (multiply-combine vs
  ground 1.0 -> effective slide mu 1.0 = your floor)
- joint dry friction pinned: ankles 0.1 Nm, hips/knees 0.0 (your MJCF values)
- actuator delay pinned: 3 physics steps = 15 ms (your measured typical);
  tv_randomization = 0 (deterministic clamp); no pushes; no mass randomization;
  no terminations; seed 42
- physics: 200 Hz (dt 5 ms), decimation 4 -> policy 50 Hz. Recorded values are
  the LAST physics substep of each policy step (torque varies within the
  decimation window; if you need sub-window torque we can re-record at 200 Hz).

## Arrays in trace_isaac_190600_vx03.npz  (T=600, N=2, J=10, F=2 feet)
- joint_names [J] — THE joint order for every J-indexed array
- default_joint_pos [J] rad — action mapping: joint_target = default + 0.5*action
- obs [T,N,43] — policy input (pgrav 3, cmd 3, q_rel 10, qd 10, imu_ang 3,
  last_action 10, gait_phase 4)
- raw_action [T,N,J] — policy output BEFORE the ±8 clip
- clipped_action [T,N,J] — after ±8 clip (what the action manager consumed)
- joint_target [T,N,J] rad — processed PD position target (default + 0.5*clipped)
- q, qd [T,N,J] rad, rad/s — measured joint state
- applied_torque [T,N,J] Nm — POST T-V-clamp torque actually applied
- tv_limit [T,N,J] Nm — live motoring ceiling at that step's |qd|
- root_pos [T,N,3] m (world), root_quat [T,N,4] (w,x,y,z world),
  root_linvel/root_angvel [T,N,3] (BODY frame), pgrav [T,N,3] (body frame)
- foot_force [T,N,F,3] N world (contact sensor net force; foot_names gives order)

## Replay protocol (three levels, coarse -> fine)
1. ACTION-level: feed clipped_action through YOUR full path (PD @ your rates,
   your delay, your T-V clamp) from the same initial state (q[0], qd[0],
   root state [0]). Divergence here includes control-path differences.
2. TARGET-level: apply joint_target as PD setpoints directly (kp=150, kd=5,
   bypassing action mapping). Isolates PD + actuator + rigid-body.
3. TORQUE-level: apply applied_torque as direct joint torques (bypass PD and
   T-V entirely). Divergence here = pure rigid-body + contact disagreement —
   the deepest and most diagnostic level.
For each level: report per-joint RMS(q_yours - q_trace) over time, the first
time it exceeds ~0.05 rad, and which gait phase (stance/swing, from
foot_force) that occurs in. Suspects to watch: hip roll during single support,
ankle during push-off.

## Sanity anchors
- trace_summary.txt has the achieved vx in Isaac for this exact recording
  (expect ~0.37 — the disagreement IS the point).
- If even TORQUE-level replay tracks for the first ~1 s then diverges slowly,
  that's chaos accumulation (expected); what matters is SYSTEMATIC divergence:
  one joint consistently lagging its trace while its torque is at tv_limit.
