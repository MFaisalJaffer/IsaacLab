# Sim episodes behind RIG_HW_STAND_ADDENDUM_SENSING.md (rig MuJoCo, walker_v5_3200, hard start, K_s 52)

Columns of `data`: t, dt_inf_ms, obs27 (jpos10, jvel10, quat4, gyro3 — raw, before the test knobs),
obs43 (the frame the policy saw — after the knobs), a_raw10, a10. `wire` = vcan tap (t, node, kind 0 cmd / 1 fb, …).

| file | sensing | plant extra | result |
|---|---|---|---|
| ep_20261004_032049_385t | stock | play 2° | stands (GC stall at tick 37 visible) |
| ep_20261004_032158_387t | stock, GC frozen | play 2° | stands, no stall |
| ep_20261004_030812_729t | IMU 20 Hz only | play 2° | stands |
| ep_20261004_030542_729t | measured (IMU 20 Hz + 20 ms, joints + 20 ms) | play 2° | stands, rocks 1.5 Hz |
| ep_20261004_031539_971t | measured | play 2°, 2 Nm roll moment | stands, rocks |
| ep_20261004_031659_435t | measured | play 2°, ankle rotor stiction 1 Nm | watchdog 9.2 s |
| ep_20261004_032452_254t | measured, GC frozen | play 2°, stiction 1 Nm | watchdog 5.3 s |
| ep_20261004_031908_969t | measured | play 2°, roll moment + stiction | stands, rocks |
| ep_20261004_031808_25t | IMU 20 Hz + 40 ms, joints + 20 ms | play 2° | watchdog 0.57 s |
| ep_20261004_031044_728t | fresh IMU + 20 ms, joints + 20 ms | play 2° | stands |
| ep_20261004_032331_966t | fresh IMU, GC frozen | play 2°, stiction 1 Nm | stands |
| ep_20261004_025757_224t | IMU 40 ms + joints 40 ms (constant) | play 2° | watchdog 4.8 s, 1.8 Hz |
| ep_20261004_031159_582t | stock | play 2°, kick at 6 s | dies in one swing |
| ep_20261004_031310_577t | measured | play 2°, kick at 6 s | sustained |
| ep_20261004_031423_580t | fresh IMU | play 2°, kick at 6 s | decays |
| ep_20261004_032559_583t | fresh IMU, GC frozen | play 2°, stiction 1 Nm, kick at 6 s | stands, 0.2° wobble |
