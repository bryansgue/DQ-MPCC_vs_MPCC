# Agent: MPCC Baseline Rates (rate control, MuJoCo SiL) — Context Document

This file gives full context to Claude Code when working inside `MPCC_baseline_rates/`.
This pipeline implements MPCC with **angular-rate commands** and runs in **MuJoCo SiL via ROS2**.

---

## What this pipeline does

MPCC controller sending `[T, ωx_cmd, ωy_cmd, ωz_cmd]` directly to MuJoCo's AcroMode.
Unlike MPCC_baseline (torques), there is NO torque allocation — the rate commands go straight
to the simulator's rate controller (first-order lag model: τ_rc=0.03s).

Execution: MiL (`MPCC_baseline_rates.py`) and **MuJoCo SiL** (`mpcc_mujoco_node.py` via ROS2).

---

## State vector x ∈ ℝ¹⁴

| idx | symbol | meaning |
|-----|--------|---------|
| 0:3 | p | position [m], inertial frame |
| 3:6 | v | linear velocity [m/s], inertial frame |
| 6:10 | q = [qw,qx,qy,qz] | unit quaternion, Hamilton |
| 10:13 | ω | angular velocity [rad/s], body frame |
| 13 | θ | arc-length progress [m] (Euler-integrated in SiL node) |

> In MuJoCo SiL, θ is NOT published by MuJoCo — it is integrated in the node:
> θ_{k+1} = θ_k + v_θ_k · (1/FREC)

## Control vector u ∈ ℝ⁵

| idx | symbol | meaning |
|-----|--------|---------|
| 0 | T | total thrust [N] |
| 1:4 | [ωx_cmd, ωy_cmd, ωz_cmd] | commanded angular rates [rad/s] |
| 4 | v_θ | arc-length progress velocity [m/s] |

Current limits: `T ∈ [0, 5g]`, `|ω_cmd| ≤ 20 rad/s`, `v_θ ∈ [0, 20] m/s`.

## Rate dynamics

```
ω̇ = (ω_cmd − ω) / τ_rc       τ_rc = 0.03 s   (first-order lag)
```
At steady state: ω ≈ ω_cmd. This is the KEY model difference from MPCC_baseline.

---

## OCP cost function

**Stage cost (k = 0 … N−1):**
```
J_stage = ec' Q_ec ec                       ← contouring error  (ec = (I−ττ')·(p_ref−p))
        + el' Q_el el                       ← lag error         (el = (τ·τ')·(p_ref−p))
        + logq' Q_q logq                   ← quaternion log error
        + U_T·(T − T_hover)²               ← thrust deviation from hover  [U_mat[0]]
        + ω_cmd' diag(U_mat[1:4]) ω_cmd    ← commanded rate penalty
        + Q_s·(v_max − v_θ)²               ← progress speed incentive
```

**Terminal cost:** ec, el, logq — NO control/progress terms.

> CRITICAL DISTINCTIONS vs MPCC_baseline:
> - Control cost uses (T − T_hover)² not T² — hover-centred
> - Rates ω_cmd are penalised, NOT torques τ
> - Q_omega (state angular velocity) is REMOVED from tunable set (ω ≈ ω_cmd at SS)
> - θ is NOT in the state vector (Euler-integrated externally in SiL node)

---

## Parameter vector p ∈ ℝ¹⁵ (tuner OCP)

```
p[ 0: 3]  Q_ec        contouring [x, y, z]
p[ 3: 6]  Q_el        lag        [x, y, z]
p[ 6: 9]  Q_q         quaternion log [x, y, z]
p[ 9:13]  U_mat       control effort [T, wx, wy, wz]  ← rates, NOT torques
p[13]     Q_s         progress speed (scalar)
p[14]     vtheta_max  max v_θ (runtime + braking logic)
```

Note: N_PARAMS=15. Previously 18 — Q_omega (3 params) removed for rate control.

Round-3 best weights (current `experiment_config.py`):
- MPCC_Q_EC=[50.9, 50.8, 274.4], MPCC_Q_EL=[183.2, 80.9, 78.1]
- MPCC_Q_Q=[0.110, 0.108, 0.980], MPCC_RATE_U_MAT=[0.153, 0.207, 0.266, 0.486]
- MPCC_Q_S=1.854

---

## Trajectory

Lissajous figure-8 (TRAJ_VALUE=15), DIFFERENT from DQ-MPCC/MPCC_baseline:
```
x(t) = 2.5·sin(0.6t) + 2.5    # [−0.5, 5.5] m
y(t) = 1.5·sin(1.2t)           # [−1.5, 1.5] m
z(t) = 0.5·sin(0.6t) + 1.2    # [ 0.7, 1.7] m
```
One full loop ≈ 17–20 m. S_MAX_MANUAL=60m (~3 loops).
Starts at P0=[2.5, 0, 1.2]. CasADi interpolators (40 waypoints).
**Requires precomputed cache**: `python3 precompute_path.py`

---

## Bilevel tuner

- **Outer**: Optuna TPE, 14 params `[Q_ec(3), Q_el(3), Q_q(3), U_mat(4), Q_s(1)]`
- **Inner**: `mpcc_mujoco_tuner_runner.run_simulation_mujoco(weights, vtheta_max)`
- Solver compiled ONCE with p∈ℝ¹⁵; weights injected at runtime via `solver.set(stage,"p",p_vec)`
- MuJoCo must be running; resets between trials via `/quadrotor/sim/reset` service

```bash
# Terminal 1
mujoco_launch.sh scene:=motors

# Terminal 2
python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner \
    --study-name round4 --n-trials 150
```

Current tuning config (round 4, single velocity, acrobatic):
- TUNING_VELOCITIES=[15.0], W_VEL=30, W_TIME=20, W_CONTOUR=5
- Q_S_RANGE=[0.5, 30], U_T_RANGE=[0.001, 0.5], U_W_RANGE=[0.001, 0.5]
- CRASH_MAX_TILT_COS=-0.4 (114°), CRASH_STILL_STEPS=300

---

## ROS2 interface

| Topic/Service | Dir | Type | Content |
|---------------|-----|------|---------|
| `/quadrotor/odom` | Sub | `nav_msgs/Odometry` | 240Hz: p,v,q,ω from MuJoCo |
| `/quadrotor/trpy_cmd` | Pub | `mav_msgs/TRPYCommand` | T, ωx_cmd, ωy_cmd, ωz_cmd |
| `/quadrotor/sim/reset` | Svc | `std_srvs/Trigger` | Reset MuJoCo simulation |

v_θ is internal to MPCC — it is NOT sent to MuJoCo.

---

## When to regenerate / recompile

| Change | Action needed |
|--------|--------------|
| N_WAYPOINTS, S_MAX_MANUAL, T_TRAJ_BUILD | `rm path_cache.npz && python3 precompute_path.py` |
| VTHETA_MAX, W_MAX, T_MAX | `rm -rf c_generated_code_mpcc_rate_mujoco* && rerun node` |
| Q_omega added back | N_PARAMS changes → update OCP, runner, tuner |

---

## Key files

| File | Role |
|------|------|
| `config/experiment_config.py` | ALL parameters (weights, limits, trajectory) |
| `models/quadrotor_mpcc_rate_model_mujoco.py` | 14-state rate model (MASS=1.08 kg) |
| `ocp/mpcc_controller_rate_mujoco.py` | Fixed-gain SiL OCP |
| `ocp/mpcc_controller_rate_mujoco_tuner.py` | Runtime-param OCP (p∈ℝ¹⁵) |
| `mpcc_mujoco_node.py` | SiL production entry point |
| `mpcc_mujoco_tuner_runner.py` | Inner runner for bilevel tuning |
| `tuning/tuning_config.py` | Search space, objective weights, crash thresholds |
| `tuning/mpcc_rate_mujoco_tuner.py` | Optuna bilevel optimizer |
| `precompute_path.py` | Build arc-length cache |

---

## Key differences vs MPCC_baseline (torque)

| Aspect | MPCC_baseline | MPCC_baseline_rates (this) |
|--------|--------------|---------------------------|
| Control u[1:4] | Torques [τx,τy,τz] | Rate commands [ωx,ωy,ωz] |
| Thrust cost | U_T·T² | U_T·(T−T_hover)² |
| Q_omega | Tunable (in p) | REMOVED (ω≈ω_cmd at SS) |
| Simulator | MiL or MuJoCo | MuJoCo SiL primary |
| θ in state | YES, x[13] | NO, external Euler integration |
| p vector size | 18 | **15** |
| Trajectory | Lissajous helix (z≈6m) | Figure-8 (z≈1.2m) |
