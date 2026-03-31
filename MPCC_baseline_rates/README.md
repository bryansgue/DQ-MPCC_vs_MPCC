# MPCC Baseline — Rate Control (MiL + MuJoCo SiL)

MPCC (Model Predictive Contouring Control) for a quadrotor UAV using **body angular-rate commands**.
Two execution modes share the same OCP formulation:

| Mode | Entry point | Simulator |
|------|-------------|-----------|
| **MiL** (Model-in-the-Loop) | `MPCC_baseline_rates.py` | Internal RK4 step (no ROS2) |
| **SiL** (MuJoCo Software-in-the-Loop) | `mpcc_mujoco_node.py` | MuJoCo via ROS2 |

---

## Context in the project

This pipeline is one of three baselines compared against the **DQ-MPCC** controller:

```
DQ-MPCC_vs_MPCC_baseline/
├── DQ-MPCC_baseline/          ← DQ-MPCC with SE(3) dual-quaternion cost
├── MPCC_baseline/             ← MPCC with torque commands (τx, τy, τz)
├── MPCC_baseline_rates/       ← THIS — MPCC with angular-rate commands
└── NMPC_baseline/             ← NMPC reference
```

The rate-control formulation was chosen because MuJoCo's `AcroMode` interface receives
`TRPYCommand` (thrust + desired angular velocity), matching `u = [T, ωx_cmd, ωy_cmd, ωz_cmd]`
directly without any intermediate torque-to-rate inversion.

---

## Drone model

**State** `x ∈ ℝ¹⁴`:

| Index | Symbol | Description |
|-------|--------|-------------|
| 0:3 | `p` | Position [m] |
| 3:6 | `v` | Linear velocity [m/s] |
| 6:10 | `q` | Quaternion `[qw, qx, qy, qz]` |
| 10:13 | `ω` | Angular velocity [rad/s] |
| 13 | `θ` | Arc-length progress [m] |

**Control** `u ∈ ℝ⁵`:

| Index | Symbol | Description |
|-------|--------|-------------|
| 0 | `T` | Total thrust [N] |
| 1:4 | `ω_cmd` | Commanded angular rates [rad/s] |
| 4 | `v_θ` | Arc-length progress velocity [m/s] |

**Rate dynamics** (first-order lag):
```
ω̇ = (ω_cmd − ω) / τ_rc      τ_rc = 0.03 s
```

**Key distinction from `MPCC_baseline/`**: that pipeline uses torques `[τx, τy, τz]` as inputs.
Here `ω_cmd` goes directly to MuJoCo's AcroMode. No torque allocation needed.

**Two separate acados models** prevent C-code conflicts:

| Model name | Used in | Mass |
|-----------|---------|------|
| `Drone_ode_mpcc_rate_mil` | MiL (`MPCC_baseline_rates.py`) | 1.0 kg |
| `Drone_ode_mpcc_rate_mujoco` | SiL (`mpcc_mujoco_node.py`) | 1.08 kg |
| `Drone_ode_mpcc_rate_mujoco_tuner` | Bilevel tuner | 1.08 kg |

---

## MPCC cost function

```
J_stage = ec' Q_ec ec  +  el' Q_el el  +  log(q_err)' Q_q log(q_err)
        + U_T·(T − T_hover)²  +  ω_cmd' diag(U_mat[1:4]) ω_cmd
        + Q_s·(v_θ_max − v_θ)²
```

| Term | Symbol | Meaning |
|------|--------|---------|
| Contouring error | `ec = (I − ττᵀ)·(p_ref − p)` | Orthogonal deviation from path |
| Lag error | `el = (τ·τᵀ)·(p_ref − p)` | Along-path tracking delay |
| Attitude error | `log(q_err)` | SO(3) log-map of quaternion error |
| Control effort | `U_mat` | Thrust deviation + rate commands |
| Progress speed | `Q_s` | Reward for path advancement |

**Terminal cost** uses `ec, el, log(q_err)` only (no control/progress terms).

> `Q_omega` (angular-velocity state cost) was intentionally **removed** from the tunable
> set. In a rate-controlled drone `ω ≈ ω_cmd` at steady state, so penalising both
> `U_mat` on `ω_cmd` and `Q_omega` on `ω` is largely redundant. Removing it shrinks the
> Optuna search space from 17 → 14 parameters.

---

## Trajectory

Lissajous figure-8 parameterised by arc-length `θ`:

```python
x(t) = 2.5·sin(0.6t) + 2.5    # X: [−0.5, 5.5] m
y(t) = 1.5·sin(1.2t)           # Y: [−1.5, 1.5] m
z(t) = 0.5·sin(0.6t) + 1.2    # Z: [ 0.7, 1.7] m
```

Frequency ratio X:Y = 1:2 → closed figure-8 in XY plane with 3D z-oscillation.
One full loop ≈ **17–20 m** arc-length at the reference parametric speed.

### Arc-length precomputation (run once)

```bash
python3 MPCC_baseline_rates/precompute_path.py
```

Saves `path_cache.npz` with waypoints, tangents, quaternions and `s_max` values.
Both MiL and SiL nodes load from this cache via `path_loader.py`.

**Must regenerate cache** when any of these change: `N_WAYPOINTS`, `S_MAX_MANUAL`,
`T_TRAJ_BUILD`, `TRAJ_VALUE`.

---

## θ virtual state in SiL

MuJoCo publishes only 13 physical states (`p, v, q, ω`).
Arc-length progress `θ` is propagated via Euler integration in the node:

```python
θ_{k+1} = θ_k + v_θ_k · t_s          # t_s = 1/FREC = 0.01 s
```

`v_θ` is taken from `u_control[4, k]` (solver output at stage 0).
The loop exits when `θ_k ≥ S_MAX_MANUAL`.

---

## File structure

```
MPCC_baseline_rates/
├── config/
│   └── experiment_config.py          # All tunable parameters (single source of truth)
├── models/
│   ├── quadrotor_mpcc_rate_model.py          # MiL model (MASS=1.0 kg)
│   └── quadrotor_mpcc_rate_model_mujoco.py   # SiL model (MASS_MUJOCO=1.08 kg)
├── ocp/
│   ├── mpcc_controller_rate.py               # MiL OCP (fixed weights)
│   ├── mpcc_controller_rate_mujoco.py        # SiL OCP (fixed weights)
│   └── mpcc_controller_rate_mujoco_tuner.py  # SiL OCP (runtime params p∈ℝ¹⁵)
├── ros2_interface/
│   ├── mujoco_interface.py    # MujocoInterface: odom subscriber + cmd publisher + PD hold
│   └── reset_sim.py           # SimControl: /quadrotor/sim/reset service wrapper
├── tuning/
│   ├── tuning_config.py              # Search space, objective weights, crash thresholds
│   ├── mpcc_rate_mujoco_tuner.py     # Optuna outer optimizer
│   └── best_weights.json             # Latest best weights from bilevel tuning
├── MPCC_baseline_rates.py    # MiL entry point
├── mpcc_mujoco_node.py       # SiL entry point (production)
├── mpcc_mujoco_tuner_runner.py   # SiL inner runner for bilevel tuning
├── precompute_path.py        # Build and cache arc-length parameterisation
├── path_loader.py            # Load cache or rebuild on the fly
└── path_cache.npz            # Precomputed waypoints (generated, not tracked in git)
```

---

## Quick start

### Terminal 1 — MuJoCo simulator
```bash
mujoco_launch.sh scene:=motors
```

### Terminal 2 — Run SiL node
```bash
cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline

# First time or after config change: regenerate path cache
python3 MPCC_baseline_rates/precompute_path.py

# Run
python3 MPCC_baseline_rates/mpcc_mujoco_node.py
```

### Run MiL (no ROS2, no MuJoCo)
```bash
python3 MPCC_baseline_rates/MPCC_baseline_rates.py
```

---

## Bilevel gain tuning

The tuner is a **two-level optimisation**:

- **Level 1 (outer)**: Optuna TPE proposes 14 weights `[Q_ec(3), Q_el(3), Q_q(3), U_mat(4), Q_s(1)]`
- **Level 2 (inner)**: Full SiL simulation with MuJoCo; solver compiled ONCE, MuJoCo reset between trials

**Parameter vector** `p ∈ ℝ¹⁵` (runtime, no recompilation needed):

```
p[ 0: 3]  Q_ec        contouring error   [x, y, z]
p[ 3: 6]  Q_el        lag error          [x, y, z]
p[ 6: 9]  Q_q         quaternion log     [x, y, z]
p[ 9:13]  U_mat       control effort     [T, wx, wy, wz]
p[13]     Q_s         progress speed
p[14]     vtheta_max  runtime v_θ max (also used for braking logic)
```

### Meta-objective (Optuna minimises):
```
J = mean_mpcc_cost
  + W_INCOMPLETE · (1 − path_completed)     if path_completed < 0.99
  + W_TIME       · (t_lap / T_ref)
  + W_VEL        · (v_max − v̄θ) / v_max
  + W_ISOTROPY   · anisotropy(RMSE_xyz)
  + W_CONTOUR    · RMSE_contorno
```

Evaluated at each velocity in `TUNING_VELOCITIES`; `J_multi = mean over velocities`.

### Run tuner
```bash
# Terminal 1: MuJoCo must be running
mujoco_launch.sh scene:=motors

# Terminal 2:
python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner \
    --study-name mpcc_rates_round3 \
    --n-trials 120

# Optional: install cmaes for CMA-ES sampler
pip install cmaes
python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner --sampler cmaes
```

Results saved to `tuning/best_weights.json` and `tuning/tuning_history.json`.
Copy best weights into `config/experiment_config.py` and **delete compiled solvers** to force recompile.

---

## Configuration reference (`experiment_config.py`)

| Variable | Current | Description |
|----------|---------|-------------|
| `S_MAX_MANUAL` | 60.0 m | Path length — drone stops when `θ ≥ S_MAX_MANUAL` |
| `T_TRAJ_BUILD` | 50 s | Time used to build arc parameterisation (not simulation time) |
| `T_FINAL` | 30 s | Simulation upper-bound (exits early via `θ` condition) |
| `N_WAYPOINTS` | 40 | CasADi interpolation points (accuracy vs. compile time) |
| `VTHETA_MAX` | 20 m/s | Max arc-length progress velocity (OCP constraint) |
| `W_MAX` | 15 rad/s | Max commanded angular rate (OCP constraint) |
| `T_MAX` | 4g N | Max thrust |
| `MASS_MUJOCO` | 1.08 kg | MuJoCo drone mass (different from MiL=1.0 kg) |
| `TAU_RC` | 0.03 s | Rate-control time constant |
| `T_PREDICTION` | 0.3 s | MPC horizon duration |
| `FREC` | 100 Hz | Control loop frequency |

### Quick configs

| Purpose | `S_MAX_MANUAL` | `T_TRAJ_BUILD` | `N_WAYPOINTS` | `T_FINAL` |
|---------|---------------|----------------|---------------|-----------|
| **Quick test** (1 loop) | 20 m | 25 s | 20 | 15 s |
| **Production** (3 loops) | 60 m | 50 s | 40 | 30 s |
| **Full production** | 100 m | 80 s | 80 | 60 s |

After changing these: **delete `path_cache.npz`** and rerun `precompute_path.py`.
After changing `VTHETA_MAX` or `W_MAX`: **delete compiled C code** and recompile.

```bash
# Force full recompile
rm -f  MPCC_baseline_rates/path_cache.npz
rm -rf MPCC_baseline_rates/c_generated_code_mpcc_rate_mujoco/
rm -rf MPCC_baseline_rates/c_generated_code_mpcc_rate_mujoco_tuner/
```

---

## ROS2 interface

| Topic / Service | Direction | Type | Description |
|----------------|-----------|------|-------------|
| `/quadrotor/odom` | Subscribe | `nav_msgs/Odometry` | 240 Hz state from MuJoCo |
| `/quadrotor/trpy_cmd` | Publish | `mav_msgs/TRPYCommand` | Thrust + angular rate commands |
| `/quadrotor/sim/reset` | Service call | `std_srvs/Trigger` | Reset MuJoCo simulation |

**State update per step**:
```
p, v, q, ω  ← /quadrotor/odom   (MuJoCo physical states)
θ_{k+1}     = θ_k + v_θ_k · t_s  (Euler integration)
```

**Command sent**:
```
T, ωx_cmd, ωy_cmd, ωz_cmd  ← solver.get(0, "u")[0:4]
v_θ is NOT sent — internal MPCC state only
```

---

## Tuning history

| Round | Trials | Velocities | v̄θ @ v=5 | v̄θ @ v=10 | RMSE_c | Key change |
|-------|--------|-----------|-----------|-----------|--------|-----------|
| R1 | 100 | [5, 7.5, 10] | 2.18 m/s (43%) | 2.75 m/s (27%) | ~10 cm | First run, wide bounds |
| R2 | 100 | [5, 7.5, 10] | 3.26 m/s (65%) | 4.04 m/s (40%) | ~13 cm | `W_VEL=8`, tighter U_W |
| R3 | 120 | [5, 12, 20] | — | — | — | `VTHETA_MAX=20`, `S_MAX=60m` |
