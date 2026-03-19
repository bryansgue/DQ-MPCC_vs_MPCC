# Experimental Protocol — DQ-MPCC vs Baseline MPCC

## Rationale

In MPCC the progress velocity $v_\theta$ is a decision variable. The controller
optimises *how fast* it traverses the path. Consequently, each controller can
finish the path in a different time. A fair comparison must normalise by
**distance**, not by time.

## Stopping criterion

**Fixed arc-length: `S_MAX_MANUAL = 100.0` m.**

Both controllers start at θ = 0 and stop when θ ≥ s_max − 0.01 m.
The full Lissajous path is ≈ 126.8 m (with VALUE=5, t_final=60 s), so 100 m
is a strict subset that is comfortably reachable.

The variable `t_final = 60 s` is a generous safety budget that should
**never** be the binding constraint. If either controller hits t_final before
completing the 100 m, this indicates a misconfiguration or a divergence.

## Single source of truth: `experiment_config.py`

All experimental parameters live in **`experiment_config.py`** at the project
root. Both `MPCC_baseline.py` and `DQ_MPCC_baseline.py` import from it.
To change ANY experimental condition, edit ONLY `experiment_config.py`.

## Initial conditions (arbitrary, user-defined)

| Parameter | Default          | Description                                |
|-----------|------------------|--------------------------------------------|
| `P0`      | [3, 0, 6]       | Initial position ℝ³ [m]                    |
| `Q0`      | [1, 0, 0, 0]    | Initial orientation (unit quaternion ℍ)     |
| `V0`      | [0, 0, 0]       | Initial linear velocity ℝ³ [m/s]           |
| `W0`      | [0, 0, 0]       | Initial angular velocity ℝ³ [rad/s]        |
| `THETA0`  | 0.0              | Initial arc-length progress [m]            |

**Note:** `P0` and `Q0` are completely arbitrary — the UAV does NOT need to
start on the path or aligned with the tangent. The MPCC controller will
drive the UAV to the nearest point on the path.

## Shared parameters (must be identical)

| Parameter        | Value   | Description                         |
|------------------|---------|-------------------------------------|
| `VALUE`          | 5       | Trajectory speed scaling            |
| `FREC`           | 100 Hz  | Control frequency                   |
| `t_s`            | 0.01 s  | Sample time (derived)               |
| `T_PREDICTION`   | 0.3 s   | MPC prediction horizon              |
| `N_prediction`   | 30      | Prediction steps (derived)          |
| `T_FINAL`        | 60 s    | Safety time budget                  |
| `S_MAX_MANUAL`   | 100.0 m | Fixed distance for experiment       |
| `N_WAYPOINTS`    | 30      | CasADi interpolation waypoints      |

## Cost weights (must be identical)

| Weight       | Value                  |
|-------------|------------------------|
| `Q_EC`      | [10, 10, 10]           |
| `Q_EL`      | [5, 5, 5]             |
| `Q_PHI/Q_Q` | [5, 5, 5]             |
| `U_MAT`     | [0.1, 250, 250, 250]  |
| `Q_OMEGA`   | [0.5, 0.5, 0.5]       |
| `Q_S`       | 0.3                    |
| `V_THETA_MAX` | 15.0 m/s             |

## KPI (Key Performance Indicators)

1. **t_lap** [s] — time to complete one full Lissajous lap (primary KPI)
2. **IAE_contouring** [m·s] — ∫₀^{t_lap} ‖ρ_cont(t)‖ dt
3. **IAE_lag** [m·s] — ∫₀^{t_lap} ‖ρ_lag(t)‖ dt
4. **IAE_orientation** [rad·s] — ∫₀^{t_lap} ‖Log(q_d⁻¹ ⊗ q)‖ dt
5. **Solver time** [ms] — mean ± std (max) per iteration
6. **v_θ statistics** — mean, max, min of progress velocity

## Data saved in `.mat` files

Both simulation scripts now save the following additional fields:

- `quat_d_theta` (4 × N) — desired quaternion evaluated at θ_k (not at t_k)
- `t_lap` (scalar) — lap completion time [s]
- `t_solver_mean`, `t_solver_max`, `t_solver_std` (scalars) — solver timing [ms]

These are consumed by `compare_results.py` to compute correct θ-indexed
orientation errors and to display the lap time / solver KPIs in the
summary table.

## How to run

```bash
# 1. Run DQ-MPCC simulation
cd DQ-MPCC_baseline && python DQ_MPCC_baseline.py

# 2. Run Baseline MPCC simulation
cd ../MPCC_baseline && python MPCC_baseline.py

# 3. Generate comparison plots
cd .. && python compare_results.py
```

## What NOT to change for a fair comparison

- `t_final` must be ≥ 40 s (safety buffer, not the stopping criterion)
- `S_MAX_MANUAL` must remain `None` (both must run the full curve)
- All cost weights must be identical in both OCP files
- The Lissajous trajectory definition must be identical
- Both must use `N_WAYPOINTS = 30` for the CasADi interpolation
