# MPCC Baseline for Quadrotor UAV

Model Predictive Contouring Control (MPCC) implementation for quadrotor trajectory tracking using acados and CasADi.

## 📋 Requirements

- **Python**: 3.10+
- **acados**: v0.5.3 (path: `/home/bryansgue/acados`)
- **CasADi**: 3.7.2
- **Python packages**: `numpy`, `scipy`, `matplotlib`, `optuna`

```bash
pip install numpy scipy matplotlib optuna casadi
```

## 🏗️ Project Structure

```
MPCC_baseline/
├── MPCC_baseline.py           # Main MPCC simulation (real-time)
├── NMPC_baseline.py           # Standard NMPC simulation (comparison)
├── MPCC_simulation_tuner.py  # Headless MPCC for gain tuning
├── ocp/
│   ├── mpcc_controller.py         # MPCC OCP (fixed gains)
│   ├── mpcc_controller_tuner.py   # MPCC OCP (runtime parameters)
│   └── nmpc_controller.py         # NMPC OCP
├── models/
│   └── quadrotor_mpcc_model.py    # 14-state dynamics + physical constants
├── utils/
│   ├── numpy_utils.py             # Arc-length, quaternions, RK4
│   ├── casadi_utils.py            # Symbolic utils (CasADi)
│   └── graficas.py                # Plotting functions
└── tuning/
    ├── mpcc_tuner.py              # Optuna bilevel optimizer
    └── best_weights.json          # Optimized gains
```

## 🚀 Quick Start

### 1. Run MPCC Simulation

```bash
python3 MPCC_baseline.py
```

**Output**: Real-time visualization (100 Hz), plots saved as PNG files, trajectory data saved to `.mat`.

### 2. Run NMPC Simulation (baseline comparison)

```bash
python3 NMPC_baseline.py
```

### 3. Run Gain Tuning

```bash
# 50 trials (default)
python3 -m tuning.mpcc_tuner

# More trials
python3 -m tuning.mpcc_tuner --n-trials 200

# CMA-ES sampler
python3 -m tuning.mpcc_tuner --sampler cmaes
```

**Tuner output**: Optimized gains saved to `tuning/best_weights.json`.

## ⚙️ MPCC Formulation

### State Vector (14D)
```
x = [px, py, pz,          # position [m]
     vx, vy, vz,          # velocity [m/s]
     qw, qx, qy, qz,      # quaternion (scalar-first)
     ωx, ωy, ωz,          # angular velocity [rad/s]
     θ]                   # arc-length progress [m]
```

### Control Vector (5D)
```
u = [T,                   # total thrust [N]
     τx, τy, τz,          # torques [N·m]
     v_θ]                 # arc-length velocity [m/s]
```

### Cost Function

The MPCC stage cost penalizes:

$$
J_k = \mathbf{e}_c^\top Q_{ec} \mathbf{e}_c + \mathbf{e}_l^\top Q_{el} \mathbf{e}_l + \log(\mathbf{q}_{err})^\top Q_q \log(\mathbf{q}_{err}) + \mathbf{u}^\top U \mathbf{u} + \boldsymbol{\omega}^\top Q_\omega \boldsymbol{\omega} + Q_s (v_{\theta,\max} - v_\theta)^2
$$

Where:
- **e_c**: Contouring error (⊥ to path)
- **e_l**: Lag error (∥ to path)
- **log(q_err)**: Quaternion logarithm error (SO(3))
- **u**: Control effort (thrust + torques)
- **ω**: Angular velocity
- **v_θ**: Arc-length velocity (penalizes slow progress)

### Constraints

```python
# Control bounds
0.0 ≤ T ≤ 29.43 N          # thrust
|τx|, |τy|, |τz| ≤ 0.1 N·m  # torques
0.0 ≤ v_θ ≤ 15.0 m/s        # arc-length velocity

# State bounds
0.0 ≤ θ ≤ s_max             # stay within trajectory
```

## 🎛️ Gain Tuning

### Objective Function

The tuner minimizes the **same cost function used by the MPCC solver**:

$$
J_{\text{tuner}} = \bar{J}_{\text{MPCC}} + W_{\text{incomplete}} \cdot (1 - p_{\text{completed}})^2
$$

Where:
- **J̄_MPCC**: Mean MPCC stage cost over the entire trajectory
- **p_completed**: Fraction of path completed [0, 1]
- **W_incomplete = 1000**: Large penalty for incomplete paths

### Search Space

| Parameter | Range | Description |
|---|---|---|
| `Q_ec[x,y,z]` | [1, 50] | Contouring error weights |
| `Q_el[x,y,z]` | [0.1, 50] | Lag error weights |
| `Q_q[roll,pitch,yaw]` | [0.1, 50] | Quaternion error weights |
| `U_mat[T,τx,τy,τz]` | [0.01-10, 10-800] | Control effort weights |
| `Q_omega[x,y,z]` | [0.01, 10] | Angular velocity weights |
| `Q_s` | [0.5, 10] | Progress speed weight |

All sampled in **log-space** for better exploration.

### Current Best Gains

```python
DEFAULT_Q_EC    = [22.01, 39.53, 41.35]
DEFAULT_Q_EL    = [33.87, 40.31, 34.90]
DEFAULT_Q_Q     = [49.68, 36.49, 16.80]
DEFAULT_U_MAT   = [0.1, 550.0, 550.0, 550.0]
DEFAULT_Q_OMEGA = [0.026, 3.41, 6.29]
DEFAULT_Q_S     = 0.789
```

## 🔧 Configuration

### Trajectory

3D Lissajous curve (edit `MPCC_baseline.py`):

```python
xd(t) = 7·sin(0.2t) + 3
yd(t) = 7·sin(0.4t)
zd(t) = 1.5·sin(0.4t) + 6
```

### Solver Settings

```python
N_horizon = 30              # prediction steps
t_prediction = 0.3 s        # prediction time
freq = 100 Hz               # control frequency
solver = "SQP_RTI"          # real-time iteration
qp_solver = "FULL_CONDENSING_HPIPM"
```

### Physical Parameters

```python
MASS = 1.0 kg
G = 9.81 m/s²
T_hover = 9.81 N
I = diag([0.0148, 0.0148, 0.0264]) kg·m²
```

## 📊 Performance Metrics

Example output from `MPCC_baseline.py`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MPCC Simulation Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Path length       : 62.82 m
  Path completed    : 100.0 %
  Simulation time   : 14.4 s
  Steps executed    : 1387
  Avg. frequency    : 99.3 Hz
  Avg. solver time  : 3.21 ms
  Solver overruns   : 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 📝 Key Differences: MPCC vs NMPC

| Feature | MPCC | NMPC |
|---|---|---|
| **States** | 14 (+ arc-length θ) | 13 |
| **Controls** | 5 (+ v_θ) | 4 |
| **Reference** | Implicit (via θ) | Explicit (fixed trajectory) |
| **Path tracking** | Contouring + Lag errors | Position error |
| **Progress** | Optimized v_θ | Passive |
| **Timing flexibility** | ✅ Adapts speed | ❌ Fixed timing |

## 🛠️ Development

### Adding New Trajectories

Edit `_trayectoria()` in `MPCC_baseline.py`:

```python
def _trayectoria(t):
    xd   = lambda t: ...  # your x(t)
    yd   = lambda t: ...  # your y(t)
    zd   = lambda t: ...  # your z(t)
    xd_p = lambda t: ...  # dx/dt
    yd_p = lambda t: ...  # dy/dt
    zd_p = lambda t: ...  # dz/dt
    return xd, yd, zd, xd_p, yd_p, zd_p
```

### Modifying Cost Weights

Edit `ocp/mpcc_controller.py`:

```python
DEFAULT_Q_EC    = [10.0, 10.0, 10.0]
DEFAULT_Q_EL    = [5.0, 5.0, 5.0]
# ...
```

Then recompile:

```bash
python3 MPCC_baseline.py  # automatic recompilation
```

## 📚 References

- **MPCC**: Model Predictive Contouring Control (Lam et al., 2010)
- **acados**: Fast embedded optimization (Verschueren et al., 2022)
- **Quaternion dynamics**: Markley & Crassidis (2014)

## 📄 License

Research code - see supervisor for usage terms.

