# Nonlinear Model Predictive Control for Quadrotor UAV: Baseline Implementation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![acados](https://img.shields.io/badge/acados-v0.5.3-orange.svg)](https://docs.acados.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.7.2-red.svg)](https://web.casadi.org/)

> **A high-performance NMPC baseline for quadrotor trajectory tracking using acados and CasADi**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Mathematical Model](#mathematical-model)
4. [Optimal Control Problem Formulation](#optimal-control-problem-formulation)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [References](#references)

---

## Abstract

This repository provides a **baseline implementation** of Nonlinear Model Predictive Control (NMPC) for quadrotor trajectory tracking. The controller is implemented using the **acados** optimization framework and **CasADi** symbolic framework, achieving real-time performance at **100 Hz** with an average solver time of **~3.5 ms** per iteration.

The implementation features:
- ✅ Full 6-DOF quadrotor dynamics with quaternion-based attitude representation
- ✅ External cost formulation with position error and quaternion logarithm error metrics
- ✅ Real-Time Iteration (RTI) scheme with SQP solver
- ✅ Automatic code generation for embedded deployment
- ✅ Comprehensive timing analysis and performance monitoring

---

## 1. Introduction

### 1.1 Motivation

Quadrotor UAVs require precise and robust control to perform complex maneuvers in 3D space. Classical control approaches (PID, LQR) often struggle with:
- Nonlinear dynamics at aggressive maneuvers
- Coupling between translational and rotational dynamics
- Constraints on actuation (thrust and torque limits)

**Model Predictive Control (MPC)** addresses these challenges by:
- Explicitly handling system constraints
- Optimizing over a finite prediction horizon
- Naturally incorporating reference trajectories

### 1.2 System Overview

This implementation targets a **generic quadrotor platform** with the following characteristics:

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Mass | m | 1.0 | kg |
| Inertia (x-axis) | J_xx | 0.00305587 | kg·m² |
| Inertia (y-axis) | J_yy | 0.00159695 | kg·m² |
| Inertia (z-axis) | J_zz | 0.00159687 | kg·m² |
| Gravity | g | 9.81 | m/s² |

### 1.3 Control Frequency

The controller operates at a fixed frequency of **100 Hz** (T_s = 10 ms), with configurable prediction horizon:

```
N_pred = floor(T_pred / T_s)
```

where `T_pred` is the user-defined prediction time (default: 0.5 s → 50 steps).

---

## 2. Mathematical Model

### 2.1 State-Space Representation

The quadrotor state vector **x ∈ ℝ¹³** is defined as:

```
x = [p; v; q; ω]
  = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z]ᵀ
```

where:
- **p** = [p_x, p_y, p_z]ᵀ ∈ ℝ³: position in the inertial frame
- **v** = [v_x, v_y, v_z]ᵀ ∈ ℝ³: linear velocity in the inertial frame
- **q** = [q_w, q_x, q_y, q_z]ᵀ ∈ S³: unit quaternion (attitude)
- **ω** = [ω_x, ω_y, ω_z]ᵀ ∈ ℝ³: angular velocity in the body frame

The control input **u ∈ ℝ⁴** is:

```
u = [T, τ_x, τ_y, τ_z]ᵀ
```

where:
- T ∈ [0, 3g]: total thrust (normalized by mass)
- τ ∈ ℝ³: body torques with |τ|_∞ ≤ 0.05 N·m

### 2.2 Continuous-Time Dynamics

The system dynamics are given by:

#### Translational Dynamics

```
ṗ = v
v̇ = -g·e₃ + (1/m)·R(q)·[0, 0, T]ᵀ
```

where **R(q) ∈ SO(3)** is the rotation matrix from body to inertial frame, computed from the quaternion as:

```
R(q) = I₃ + 2[q_v]×² + 2q_w[q_v]×
```

with q_v = [q_x, q_y, q_z]ᵀ and [·]× denoting the skew-symmetric matrix operator.

#### Rotational Dynamics (Quaternion Kinematics)

```
q̇ = (1/2)·q ⊗ [0, ω]ᵀ
ω̇ = J⁻¹·(τ - ω × (J·ω))
```

where:
- ⊗ denotes quaternion multiplication (Hamilton product)
- **J** = diag(J_xx, J_yy, J_zz) is the inertia matrix
- × denotes the vector cross product

### 2.3 Discretization

The continuous dynamics are discretized using a **4th-order Runge-Kutta (RK4)** integrator:

```
x_{k+1} = x_k + (T_s/6)·(k₁ + 2k₂ + 2k₃ + k₄)
```

where:

```
k₁ = f(x_k, u_k)
k₂ = f(x_k + (T_s/2)·k₁, u_k)
k₃ = f(x_k + (T_s/2)·k₂, u_k)
k₄ = f(x_k + T_s·k₃, u_k)
```

---

## 3. Optimal Control Problem Formulation

### 3.1 Cost Function

The NMPC problem minimizes a stage cost and terminal cost over the prediction horizon N:

```
min_{u_{0:N-1}} Σ_{k=0}^{N-1} ℓ(x_k, u_k, x_k^ref) + ℓ_N(x_N, x_N^ref)
```

#### Stage Cost

```
ℓ(x, u, x^ref) = ‖p - p^ref‖²_Q + ‖u‖²_R + ‖Log(q_err)‖²_K
```

where:
- **Position error**: e_p = p - p^ref
- **Quaternion error**: q_err = q⁻¹ ⊗ q^ref
- **Quaternion logarithm**: Log(q_err) = 2·(q_v/‖q_v‖)·arctan(‖q_v‖/q_w)

**Cost Matrices:**

```
Q = diag(25, 25, 30)      # Position weights
K = diag(12, 12, 12)      # Orientation weights
R = diag(1, 800, 800, 800) # Control effort weights
```

**Design Rationale:**
- High Q(3,3) for altitude tracking priority
- High R for smooth control (aggressive penalization of torques)
- K balances orientation error across all axes

#### Terminal Cost

```
ℓ_N(x_N, x_N^ref) = ‖p_N - p_N^ref‖²_Q + ‖Log(q_err,N)‖²_K
```

(No control penalization at the terminal state)

### 3.2 Constraints

#### Control Constraints (Box Constraints)

```
0 ≤ T ≤ 3g = 29.43 N
|τᵢ| ≤ 0.05 N·m, i ∈ {x, y, z}
```

#### State Constraints (Optional, currently disabled)

```
z_min = 1.5 m ≤ p_z ≤ z_max = 50 m
```

#### Initial Condition Constraint

```
x₀ = x_meas
```

### 3.3 Solver Configuration

The optimal control problem is solved using:

- **Solver**: Sequential Quadratic Programming with Real-Time Iteration (SQP-RTI)
- **QP Solver**: FULL_CONDENSING_HPIPM (High-Performance Interior Point Method)
- **Hessian Approximation**: Gauss-Newton
- **Integrator**: Explicit Runge-Kutta (ERK) 4th order
- **Tolerance**: ε = 10⁻⁴

**SQP-RTI Scheme:**

The RTI variant performs only **one SQP iteration** per sampling instant, which:
- Ensures deterministic execution time
- Enables real-time performance (solver completes within T_s = 10 ms)
- Relies on warm-starting from the previous solution

---

## 4. Implementation Details

### 4.1 Software Architecture

```
MPC_baseline/
├── T_UAV_baseline.py          # Main controller implementation
├── graficas.py                # Plotting and visualization utilities
├── acados_ocp_*.json          # acados OCP configuration
└── c_generated_code/          # Auto-generated C code for solver
    ├── libacados_ocp_solver_*.so
    └── acados_solver_*.{c,h}
```

### 4.2 Key Implementation Features

#### 4.2.1 Quaternion Normalization

To maintain numerical stability, quaternions are normalized and sign-corrected:

```python
q_normalized = q / norm_2(q)
q = if_else(q[0] < 0, -q, q)  # Enforce positive scalar part
```

#### 4.2.2 Quaternion Logarithm (SO(3) Error Metric)

The quaternion error is mapped to the tangent space of SO(3) using:

```
Log(q) = 2·(q_v/‖q_v‖)·θ,  where θ = arctan2(‖q_v‖, q_w)
```

This provides a **geodesic distance** on the unit sphere, avoiding gimbal lock.

#### 4.2.3 Rate Control Loop

Instead of fixed `sleep(T_s)`, the implementation uses **adaptive rate control**:

```python
elapsed = time.time() - tic
remaining = T_s - elapsed
if remaining > 0:
    time.sleep(remaining)
```

This ensures the loop frequency stays close to 100 Hz even with variable solver times.

### 4.3 Performance Monitoring

The implementation tracks three key metrics:

1. **Solver Time** (t_solver): Pure optimization time
2. **Loop Time** (t_loop): Total iteration time (compute + sleep)
3. **Overruns**: Iterations where t_loop > 1.05·T_s

Output format:
```
[k=0042]  solver= 3.45 ms  |  loop= 10.02 ms  |  99.8 Hz
```

---

## 5. Results

### 5.1 Trajectory Tracking Performance

The controller successfully tracks a 3D Lissajous trajectory:

```
p_x(t) = 4·sin(0.6t) + 3
p_y(t) = 4·sin(1.2t)
p_z(t) = 2·sin(1.2t) + 6
```

with desired yaw:

```
ψ_d(t) = arctan2(ṗ_y, ṗ_x)
```

**Generated Figures:**
- `1_pose.png`: 3D trajectory and position vs. time
- `2_control_actions.png`: Thrust and torque commands
- `3_vel_lineal.png`: Linear velocities
- `4_vel_angular.png`: Angular velocities
- `5_CBF.png`: Control Barrier Function values (if enabled)
- `6_timing.png`: Solver and loop timing analysis

### 5.2 Computational Performance

**Benchmark Configuration:**
- Platform: Ubuntu 22.04 / Python 3.10
- Prediction Horizon: T_pred = 0.5 s (50 steps)
- Control Frequency: 100 Hz
- Simulation Duration: 30 s (2971 iterations)

**Timing Statistics:**

| Metric | Mean | Max | Std Dev | Unit |
|--------|------|-----|---------|------|
| Solver Time | 3.49 | 6.89 | 0.35 | ms |
| Loop Time | 10.08 | 10.92 | 0.04 | ms |
| Effective Frequency | 99.2 | - | - | Hz |
| Overruns | 3 / 2971 | - | - | (0.1%) |

**Analysis:**
- ✅ Solver time is **~35% of the sampling time** → margin for safety
- ✅ Loop time is consistently **≈10 ms** → excellent rate stability
- ✅ Overruns are **negligible** (0.1%) → real-time guarantees met

### 5.3 Scalability Analysis

| T_pred | N_pred | Solver Time (est.) |
|--------|--------|--------------------|
| 0.1 s | 10 steps | ~1.2 ms |
| 0.2 s | 20 steps | ~2.0 ms |
| 0.3 s | 30 steps | ~2.8 ms |
| 0.5 s | 50 steps | ~3.5 ms |
| 1.0 s | 100 steps | ~6.8 ms |
| 2.0 s | 200 steps | ~14 ms ⚠️ |

⚠️ At T_pred > 1.5 s, solver time may exceed the sampling period → reduce control frequency or use longer horizon MPC.

---

## 6. Installation

### 6.1 Dependencies

#### System Requirements
- **OS**: Linux (tested on Ubuntu 20.04/22.04)
- **Python**: 3.8+
- **Compiler**: GCC 7.5+ or Clang 10+

#### Required Packages

```bash
# acados (optimization framework)
acados == 0.5.3

# CasADi (symbolic framework)
casadi == 3.7.2

# Python scientific stack
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.5.0
```

### 6.2 acados Installation

```bash
# Clone acados repository
git clone https://github.com/acados/acados.git
cd acados
git checkout v0.5.3
git submodule update --recursive --init

# Build acados
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON \
      -DACADOS_WITH_HPIPM=ON \
      -DACADOS_WITH_OSQP=ON \
      ..
make -j4 && sudo make install

# Set environment variables (add to ~/.bashrc)
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib"

# Install Python interface
pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template
```

### 6.3 CasADi Installation

```bash
pip install casadi==3.7.2
```

### 6.4 Clone This Repository

```bash
git clone https://github.com/bryansgue/CLF_CBF_MPC_UAV_Acados.git
cd CLF_CBF_MPC_UAV_Acados/MPC_baseline
```

---

## 7. Usage

### 7.1 Basic Execution

```bash
cd /path/to/MPC_baseline
python3 T_UAV_baseline.py
```

**Expected Output:**
```
[CONFIG]  frec=100 Hz  |  t_s=10.00 ms  |  t_prediction=0.5 s  |  N_prediction=50 steps
Ready!!!
[k=0000]  solver= 3.52 ms  |  loop= 10.05 ms  |   99.5 Hz
[k=0001]  solver= 3.48 ms  |  loop= 10.03 ms  |   99.7 Hz
...
Generating figures...
✓ Saved 1_pose.png
✓ Saved 6_timing.png

╔══════════════════════════════════════════════════════════════════╗
║                     TIMING STATISTICS                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Nominal t_s = 10.00 ms  (100 Hz)                              ║
╠══════════════════════════════════════════════════════════════════╣
║  [Solver]  mean= 3.49  max= 6.89  std=0.35  ms     ║
║  [Loop  ]  mean=10.08  max=10.92  std=0.04  ms     ║
║  Freq real :  99.2 Hz                                  ║
║  Overruns  :    3 / 2971 iters (0.1 %)                 ║
╚══════════════════════════════════════════════════════════════════╝
```

### 7.2 Configuration Parameters

Edit `T_UAV_baseline.py` (lines 570-578):

```python
def main():
    # Simulation Time
    t_final = 30                     # Total simulation duration [s]
    
    # Control Frequency
    frec = 100                       # Sampling frequency [Hz]
    t_s = 1/frec                     # Sampling time [s]
    
    # Prediction Horizon
    t_prediction = 0.5               # Prediction time [s] ← MODIFY HERE
    N_prediction = int(round(t_prediction / t_s))  # Auto-computed
```

**Parameters to Tune:**
- `t_final`: Simulation duration (default: 30 s)
- `frec`: Control frequency (default: 100 Hz)
- `t_prediction`: MPC horizon (default: 0.5 s → 50 steps)

**Weight Matrices** (lines 408-421):
```python
# Position tracking
Q_mat[0, 0] = 25    # X position
Q_mat[1, 1] = 25    # Y position
Q_mat[2, 2] = 30    # Z position (higher for altitude priority)

# Orientation tracking
K_mat[0, 0] = 12    # Roll error
K_mat[1, 1] = 12    # Pitch error
K_mat[2, 2] = 12    # Yaw error

# Control effort
R_mat[0, 0] = 1.0   # Thrust penalty
R_mat[1, 1] = 800   # Torque X penalty (high for smoothness)
R_mat[2, 2] = 800   # Torque Y penalty
R_mat[3, 3] = 800   # Torque Z penalty
```

### 7.3 Reference Trajectory

The trajectory is defined in `main()` (lines 605-615):

```python
# Lissajous curve parameters
value = 15
xd = lambda t: 4 * np.sin(value*0.04*t) + 3
yd = lambda t: 4 * np.sin(value*0.08*t)
zd = lambda t: 2 * np.sin(value*0.08*t) + 6
```

**To define a custom trajectory:**
1. Modify the `xd`, `yd`, `zd` lambda functions
2. Ensure continuity (C¹ or C²) for smooth tracking
3. Compute derivatives `xdp`, `ydp`, `zdp` for yaw reference

### 7.4 Visualization

All figures are automatically saved as PNG in the current directory:

```bash
ls *.png
# Output:
# 1_pose.png  2_control_actions.png  3_vel_lineal.png
# 4_vel_angular.png  5_CBF.png  6_timing.png
```

To customize plots, edit `graficas.py`.

---

## 8. Code Structure

### 8.1 Main Functions

| Function | Description | Lines |
|----------|-------------|-------|
| `f_system_model()` | Defines quadrotor dynamics in CasADi | 150-285 |
| `create_ocp_solver_description()` | Configures NMPC problem | 390-465 |
| `f_d()` | RK4 discrete-time integrator | 290-297 |
| `quaternion_multiply()` | Hamilton product q₁ ⊗ q₂ | 115-135 |
| `quat_p()` | Quaternion kinematics q̇ | 140-145 |
| `QuatToRot()` | Quaternion → Rotation matrix | 90-110 |
| `log_cuaternion_casadi()` | SO(3) logarithm map | 360-385 |
| `main()` | Main control loop | 570-829 |

### 8.2 External Dependencies

```python
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, Function, norm_2, cos, sin, ...
from graficas import plot_pose, plot_timing, ...
import numpy as np
```

---

## 9. Extensions and Variants

### 9.1 Control Barrier Functions (CBF)

See `T_UAV_CBF_baseline.py` for a variant with **safety constraints** via CBF:

```
h(x) ≥ 0  ⟹  ḣ(x) + α·h(x) ≥ 0
```

Useful for:
- Obstacle avoidance
- Geofencing
- Safe speed limits

### 9.2 Potential Extensions

**Short-term:**
- [ ] Add wind disturbance rejection
- [ ] Implement trajectory re-planning
- [ ] Multi-UAV formation control

**Long-term:**
- [ ] Learning-based model adaptation (NMPC + Neural Networks)
- [ ] Embedded deployment (C code export)
- [ ] Hardware-in-the-loop (HIL) testing

---

## 10. Troubleshooting

### 10.1 Common Issues

**Problem:** Solver time exceeds sampling period

**Solution:**
- Reduce prediction horizon: `t_prediction = 0.3` (30 steps)
- Use partial condensing: `qp_solver = "PARTIAL_CONDENSING_HPIPM"`
- Decrease tolerance: `tol = 1e-3`

---

**Problem:** Quaternion norm drift

**Solution:**
- The implementation already normalizes quaternions in `QuatToRot()`
- For long simulations, add explicit normalization constraint

---

**Problem:** `acados_ocp_solver` not found

**Solution:**
```bash
# Regenerate solver
rm -rf c_generated_code/ acados_ocp_*.json
python3 T_UAV_baseline.py
```

---



**Last Updated:** March 16, 2026  
**Version:** 1.0.0  
**Status:** Stable ✅
# NMPC_baseline
# NMPC_baseline
