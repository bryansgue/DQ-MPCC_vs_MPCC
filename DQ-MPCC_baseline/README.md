# DQ-MPCC Baseline

**Dual-Quaternion Model Predictive Contouring Control** for quadrotor trajectory tracking.

A Lie-invariant MPCC formulation that combines dual-quaternion SE(3) dynamics with arc-length–parameterised contouring control. The pose error is computed via the **exact se(3) logarithmic map**, and the translational component is decomposed into lag and contouring errors in the desired body frame.

---

## Mathematical formulation

### State and control

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\mathbf{Q}$ | 8 | Unit dual quaternion $\mathbf{Q} = \mathbf{q}_r + \varepsilon\,\tfrac{1}{2}\,\mathbf{t}\ast\mathbf{q}_r \in SE(3)$ |
| $\boldsymbol{\xi}$ | 6 | Body-frame twist $[\omega_x,\omega_y,\omega_z,v_x,v_y,v_z]$ |
| $\theta$ | 1 | Arc-length progress along the reference path |
| $\mathbf{u}$ | 5 | $[T,\tau_x,\tau_y,\tau_z,v_\theta]$ — thrust, torques, progress velocity |

Full state: $\mathbf{x}\in\mathbb{R}^{15}=[\mathbf{Q}(8),\;\boldsymbol{\xi}(6),\;\theta]$

### Dynamics

$$\dot{\mathbf{Q}} = \tfrac{1}{2}\,\mathbf{Q}\otimes\boldsymbol{\Xi}, \qquad \dot{\boldsymbol{\xi}} = f(\boldsymbol{\xi},\mathbf{u}), \qquad \dot{\theta} = v_\theta$$

where $\boldsymbol{\Xi} = [0,\boldsymbol{\omega}] + \varepsilon[0,\mathbf{v}]$ is the body twist as a pure dual quaternion.

### Pose error (exact se(3) Log map)

$$\mathbf{Q}_{\text{err}} = \mathbf{Q}_d^* \otimes \mathbf{Q}$$

$$[\boldsymbol{\varphi};\;\boldsymbol{\rho}] = \operatorname{Log}(\mathbf{Q}_{\text{err}}) \in \mathfrak{se}(3)$$

where $\boldsymbol{\varphi}\in\mathfrak{so}(3)$ is the rotation error and $\boldsymbol{\rho} = J_l^{-1}(\boldsymbol{\varphi})\,\mathbf{t}_{\text{err}}$ is the exact translational error using the closed-form left Jacobian inverse (Solà 2018):

$$J_l^{-1}(\boldsymbol{\varphi}) = \alpha\,I_3 - \tfrac{\|\boldsymbol{\varphi}\|}{2}[\hat{\mathbf{n}}]_\times + (1-\alpha)\,\hat{\mathbf{n}}\hat{\mathbf{n}}^\top, \qquad \alpha = \tfrac{\|\boldsymbol{\varphi}\|}{2}\cot\!\tfrac{\|\boldsymbol{\varphi}\|}{2}$$

### Lag-contouring decomposition

The translational error $\boldsymbol{\rho}$ (expressed in the desired body frame) is decomposed into:

$$\boldsymbol{\rho}_{\text{lag}} = (\boldsymbol{\rho}^\top\hat{\mathbf{t}}_b)\,\hat{\mathbf{t}}_b, \qquad \boldsymbol{\rho}_{\text{cont}} = \boldsymbol{\rho} - \boldsymbol{\rho}_{\text{lag}}$$

where $\hat{\mathbf{t}}_b = R_d^\top\,\hat{\mathbf{t}}$ is the path tangent rotated into the desired body frame.

### Cost function

$$L = \boldsymbol{\varphi}^\top Q_\varphi\,\boldsymbol{\varphi} + \boldsymbol{\rho}_{\text{lag}}^\top Q_l\,\boldsymbol{\rho}_{\text{lag}} + \boldsymbol{\rho}_{\text{cont}}^\top Q_c\,\boldsymbol{\rho}_{\text{cont}} + \mathbf{u}_{1:4}^\top R\,\mathbf{u}_{1:4} + \boldsymbol{\omega}^\top Q_\omega\,\boldsymbol{\omega} + Q_s(v_{\max} - v_\theta)^2$$

The reference is a **symbolic CasADi function of** $\theta$, so the solver differentiates through:

$$v_\theta \;\to\; \dot\theta = v_\theta \;\to\; \theta \;\to\; \gamma(\theta) \;\to\; \text{error} \;\to\; \text{cost}$$

---

## Project structure

```
DQ-MPCC_baseline/
├── DQ_MPCC_baseline.py              # Main simulation script
├── README.md
├── models/
│   ├── __init__.py
│   └── dq_quadrotor_mpcc_model.py   # 15-state DQ + θ model (physical params + acados model)
├── ocp/
│   ├── __init__.py
│   └── dq_mpcc_controller.py        # Lie-invariant MPCC OCP formulation + solver builder
└── utils/
    ├── __init__.py
    ├── casadi_utils.py              # θ → position/tangent/quaternion CasADi interpolators
    ├── dq_casadi_utils.py           # DQ symbolic operations, Log map, lag-contouring
    ├── dq_numpy_utils.py            # DQ NumPy operations, RK4, state conversion
    ├── graficas.py                  # Plotting functions
    └── numpy_utils.py               # Arc-length parameterisation, waypoints, errors
```

---

## Dependencies

- Python 3.10+
- [acados](https://github.com/acados/acados) (with Python interface and Cython)
- [CasADi](https://web.casadi.org/)
- NumPy, SciPy, Matplotlib

## Physical parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Mass | 1.0 | kg |
| $J_{xx}$ | 0.00305587 | kg·m² |
| $J_{yy}$ | 0.00159695 | kg·m² |
| $J_{zz}$ | 0.00159687 | kg·m² |
| $g$ | 9.81 | m/s² |

## Running

```bash
python3 DQ_MPCC_baseline.py
```

On first run, acados generates and compiles the C code in `c_generated_code/`. Subsequent runs reuse the compiled solver.

## Default tuning

| Weight | Values | Description |
|--------|--------|-------------|
| $Q_\varphi$ | diag(5, 5, 5) | Orientation error |
| $Q_c$ | diag(15, 45, 45) | Contouring error |
| $Q_l$ | diag(10, 4, 0.12) | Lag error |
| $R$ | diag(0.1, 500, 500, 500) | Control effort |
| $Q_\omega$ | diag(0.5, 0.5, 0.5) | Angular velocity |
| $Q_s$ | 0.5 | Progress speed incentive |

## Solver configuration

| Parameter | Value |
|-----------|-------|
| Frequency | 100 Hz |
| Prediction horizon | 0.3 s (N=30) |
| QP solver | FULL_CONDENSING_HPIPM |
| Integrator | ERK |
| NLP solver | SQP_RTI |
| Hessian approx | GAUSS_NEWTON |

## References

- J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, 2018.
- D. Lam, C. Manzie, M. Good, *Model Predictive Contouring Control*, CDC 2010.
