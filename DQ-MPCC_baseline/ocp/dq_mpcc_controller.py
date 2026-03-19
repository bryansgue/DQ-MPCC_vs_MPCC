"""
DQ-MPCC (Dual-Quaternion Model Predictive Contouring Control) – Lie-invariant
formulation with symbolic trajectory interpolation.

Key DQ-MPCC features:
  • θ (arc-length progress) is an optimisation STATE   (x[14])
  • v_θ (progress velocity) is an optimisation CONTROL (u[4])
  • θ̇ = v_θ  (augmented dynamics)
  • Pose error via exact se(3) logarithmic map:
        [φ; ρ] = Log(Q_d* ⊗ Q)
  • Lag-contouring decomposition of ρ in the desired body frame
  • The reference is a **symbolic CasADi function of θ** so the solver
    can differentiate through:
        v_θ  →  θ̇=v_θ  →  θ  →  reference(θ)  →  error  →  cost

State   x ∈ ℝ¹⁵ = [dq(8), twist(6), θ]
Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

Cost structure:
    L = φᵀ Q_φ φ  +  ρ_lag ᵀ Q_l  ρ_lag  +  ρ_cont ᵀ Q_c  ρ_cont
      + u[:4]ᵀ R  u[:4]  +  ωᵀ Q_ω  ω
      + Q_s (v_max − v_θ)²
"""

import numpy as np
from casadi import MX, vertcat, norm_2, dot
from acados_template import AcadosOcp, AcadosOcpSolver

from models.dq_quadrotor_mpcc_model import MASS, f_dq_system_model_mpcc
from utils.dq_casadi_utils import (
    dq_error_casadi,
    ln_dual_casadi,
    dq_from_pose_casadi,
    dq_get_quaternion_casadi,
    rotate_tangent_to_desired_frame,
    lag_contouring_decomposition,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights
# ──────────────────────────────────────────────────────────────────────────────

# Orientation error φ ∈ so(3)  [φ_x, φ_y, φ_z]
DEFAULT_Q_PHI   = [5.0, 5.0, 5.0]

# Contouring error ρ_cont  (⊥ to path)  [ρ_cx, ρ_cy, ρ_cz]
DEFAULT_Q_EC    = [10.0, 10.0, 10.0]  

# Lag error ρ_lag  (‖ to path)  [ρ_lx, ρ_ly, ρ_lz]
DEFAULT_Q_EL    = [5.0, 5.0, 5.0]    

# Control effort [T, τx, τy, τz]
DEFAULT_U_MAT   = [0.1, 250.0, 250.0, 250.0] 

# Angular velocity cost [ωx, ωy, ωz]
DEFAULT_Q_OMEGA = [0.5, 0.5, 0.5]

# Progress speed incentive:  Q_s * (v_max − v_θ)²
DEFAULT_Q_S     = 0.3

G = 9.81
DEFAULT_T_MAX      = 5 * G
DEFAULT_T_MIN      = 0.0
DEFAULT_TAUX_MAX   = 0.1
DEFAULT_TAUY_MAX   = 0.1
DEFAULT_TAUZ_MAX   = 0.1
DEFAULT_VTHETA_MIN = 0.0
DEFAULT_VTHETA_MAX = 15.0


# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder
# ══════════════════════════════════════════════════════════════════════════════

def create_dq_mpcc_ocp_description(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Create DQ-MPCC OCP with Lie-invariant cost on se(3).

    The desired dual quaternion, tangent, and quaternion are all symbolic
    CasADi functions of θ = x[14].  The solver differentiates through
    θ → reference → error → cost, making v_θ truly optimisable.

    Cost structure:
        L = φᵀ Q_φ φ  +  ρ_lag ᵀ Q_l  ρ_lag  +  ρ_cont ᵀ Q_c  ρ_cont
          + u[:4]ᵀ R  u[:4]  +  ωᵀ Q_ω  ω
          + Q_s (v_max − v_θ)²
    """
    ocp = AcadosOcp()
    model, _, _, _ = f_dq_system_model_mpcc()
    ocp.model = model

    ocp.solver_options.N_horizon = N_horizon

    # ── Runtime parameter: v_theta_max ───────────────────────────────────
    p_vtheta_max = MX.sym('p_vtheta_max')
    model.p = p_vtheta_max
    ocp.parameter_values = np.array([DEFAULT_VTHETA_MAX])

    # ── Weights ──────────────────────────────────────────────────────────
    Q_phi   = np.diag(DEFAULT_Q_PHI)          # 3×3  orientation
    Q_ec    = np.diag(DEFAULT_Q_EC)           # 3×3  contouring
    Q_el    = np.diag(DEFAULT_Q_EL)           # 3×3  lag
    U_mat   = np.diag(DEFAULT_U_MAT)          # 4×4  control effort
    Q_omega = np.diag(DEFAULT_Q_OMEGA)        # 3×3  angular velocity
    Q_s     = DEFAULT_Q_S                     # scalar speed incentive

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── Symbolic reference from θ ────────────────────────────────────────
    theta_state = model.x[14]                 # θ is the 15th state
    sd      = gamma_pos(theta_state)          # (3,)  reference position
    tangent = gamma_vel(theta_state)          # (3,)  unit tangent (inertial)
    qd      = gamma_quat(theta_state)        # (4,)  desired quaternion

    # Build desired dual quaternion from position + quaternion interpolators
    dq_desired = dq_from_pose_casadi(qd, sd)  # (8,)

    # Current dual quaternion from state
    dq_actual = model.x[0:8]

    # ── Dual-quaternion error + exact Log ────────────────────────────────
    dq_err = dq_error_casadi(dq_desired, dq_actual)   # Q_err = Q_d* ⊗ Q
    log_err = ln_dual_casadi(dq_err)                    # [φ(3); ρ(3)] ∈ se(3)

    phi = log_err[0:3]    # rotation vector (axis·angle)
    rho = log_err[3:6]    # translational error (desired body frame)

    # ── Lag-contouring decomposition in the desired body frame ───────────
    # Rotate inertial tangent into desired body frame:  t_body = R_d^T · t
    tangent_body = rotate_tangent_to_desired_frame(tangent, qd)
    rho_lag, rho_cont = lag_contouring_decomposition(rho, tangent_body)

    # ── Twist and control ────────────────────────────────────────────────
    omega   = model.x[8:11]                   # angular velocity [ωx, ωy, ωz]
    v_theta = model.u[4]                      # progress velocity

    # ── Cost terms ───────────────────────────────────────────────────────
    orientation_cost  = phi.T @ Q_phi @ phi
    contouring_cost   = rho_cont.T @ Q_ec @ rho_cont
    lag_cost          = rho_lag.T @ Q_el @ rho_lag
    control_cost      = model.u[0:4].T @ U_mat @ model.u[0:4]
    omega_cost        = omega.T @ Q_omega @ omega
    arc_speed_penalty = Q_s * (p_vtheta_max - v_theta)**2

    # ── Stage cost ───────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost = (
        orientation_cost + contouring_cost + lag_cost
        + control_cost + omega_cost
        + arc_speed_penalty
    )

    # ── Terminal cost (no control or speed terms) ────────────────────────
    ocp.model.cost_expr_ext_cost_e = (
        orientation_cost + contouring_cost + lag_cost + omega_cost
    )

    # ── Control constraints ──────────────────────────────────────────────
    ocp.constraints.lbu = np.array([
        DEFAULT_T_MIN, -DEFAULT_TAUX_MAX, -DEFAULT_TAUY_MAX, -DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MIN,
    ])
    ocp.constraints.ubu = np.array([
        DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX, DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MAX,
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # ── State constraints: θ ∈ [0, s_max] ───────────────────────────────
    ocp.constraints.lbx   = np.array([0.0])
    ocp.constraints.ubx   = np.array([s_max])
    ocp.constraints.idxbx = np.array([14])          # θ is state index 14

    # ── Nonlinear constraint: ‖q_r‖ = 1 (soft) ──────────────────────────
    qr = model.x[0:4]
    ocp.model.con_h_expr = norm_2(qr)
    ocp.constraints.lh = np.array([0.99])           # soft lower
    ocp.constraints.uh = np.array([1.01])           # soft upper

    # Slack variables for the soft constraint
    ns = 1
    ocp.constraints.lsh = np.zeros(ns)
    ocp.constraints.ush = np.zeros(ns)
    ocp.constraints.idxsh = np.array(range(ns))

    # L2 and L1 penalty on slack (make it soft)
    ocp.cost.zl = 100.0 * np.ones(ns)              # L1 lower
    ocp.cost.zu = 100.0 * np.ones(ns)              # L1 upper
    ocp.cost.Zl = 100.0 * np.ones(ns)              # L2 lower
    ocp.cost.Zu = 100.0 * np.ones(ns)              # L2 upper

    # ── Initial state ────────────────────────────────────────────────────
    ocp.constraints.x0 = x0

    # ── Solver options ───────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.Tsim             = t_horizon / N_horizon
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = t_horizon

    return ocp


# ══════════════════════════════════════════════════════════════════════════════
#  Solver factory
# ══════════════════════════════════════════════════════════════════════════════

def build_dq_mpcc_solver(x0, N_prediction, t_prediction, s_max,
                         gamma_pos, gamma_vel, gamma_quat,
                         use_cython=True):
    """Create, code-generate, compile and return the DQ-MPCC solver."""
    ocp = create_dq_mpcc_ocp_description(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_dq_system_model_mpcc()

    solver_json = 'acados_ocp_' + model.name + '.json'

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, ocp, model, f_system
