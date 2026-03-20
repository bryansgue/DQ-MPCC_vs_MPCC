"""
DQ-MPCC controller with TUNABLE gains via acados runtime parameters.

This is a COPY of dq_mpcc_controller.py modified so that all cost weights
are declared as `model.p` (acados runtime parameters).  This allows
changing the gains at runtime via `solver.set(stage, "p", gains_vec)`
WITHOUT recompiling the C code — enabling fast bilevel optimisation.

Parameter vector  p ∈ ℝ¹⁷:
    p[ 0: 3]  →  Q_phi  = [Q_φx, Q_φy, Q_φz]        orientation error (so(3))
    p[ 3: 6]  →  Q_ec   = [Q_ecx, Q_ecy, Q_ecz]      contouring error
    p[ 6: 9]  →  Q_el   = [Q_elx, Q_ely, Q_elz]      lag error
    p[ 9:13]  →  U_mat  = [U_T,   U_tx,  U_ty, U_tz]  control effort
    p[13:16]  →  Q_omega= [Q_wx,  Q_wy,  Q_wz]        angular velocity
    p[16]     →  Q_s                                    progress speed

Original file: ocp/dq_mpcc_controller.py  (UNTOUCHED)
"""

import os
import sys
import shutil
import numpy as np
from casadi import MX, vertcat, norm_2, dot, diag as casadi_diag
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

# ── Shared tuning configuration (control limits, v_theta_max, etc.) ──────────
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)
from tuning_config import (
    T_MAX as _CFG_T_MAX, T_MIN as _CFG_T_MIN,
    TAUX_MAX as _CFG_TAUX_MAX, TAUY_MAX as _CFG_TAUY_MAX,
    TAUZ_MAX as _CFG_TAUZ_MAX,
    VTHETA_MIN as _CFG_VTHETA_MIN, VTHETA_MAX as _CFG_VTHETA_MAX,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights  (same as dq_mpcc_controller.py — used as initial values)
# ──────────────────────────────────────────────────────────────────────────────
N_PARAMS = 17   # total number of runtime parameters

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

# ── Control limits from shared tuning_config.py ──────────────────────────────
DEFAULT_T_MAX      = _CFG_T_MAX
DEFAULT_T_MIN      = _CFG_T_MIN
DEFAULT_TAUX_MAX   = _CFG_TAUX_MAX
DEFAULT_TAUY_MAX   = _CFG_TAUY_MAX
DEFAULT_TAUZ_MAX   = _CFG_TAUZ_MAX
DEFAULT_VTHETA_MIN = _CFG_VTHETA_MIN
DEFAULT_VTHETA_MAX = _CFG_VTHETA_MAX


def weights_to_param_vector(weights: dict | None = None) -> np.ndarray:
    """Convert a weights dict to the p ∈ ℝ¹⁷ parameter vector.

    Parameters
    ----------
    weights : dict or None
        Optional overrides.  Keys:
            'Q_phi', 'Q_ec', 'Q_el', 'U_mat', 'Q_omega', 'Q_s'

    Returns
    -------
    np.ndarray of shape (17,)
    """
    w = weights or {}
    p = np.zeros(N_PARAMS)
    p[0:3]   = w.get('Q_phi',   DEFAULT_Q_PHI)
    p[3:6]   = w.get('Q_ec',    DEFAULT_Q_EC)
    p[6:9]   = w.get('Q_el',    DEFAULT_Q_EL)
    p[9:13]  = w.get('U_mat',   DEFAULT_U_MAT)
    p[13:16] = w.get('Q_omega', DEFAULT_Q_OMEGA)
    p[16]    = w.get('Q_s',     DEFAULT_Q_S)
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder  (gains are SYMBOLIC — read from model.p at runtime)
# ══════════════════════════════════════════════════════════════════════════════

def create_dq_mpcc_ocp_description_tunable(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Build the DQ-MPCC OCP with symbolic cost weights as runtime parameters.

    All 17 cost weights are declared as `model.p`, enabling runtime
    gain changes WITHOUT recompilation.

    Cost structure (identical to dq_mpcc_controller.py):
        L = φᵀ Q_φ φ  +  ρ_lag ᵀ Q_l  ρ_lag  +  ρ_cont ᵀ Q_c  ρ_cont
          + u[:4]ᵀ R  u[:4]  +  ωᵀ Q_ω  ω
          + Q_s (v_max − v_θ)²
    """
    ocp = AcadosOcp()
    model, _, _, _ = f_dq_system_model_mpcc()

    # ── Declare runtime parameters p ∈ ℝ¹⁷ ──────────────────────────────
    p_sym = MX.sym('p', N_PARAMS)
    model.p = p_sym
    ocp.model = model

    # Absolute path so the compiled solver always lives next to the ocp/ folder,
    # regardless of what cwd is when the script runs.
    _OCP_DIR  = os.path.dirname(os.path.abspath(__file__))
    _PROJ_DIR = os.path.dirname(_OCP_DIR)
    ocp.code_export_directory = os.path.join(_PROJ_DIR, 'c_generated_code_dq_tuner')

    ocp.solver_options.N_horizon = N_horizon

    # ── Extract symbolic weight matrices from p ─────────────────────────
    Q_phi_diag   = p_sym[0:3]       # [3]  orientation
    Q_ec_diag    = p_sym[3:6]       # [3]  contouring
    Q_el_diag    = p_sym[6:9]       # [3]  lag
    U_mat_diag   = p_sym[9:13]      # [4]  control effort
    Q_omega_diag = p_sym[13:16]     # [3]  angular velocity
    Q_s_sym      = p_sym[16]        # scalar  progress speed

    # Build diagonal matrices symbolically
    Q_phi   = casadi_diag(Q_phi_diag)      # 3×3
    Q_ec    = casadi_diag(Q_ec_diag)       # 3×3
    Q_el    = casadi_diag(Q_el_diag)       # 3×3
    U_mat   = casadi_diag(U_mat_diag)      # 4×4
    Q_omega = casadi_diag(Q_omega_diag)    # 3×3

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
    tangent_body = rotate_tangent_to_desired_frame(tangent, qd)
    rho_lag, rho_cont = lag_contouring_decomposition(rho, tangent_body)

    # ── Twist and control ────────────────────────────────────────────────
    omega   = model.x[8:11]                   # angular velocity [ωx, ωy, ωz]
    v_theta = model.u[4]                      # progress velocity

    # ── Cost terms (using symbolic p) ────────────────────────────────────
    orientation_cost  = phi.T @ Q_phi @ phi
    contouring_cost   = rho_cont.T @ Q_ec @ rho_cont
    lag_cost          = rho_lag.T @ Q_el @ rho_lag
    control_cost      = model.u[0:4].T @ U_mat @ model.u[0:4]
    omega_cost        = omega.T @ Q_omega @ omega
    arc_speed_penalty = Q_s_sym * (DEFAULT_VTHETA_MAX - v_theta)**2

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

    # ── Set default parameter values for all stages ──────────────────────
    p_default = weights_to_param_vector()
    ocp.parameter_values = p_default

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

def build_dq_mpcc_solver_tunable(x0, N_prediction, t_prediction, s_max,
                                  gamma_pos, gamma_vel, gamma_quat,
                                  use_cython=True):
    """Create, compile and return the tunable DQ-MPCC solver.

    Compiles ONCE.  After that, change gains via:
        p_vec = weights_to_param_vector(my_weights)
        for stage in range(N+1):
            solver.set(stage, "p", p_vec)
    """
    ocp = create_dq_mpcc_ocp_description_tunable(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_dq_system_model_mpcc()

    solver_json = os.path.join(
        os.path.dirname(ocp.code_export_directory),
        'acados_ocp_' + model.name + '_tuner.json',
    )

    if use_cython:
        # Clean stale code to force a fresh build
        if os.path.isdir(ocp.code_export_directory):
            shutil.rmtree(ocp.code_export_directory)
        if os.path.isfile(solver_json):
            os.remove(solver_json)
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        # ctypes path: build=True + generate=True already force full rebuild
        acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file=solver_json,
            build=True, generate=True,
        )

    return acados_ocp_solver, ocp, model, f_system
