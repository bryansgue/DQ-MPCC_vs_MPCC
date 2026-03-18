"""
MPCC controller with TUNABLE gains via acados runtime parameters.

This is a COPY of mpcc_controller.py modified so that all cost weights
are declared as `model.p` (acados runtime parameters).  This allows
changing the gains at runtime via `solver.set(stage, "p", gains_vec)`
WITHOUT recompiling the C code — enabling fast bilevel optimisation.

Parameter vector  p ∈ ℝ¹⁷:
    p[ 0: 3]  →  Q_ec   = [Q_ecx, Q_ecy, Q_ecz]     contouring error
    p[ 3: 6]  →  Q_el   = [Q_elx, Q_ely, Q_elz]     lag error
    p[ 6: 9]  →  Q_q    = [Q_qx,  Q_qy,  Q_qz]      quaternion log error
    p[ 9:13]  →  U_mat  = [U_T,   U_tx,  U_ty, U_tz] control effort
    p[13:16]  →  Q_omega= [Q_wx,  Q_wy,  Q_wz]       angular velocity
    p[16]     →  Q_s                                   progress speed

Original file: ocp/mpcc_controller.py  (UNTOUCHED)
"""

import numpy as np
from casadi import MX, dot, vertcat, diag as casadi_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_mpcc_model import f_system_model_mpcc, MASS
from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi   as log_cuaternion_casadi,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights  (same as mpcc_controller.py — used as initial values)
# ──────────────────────────────────────────────────────────────────────────────
N_PARAMS = 17   # total number of runtime parameters

DEFAULT_Q_EC    = [22.00736091362432, 39.52929117895614, 41.351252481945046]
DEFAULT_Q_EL    = [33.874832397312225, 40.30522170732158, 34.90225580204642]
DEFAULT_Q_Q     = [49.68367341061377, 36.49395435837781, 16.803648807763942]
DEFAULT_U_MAT   = [0.1, 550.0, 550.0, 550.0]
DEFAULT_Q_OMEGA = [0.025650491345221405, 3.4126047002592546, 6.291153914894247]
DEFAULT_Q_S     = 0.7891639770382975                         # Progress: Q_s*(v_max-v_θ)²

G = 9.81
DEFAULT_T_MAX      = 5 * G
DEFAULT_T_MIN      = 0.0
DEFAULT_TAUX_MAX   = 0.1                         # ← was 0.03 — too restrictive!
DEFAULT_TAUY_MAX   = 0.1                         # ← was 0.03
DEFAULT_TAUZ_MAX   = 0.1                         # ← was 0.03
DEFAULT_VTHETA_MIN = 0.0
DEFAULT_VTHETA_MAX = 15.0


def weights_to_param_vector(weights: dict | None = None) -> np.ndarray:
    """Convert a weights dict to the p ∈ ℝ¹⁷ parameter vector.

    Parameters
    ----------
    weights : dict or None
        Optional overrides.  Keys match the DEFAULT_* names (lower-case):
            'Q_ec', 'Q_el', 'Q_q', 'U_mat', 'Q_omega', 'Q_s'

    Returns
    -------
    np.ndarray of shape (17,)
    """
    w = weights or {}
    p = np.zeros(N_PARAMS)
    p[0:3]   = w.get('Q_ec',    DEFAULT_Q_EC)
    p[3:6]   = w.get('Q_el',    DEFAULT_Q_EL)
    p[6:9]   = w.get('Q_q',     DEFAULT_Q_Q)
    p[9:13]  = w.get('U_mat',   DEFAULT_U_MAT)
    p[13:16] = w.get('Q_omega', DEFAULT_Q_OMEGA)
    p[16]    = w.get('Q_s',     DEFAULT_Q_S)
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder  (gains are SYMBOLIC — read from model.p at runtime)
# ══════════════════════════════════════════════════════════════════════════════

def create_mpcc_ocp_description_tunable(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Build the MPCC OCP with symbolic cost weights as runtime parameters."""

    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc()

    # ── Declare runtime parameters p ∈ ℝ¹⁷ ──────────────────────────────
    p_sym = MX.sym('p', N_PARAMS)
    model.p = p_sym
    ocp.model = model

    ocp.solver_options.N_horizon = N_horizon

    # ── Extract symbolic weight matrices from p ─────────────────────────
    Q_ec_diag    = p_sym[0:3]       # [3]
    Q_el_diag    = p_sym[3:6]       # [3]
    Q_q_diag     = p_sym[6:9]       # [3]
    U_mat_diag   = p_sym[9:13]      # [4]
    Q_omega_diag = p_sym[13:16]     # [3]
    Q_s_sym      = p_sym[16]        # scalar

    # Build diagonal matrices symbolically
    Q_ec    = casadi_diag(Q_ec_diag)       # 3×3
    Q_el    = casadi_diag(Q_el_diag)       # 3×3
    Q_q     = casadi_diag(Q_q_diag)        # 3×3
    U_mat   = casadi_diag(U_mat_diag)      # 4×4
    Q_omega = casadi_diag(Q_omega_diag)    # 3×3

    T_hover = MASS * 9.81

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── Symbolic reference from θ ────────────────────────────────────────
    theta_state = model.x[13]
    sd      = gamma_pos(theta_state)
    tangent = gamma_vel(theta_state)
    qd      = gamma_quat(theta_state)

    # ── Errors ───────────────────────────────────────────────────────────
    quat_err = quaternion_error(model.x[6:10], qd)
    log_q    = log_cuaternion_casadi(quat_err)

    e_t = sd - model.x[0:3]

    # Lag error: projection of e_t onto the tangent → 3-vector
    e_lag = dot(tangent, e_t) * tangent
    # Contouring error: component orthogonal to tangent
    P_ec = MX.eye(3) - tangent @ tangent.T
    ec   = P_ec @ e_t

    omega   = model.x[10:13]
    v_theta = model.u[4]
    u_dev   = model.u[0:4]

    # ── Cost terms (using symbolic p) ────────────────────────────────────
    control_cost      = u_dev.T @ U_mat @ u_dev
    actitud_cost      = log_q.T @ Q_q @ log_q
    error_contorno    = ec.T @ Q_ec @ ec
    error_lag         = e_lag.T @ Q_el @ e_lag
    omega_cost        = omega.T @ Q_omega @ omega
    arc_speed_penalty = Q_s_sym * (DEFAULT_VTHETA_MAX - v_theta)**2

    # ── Stage cost ───────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost = (
        error_contorno + error_lag + actitud_cost
        + control_cost + omega_cost
        + arc_speed_penalty
    )

    # ── Terminal cost ────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost_e = (
        error_contorno + error_lag + actitud_cost + omega_cost
    )

    # ── Set default parameter values for all stages ──────────────────────
    p_default = weights_to_param_vector()
    ocp.parameter_values = p_default

    # ── Constraints ──────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([
        DEFAULT_T_MIN, -DEFAULT_TAUX_MAX, -DEFAULT_TAUY_MAX, -DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MIN,
    ])
    ocp.constraints.ubu = np.array([
        DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX, DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MAX,
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.lbx   = np.array([0.0])
    ocp.constraints.ubx   = np.array([s_max])
    ocp.constraints.idxbx = np.array([13])

    ocp.constraints.x0 = x0

    # ── Solver options ───────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = t_horizon
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.Tsim             = t_horizon / N_horizon

    return ocp


# ══════════════════════════════════════════════════════════════════════════════
#  Solver factory
# ══════════════════════════════════════════════════════════════════════════════

def build_mpcc_solver_tunable(x0, N_prediction, t_prediction, s_max,
                              gamma_pos, gamma_vel, gamma_quat,
                              use_cython=True):
    """Create, compile and return the tunable MPCC solver.

    Compiles ONCE.  After that, change gains via:
        p_vec = weights_to_param_vector(my_weights)
        for stage in range(N+1):
            solver.set(stage, "p", p_vec)
    """
    ocp = create_mpcc_ocp_description_tunable(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_system_model_mpcc()

    solver_json = 'acados_ocp_' + model.name + '.json'

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, ocp, model, f_system
