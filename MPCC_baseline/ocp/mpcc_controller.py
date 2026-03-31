"""
Canonical MPCC controller with runtime parameters p ∈ R^18.

This is the OCP that should be used by BOTH:
  - production simulation / execution
  - bilevel tuning

Compile ONCE, then update numeric values at runtime via:
    solver.set(stage, "p", p_vec)

Parameter vector:
    p[ 0: 3]  -> Q_ec      contouring error weights
    p[ 3: 6]  -> Q_el      lag error weights
    p[ 6: 9]  -> Q_q       quaternion-log error weights
    p[ 9:13]  -> U_mat     control effort weights [T, taux, tauy, tauz]
    p[13:16]  -> Q_omega   angular-velocity weights
    p[16]     -> Q_s       linear progress weight in -Q_s * v_theta
    p[17]     -> vtheta_max runtime upper bound helper (not part of the cost)

Important:
  - trajectory interpolation is still part of the symbolic graph, so changing
    the hardcoded trajectory / horizon / waypoint structure still requires
    rebuilding once
  - weights do NOT require rebuilding
"""

import os
import sys
import shutil
import numpy as np
from casadi import MX, dot, diag as casadi_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_mpcc_model import f_system_model_mpcc
from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi as log_cuaternion_casadi,
)

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiment_config import (
    T_MAX, T_MIN,
    TAUX_MAX, TAUY_MAX, TAUZ_MAX,
    VTHETA_MIN, VTHETA_MAX, ATHETA_MIN, ATHETA_MAX,
    MPCC_Q_EC, MPCC_Q_EL, MPCC_Q_Q, MPCC_U_MAT, MPCC_Q_OMEGA, MPCC_Q_S,
    MPCC_Q_ATHETA,
)


N_PARAMS = 18

DEFAULT_Q_EC = MPCC_Q_EC
DEFAULT_Q_EL = MPCC_Q_EL
DEFAULT_Q_Q = MPCC_Q_Q
DEFAULT_U_MAT = MPCC_U_MAT
DEFAULT_Q_OMEGA = MPCC_Q_OMEGA
DEFAULT_Q_S = MPCC_Q_S
DEFAULT_VTHETA_MAX = VTHETA_MAX


def weights_to_param_vector(weights: dict | None = None) -> np.ndarray:
    """Pack runtime weights into p ∈ R^18."""
    w = weights or {}
    p = np.zeros(N_PARAMS)
    p[0:3] = w.get("Q_ec", DEFAULT_Q_EC)
    p[3:6] = w.get("Q_el", DEFAULT_Q_EL)
    p[6:9] = w.get("Q_q", DEFAULT_Q_Q)
    p[9:13] = w.get("U_mat", DEFAULT_U_MAT)
    p[13:16] = w.get("Q_omega", DEFAULT_Q_OMEGA)
    p[16] = w.get("Q_s", DEFAULT_Q_S)
    p[17] = w.get("vtheta_max", DEFAULT_VTHETA_MAX)
    return p


def apply_input_bounds(
    solver,
    n_prediction: int,
    s_max: float,
    vtheta_max: float | None = None,
) -> None:
    """Apply runtime numeric bounds via constraints_set.

    Bounds are numeric data, not part of the symbolic cost graph. Therefore,
    changing them does not require rebuilding the OCP as long as the structure
    of the constraints (same constrained inputs / same dimensions) stays fixed.
    """
    vtheta_max_eff = float(VTHETA_MAX if vtheta_max is None else vtheta_max)
    lbu = np.array([T_MIN, -TAUX_MAX, -TAUY_MAX, -TAUZ_MAX, ATHETA_MIN])
    ubu = np.array([T_MAX, TAUX_MAX, TAUY_MAX, TAUZ_MAX, ATHETA_MAX])
    lbx = np.array([0.0, VTHETA_MIN])
    ubx = np.array([s_max, vtheta_max_eff])
    for stage in range(n_prediction):
        solver.constraints_set(stage, "lbu", lbu)
        solver.constraints_set(stage, "ubu", ubu)
    for stage in range(1, n_prediction):
        solver.constraints_set(stage, "lbx", lbx)
        solver.constraints_set(stage, "ubx", ubx)


def create_mpcc_ocp_description(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Build MPCC OCP with runtime parameters for all numeric weights."""
    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc()
    # Use a dedicated model/build name for the p ∈ R^18 runtime-parameter OCP.
    # This avoids collisions with older cached builds that used a different
    # parameter dimension (for example the previous p ∈ R^1 variant).
    model.name = "Drone_ode_complete_runtime_bspline"
    ocp.model = model

    _OCP_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJ_DIR = os.path.dirname(_OCP_DIR)
    ocp.code_export_directory = os.path.join(_PROJ_DIR, "c_generated_code_mpcc_runtime_bspline")
    ocp.solver_options.N_horizon = N_horizon

    p_sym = MX.sym("p", N_PARAMS)
    model.p = p_sym

    Q_ec_diag = p_sym[0:3]
    Q_el_diag = p_sym[3:6]
    Q_q_diag = p_sym[6:9]
    U_mat_diag = p_sym[9:13]
    Q_omega_diag = p_sym[13:16]
    Q_s = p_sym[16]
    Q_ec = casadi_diag(Q_ec_diag)
    Q_el = casadi_diag(Q_el_diag)
    Q_q = casadi_diag(Q_q_diag)
    U_mat = casadi_diag(U_mat_diag)
    Q_omega = casadi_diag(Q_omega_diag)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    theta_state = model.x[13]
    v_theta_state = model.x[14]
    sd = gamma_pos(theta_state)
    tangent = gamma_vel(theta_state)
    qd = gamma_quat(theta_state)

    quat_err = quaternion_error(model.x[6:10], qd)
    log_q = log_cuaternion_casadi(quat_err)

    e_t = sd - model.x[0:3]
    e_lag = dot(tangent, e_t) * tangent
    P_ec = MX.eye(3) - tangent @ tangent.T
    ec = P_ec @ e_t

    omega = model.x[10:13]
    a_theta = model.u[4]

    control_cost = model.u[0:4].T @ U_mat @ model.u[0:4]
    attitude_cost = log_q.T @ Q_q @ log_q
    contour_cost = ec.T @ Q_ec @ ec
    lag_cost = e_lag.T @ Q_el @ e_lag
    omega_cost = omega.T @ Q_omega @ omega
    accel_progress_cost = MPCC_Q_ATHETA * a_theta**2
    # Classical MPCC-style progress incentive: maximize progress linearly.
    # Since acados minimizes the cost, this enters as a negative linear term.
    progress_cost = -Q_s * v_theta_state

    ocp.model.cost_expr_ext_cost = (
        contour_cost + lag_cost + attitude_cost
        + control_cost + omega_cost + accel_progress_cost + progress_cost
    )
    ocp.model.cost_expr_ext_cost_e = (
        contour_cost + lag_cost + attitude_cost + omega_cost
    )

    ocp.parameter_values = weights_to_param_vector()

    ocp.constraints.lbu = np.array([
        T_MIN, -TAUX_MAX, -TAUY_MAX, -TAUZ_MAX, ATHETA_MIN
    ])
    ocp.constraints.ubu = np.array([
        T_MAX, TAUX_MAX, TAUY_MAX, TAUZ_MAX, ATHETA_MAX
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.lbx = np.array([0.0, VTHETA_MIN])
    ocp.constraints.ubx = np.array([s_max, VTHETA_MAX])
    ocp.constraints.idxbx = np.array([13, 14])
    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = t_horizon / N_horizon
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = t_horizon
    ocp.solver_options.sim_method_num_stages = 4

    return ocp


def build_mpcc_solver(
    x0, N_prediction, t_prediction, s_max,
    gamma_pos, gamma_vel, gamma_quat,
    use_cython=True, force_rebuild=False,
):
    """Build or reuse the canonical MPCC solver."""
    ocp = create_mpcc_ocp_description(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_system_model_mpcc()

    solver_json = os.path.join(
        os.path.dirname(ocp.code_export_directory),
        "acados_ocp_" + model.name + ".json",
    )

    if use_cython:
        already_built = (
            os.path.isdir(ocp.code_export_directory)
            and os.path.isfile(solver_json)
        )
        if force_rebuild or not already_built:
            if os.path.isdir(ocp.code_export_directory):
                shutil.rmtree(ocp.code_export_directory)
            if os.path.isfile(solver_json):
                os.remove(solver_json)
            AcadosOcpSolver.generate(ocp, json_file=solver_json)
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        else:
            print(f"[SOLVER] Reusing cached build at {ocp.code_export_directory}")
        solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        solver = AcadosOcpSolver(
            ocp,
            json_file=solver_json,
            build=not (not force_rebuild and os.path.isfile(solver_json)),
            generate=not (not force_rebuild and os.path.isfile(solver_json)),
        )

    return solver, ocp, model, f_system
