"""
Canonical DQ-MPCC controller with runtime parameters p ∈ R^18.

This is the OCP that should be used by BOTH:
  - production simulation / execution
  - bilevel tuning

Compile ONCE, then update numeric values at runtime via:
    solver.set(stage, "p", p_vec)

Parameter vector:
    p[ 0: 3]  -> Q_phi     rotation-log error weights
    p[ 3: 6]  -> Q_ec      contouring error weights
    p[ 6: 9]  -> Q_el      lag error weights
    p[ 9:13]  -> U_mat     control effort weights [T, taux, tauy, tauz]
    p[13:16]  -> Q_omega   angular-velocity weights
    p[16]     -> Q_s       linear progress weight in -Q_s * v_theta
    p[17]     -> vtheta_max runtime upper bound helper (not part of the cost)
"""

import os
import sys
import shutil
import numpy as np
from casadi import MX, norm_2, diag as casadi_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from models.dq_quadrotor_mpcc_model import f_dq_system_model_mpcc
from utils.dq_casadi_utils import (
    dq_error_casadi,
    ln_dual_casadi,
    dq_from_pose_casadi,
    rotate_tangent_to_desired_frame,
    lag_contouring_decomposition,
)

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from experiment_config import (
    T_MAX, T_MIN,
    TAUX_MAX, TAUY_MAX, TAUZ_MAX,
    VTHETA_MIN, VTHETA_MAX, ATHETA_MIN, ATHETA_MAX,
    DQ_Q_PHI, DQ_Q_EC, DQ_Q_EL, DQ_U_MAT, DQ_Q_OMEGA, DQ_Q_S,
    DQ_Q_ATHETA,
)


N_PARAMS = 18

DEFAULT_Q_PHI = DQ_Q_PHI
DEFAULT_Q_EC = DQ_Q_EC
DEFAULT_Q_EL = DQ_Q_EL
DEFAULT_U_MAT = DQ_U_MAT
DEFAULT_Q_OMEGA = DQ_Q_OMEGA
DEFAULT_Q_S = DQ_Q_S
DEFAULT_VTHETA_MAX = VTHETA_MAX


def weights_to_param_vector(weights: dict | None = None) -> np.ndarray:
    """Pack runtime weights into p ∈ R^18."""
    w = weights or {}
    p = np.zeros(N_PARAMS)
    p[0:3] = w.get("Q_phi", DEFAULT_Q_PHI)
    p[3:6] = w.get("Q_ec", DEFAULT_Q_EC)
    p[6:9] = w.get("Q_el", DEFAULT_Q_EL)
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
    """Apply runtime numeric bounds via constraints_set."""
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


def create_dq_mpcc_ocp_description(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Build DQ-MPCC OCP with runtime parameters for all numeric weights."""
    ocp = AcadosOcp()
    model, _, _, _ = f_dq_system_model_mpcc()
    model.name = "DQ_Drone_MPCC_runtime_accel"
    ocp.model = model

    ocp_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(ocp_dir)
    ocp.code_export_directory = os.path.join(
        proj_dir, "c_generated_code_dq_runtime_accel"
    )
    ocp.solver_options.N_horizon = N_horizon

    p_sym = MX.sym("p", N_PARAMS)
    model.p = p_sym

    Q_phi = casadi_diag(p_sym[0:3])
    Q_ec = casadi_diag(p_sym[3:6])
    Q_el = casadi_diag(p_sym[6:9])
    U_mat = casadi_diag(p_sym[9:13])
    Q_omega = casadi_diag(p_sym[13:16])
    Q_s = p_sym[16]

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    theta_state = model.x[14]
    v_theta_state = model.x[15]
    sd = gamma_pos(theta_state)
    tangent = gamma_vel(theta_state)
    qd = gamma_quat(theta_state)

    dq_desired = dq_from_pose_casadi(qd, sd)
    dq_actual = model.x[0:8]
    dq_err = dq_error_casadi(dq_desired, dq_actual)
    log_err = ln_dual_casadi(dq_err)

    phi = log_err[0:3]
    rho = log_err[3:6]
    tangent_body = rotate_tangent_to_desired_frame(tangent, qd)
    rho_lag, rho_cont = lag_contouring_decomposition(rho, tangent_body)

    omega = model.x[8:11]
    a_theta = model.u[4]

    orientation_cost = phi.T @ Q_phi @ phi
    contouring_cost = rho_cont.T @ Q_ec @ rho_cont
    lag_cost = rho_lag.T @ Q_el @ rho_lag
    control_cost = model.u[0:4].T @ U_mat @ model.u[0:4]
    omega_cost = omega.T @ Q_omega @ omega
    accel_progress_cost = DQ_Q_ATHETA * a_theta**2
    progress_cost = -Q_s * v_theta_state

    ocp.model.cost_expr_ext_cost = (
        orientation_cost + contouring_cost + lag_cost
        + control_cost + omega_cost + accel_progress_cost + progress_cost
    )
    ocp.model.cost_expr_ext_cost_e = (
        orientation_cost + contouring_cost + lag_cost + omega_cost
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
    ocp.constraints.idxbx = np.array([14, 15])

    qr = model.x[0:4]
    ocp.model.con_h_expr = norm_2(qr)
    ocp.constraints.lh = np.array([0.99])
    ocp.constraints.uh = np.array([1.01])
    ns = 1
    ocp.constraints.lsh = np.zeros(ns)
    ocp.constraints.ush = np.zeros(ns)
    ocp.constraints.idxsh = np.array(range(ns))
    ocp.cost.zl = 100.0 * np.ones(ns)
    ocp.cost.zu = 100.0 * np.ones(ns)
    ocp.cost.Zl = 100.0 * np.ones(ns)
    ocp.cost.Zu = 100.0 * np.ones(ns)

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

    return ocp


def build_dq_mpcc_solver(
    x0, N_prediction, t_prediction, s_max,
    gamma_pos, gamma_vel, gamma_quat,
    use_cython=True, force_rebuild=False,
):
    """Build or reuse the canonical DQ-MPCC solver."""
    ocp = create_dq_mpcc_ocp_description(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_dq_system_model_mpcc()

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
