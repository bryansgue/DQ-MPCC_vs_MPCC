"""Strict MPCC OCP for the rate-control MuJoCo SiL baseline.

Same cost formulation as mpcc_controller_rate.py but uses the MuJoCo model
(MASS_MUJOCO) and writes generated code to a separate directory so the MiL
solver is never overwritten.
"""

import os
import shutil
import sys

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, dot

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SHARED_MPCC_ROOT = os.path.join(_WORKSPACE_ROOT, "MPCC_baseline")
for _path in (_WORKSPACE_ROOT, _SHARED_MPCC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi as quat_log,
)
from MPCC_baseline_rates.config.experiment_config import (
    MASS_MUJOCO,
    MPCC_Q_EC,
    MPCC_Q_EL,
    MPCC_Q_OMEGA,
    MPCC_Q_Q,
    MPCC_Q_S,
    MPCC_RATE_U_MAT,
    T_MAX,
    T_MIN,
    VTHETA_MAX,
    VTHETA_MIN,
    W_MAX,
    G,
)
from MPCC_baseline_rates.models.quadrotor_mpcc_rate_model_mujoco import (
    f_system_model_mpcc_rate_mujoco,
)


def create_mpcc_rate_ocp_description_mujoco(
    x0, N_horizon, t_horizon, s_max, gamma_pos, gamma_vel, gamma_quat
) -> AcadosOcp:
    """Create the strict MPCC OCP for body-rate commands (MuJoCo mass)."""
    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc_rate_mujoco()
    ocp.model = model

    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ocp.code_export_directory = os.path.join(proj_dir, "c_generated_code_mpcc_rate_mujoco")
    ocp.solver_options.N_horizon = N_horizon

    p_vtheta_max = MX.sym("p_vtheta_max")
    model.p = p_vtheta_max
    ocp.parameter_values = np.array([VTHETA_MAX])

    Q_q = np.diag(MPCC_Q_Q)
    Q_el = np.diag(MPCC_Q_EL)
    Q_ec = np.diag(MPCC_Q_EC)
    U_mat = np.diag(MPCC_RATE_U_MAT)
    Q_omega = np.diag(MPCC_Q_OMEGA)
    T_hover = MASS_MUJOCO * G

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    theta_state = model.x[13]
    sd = gamma_pos(theta_state)
    tangent = gamma_vel(theta_state)
    qd = gamma_quat(theta_state)

    quat_err = quaternion_error(model.x[6:10], qd)
    log_q = quat_log(quat_err)

    e_t = sd - model.x[0:3]
    e_lag = dot(tangent, e_t) * tangent
    ec = (MX.eye(3) - tangent @ tangent.T) @ e_t
    omega = model.x[10:13]
    v_theta = model.u[4]

    thrust_error = model.u[0] - T_hover
    rates_cmd = model.u[1:4]
    control_cost = U_mat[0, 0] * thrust_error**2 + rates_cmd.T @ U_mat[1:4, 1:4] @ rates_cmd
    contour_cost = ec.T @ Q_ec @ ec
    lag_cost = e_lag.T @ Q_el @ e_lag
    attitude_cost = log_q.T @ Q_q @ log_q
    omega_cost = omega.T @ Q_omega @ omega
    progress_cost = MPCC_Q_S * (p_vtheta_max - v_theta) ** 2

    ocp.model.cost_expr_ext_cost = (
        contour_cost + lag_cost + attitude_cost + control_cost + omega_cost + progress_cost
    )
    ocp.model.cost_expr_ext_cost_e = contour_cost + lag_cost + attitude_cost + omega_cost

    ocp.constraints.lbu = np.array([T_MIN, -W_MAX, -W_MAX, -W_MAX, VTHETA_MIN])
    ocp.constraints.ubu = np.array([T_MAX, W_MAX, W_MAX, W_MAX, VTHETA_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([s_max])
    ocp.constraints.idxbx = np.array([13])
    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = max(1, N_horizon // 4)
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.Tsim = t_horizon / N_horizon
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = t_horizon
    return ocp


def build_mpcc_rate_solver_mujoco(
    x0, N_prediction, t_prediction, s_max, gamma_pos, gamma_vel, gamma_quat, use_cython=True
):
    """Build and return the acados MPCC rate solver (MuJoCo version)."""
    ocp = create_mpcc_rate_ocp_description_mujoco(
        x0, N_prediction, t_prediction, s_max, gamma_pos, gamma_vel, gamma_quat
    )
    model = ocp.model
    _, f_system, _, _ = f_system_model_mpcc_rate_mujoco()

    solver_json = os.path.join(
        os.path.dirname(ocp.code_export_directory),
        f"acados_ocp_{model.name}.json",
    )

    if use_cython:
        if os.path.isdir(ocp.code_export_directory):
            shutil.rmtree(ocp.code_export_directory)
        if os.path.isfile(solver_json):
            os.remove(solver_json)
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        solver = AcadosOcpSolver(ocp, json_file=solver_json, build=True, generate=True)

    return solver, ocp, model, f_system
