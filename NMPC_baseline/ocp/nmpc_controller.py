"""
NMPC formulation for the quadrotor UAV.

Builds an AcadosOcp with:
  • External cost  (position + quaternion-log + control effort)
  • Box constraints on controls  (thrust & torques)
  • SQP-RTI solver  with FULL_CONDENSING_HPIPM
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_model import f_system_model
from utils.casadi_utils import quat_error_casadi as quaternion_error, quat_log_casadi as log_cuaternion_casadi


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_Q = [25, 25, 30]              # Position  [x, y, z]
DEFAULT_K = [12, 12, 12]              # Orientation (SO(3) log)
DEFAULT_R = [1.0, 800, 800, 800]      # Control [T, τx, τy, τz]

# ──────────────────────────────────────────────────────────────────────────────
#  Default actuation limits
# ──────────────────────────────────────────────────────────────────────────────

G = 9.81
DEFAULT_T_MAX   = 3 * G     # [N]
DEFAULT_T_MIN   = 0.0       # [N]
DEFAULT_TAU_MAX = 0.05      # [N·m]


# ──────────────────────────────────────────────────────────────────────────────
#  OCP builder
# ──────────────────────────────────────────────────────────────────────────────

def create_ocp_solver_description(
    x0,
    N_horizon,
    t_horizon,
) -> AcadosOcp:
    """Create the acados OCP description.

    Parameters
    ----------
    x0        : ndarray (13,)  – initial state
    N_horizon : int            – prediction steps
    t_horizon : float          – prediction time [s]

    Returns
    -------
    ocp : AcadosOcp
    """
    ocp = AcadosOcp()

    model, f_system, f_x, g_x = f_system_model()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    np_param = model.p.size()[0]

    # ── Horizon ───────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N_horizon

    # ── Cost matrices ─────────────────────────────────────────────────────
    Q_mat = np.diag(DEFAULT_Q)
    K_mat = np.diag(DEFAULT_K)
    R_mat = np.diag(DEFAULT_R)

    # ── External cost ─────────────────────────────────────────────────────
    ocp.parameter_values = np.zeros(np_param)
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = model.p[0:3] - model.x[0:3]
    quat_error = quaternion_error(model.x[6:10], model.p[6:10])
    log_q      = log_cuaternion_casadi(quat_error)

    # Stage cost
    ocp.model.cost_expr_ext_cost = (
        error_pose.T @ Q_mat @ error_pose
        + model.u.T @ R_mat @ model.u
        + log_q.T @ K_mat @ log_q
    )
    # Terminal cost (no control term)
    ocp.model.cost_expr_ext_cost_e = (
        error_pose.T @ Q_mat @ error_pose
        + log_q.T @ K_mat @ log_q
    )

    # ── Control constraints (box) ─────────────────────────────────────────
    ocp.constraints.lbu = np.array([DEFAULT_T_MIN, -DEFAULT_TAU_MAX, -DEFAULT_TAU_MAX, -DEFAULT_TAU_MAX])
    ocp.constraints.ubu = np.array([DEFAULT_T_MAX,  DEFAULT_TAU_MAX,  DEFAULT_TAU_MAX,  DEFAULT_TAU_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ── Initial state constraint ──────────────────────────────────────────
    ocp.constraints.x0 = x0

    # ── Solver options ────────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = t_horizon

    return ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Solver factory
# ──────────────────────────────────────────────────────────────────────────────

def build_ocp_solver(x0, N_prediction, t_prediction, use_cython=True):
    """Create, code-generate, compile and return an acados OCP solver.

    Parameters
    ----------
    x0            : ndarray (13,)
    N_prediction  : int
    t_prediction  : float  [s]
    use_cython    : bool   – compile Cython solver (faster calls)

    Returns
    -------
    acados_ocp_solver : AcadosOcpSolver
    ocp               : AcadosOcp
    model             : AcadosModel
    f_system          : casadi.Function
    """
    model, f_system, f_x, g_x = f_system_model()
    ocp = create_ocp_solver_description(x0, N_prediction, t_prediction)

    solver_json = 'acados_ocp_' + model.name + '.json'

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, ocp, model, f_system
