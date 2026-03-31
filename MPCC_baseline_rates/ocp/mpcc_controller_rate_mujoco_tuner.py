"""MPCC rate OCP with TUNABLE gains — MuJoCo SiL version.

All cost weights are declared as acados runtime parameters p ∈ ℝ¹⁵,
so gains can be changed via solver.set(stage, "p", p_vec) without
recompiling the C code.  This enables fast bilevel optimisation with
MuJoCo SiL.

Q_omega (angular-velocity state cost) is intentionally REMOVED from the
tunable set.  In a rate-controlled drone, ω ≈ ω_cmd in steady state, so
penalising both the commanded rates (U_mat[1:4]) and the actual angular
velocity (Q_omega) is largely redundant.  Dropping Q_omega reduces the
Optuna search space from 17 → 14 parameters, improving convergence.

Parameter vector layout:
    p[ 0: 3]  → Q_ec        contouring error   [x, y, z]
    p[ 3: 6]  → Q_el        lag error           [x, y, z]
    p[ 6: 9]  → Q_q         quaternion log map  [x, y, z]
    p[ 9:13]  → U_mat       control effort      [T, wx, wy, wz]
    p[13]     → Q_s         progress speed
    p[14]     → vtheta_max  max arc-length velocity (also used for braking)

Original (fixed-weight) file: ocp/mpcc_controller_rate_mujoco.py  (UNTOUCHED)
Generated code exported to:   c_generated_code_mpcc_rate_mujoco_tuner/
"""

import os
import shutil
import sys

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import MX, dot, diag as casadi_diag

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_SHARED_MPCC_ROOT = os.path.join(_WORKSPACE_ROOT, "MPCC_baseline")
for _path in (_WORKSPACE_ROOT, _SHARED_MPCC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi   as quat_log,
)
from MPCC_baseline_rates.config.experiment_config import (
    MASS_MUJOCO, G,
    MPCC_Q_EC, MPCC_Q_EL, MPCC_Q_Q, MPCC_RATE_U_MAT, MPCC_Q_S,
    T_MAX, T_MIN, W_MAX, VTHETA_MIN, VTHETA_MAX,
)
from MPCC_baseline_rates.models.quadrotor_mpcc_rate_model_mujoco import (
    f_system_model_mpcc_rate_mujoco,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter vector helpers
# ──────────────────────────────────────────────────────────────────────────────

N_PARAMS = 15   # 3+3+3+4+1+1  (Q_omega removed)

DEFAULT_Q_EC       = list(MPCC_Q_EC)
DEFAULT_Q_EL       = list(MPCC_Q_EL)
DEFAULT_Q_Q        = list(MPCC_Q_Q)
DEFAULT_U_MAT      = list(MPCC_RATE_U_MAT)
DEFAULT_Q_S        = float(MPCC_Q_S)
DEFAULT_VTHETA_MAX = float(VTHETA_MAX)
DEFAULT_T_MAX      = float(T_MAX)
DEFAULT_W_MAX      = float(W_MAX)
DEFAULT_VTHETA_MIN = float(VTHETA_MIN)
DEFAULT_T_MIN      = float(T_MIN)


def weights_to_param_vector(weights: dict | None = None) -> np.ndarray:
    """Convert a weights dict to the p ∈ ℝ¹⁵ parameter vector.

    Parameters
    ----------
    weights : dict or None
        Optional overrides.  Recognised keys:
            'Q_ec', 'Q_el', 'Q_q', 'U_mat', 'Q_s', 'vtheta_max'

    Returns
    -------
    np.ndarray of shape (15,)
    """
    w = weights or {}
    p = np.zeros(N_PARAMS)
    p[0:3]  = w.get('Q_ec',       DEFAULT_Q_EC)
    p[3:6]  = w.get('Q_el',       DEFAULT_Q_EL)
    p[6:9]  = w.get('Q_q',        DEFAULT_Q_Q)
    p[9:13] = w.get('U_mat',      DEFAULT_U_MAT)
    p[13]   = w.get('Q_s',        DEFAULT_Q_S)
    p[14]   = w.get('vtheta_max', DEFAULT_VTHETA_MAX)
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder
# ══════════════════════════════════════════════════════════════════════════════

def create_mpcc_rate_ocp_description_mujoco_tuner(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Build the tunable MPCC rate OCP for MuJoCo SiL (symbolic weights)."""
    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc_rate_mujoco()
    model.name = "Drone_ode_mpcc_rate_mujoco_tuner"   # unique → no C-code conflict

    # ── Declare all weights as symbolic runtime parameters ────────────────
    p_sym = MX.sym('p', N_PARAMS)
    model.p = p_sym
    ocp.model = model

    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ocp.code_export_directory = os.path.join(
        proj_dir, "c_generated_code_mpcc_rate_mujoco_tuner"
    )
    ocp.solver_options.N_horizon = N_horizon

    # ── Extract symbolic weight matrices from p ─────────────────────────
    Q_ec_diag  = p_sym[0:3]
    Q_el_diag  = p_sym[3:6]
    Q_q_diag   = p_sym[6:9]
    U_mat_diag = p_sym[9:13]
    Q_s_sym    = p_sym[13]
    p_vtheta_max = p_sym[14]

    Q_ec    = casadi_diag(Q_ec_diag)
    Q_el    = casadi_diag(Q_el_diag)
    Q_q_mat = casadi_diag(Q_q_diag)
    U_mat   = casadi_diag(U_mat_diag)
    T_hover = MASS_MUJOCO * G

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── Path reference from θ ─────────────────────────────────────────────
    theta_state = model.x[13]
    sd      = gamma_pos(theta_state)
    tangent = gamma_vel(theta_state)
    qd      = gamma_quat(theta_state)

    # ── Errors ────────────────────────────────────────────────────────────
    quat_err = quaternion_error(model.x[6:10], qd)
    log_q    = quat_log(quat_err)

    e_t   = sd - model.x[0:3]
    e_lag = dot(tangent, e_t) * tangent
    ec    = (MX.eye(3) - tangent @ tangent.T) @ e_t

    omega        = model.x[10:13]
    v_theta      = model.u[4]
    thrust_error = model.u[0] - T_hover
    rates_cmd    = model.u[1:4]

    # ── Cost terms ────────────────────────────────────────────────────────
    contour_cost  = ec.T @ Q_ec @ ec
    lag_cost      = e_lag.T @ Q_el @ e_lag
    attitude_cost = log_q.T @ Q_q_mat @ log_q
    control_cost  = (U_mat[0, 0] * thrust_error**2
                     + rates_cmd.T @ U_mat[1:4, 1:4] @ rates_cmd)
    progress_cost = Q_s_sym * (p_vtheta_max - v_theta) ** 2

    ocp.model.cost_expr_ext_cost = (
        contour_cost + lag_cost + attitude_cost + control_cost + progress_cost
    )
    ocp.model.cost_expr_ext_cost_e = (
        contour_cost + lag_cost + attitude_cost
    )

    # ── Default parameter values ──────────────────────────────────────────
    ocp.parameter_values = weights_to_param_vector()

    # ── Constraints ───────────────────────────────────────────────────────
    ocp.constraints.lbu    = np.array([T_MIN, -W_MAX, -W_MAX, -W_MAX, VTHETA_MIN])
    ocp.constraints.ubu    = np.array([T_MAX,  W_MAX,  W_MAX,  W_MAX, VTHETA_MAX])
    ocp.constraints.idxbu  = np.array([0, 1, 2, 3, 4])

    ocp.constraints.lbx    = np.array([0.0])
    ocp.constraints.ubx    = np.array([s_max])
    ocp.constraints.idxbx  = np.array([13])
    ocp.constraints.x0     = x0

    # ── Solver options ────────────────────────────────────────────────────
    ocp.solver_options.qp_solver         = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N  = max(1, N_horizon // 4)
    ocp.solver_options.hessian_approx    = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type   = "ERK"
    ocp.solver_options.nlp_solver_type   = "SQP_RTI"
    ocp.solver_options.Tsim              = t_horizon / N_horizon
    ocp.solver_options.tol               = 1e-4
    ocp.solver_options.tf                = t_horizon
    return ocp


# ══════════════════════════════════════════════════════════════════════════════
#  Solver factory
# ══════════════════════════════════════════════════════════════════════════════

def build_mpcc_rate_solver_mujoco_tuner(
    x0, N_prediction, t_prediction, s_max,
    gamma_pos, gamma_vel, gamma_quat,
    use_cython: bool = False,
):
    """Build and return the tunable MPCC rate solver (MuJoCo SiL).

    Compiles ONCE.  After that, update gains via:
        p_vec = weights_to_param_vector(my_weights)
        for stage in range(N+1):
            solver.set(stage, "p", p_vec)
    """
    ocp = create_mpcc_rate_ocp_description_mujoco_tuner(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
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
