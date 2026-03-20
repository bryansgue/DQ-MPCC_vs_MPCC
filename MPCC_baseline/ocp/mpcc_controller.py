"""
MPCC (Model Predictive Contouring Control) – strict formulation with
symbolic trajectory interp    # Lag error: projection of e_t onto the tangent direction → 3-vector
    e_lag    = dot(tangent, e_t) * tangent      # (t·e_t) · t  ∈ ℝ³
    # Contouring error (component orthogonal to tangent)
    P_ec = MX.eye(3) - tangent @ tangent.T
    ec   = P_ec @ e_t                           # 3-vector ⊥ tangenton.

Key MPCC features:
  • θ (arc-length progress) is an optimisation STATE   (x[13])
  • v_θ (progress velocity) is an optimisation CONTROL (u[4])
  • θ̇ = v_θ  (augmented dynamics)
  • The reference is a **symbolic CasADi function of θ** so the solver
    can differentiate through:
        v_θ  →  θ̇=v_θ  →  θ  →  reference(θ)  →  error  →  cost
    This enables true optimisation of the progress speed v_θ.

Augmented vectors
-----------------
  State   x ∈ ℝ¹⁴ = [p(3), v(3), q(4), ω(3), θ]
  Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

No runtime parameters — the reference comes from the symbolic θ interpolation.
"""

import os
import sys
import shutil
import numpy as np
from casadi import MX, dot, vertcat
from acados_template import AcadosOcp, AcadosOcpSolver

from models.quadrotor_mpcc_model import f_system_model_mpcc, MASS
from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi   as log_cuaternion_casadi,
)

# ── Shared control limits from tuning_config.py ──────────────────────────────
# IMPORTANT: must match the limits used during tuning (mpcc_controller_tuner.py)
#            so that the optimised weights are valid for this OCP.
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
#  Default cost weights  (best result from bilevel tuning — J=32.71)
#  Tuned with τ_max=0.5 N·m  (see tuning_config.py → TAUX/Y/Z_MAX)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_Q_EC    = [13.57738255445416, 1.8890124348260142, 1.6715023759813776]
DEFAULT_Q_EL    = [12.6214175909978, 28.348657413534657, 1.1913008526216196]
DEFAULT_Q_Q     = [0.12810529227483344, 0.13436063883116614, 4.087587428847498]
DEFAULT_U_MAT   = [0.010021926260341494, 87.89317517538485, 11.75052384151524, 113.50974128305475]
DEFAULT_Q_OMEGA = [0.017449843366922332, 0.03216712507119171, 0.04936032393158435]
DEFAULT_Q_S     = 0.5342721333724649

# ── Control limits — read from tuning_config so tuner & production are identical
DEFAULT_T_MAX      = _CFG_T_MAX
DEFAULT_T_MIN      = _CFG_T_MIN
DEFAULT_TAUX_MAX   = _CFG_TAUX_MAX
DEFAULT_TAUY_MAX   = _CFG_TAUY_MAX
DEFAULT_TAUZ_MAX   = _CFG_TAUZ_MAX
DEFAULT_VTHETA_MIN = _CFG_VTHETA_MIN
DEFAULT_VTHETA_MAX = _CFG_VTHETA_MAX

# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder
# ══════════════════════════════════════════════════════════════════════════════

def create_mpcc_ocp_description(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Create strict MPCC OCP with symbolic trajectory interpolation.

    The reference sd(θ), tangent(θ), qd(θ) are CasADi functions of the
    state θ=x[13].  The solver differentiates through θ→reference→error,
    which makes v_θ truly optimisable.
    """
    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc()
    ocp.model = model

    # Absolute path so the compiled solver always lives next to the ocp/ folder,
    # regardless of what cwd is when the script runs.
    _OCP_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJ_DIR = os.path.dirname(_OCP_DIR)
    ocp.code_export_directory = os.path.join(_PROJ_DIR, 'c_generated_code_mpcc')

    ocp.solver_options.N_horizon = N_horizon

    # ── Runtime parameter: v_theta_max ───────────────────────────────────
    p_vtheta_max = MX.sym('p_vtheta_max')
    model.p = p_vtheta_max
    ocp.parameter_values = np.array([DEFAULT_VTHETA_MAX])

    # Weights — one independent scalar per axis
    Q_q     = np.diag(DEFAULT_Q_Q)
    Q_el    = np.diag(DEFAULT_Q_EL)           # 3×3 diagonal: lag error per axis
    Q_ec    = np.diag(DEFAULT_Q_EC)
    U_mat   = np.diag(DEFAULT_U_MAT)
    Q_omega = np.diag(DEFAULT_Q_OMEGA)
    Q_s     = DEFAULT_Q_S

    T_hover = MASS * 9.81                     # hover thrust [N] ≈ 9.81

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── Symbolic reference from θ ────────────────────────────────────────
    theta_state = model.x[13]
    sd      = gamma_pos(theta_state)      # [3]  reference position
    tangent = gamma_vel(theta_state)      # [3]  unit tangent
    qd      = gamma_quat(theta_state)     # [4]  desired quaternion

    # ── Errors ───────────────────────────────────────────────────────────
    quat_err = quaternion_error(model.x[6:10], qd)
    log_q    = log_cuaternion_casadi(quat_err)

    e_t = sd - model.x[0:3]            # 3-vector: reference minus position

    # Lag error: projection of e_t onto the tangent direction → 3-vector
    e_lag    = dot(tangent, e_t) * tangent      # (t·e_t)·t  ∈ ℝ³  ∥ tangent
    # Contouring error (component orthogonal to tangent)
    P_ec = MX.eye(3) - tangent @ tangent.T
    ec   = P_ec @ e_t                           # 3-vector ⊥ tangent

    omega   = model.x[10:13]
    v_theta = model.u[4]

    # ── Control cost (absolute, same as example) ────────────────────────
    # Penalise absolute control values. With Q_T small (0.1) the hover
    # thrust cost ≈ 0.1·9.81² ≈ 9.6 is negligible vs tracking errors.
    # ── Cost terms ───────────────────────────────────────────────────────
    control_cost      = model.u[0:4].T @ U_mat @ model.u[0:4]
    actitud_cost      = log_q.T @ Q_q @ log_q
    error_contorno    = ec.T @ Q_ec @ ec
    error_lag         = e_lag.T @ Q_el @ e_lag  # e_lag' Q_el e_lag  ∈ ℝ
    omega_cost        = omega.T @ Q_omega @ omega
    arc_speed_penalty = Q_s * (p_vtheta_max - v_theta)**2

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

def build_mpcc_solver(x0, N_prediction, t_prediction, s_max,
                      gamma_pos, gamma_vel, gamma_quat,
                      use_cython=True):
    """Create, code-generate, compile and return the MPCC solver."""
    ocp = create_mpcc_ocp_description(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_system_model_mpcc()

    solver_json = os.path.join(
        os.path.dirname(ocp.code_export_directory),
        'acados_ocp_' + model.name + '.json',
    )

    if use_cython:
        # Cython path has no build/generate flags — must remove stale code
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
