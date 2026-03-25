"""
Quadrotor 6-DOF dynamic model (quaternion-based, rate-controlled).

State  x ∈ ℝ¹³ = [p(3), v(3), q(4), ω(3)]
Input  u ∈ ℝ⁴  = [T, wx_cmd, wy_cmd, wz_cmd]

Angular velocity dynamics model MuJoCo's rate controller as a
first-order system:  ω̇ = (u_ω − ω) / τ_rc

Returns an AcadosModel together with CasADi functions for
simulation (f_system) and control analysis (f_x, g_x).
"""

from acados_template import AcadosModel
from casadi import (
    MX, vertcat, Function,
    substitute, jacobian,
)
import numpy as np

from utils.casadi_utils import quat_to_rot_casadi as QuatToRot, quat_kinematics_casadi as quat_p
from config.experiment_config import MASS_MUJOCO as MASS, G, TAU_RC


# ──────────────────────────────────────────────────────────────────────────────
#  Model builder
# ──────────────────────────────────────────────────────────────────────────────

def f_system_model():
    """Build the quadrotor CasADi / acados model (rate-control inputs).

    Returns
    -------
    model    : AcadosModel   – ready for OCP formulation
    f_system : casadi.Function(x, u) → ẋ  – continuous dynamics
    f_x      : casadi.Function(x) → f₀(x) – drift (u = 0)
    g_x      : casadi.Function(x) → ∂f/∂u – input matrix
    """
    model_name = 'Drone_ode_rate_ctrl'

    m = MASS
    e = MX([0, 0, 1])
    g = G
    tau_rc = TAU_RC

    # ── States ────────────────────────────────────────────────────────────
    p1 = MX.sym('p1');  p2 = MX.sym('p2');  p3 = MX.sym('p3')
    v1 = MX.sym('v1');  v2 = MX.sym('v2');  v3 = MX.sym('v3')
    q0 = MX.sym('q0');  q1 = MX.sym('q1');  q2 = MX.sym('q2');  q3 = MX.sym('q3')
    w1 = MX.sym('w1');  w2 = MX.sym('w2');  w3 = MX.sym('w3')

    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3)

    # ── Controls (thrust + desired angular velocity) ─────────────────────
    Tt     = MX.sym('Tt')
    w1_cmd = MX.sym('w1_cmd');  w2_cmd = MX.sym('w2_cmd');  w3_cmd = MX.sym('w3_cmd')

    u = vertcat(Tt, w1_cmd, w2_cmd, w3_cmd)

    # ── State derivative (implicit) ───────────────────────────────────────
    p1_p = MX.sym('p1_p');  p2_p = MX.sym('p2_p');  p3_p = MX.sym('p3_p')
    v1_p = MX.sym('v1_p');  v2_p = MX.sym('v2_p');  v3_p = MX.sym('v3_p')
    q0_p = MX.sym('q0_p');  q1_p = MX.sym('q1_p');  q2_p = MX.sym('q2_p');  q3_p = MX.sym('q3_p')
    w1_p = MX.sym('w1_p');  w2_p = MX.sym('w2_p');  w3_p = MX.sym('w3_p')

    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p,
                  q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p)

    # ── Reference parameter vector (for external cost) ───────────────────
    p1_d = MX.sym('p1_d');  p2_d = MX.sym('p2_d');  p3_d = MX.sym('p3_d')
    v1_d = MX.sym('v1_d');  v2_d = MX.sym('v2_d');  v3_d = MX.sym('v3_d')
    q0_d = MX.sym('q0_d');  q1_d = MX.sym('q1_d');  q2_d = MX.sym('q2_d');  q3_d = MX.sym('q3_d')
    w1_d = MX.sym('w1_d');  w2_d = MX.sym('w2_d');  w3_d = MX.sym('w3_d')
    T_d  = MX.sym('T_d');   w1_cd = MX.sym('w1_cd');  w2_cd = MX.sym('w2_cd');  w3_cd = MX.sym('w3_cd')

    p_param = vertcat(p1_d, p2_d, p3_d, v1_d, v2_d, v3_d,
                      q0_d, q1_d, q2_d, q3_d, w1_d, w2_d, w3_d,
                      T_d, w1_cd, w2_cd, w3_cd)

    # ── Dynamics ──────────────────────────────────────────────────────────
    quat  = vertcat(q0, q1, q2, q3)
    w     = vertcat(w1, w2, w3)
    w_cmd = vertcat(w1_cmd, w2_cmd, w3_cmd)
    Rot   = QuatToRot(quat)

    u1 = vertcat(0, 0, Tt)

    p_p   = vertcat(v1, v2, v3)
    v_p   = -e * g + (Rot @ u1) / m
    q_dot = quat_p(quat, w)
    w_p   = (w_cmd - w) / tau_rc             # first-order rate controller

    f_expl = vertcat(p_p, v_p, q_dot, w_p)

    # ── Drift & input-matrix functions (for control analysis) ──────────
    u_zero = MX.zeros(u.shape[0], 1)
    f0_expr = substitute(f_expl, u, u_zero)

    f_x = Function('f0', [x], [f0_expr])
    g_x = Function('g',  [x], [jacobian(f_expl, u)])

    f_system = Function('system', [x, u], [f_expl])

    # ── Acados model ─────────────────────────────────────────────────────
    f_impl = x_p - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x    = x
    model.xdot = x_p
    model.u    = u
    model.name = model_name
    model.p    = p_param

    return model, f_system, f_x, g_x
