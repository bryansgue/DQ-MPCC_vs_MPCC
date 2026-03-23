"""
Quadrotor 6-DOF model augmented with arc-length state θ for MPCC.

State  x ∈ ℝ¹⁴ = [p(3), v(3), q(4), ω(3), θ]
Input  u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

The extra dynamics are simply  θ̇ = v_θ.

Returns an AcadosModel together with CasADi functions for
simulation (f_system) and analysis (f_x, g_x).
"""

from acados_template import AcadosModel
from casadi import (
    MX, vertcat, horzcat, Function,
    inv, cross, substitute, jacobian,
)
import numpy as np

from utils.casadi_utils import (
    quat_to_rot_casadi  as QuatToRot,
    quat_kinematics_casadi as quat_p,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Physical parameters — imported from experiment_config.py
# ──────────────────────────────────────────────────────────────────────────────
import os as _os, sys as _sys
_WORKSPACE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
if _WORKSPACE_ROOT not in _sys.path:
    _sys.path.insert(0, _WORKSPACE_ROOT)
from experiment_config import MASS, G, JXX, JYY, JZZ


def f_system_model_mpcc():
    """Build the augmented quadrotor + θ model for strict MPCC.

    Returns
    -------
    model    : AcadosModel           – 14-state / 5-control model (no params).
    f_system : casadi.Function(x, u) → ẋ  – continuous dynamics.
    f_x      : casadi.Function(x)    → f₀  – drift (u = 0).
    g_x      : casadi.Function(x)    → ∂f/∂u  – input matrix.
    """
    model_name = 'Drone_ode_complete'
    m  = MASS
    e3 = MX([0, 0, 1])
    g  = G

    # ── States (14) ──────────────────────────────────────────────────────
    p1 = MX.sym('p1');  p2 = MX.sym('p2');  p3 = MX.sym('p3')
    v1 = MX.sym('v1');  v2 = MX.sym('v2');  v3 = MX.sym('v3')
    q0 = MX.sym('q0');  q1 = MX.sym('q1');  q2 = MX.sym('q2');  q3 = MX.sym('q3')
    w1 = MX.sym('w1');  w2 = MX.sym('w2');  w3 = MX.sym('w3')
    theta = MX.sym('theta')
    x = vertcat(p1, p2, p3, v1, v2, v3,
                q0, q1, q2, q3, w1, w2, w3, theta)

    # ── Controls (5) ─────────────────────────────────────────────────────
    Tt      = MX.sym('Tt')
    tau1    = MX.sym('tau1');  tau2 = MX.sym('tau2');  tau3 = MX.sym('tau3')
    v_theta = MX.sym('v_theta')
    u = vertcat(Tt, tau1, tau2, tau3, v_theta)

    # ── State derivatives (symbolic, for implicit form) ──────────────────
    p1_p = MX.sym('p1_p');  p2_p = MX.sym('p2_p');  p3_p = MX.sym('p3_p')
    v1_p = MX.sym('v1_p');  v2_p = MX.sym('v2_p');  v3_p = MX.sym('v3_p')
    q0_p = MX.sym('q0_p');  q1_p = MX.sym('q1_p')
    q2_p = MX.sym('q2_p');  q3_p = MX.sym('q3_p')
    w1_p = MX.sym('w1_p');  w2_p = MX.sym('w2_p');  w3_p = MX.sym('w3_p')
    theta_p = MX.sym('theta_p')
    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p,
                  q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p, theta_p)

    # ── Dynamics ─────────────────────────────────────────────────────────
    quat = vertcat(q0, q1, q2, q3)
    w    = vertcat(w1, w2, w3)
    Rot  = QuatToRot(quat)

    I_inertia = vertcat(
        horzcat(MX(JXX), MX(0),   MX(0)),
        horzcat(MX(0),   MX(JYY), MX(0)),
        horzcat(MX(0),   MX(0),   MX(JZZ)),
    )

    dp     = vertcat(v1, v2, v3)
    dv     = -e3 * g + (Rot @ vertcat(MX(0), MX(0), Tt)) / m
    dq     = quat_p(quat, w)
    dw     = inv(I_inertia) @ (vertcat(tau1, tau2, tau3) - cross(w, I_inertia @ w))
    dtheta = v_theta                            # θ̇ = v_θ

    f_expl = vertcat(dp, dv, dq, dw, dtheta)

    # ── Auxiliary CasADi functions ───────────────────────────────────────
    u_zero   = MX.zeros(u.size1(), 1)
    f_x_func = Function('f0',     [x],    [substitute(f_expl, u, u_zero)])
    g_x_func = Function('g',      [x],    [jacobian(f_expl, u)])
    f_system = Function('system', [x, u], [f_expl])

    # ── AcadosModel (no runtime parameters — baseline MPCC) ────────────
    model              = AcadosModel()
    model.f_impl_expr  = x_p - f_expl
    model.f_expl_expr  = f_expl
    model.x            = x
    model.xdot         = x_p
    model.u            = u
    model.name         = model_name

    return model, f_system, f_x_func, g_x_func
