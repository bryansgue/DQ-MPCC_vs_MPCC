"""
Quadrotor 6-DOF model (dual-quaternion) augmented with arc-length θ for MPCC.

State  x ∈ ℝ¹⁵ = [dq(8), twist(6), θ]
    dq    = [qr_w, qr_x, qr_y, qr_z, qd_w, qd_x, qd_y, qd_z]
    twist = [ωx, ωy, ωz, vx, vy, vz]   (body-frame angular + body-frame linear)
    θ     = arc-length progress along the reference path

Input  u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

The extra dynamics are simply  θ̇ = v_θ.

Returns an AcadosModel together with a CasADi function for simulation.
No runtime parameters — the reference comes from symbolic θ interpolation.
"""

from acados_template import AcadosModel
from casadi import MX, vertcat, horzcat, Function, inv, substitute, jacobian, DM
import numpy as np

from utils.dq_casadi_utils import (
    dq_kinematics_casadi,
    dq_acceleration_casadi,
    dq_get_quaternion_casadi,
    rotation_expr,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Physical parameters
# ──────────────────────────────────────────────────────────────────────────────
MASS = 1.0        # [kg]
G    = 9.81       # [m/s²]
JXX  = 0.00305587 # [kg·m²]
JYY  = 0.00159695
JZZ  = 0.00159687

# System parameter vector  [m, Jxx, Jyy, Jzz, g]
L_PARAMS = [MASS, JXX, JYY, JZZ, G]


def f_dq_system_model_mpcc():
    """Build the DQ quadrotor + θ model for strict DQ-MPCC.

    Returns
    -------
    model    : AcadosModel           – 15-state / 5-control model (no params).
    f_system : casadi.Function(x, u) → ẋ  – continuous dynamics.
    f_x      : casadi.Function(x)    → f₀  – drift (u = 0).
    g_x      : casadi.Function(x)    → ∂f/∂u  – input matrix.
    """
    import casadi as ca
    model_name = 'DQ_Drone_MPCC'

    # ── States (15) ──────────────────────────────────────────────────────
    # Dual quaternion (8)
    qr_w = MX.sym('qr_w'); qr_x = MX.sym('qr_x')
    qr_y = MX.sym('qr_y'); qr_z = MX.sym('qr_z')
    qd_w = MX.sym('qd_w'); qd_x = MX.sym('qd_x')
    qd_y = MX.sym('qd_y'); qd_z = MX.sym('qd_z')

    # Body twist (6): [ωx, ωy, ωz, vx, vy, vz]
    wx = MX.sym('wx'); wy = MX.sym('wy'); wz = MX.sym('wz')
    vx = MX.sym('vx'); vy = MX.sym('vy'); vz = MX.sym('vz')

    # Arc-length progress
    theta = MX.sym('theta')

    x = vertcat(qr_w, qr_x, qr_y, qr_z,
                qd_w, qd_x, qd_y, qd_z,
                wx, wy, wz, vx, vy, vz,
                theta)

    dq    = x[0:8]
    twist = x[8:14]

    # ── Controls (5) ─────────────────────────────────────────────────────
    Tt      = MX.sym('Tt')
    tau1    = MX.sym('tau1'); tau2 = MX.sym('tau2'); tau3 = MX.sym('tau3')
    v_theta = MX.sym('v_theta')
    u = vertcat(Tt, tau1, tau2, tau3, v_theta)

    # ── State derivatives (symbolic, for implicit form) ──────────────────
    qr_w_d = MX.sym('qr_w_d'); qr_x_d = MX.sym('qr_x_d')
    qr_y_d = MX.sym('qr_y_d'); qr_z_d = MX.sym('qr_z_d')
    qd_w_d = MX.sym('qd_w_d'); qd_x_d = MX.sym('qd_x_d')
    qd_y_d = MX.sym('qd_y_d'); qd_z_d = MX.sym('qd_z_d')
    wx_d = MX.sym('wx_d'); wy_d = MX.sym('wy_d'); wz_d = MX.sym('wz_d')
    vx_d = MX.sym('vx_d'); vy_d = MX.sym('vy_d'); vz_d = MX.sym('vz_d')
    theta_d = MX.sym('theta_d')

    x_dot = vertcat(qr_w_d, qr_x_d, qr_y_d, qr_z_d,
                    qd_w_d, qd_x_d, qd_y_d, qd_z_d,
                    wx_d, wy_d, wz_d, vx_d, vy_d, vz_d,
                    theta_d)

    # ── Dynamics ─────────────────────────────────────────────────────────
    dq_dot    = dq_kinematics_casadi(dq, twist)
    twist_dot = dq_acceleration_casadi(dq, twist, u[0:4], L_PARAMS)
    dtheta    = v_theta                            # θ̇ = v_θ

    f_expl = vertcat(dq_dot, twist_dot, dtheta)

    # ── Auxiliary CasADi functions ───────────────────────────────────────
    u_zero   = MX.zeros(u.size1(), 1)
    f_x_func = Function('f0',     [x],    [substitute(f_expl, u, u_zero)])
    g_x_func = Function('g',      [x],    [jacobian(f_expl, u)])
    f_system = Function('system', [x, u], [f_expl])

    # ── AcadosModel (no runtime parameters — baseline DQ-MPCC) ──────────
    model              = AcadosModel()
    model.f_impl_expr  = x_dot - f_expl
    model.f_expl_expr  = f_expl
    model.x            = x
    model.xdot         = x_dot
    model.u            = u
    model.name         = model_name

    return model, f_system, f_x_func, g_x_func
