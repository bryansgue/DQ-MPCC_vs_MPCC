"""Augmented MPCC quadrotor model with rate-control inputs for MuJoCo SiL.

Same dynamics as quadrotor_mpcc_rate_model.py but uses MASS_MUJOCO and a
distinct model_name so the generated C code does not conflict with the MiL
solver.
"""

import os
import sys

from acados_template import AcadosModel
from casadi import MX, Function, jacobian, substitute, vertcat

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SHARED_MPCC_ROOT = os.path.join(_WORKSPACE_ROOT, "MPCC_baseline")
for _path in (_WORKSPACE_ROOT, _SHARED_MPCC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.casadi_utils import (
    quat_kinematics_casadi as quat_p,
    quat_to_rot_casadi as quat_to_rot,
)
from MPCC_baseline_rates.config.experiment_config import G, MASS_MUJOCO, TAU_RC


def f_system_model_mpcc_rate_mujoco():
    """Build the 14-state MPCC model with body-rate commands (MuJoCo mass)."""
    model_name = "Drone_ode_mpcc_rate_mujoco"

    e3 = MX([0, 0, 1])

    p1 = MX.sym("p1")
    p2 = MX.sym("p2")
    p3 = MX.sym("p3")
    v1 = MX.sym("v1")
    v2 = MX.sym("v2")
    v3 = MX.sym("v3")
    q0 = MX.sym("q0")
    q1 = MX.sym("q1")
    q2 = MX.sym("q2")
    q3 = MX.sym("q3")
    w1 = MX.sym("w1")
    w2 = MX.sym("w2")
    w3 = MX.sym("w3")
    theta = MX.sym("theta")
    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3, theta)

    Tt = MX.sym("Tt")
    w1_cmd = MX.sym("w1_cmd")
    w2_cmd = MX.sym("w2_cmd")
    w3_cmd = MX.sym("w3_cmd")
    v_theta = MX.sym("v_theta")
    u = vertcat(Tt, w1_cmd, w2_cmd, w3_cmd, v_theta)

    xdot = MX.sym("xdot", 14, 1)

    quat = vertcat(q0, q1, q2, q3)
    omega = vertcat(w1, w2, w3)
    omega_cmd = vertcat(w1_cmd, w2_cmd, w3_cmd)
    rot = quat_to_rot(quat)

    dp = vertcat(v1, v2, v3)
    dv = -G * e3 + (rot @ vertcat(MX(0), MX(0), Tt)) / MASS_MUJOCO
    dq = quat_p(quat, omega)
    dw = (omega_cmd - omega) / TAU_RC
    dtheta = v_theta
    f_expl = vertcat(dp, dv, dq, dw, dtheta)

    u_zero = MX.zeros(u.size1(), 1)
    f_x = Function("f0_rate_mpcc_muj", [x], [substitute(f_expl, u, u_zero)])
    g_x = Function("g_rate_mpcc_muj", [x], [jacobian(f_expl, u)])
    f_system = Function("system_rate_mpcc_muj", [x, u], [f_expl])

    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    return model, f_system, f_x, g_x
