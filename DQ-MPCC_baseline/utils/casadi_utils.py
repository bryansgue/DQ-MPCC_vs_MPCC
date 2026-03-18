"""
casadi_utils.py – CasADi symbolic utilities for quadrotor MPCC.

All functions work on CasADi MX / SX symbolic types and are suitable for
embedding inside acados OCP formulations.

Quaternion convention: Hamilton form  q = [qw, qx, qy, qz]  (scalar first).

Sections
--------
1. Rotation matrices
2. Quaternion algebra  (multiply, kinematics, error, logarithm)
3. Trajectory interpolation  (θ → position / tangent / quaternion)
"""

import numpy as np
import casadi as ca
from casadi import (
    MX, vertcat, vertsplit, Function,
    cos, sin, norm_2, if_else, atan2,
)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Rotation matrices
# ══════════════════════════════════════════════════════════════════════════════

def rot_zyx_casadi(x):
    """ZYX Euler rotation matrix  R = Rz(ψ) Ry(θ) Rx(φ)  (CasADi symbolic).

    Parameters
    ----------
    x : MX  – state vector whose indices [3,4,5] = [φ, θ, ψ] (rad).
    """
    phi   = x[3, 0]
    theta = x[4, 0]
    psi   = x[5, 0]

    RotX = MX.zeros(3, 3)
    RotX[0, 0] = 1.0
    RotX[1, 1] =  cos(phi);  RotX[1, 2] = -sin(phi)
    RotX[2, 1] =  sin(phi);  RotX[2, 2] =  cos(phi)

    RotY = MX.zeros(3, 3)
    RotY[0, 0] = cos(theta);  RotY[0, 2] = sin(theta)
    RotY[1, 1] = 1.0
    RotY[2, 0] = -sin(theta); RotY[2, 2] = cos(theta)

    RotZ = MX.zeros(3, 3)
    RotZ[0, 0] = cos(psi);  RotZ[0, 1] = -sin(psi)
    RotZ[1, 0] = sin(psi);  RotZ[1, 1] =  cos(psi)
    RotZ[2, 2] = 1.0

    return RotZ @ RotY @ RotX


def quat_to_rot_casadi(quat):
    """Quaternion → rotation matrix  R ∈ SO(3)  (CasADi symbolic).

    Uses the skew-symmetric (hat) parametrisation:
        R = I + 2 q̂² + 2 qw q̂
    where q̂ is the skew-symmetric matrix of the vector part.

    Parameters
    ----------
    quat : MX (4,)  – unit quaternion [qw, qx, qy, qz].

    Returns
    -------
    Rot : MX (3,3)  – rotation matrix (body → inertial).
    """
    q_norm       = norm_2(quat)
    q            = quat / q_norm

    q_hat        = MX.zeros(3, 3)
    q_hat[0, 1]  = -q[3];  q_hat[0, 2] =  q[2]
    q_hat[1, 0]  =  q[3];  q_hat[1, 2] = -q[1]
    q_hat[2, 0]  = -q[2];  q_hat[2, 1] =  q[1]

    return MX.eye(3) + 2 * q_hat @ q_hat + 2 * q[0] * q_hat


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Quaternion algebra
# ══════════════════════════════════════════════════════════════════════════════

def quat_multiply_casadi(q1, q2):
    """Hamilton product  q1 ⊗ q2  (CasADi symbolic).

    Parameters
    ----------
    q1, q2 : MX (4,)  – quaternions [qw, qx, qy, qz].
    """
    w0, x0, y0, z0 = vertsplit(q1)
    w1, x1, y1, z1 = vertsplit(q2)

    return vertcat(
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    )


def quat_kinematics_casadi(quat, omega):
    """Quaternion kinematics:  q̇ = ½ q ⊗ [0, ω]  (CasADi symbolic).

    Parameters
    ----------
    quat  : MX (4,)  – current quaternion [qw, qx, qy, qz].
    omega : MX (3,)  – body angular velocity [ωx, ωy, ωz].
    """
    omega_quat = vertcat(MX(0), omega)
    return 0.5 * quat_multiply_casadi(quat, omega_quat)


def quat_error_casadi(q_real, q_desired):
    """Quaternion error:  q_err = q_real⁻¹ ⊗ q_desired  (CasADi symbolic).

    The inverse of a unit quaternion is its conjugate.

    Parameters
    ----------
    q_real    : MX (4,)  – current quaternion.
    q_desired : MX (4,)  – desired quaternion.
    """
    norm_q = norm_2(q_real)
    q_inv  = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    return quat_multiply_casadi(q_inv, q_desired)


def quat_log_casadi(q):
    """Quaternion logarithm on SO(3):  Log(q) = 2·atan2(‖q_v‖, qw)·q_v/‖q_v‖
    (CasADi symbolic).

    Safe at the identity (q_v → 0): the denominator is regularised with a
    small ε so CasADi produces a finite Jacobian instead of 0/0.

    Parameters
    ----------
    q : MX (4,)  – unit quaternion [qw, qx, qy, qz].

    Returns
    -------
    log_q : MX (3,)  – rotation vector (element of so(3)).
    """
    # Enforce positive scalar part (double cover: q ≡ −q)
    q_w = q[0]
    q   = if_else(q_w < 0, -q, q)
    q_w = q[0]
    q_v = q[1:]

    norm_q_v  = norm_2(q_v)
    theta     = atan2(norm_q_v, q_w)
    safe_norm = norm_q_v + 1e-9          # finite Jacobian at identity
    return 2.0 * q_v * theta / safe_norm


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Trajectory interpolation  (θ → reference)
#
#  The three functions below build CasADi Function objects that map a scalar
#  arc-length parameter  θ ∈ [0, s_max]  to the trajectory reference.
#  Being CasADi symbolic functions they are fully differentiable, which lets
#  the MPCC solver optimise  v_θ  by differentiating through:
#      v_θ  →  θ̇ = v_θ  →  θ  →  γ(θ)  →  error  →  cost
# ══════════════════════════════════════════════════════════════════════════════

def _piecewise_linear(s_sym, s_wp, values_wp):
    """Build a piecewise-linear CasADi expression for a 1-D array of values.

    Iterates from the last segment backwards so that the final if_else chain
    selects the correct segment for any s ∈ [s_wp[0], s_wp[-1]].

    Parameters
    ----------
    s_sym    : MX scalar         – the symbolic arc-length variable.
    s_wp     : ndarray (N,)      – monotonically increasing knot positions.
    values_wp: ndarray (N,)      – scalar values at each knot.

    Returns
    -------
    expr : MX scalar  – piecewise-linear interpolation of values_wp at s_sym.
    """
    n    = len(s_wp)
    s_c  = ca.fmin(ca.fmax(s_sym, s_wp[0]), s_wp[-1])
    expr = MX(values_wp[-1])

    for i in range(n - 2, -1, -1):
        s0, s1 = s_wp[i], s_wp[i + 1]
        a      = (s_c - s0) / (s1 - s0 + 1e-10)
        a      = ca.fmin(ca.fmax(a, 0.0), 1.0)
        vi     = (1 - a) * values_wp[i] + a * values_wp[i + 1]
        expr   = ca.if_else(s_c < s1, vi, expr)

    return expr


def create_position_interpolator_casadi(s_waypoints, pos_waypoints):
    """Build  γ_pos(θ) : ℝ → ℝ³  (piecewise-linear, CasADi symbolic).

    Parameters
    ----------
    s_waypoints  : ndarray (N,)   – arc-length values of the waypoints [m].
    pos_waypoints: ndarray (3, N) – Cartesian positions at each waypoint [m].

    Returns
    -------
    gamma_pos : casadi.Function  (scalar θ) → (3,) position vector.
    """
    s = MX.sym('s')
    px = _piecewise_linear(s, s_waypoints, pos_waypoints[0, :])
    py = _piecewise_linear(s, s_waypoints, pos_waypoints[1, :])
    pz = _piecewise_linear(s, s_waypoints, pos_waypoints[2, :])
    return Function('gamma_pos', [s], [vertcat(px, py, pz)])


def create_tangent_interpolator_casadi(s_waypoints, tang_waypoints):
    """Build  γ_vel(θ) : ℝ → ℝ³  (piecewise-linear + unit-normalisation, CasADi).

    Parameters
    ----------
    s_waypoints   : ndarray (N,)   – arc-length values of the waypoints [m].
    tang_waypoints: ndarray (3, N) – unit tangent vectors at each waypoint.

    Returns
    -------
    gamma_vel : casadi.Function  (scalar θ) → (3,) unit tangent vector.
    """
    s = MX.sym('s')
    tx = _piecewise_linear(s, s_waypoints, tang_waypoints[0, :])
    ty = _piecewise_linear(s, s_waypoints, tang_waypoints[1, :])
    tz = _piecewise_linear(s, s_waypoints, tang_waypoints[2, :])
    tn = ca.sqrt(tx**2 + ty**2 + tz**2 + 1e-10)
    return Function('gamma_vel', [s], [vertcat(tx / tn, ty / tn, tz / tn)])


def create_quat_interpolator_casadi(s_waypoints, quat_waypoints):
    """Build  γ_quat(θ) : ℝ → ℝ⁴  (linear SLERP-lite + normalisation, CasADi).

    Performs component-wise linear interpolation followed by normalisation.
    Hemisphere consistency must be ensured *before* calling this function
    (i.e. consecutive waypoints must satisfy  dot(q_i, q_{i-1}) > 0).

    Parameters
    ----------
    s_waypoints   : ndarray (N,)   – arc-length values of the waypoints [m].
    quat_waypoints: ndarray (4, N) – unit quaternions [qw, qx, qy, qz] per wp.

    Returns
    -------
    gamma_quat : casadi.Function  (scalar θ) → (4,) unit quaternion.
    """
    s  = MX.sym('s')
    qw = _piecewise_linear(s, s_waypoints, quat_waypoints[0, :])
    qx = _piecewise_linear(s, s_waypoints, quat_waypoints[1, :])
    qy = _piecewise_linear(s, s_waypoints, quat_waypoints[2, :])
    qz = _piecewise_linear(s, s_waypoints, quat_waypoints[3, :])
    qn = ca.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + 1e-10)
    return Function('gamma_quat', [s], [vertcat(qw/qn, qx/qn, qy/qn, qz/qn)])


# ── Backward-compatible aliases (old names used in mpcc_controller.py) ────────
# Remove these once all callers have been updated to the new names.
def create_casadi_position_interpolator(s_waypoints, pos_waypoints):
    return create_position_interpolator_casadi(s_waypoints, pos_waypoints)

def create_casadi_tangent_interpolator(s_waypoints, tang_waypoints):
    return create_tangent_interpolator_casadi(s_waypoints, tang_waypoints)

def create_casadi_quat_interpolator(s_waypoints, quat_waypoints):
    return create_quat_interpolator_casadi(s_waypoints, quat_waypoints)
