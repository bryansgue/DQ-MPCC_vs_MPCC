"""
numpy_utils.py – NumPy / pure-Python utilities for quadrotor MPCC.

All functions work on standard Python / NumPy types (no CasADi).

Sections
--------
1. Quaternion conversions          (Euler ↔ quaternion, angle wrap)
2. Angular kinematics              (Euler-rate transformation)
3. Arc-length parameterisation     (cubic-spline arc ↔ time)
4. MPCC error decomposition        (contouring / lag errors)
5. Numerical integrators           (RK4 for 13- and 14-state models)
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Quaternion conversions
# ══════════════════════════════════════════════════════════════════════════════

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list:
    """ZYX Euler angles → quaternion  [qw, qx, qy, qz]  (scalar-first).

    Parameters
    ----------
    roll  : float  – rotation about x [rad]
    pitch : float  – rotation about y [rad]
    yaw   : float  – rotation about z [rad]

    Returns
    -------
    [qw, qx, qy, qz] : list of floats
    """
    cy = math.cos(yaw   * 0.5);  sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5);  sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5);  sr = math.sin(roll  * 0.5)

    qw =  cr * cp * cy + sr * sp * sy
    qx =  sr * cp * cy - cr * sp * sy
    qy =  cr * sp * cy + sr * cp * sy
    qz =  cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [qw, qx, qy, qz] → ZYX Euler angles [roll, pitch, yaw] (rad).

    Parameters
    ----------
    q : array-like (4,)  – unit quaternion [qw, qx, qy, qz].

    Returns
    -------
    euler : ndarray (3,)  – [roll, pitch, yaw] in radians.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # roll (x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    # yaw (z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def wrap_angle(angle: float) -> float:
    """Wrap an angle to  (−π, π].

    Parameters
    ----------
    angle : float  – angle in radians (any value).
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_error_numpy(q_real: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
    """Quaternion error:  q_err = q_real⁻¹ ⊗ q_desired  (NumPy).

    Parameters
    ----------
    q_real    : ndarray (4,)  – current quaternion [qw, qx, qy, qz].
    q_desired : ndarray (4,)  – desired quaternion [qw, qx, qy, qz].

    Returns
    -------
    q_err : ndarray (4,)  – error quaternion [qw, qx, qy, qz].
    """
    norm_q = np.linalg.norm(q_real)
    q_inv = np.array([q_real[0], -q_real[1], -q_real[2], -q_real[3]]) / norm_q

    # Hamilton product  q_inv ⊗ q_desired
    w1, x1, y1, z1 = q_inv
    w2, x2, y2, z2 = q_desired
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_log_numpy(q: np.ndarray) -> np.ndarray:
    """Quaternion logarithm:  Log(q) = 2·atan2(‖q_v‖, qw) · q_v / ‖q_v‖ (NumPy).

    Safe at the identity (q_v → 0).

    Parameters
    ----------
    q : ndarray (4,)  – unit quaternion [qw, qx, qy, qz].

    Returns
    -------
    log_q : ndarray (3,)  – rotation vector (element of so(3)).
    """
    # Enforce positive scalar part (double cover)
    if q[0] < 0:
        q = -q
    q_w = q[0]
    q_v = q[1:]

    norm_q_v = np.linalg.norm(q_v)
    theta = math.atan2(norm_q_v, q_w)
    safe_norm = norm_q_v + 1e-9
    return 2.0 * q_v * theta / safe_norm


def quaternion_hemisphere_correction(quats: np.ndarray) -> np.ndarray:
    """Ensure consecutive quaternions lie in the same hemisphere.

    Flips  q_i  if  dot(q_i, q_{i-1}) < 0  to guarantee shortest-path
    interpolation.

    Parameters
    ----------
    quats : ndarray (4, N)  – sequence of unit quaternions (columns).

    Returns
    -------
    quats_fixed : ndarray (4, N)  – hemisphere-corrected copy.
    """
    q = quats.copy()
    for i in range(1, q.shape[1]):
        if np.dot(q[:, i], q[:, i - 1]) < 0:
            q[:, i] *= -1
    return q


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Angular kinematics
# ══════════════════════════════════════════════════════════════════════════════

def euler_rate_matrix(euler: np.ndarray) -> np.ndarray:
    """Transformation matrix  W  such that  ω_body = W · η̇  (ZYX Euler).

    Inverse of the body-angular-velocity → Euler-rate map.

    Parameters
    ----------
    euler : array-like (3,)  – [roll, pitch, yaw] in radians.
    """
    phi, theta = euler[0], euler[1]
    W = np.array([
        [1, math.sin(phi) * math.tan(theta), math.cos(phi) * math.tan(theta)],
        [0, math.cos(phi),                   -math.sin(phi)],
        [0, math.sin(phi) / math.cos(theta),  math.cos(phi) / math.cos(theta)],
    ])
    return W


def euler_dot(omega: np.ndarray, euler: np.ndarray) -> np.ndarray:
    """Angular-velocity → Euler-rate:  η̇ = W(euler) · ω.

    Parameters
    ----------
    omega : ndarray (3,)  – body angular velocity [ωx, ωy, ωz].
    euler : ndarray (3,)  – current Euler angles [roll, pitch, yaw].
    """
    return euler_rate_matrix(euler) @ omega


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Arc-length parameterisation
# ══════════════════════════════════════════════════════════════════════════════

def build_arc_length_parameterisation(
    xd, yd, zd,
    xd_p, yd_p, zd_p,
    t_range: np.ndarray,
):
    """Build cubic-spline arc-length ↔ time mapping for a parametric curve.

    The curve is sampled at each value in *t_range*.  Arc lengths are computed
    by numerical integration of  ‖r'(t)‖.

    Parameters
    ----------
    xd, yd, zd     : callable (t) → float  – position components.
    xd_p, yd_p, zd_p : callable (t) → float  – velocity components.
    t_range        : ndarray (M,)  – parameter values (e.g. np.linspace(0, T, N)).

    Returns
    -------
    arc_lengths      : ndarray (M,)       – cumulative arc lengths at each t.
    positions        : ndarray (3, M)     – position array along the path.
    position_by_arc  : callable (s) → ndarray (3,)  – position at arc-length s.
    tangent_by_arc   : callable (s) → ndarray (3,)  – unit tangent at arc-length s.
    s_max            : float              – total arc length of the curve.
    """
    r       = lambda t: np.array([xd(t), yd(t), zd(t)])
    r_prime = lambda t: np.array([xd_p(t), yd_p(t), zd_p(t)])

    def _arc_length(tk, t0=0.0):
        length, _ = quad(lambda t: np.linalg.norm(r_prime(t)), t0, tk, limit=100)
        return length

    arc_lengths = np.array([_arc_length(tk) for tk in t_range])
    positions   = np.array([r(tk) for tk in t_range]).T  # (3, M)

    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    def position_by_arc(s: float) -> np.ndarray:
        """Position at arc-length s (clamped to [0, s_max])."""
        s  = np.clip(s, arc_lengths[0], arc_lengths[-1])
        te = spline_t(s)
        return np.array([spline_x(te), spline_y(te), spline_z(te)])

    def tangent_by_arc(s: float, ds: float = 1e-4) -> np.ndarray:
        """Unit tangent at arc-length s (finite-difference, clamped)."""
        s_lo = np.clip(s - ds, arc_lengths[0], arc_lengths[-1])
        s_hi = np.clip(s + ds, arc_lengths[0], arc_lengths[-1])
        tang = (position_by_arc(s_hi) - position_by_arc(s_lo)) / (s_hi - s_lo + 1e-10)
        norm = np.linalg.norm(tang)
        return tang / norm if norm > 1e-8 else tang

    return arc_lengths, positions, position_by_arc, tangent_by_arc, arc_lengths[-1]


def build_waypoints(
    s_max: float,
    n_waypoints: int,
    position_by_arc,
    tangent_by_arc,
    euler_to_quat_fn=None,
):
    """Sample the path uniformly in arc-length and build waypoint arrays.

    Computes positions, unit tangents, and yaw-aligned quaternions at
    *n_waypoints* evenly-spaced arc-length values.  Quaternion hemisphere
    consistency is enforced automatically.

    Parameters
    ----------
    s_max          : float      – total arc length [m].
    n_waypoints    : int        – number of waypoints.
    position_by_arc: callable   – from  build_arc_length_parameterisation.
    tangent_by_arc : callable   – from  build_arc_length_parameterisation.
    euler_to_quat_fn : callable (roll, pitch, yaw) → [qw,qx,qy,qz], optional.
        Defaults to the local  euler_to_quaternion.

    Returns
    -------
    s_wp   : ndarray (N,)    – arc-length knots.
    pos_wp : ndarray (3, N)  – positions.
    tang_wp: ndarray (3, N)  – unit tangents.
    quat_wp: ndarray (4, N)  – hemisphere-consistent quaternions.
    """
    if euler_to_quat_fn is None:
        euler_to_quat_fn = euler_to_quaternion

    s_wp    = np.linspace(0.0, s_max, n_waypoints)
    pos_wp  = np.zeros((3, n_waypoints))
    tang_wp = np.zeros((3, n_waypoints))
    quat_wp = np.zeros((4, n_waypoints))

    for i, sv in enumerate(s_wp):
        pos_wp[:, i]  = position_by_arc(sv)
        tang_wp[:, i] = tangent_by_arc(sv)
        psi_i         = np.arctan2(tang_wp[1, i], tang_wp[0, i])
        quat_wp[:, i] = euler_to_quat_fn(0.0, 0.0, psi_i)

    quat_wp = quaternion_hemisphere_correction(quat_wp)
    return s_wp, pos_wp, tang_wp, quat_wp


# ══════════════════════════════════════════════════════════════════════════════
#  4.  MPCC error decomposition
# ══════════════════════════════════════════════════════════════════════════════

def mpcc_errors(
    position: np.ndarray,
    tangent: np.ndarray,
    reference: np.ndarray,
):
    """Decompose the position error into contouring (⊥) and lag (‖) components.

    Given the 3-D position error  e_t = reference − position,
    the errors are:

        lag error       :  e_l = (t · e_t) · t        (scalar projection × tangent)
        contouring error:  e_c = e_t − e_l = (I − t tᵀ) e_t

    Parameters
    ----------
    position  : ndarray (3,)  – current UAV position.
    tangent   : ndarray (3,)  – unit tangent of the path at the current θ.
    reference : ndarray (3,)  – desired position on the path at the current θ.

    Returns
    -------
    e_c     : ndarray (3,)  – contouring error vector  (⊥ to path).
    e_l     : ndarray (3,)  – lag error vector         (‖ to path).
    e_total : ndarray (3,)  – total error = e_c + e_l.
    """
    e_t     = reference - position
    e_lag_s = np.dot(tangent, e_t)          # scalar projection
    e_l     = e_lag_s * tangent             # lag vector
    e_c     = e_t - e_l                     # contouring vector
    return e_c, e_l, e_c + e_l


def contouring_lag_scalar(
    position: np.ndarray,
    tangent: np.ndarray,
    reference: np.ndarray,
):
    """Return the scalar contouring and lag errors.

    Parameters
    ----------
    position  : ndarray (3,)
    tangent   : ndarray (3,)
    reference : ndarray (3,)

    Returns
    -------
    e_c_norm : float  – ‖contouring error‖
    e_lag    : float  – signed lag error  (t · (ref − pos))
    """
    e_t  = reference - position
    e_lag = float(np.dot(tangent, e_t))
    e_c   = e_t - e_lag * tangent
    return float(np.linalg.norm(e_c)), e_lag


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Numerical integrators (RK4)
# ══════════════════════════════════════════════════════════════════════════════

def rk4_step(f, x: np.ndarray, u: np.ndarray, ts: float) -> np.ndarray:
    """Generic one-step RK4 integrator.

    Parameters
    ----------
    f  : callable (x, u) → ẋ  – continuous dynamics (CasADi or NumPy).
    x  : ndarray  – current state.
    u  : ndarray  – control input.
    ts : float    – sampling time [s].

    Returns
    -------
    x_next : ndarray  – state after one RK4 step (same shape as x).
    """
    k1 = np.array(f(x, u)).flatten()
    k2 = np.array(f(x + (ts / 2) * k1, u)).flatten()
    k3 = np.array(f(x + (ts / 2) * k2, u)).flatten()
    k4 = np.array(f(x + ts * k3, u)).flatten()
    return x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_quadrotor(x: np.ndarray, u: np.ndarray,
                        ts: float, f_sys) -> np.ndarray:
    """RK4 step for the 13-state quadrotor model.

    Parameters
    ----------
    x     : ndarray (13,)  – [p, v, q, ω]
    u     : ndarray (4,)   – [T, τx, τy, τz]
    ts    : float          – sampling time [s]
    f_sys : casadi.Function (x13, u4) → ẋ13
    """
    k1 = f_sys(x, u)
    k2 = f_sys(x + (ts / 2) * k1, u)
    k3 = f_sys(x + (ts / 2) * k2, u)
    k4 = f_sys(x + ts * k3, u)
    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(x_next[:, 0]).reshape((13,))


def rk4_step_mpcc(x: np.ndarray, u: np.ndarray,
                   ts: float, f_sys) -> np.ndarray:
    """RK4 step for the 14-state augmented MPCC model (+ θ state).

    Parameters
    ----------
    x     : ndarray (14,)  – [p, v, q, ω, θ]
    u     : ndarray (5,)   – [T, τx, τy, τz, v_θ]
    ts    : float          – sampling time [s]
    f_sys : casadi.Function (x14, u5) → ẋ14
    """
    k1 = f_sys(x, u)
    k2 = f_sys(x + (ts / 2) * k1, u)
    k3 = f_sys(x + (ts / 2) * k2, u)
    k4 = f_sys(x + ts * k3, u)
    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(x_next[:, 0]).reshape((14,))


# ══════════════════════════════════════════════════════════════════════════════
#  Path geometry
# ══════════════════════════════════════════════════════════════════════════════

def compute_curvature(position_by_arc, s_max: float,
                      N_samples: int = 500, ds: float = 1e-3) -> np.ndarray:
    """Compute curvature κ(s) along an arc-length-parameterised path.

    κ = ‖r''(s)‖  (since r'(s) is unit tangent for arc-length param).

    Parameters
    ----------
    position_by_arc : callable (s) → ndarray (3,)
    s_max           : float          – total arc length
    N_samples       : int            – number of uniform samples
    ds              : float          – finite-difference step

    Returns
    -------
    curvature : ndarray (N_samples,) – κ at each sample
    """
    s_vals = np.linspace(0, s_max, N_samples)
    curvature = np.zeros(N_samples)

    for i, s in enumerate(s_vals):
        s_lo = np.clip(s - ds, 0, s_max)
        s_hi = np.clip(s + ds, 0, s_max)
        s_mid_lo = np.clip(s - ds / 2, 0, s_max)
        s_mid_hi = np.clip(s + ds / 2, 0, s_max)

        # First derivatives (tangent approximations)
        t_lo = (position_by_arc(s_mid_hi) - position_by_arc(s_mid_lo))
        # Normalise to get "unit tangent" differences
        h = s_hi - s_lo
        if h > 1e-10:
            # Second derivative via finite difference of first derivative
            p_lo = position_by_arc(s_lo)
            p_mid = position_by_arc(s)
            p_hi = position_by_arc(s_hi)
            r_pp = (p_hi - 2 * p_mid + p_lo) / (ds ** 2)
            curvature[i] = np.linalg.norm(r_pp)

    return curvature


# ── Backward-compatible aliases ───────────────────────────────────────────────
# These match the old names from quaternion_utils.py so that callers that
# import from this module still work.
Euler_p  = euler_dot
Angulo   = wrap_angle
