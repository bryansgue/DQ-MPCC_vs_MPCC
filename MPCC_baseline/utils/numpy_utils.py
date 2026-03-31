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


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [qw, qx, qy, qz]."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm_q = np.linalg.norm(q)
    return q / (norm_q + 1e-12)


def quat_interp_by_arc(s: float, s_wp: np.ndarray, quat_wp: np.ndarray) -> np.ndarray:
    """Piecewise-linear quaternion interpolation at arc-length s."""
    s = np.clip(s, s_wp[0], s_wp[-1])
    idx = np.searchsorted(s_wp, s, side='right') - 1
    idx = np.clip(idx, 0, len(s_wp) - 2)
    alpha = (s - s_wp[idx]) / (s_wp[idx + 1] - s_wp[idx] + 1e-12)
    q0 = quat_wp[:, idx]
    q1 = quat_wp[:, idx + 1]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / (np.linalg.norm(q) + 1e-12)


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
    reference_speed: float = 0.0,
    gravity: float = 9.81,
    max_tilt_deg: float = 60.0,
):
    """Sample the path uniformly in arc-length and build waypoint arrays.

    Computes positions, unit tangents, and dynamically compatible reference
    quaternions at *n_waypoints* evenly-spaced arc-length values.

    The quaternion reference is not built from yaw alone. Instead:
      1. the tangent defines the desired heading direction
      2. the local curvature defines a nominal lateral acceleration
      3. the desired body z-axis aligns with the corresponding thrust direction

    This avoids the old contradiction of asking the vehicle to follow the path
    tangent while simultaneously penalising a hover-like roll/pitch attitude.

    Parameters
    ----------
    s_max          : float      – total arc length [m].
    n_waypoints    : int        – number of waypoints.
    position_by_arc: callable   – from  build_arc_length_parameterisation.
    tangent_by_arc : callable   – from  build_arc_length_parameterisation.
    euler_to_quat_fn : kept for backward compatibility, unused by the new
        attitude reference construction.
    reference_speed : nominal path speed [m/s] used to turn curvature into a
        lateral acceleration reference.
    gravity         : gravity constant [m/s²].
    max_tilt_deg    : maximum allowed tilt used to keep the reference feasible.

    Returns
    -------
    s_wp   : ndarray (N,)    – arc-length knots.
    pos_wp : ndarray (3, N)  – positions.
    tang_wp: ndarray (3, N)  – unit tangents.
    quat_wp: ndarray (4, N)  – hemisphere-consistent quaternions.
    """
    s_wp    = np.linspace(0.0, s_max, n_waypoints)
    pos_wp  = np.zeros((3, n_waypoints))
    tang_wp = np.zeros((3, n_waypoints))
    quat_wp = np.zeros((4, n_waypoints))

    for i, sv in enumerate(s_wp):
        pos_wp[:, i]  = position_by_arc(sv)
        tang_wp[:, i] = tangent_by_arc(sv)

    ds = s_wp[1] - s_wp[0] if n_waypoints > 1 else 1.0
    curvature_wp = np.zeros_like(tang_wp)
    if n_waypoints > 1:
        curvature_wp[:, 0] = (tang_wp[:, 1] - tang_wp[:, 0]) / ds
        curvature_wp[:, -1] = (tang_wp[:, -1] - tang_wp[:, -2]) / ds
    for i in range(1, n_waypoints - 1):
        curvature_wp[:, i] = (tang_wp[:, i + 1] - tang_wp[:, i - 1]) / (2.0 * ds)

    max_tilt_rad = math.radians(max_tilt_deg)
    e3 = np.array([0.0, 0.0, 1.0])

    for i in range(n_waypoints):
        tang_i = tang_wp[:, i]
        tang_i = tang_i / (np.linalg.norm(tang_i) + 1e-12)
        a_lat = (reference_speed ** 2) * curvature_wp[:, i]

        thrust_dir = np.array([a_lat[0], a_lat[1], gravity + a_lat[2]], dtype=float)
        horiz_norm = np.linalg.norm(thrust_dir[:2])
        max_horiz = max(1e-9, thrust_dir[2] * math.tan(max_tilt_rad))
        if horiz_norm > max_horiz:
            thrust_dir[:2] *= max_horiz / horiz_norm
        b3 = thrust_dir / (np.linalg.norm(thrust_dir) + 1e-12)

        b1 = tang_i - np.dot(tang_i, b3) * b3
        if np.linalg.norm(b1) < 1e-8:
            yaw_i = math.atan2(tang_i[1], tang_i[0])
            heading = np.array([math.cos(yaw_i), math.sin(yaw_i), 0.0])
            b1 = heading - np.dot(heading, b3) * b3
        if np.linalg.norm(b1) < 1e-8:
            b1 = np.array([1.0, 0.0, 0.0]) - b3[0] * b3
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)
        b2 = np.cross(b3, b1)
        b2 = b2 / (np.linalg.norm(b2) + 1e-12)
        b1 = np.cross(b2, b3)
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)

        R = np.column_stack((b1, b2, b3))
        quat_wp[:, i] = rotation_matrix_to_quaternion(R)

    quat_wp = quaternion_hemisphere_correction(quat_wp)
    return s_wp, pos_wp, tang_wp, quat_wp


def build_terminally_extended_path(
    position_by_arc,
    tangent_by_arc,
    s_path_end: float,
    s_extended_end: float,
    s_original_end: float | None = None,
):
    """Extend the path beyond its active limit using the available geometry.

    Behaviour:
      1. for s <= min(s_path_end, s_original_end) use the original path
      2. if the original path still exists beyond s_path_end, keep following it
         up to s_original_end
      3. only after the true geometric end is reached, extend linearly using
         the final tangent

    This avoids the pathological case where the controller crosses a manual
    active limit s_path_end while the real path still curves, but the solver
    horizon sees only a straight tangent continuation.
    """
    s_path_end = float(s_path_end)
    s_extended_end = float(max(s_extended_end, s_path_end))
    if s_original_end is None:
        s_original_end = s_path_end
    s_original_end = float(max(s_original_end, s_path_end))

    pos_end = np.array(position_by_arc(s_original_end), dtype=float)
    tang_end = np.array(tangent_by_arc(s_original_end), dtype=float)
    tang_norm = np.linalg.norm(tang_end)
    if tang_norm > 1e-8:
        tang_end = tang_end / tang_norm

    def position_by_arc_extended(s: float) -> np.ndarray:
        s_clamped = float(np.clip(s, 0.0, s_extended_end))
        if s_clamped <= s_original_end:
            return position_by_arc(s_clamped)
        return pos_end + (s_clamped - s_original_end) * tang_end

    def tangent_by_arc_extended(s: float) -> np.ndarray:
        s_clamped = float(np.clip(s, 0.0, s_extended_end))
        if s_clamped <= s_original_end:
            return tangent_by_arc(s_clamped)
        return tang_end

    return position_by_arc_extended, tangent_by_arc_extended


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
    """RK4 step for the 15-state augmented MPCC model (+ θ, v_θ states).

    Parameters
    ----------
    x     : ndarray (15,)  – [p, v, q, ω, θ, v_θ]
    u     : ndarray (5,)   – [T, τx, τy, τz, a_θ]
    ts    : float          – sampling time [s]
    f_sys : casadi.Function (x15, u5) → ẋ15
    """
    k1 = f_sys(x, u)
    k2 = f_sys(x + (ts / 2) * k1, u)
    k3 = f_sys(x + (ts / 2) * k2, u)
    k4 = f_sys(x + ts * k3, u)
    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(x_next[:, 0]).reshape((15,))


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
