"""
dq_numpy_utils.py – NumPy utilities for dual-quaternion NMPC.

All functions work on standard Python / NumPy types (no CasADi).

Dual Quaternion convention:
    Q = [qr(4), qd(4)]  (8-vector)
    Pose:  Q = q_rot + ε (1/2) t * q_rot

Quaternion convention: Hamilton form  q = [qw, qx, qy, qz]  (scalar first).

Sections
--------
1. Quaternion operations    (product, conjugate, from Euler)
2. Dual quaternion ops      (from_pose, get_trans, get_quat)
3. Dual quaternion RK4      (kinematics integration)
"""

import math
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Quaternion operations
# ══════════════════════════════════════════════════════════════════════════════

def quat_product_numpy(p, q):
    """Hamilton product  p ⊗ q  (NumPy).

    Parameters
    ----------
    p, q : ndarray (4,)  – quaternions [qw, qx, qy, qz].

    Returns
    -------
    result : ndarray (4,)
    """
    H = np.array([
        [p[0], -p[1], -p[2], -p[3]],
        [p[1],  p[0], -p[3],  p[2]],
        [p[2],  p[3],  p[0], -p[1]],
        [p[3], -p[2],  p[1],  p[0]],
    ])
    return H @ q


def euler_to_quaternion(roll, pitch, yaw):
    """ZYX Euler angles → quaternion [qw, qx, qy, qz] (scalar first)."""
    cy = math.cos(yaw   * 0.5);  sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5);  sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5);  sr = math.sin(roll  * 0.5)

    qw =  cr * cp * cy + sr * sp * sy
    qx =  sr * cp * cy - cr * sp * sy
    qy =  cr * sp * cy + sr * cp * sy
    qz =  cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    """Quaternion [qw, qx, qy, qz] → ZYX Euler [roll, pitch, yaw] (rad)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quat_rotate_numpy(q, v):
    """Rotate vector v by quaternion q:  v' = q ⊗ [0,v] ⊗ q*.

    Parameters
    ----------
    q : ndarray (4,)  – quaternion [qw, qx, qy, qz].
    v : ndarray (3,)  – 3D vector.

    Returns
    -------
    v_rot : ndarray (3,)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R @ v


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Dual quaternion operations
# ══════════════════════════════════════════════════════════════════════════════

def dq_from_pose_numpy(quat, trans):
    """Build a unit dual quaternion from rotation quaternion + translation.

    Q = q_rot + ε (1/2) t * q_rot

    Parameters
    ----------
    quat  : ndarray (4,)  – rotation quaternion [qw, qx, qy, qz].
    trans : ndarray (3,)  – translation [tx, ty, tz].

    Returns
    -------
    dq : ndarray (8,)  – dual quaternion [qr(4), qd(4)].
    """
    t_pure = np.array([0.0, trans[0], trans[1], trans[2]])
    q_r = quat
    q_d = 0.5 * quat_product_numpy(t_pure, q_r)
    return np.concatenate([q_r, q_d])


def dq_get_translation_numpy(dq):
    """Extract translation quaternion [0, tx, ty, tz] from a dual quaternion.

    t = 2 · Qd ⊗ Qr*

    Parameters
    ----------
    dq : ndarray (8,)  – dual quaternion.

    Returns
    -------
    t : ndarray (4,)  – [0, tx, ty, tz].
    """
    qr = dq[0:4]
    qd = dq[4:8]
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]])
    return 2.0 * quat_product_numpy(qd, qr_c)


def dq_get_position_numpy(dq):
    """Extract position [tx, ty, tz] from a dual quaternion.

    Parameters
    ----------
    dq : ndarray (8,)  – dual quaternion.

    Returns
    -------
    pos : ndarray (3,)  – [tx, ty, tz].
    """
    t = dq_get_translation_numpy(dq)
    return t[1:4]


def dq_get_quaternion_numpy(dq):
    """Extract rotation quaternion from a dual quaternion.

    Parameters
    ----------
    dq : ndarray (8,)  – dual quaternion.

    Returns
    -------
    q : ndarray (4,)  – rotation quaternion [qw, qx, qy, qz].
    """
    return dq[0:4]


def dq_hemisphere_correction(dq_current, dq_previous):
    """Ensure the current dual quaternion is in the same hemisphere as previous.

    Flips dq if  dot(qr_current, qr_previous) < 0.

    Parameters
    ----------
    dq_current  : ndarray (8,)
    dq_previous : ndarray (8,)

    Returns
    -------
    dq_corrected : ndarray (8,)
    """
    if np.dot(dq_current[0:4], dq_previous[0:4]) < 0:
        return -dq_current
    return dq_current


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Dual quaternion RK4 integration
# ══════════════════════════════════════════════════════════════════════════════

def _dq_dot_numpy(dq, twist):
    """Dual quaternion derivative (NumPy).

    Parameters
    ----------
    dq    : ndarray (8,)  – dual quaternion.
    twist : ndarray (6,)  – [ωx, ωy, ωz, vx, vy, vz].

    Returns
    -------
    dq_dot : ndarray (8,)
    """
    qr = dq[0:4]
    qd = dq[4:8]

    K_quat = 10.0
    norm_r = np.linalg.norm(qr)
    quat_error = 1.0 - norm_r
    correction = np.concatenate([qr * (K_quat * quat_error), np.zeros(4)])

    # Build H+ for real and dual
    def H_plus(q):
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1],  q[0], -q[3],  q[2]],
            [q[2],  q[3],  q[0], -q[1]],
            [q[3], -q[2],  q[1],  q[0]],
        ])

    H_r = H_plus(qr)
    H_d = H_plus(qd)
    zeros = np.zeros((4, 4))
    H_dual = np.block([
        [H_r, zeros],
        [H_d, H_r],
    ])

    omega_dq = np.array([
        0.0, twist[0], twist[1], twist[2],
        0.0, twist[3], twist[4], twist[5],
    ])

    return 0.5 * (H_dual @ omega_dq) + correction


def rk4_step_dq(dq, twist, ts):
    """RK4 integration step for dual quaternion kinematics only.

    Parameters
    ----------
    dq    : ndarray (8,)  – dual quaternion.
    twist : ndarray (6,)  – body twist [ωx, ωy, ωz, vx, vy, vz].
    ts    : float         – time step.

    Returns
    -------
    dq_next : ndarray (8,)
    """
    k1 = _dq_dot_numpy(dq, twist)
    k2 = _dq_dot_numpy(dq + 0.5 * ts * k1, twist)
    k3 = _dq_dot_numpy(dq + 0.5 * ts * k2, twist)
    k4 = _dq_dot_numpy(dq + ts * k3, twist)
    return dq + (ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_step_dq_full(x, u, ts, f_sys):
    """RK4 integration for the full 14-state dual-quaternion model.

    State x = [dq(8), twist(6)]  (14-dim)
    Control u = [T, τx, τy, τz]  (4-dim)

    Parameters
    ----------
    x     : ndarray (14,)  – full state.
    u     : ndarray (4,)   – control.
    ts    : float           – time step.
    f_sys : CasADi Function(x, u) → ẋ  (14-dim).

    Returns
    -------
    x_next : ndarray (14,)
    """
    k1 = np.array(f_sys(x, u)).flatten()
    k2 = np.array(f_sys(x + 0.5 * ts * k1, u)).flatten()
    k3 = np.array(f_sys(x + 0.5 * ts * k2, u)).flatten()
    k4 = np.array(f_sys(x + ts * k3, u)).flatten()
    return x + (ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  MPCC-specific helpers  (15-state model)
# ══════════════════════════════════════════════════════════════════════════════

def dq_normalize(dq):
    """Normalize the real part of a dual quaternion + fix dual orthogonality.

    Parameters
    ----------
    dq : ndarray (8,)  – dual quaternion.

    Returns
    -------
    dq_norm : ndarray (8,)  – normalized dual quaternion.
    """
    qr = dq[0:4].copy()
    qd = dq[4:8].copy()
    qr_norm = np.linalg.norm(qr)
    if qr_norm > 1e-8:
        qr /= qr_norm
        # Fix dual part:  dot(qr, qd) = 0
        dot_rd = np.dot(qr, qd)
        qd -= dot_rd * qr
    return np.concatenate([qr, qd])


def rk4_step_dq_mpcc(x, u, ts, f_sys):
    """RK4 integration for the 15-state DQ-MPCC model.

    State  x = [dq(8), twist(6), θ]  (15-dim)
    Control u = [T, τx, τy, τz, v_θ]  (5-dim)

    Parameters
    ----------
    x     : ndarray (15,)
    u     : ndarray (5,)
    ts    : float
    f_sys : CasADi Function(x15, u5) → ẋ15
    """
    k1 = np.array(f_sys(x, u)).flatten()
    k2 = np.array(f_sys(x + 0.5 * ts * k1, u)).flatten()
    k3 = np.array(f_sys(x + 0.5 * ts * k2, u)).flatten()
    k4 = np.array(f_sys(x + ts * k3, u)).flatten()
    return x + (ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def state15_to_standard13(x15):
    """Convert 15-dim DQ-MPCC state to standard 13-dim state for plotting.

    DQ-MPCC state:  [dq(8), twist(6), θ]
    Std state:      [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

    Body-frame linear velocity is rotated to the inertial frame.

    Parameters
    ----------
    x15 : ndarray (15,)  – full DQ-MPCC state.

    Returns
    -------
    x13 : ndarray (13,)  – standard state for plotting.
    """
    dq    = x15[0:8]
    twist = x15[8:14]

    pos  = dq_get_position_numpy(dq)
    quat = dq_get_quaternion_numpy(dq)
    w_body = twist[0:3]
    v_body = twist[3:6]

    v_inertial = quat_rotate_numpy(quat, v_body)

    return np.concatenate([pos, v_inertial, quat, w_body])


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Dual quaternion error & logarithmic map (NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def dq_conjugate_numpy(dq):
    """Dual quaternion conjugate  Q* = conj(Qr) + ε conj(Qd)."""
    return np.array([
        dq[0], -dq[1], -dq[2], -dq[3],
        dq[4], -dq[5], -dq[6], -dq[7],
    ])


def dq_product_numpy(dq1, dq2):
    """Dual quaternion product  dq1 * dq2  (NumPy)."""
    q1r, q1d = dq1[0:4], dq1[4:8]
    q2r, q2d = dq2[0:4], dq2[4:8]
    real = quat_product_numpy(q1r, q2r)
    dual = quat_product_numpy(q1r, q2d) + quat_product_numpy(q1d, q2r)
    return np.concatenate([real, dual])


def dq_error_numpy(dq_desired, dq_actual):
    """Dual quaternion error:  Q_err = Q_d* ⊗ Q_actual  (NumPy).

    Parameters
    ----------
    dq_desired : ndarray (8,)
    dq_actual  : ndarray (8,)

    Returns
    -------
    dq_err : ndarray (8,)
    """
    return dq_product_numpy(dq_conjugate_numpy(dq_desired), dq_actual)


def _left_jacobian_SO3_inv_numpy(phi):
    """Analytical inverse of the left Jacobian of SO(3)  (NumPy).

    J_l(φ)⁻¹ = (θ/2)·cot(θ/2)·I₃ − (θ/2)·[n̂]× + (1 − (θ/2)·cot(θ/2))·n̂·n̂ᵀ

    Parameters
    ----------
    phi : ndarray (3,)

    Returns
    -------
    J_l_inv : ndarray (3,3)
    """
    theta = np.linalg.norm(phi)
    I3 = np.eye(3)

    if theta < 1e-7:
        # Small-angle approximation: J_l⁻¹ ≈ I₃ − 0.5·[φ]×
        phi_x = np.array([
            [0,      -phi[2],  phi[1]],
            [phi[2],  0,      -phi[0]],
            [-phi[1], phi[0],  0     ],
        ])
        return I3 - 0.5 * phi_x

    phi_x = np.array([
        [0,      -phi[2],  phi[1]],
        [phi[2],  0,      -phi[0]],
        [-phi[1], phi[0],  0     ],
    ])

    n_hat = phi / theta
    nnT = np.outer(n_hat, n_hat)
    n_hat_x = phi_x / theta

    half_theta = 0.5 * theta
    alpha = half_theta * np.cos(half_theta) / (np.sin(half_theta) + 1e-14)

    return alpha * I3 - half_theta * n_hat_x + (1 - alpha) * nnT


def ln_dual_numpy(dq_error):
    """Logarithmic map of a unit dual quaternion error → ℝ⁶  (NumPy).

    log(Q_err) = [φ; ρ] ∈ se(3)

    Parameters
    ----------
    dq_error : ndarray (8,)

    Returns
    -------
    log_vec : ndarray (6,)  – [φ_x, φ_y, φ_z, ρ_x, ρ_y, ρ_z]
    """
    q_real = dq_error[0:4].copy()
    q_dual = dq_error[4:8].copy()

    # Enforce positive scalar part (double cover)
    if q_real[0] < 0:
        q_real = -q_real
        q_dual = -q_dual

    q_real_c = np.array([q_real[0], -q_real[1], -q_real[2], -q_real[3]])

    # ── Rotational part (φ) ──
    eps = 1e-10
    norm_v = np.linalg.norm(q_real[1:4]) + eps
    angle = 2.0 * np.arctan2(norm_v, q_real[0])
    phi = 0.5 * angle * q_real[1:4] / norm_v

    # ── Translational part (ρ) ──
    t_err = 2.0 * quat_product_numpy(q_dual, q_real_c)
    t_vec = t_err[1:4]

    # Apply J_l(φ)⁻¹ correction:  ρ = J_l(φ)⁻¹ · t_vec
    J_l_inv = _left_jacobian_SO3_inv_numpy(phi)
    rho = J_l_inv @ t_vec

    return np.concatenate([phi, rho])


def rotate_tangent_to_desired_frame_numpy(tangent_world, quat_desired):
    """Rotate an inertial-frame tangent into the desired body frame (NumPy).

    t_body = R_d^T · t_world

    Parameters
    ----------
    tangent_world : ndarray (3,)
    quat_desired  : ndarray (4,)

    Returns
    -------
    tangent_body : ndarray (3,)
    """
    q_inv = np.array([quat_desired[0], -quat_desired[1],
                      -quat_desired[2], -quat_desired[3]])
    return quat_rotate_numpy(q_inv, tangent_world)


def lag_contouring_decomposition_numpy(rho, tangent_body):
    """Decompose ρ into lag (‖) and contouring (⊥) components (NumPy).

    Parameters
    ----------
    rho          : ndarray (3,)
    tangent_body : ndarray (3,)

    Returns
    -------
    rho_lag  : ndarray (3,)
    rho_cont : ndarray (3,)
    """
    proj = np.dot(tangent_body, rho)
    rho_lag  = proj * tangent_body
    rho_cont = rho - rho_lag
    return rho_lag, rho_cont
