"""
dq_casadi_utils.py – CasADi symbolic utilities for dual-quaternion NMPC.

All functions work on CasADi MX symbolic types and are suitable for
embedding inside acados OCP formulations.

Dual Quaternion convention:
    Q = Qr + ε Qd   (8-vector: [qr(4), qd(4)])
    
    Rigid body pose:  Q = q_rot + ε (1/2) t * q_rot
    where q_rot is the rotation quaternion and t = [0, tx, ty, tz] is the
    pure quaternion of translation.

Quaternion convention: Hamilton form  q = [qw, qx, qy, qz]  (scalar first).

Sections
--------
1. Quaternion Hamilton product  (H+ matrix form)
2. Dual quaternion operations   (product, conjugate, from_pose)
3. Dual quaternion kinematics   (quatdot, acceleration)
4. Extraction functions         (get_trans, get_quat)
5. Rotation functions           (body ↔ inertial)
6. Logarithmic map              (ln_dual → ℝ⁶ error)
7. Error computation            (error_dual, cost metrics)
"""

import numpy as np
import casadi as ca
from casadi import MX, vertcat, horzcat, Function, norm_2, if_else, atan2, DM


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Quaternion Hamilton product  (H+ matrix form)
# ══════════════════════════════════════════════════════════════════════════════

def H_plus_matrix(q):
    """Build the left-multiplication Hamilton matrix H+(q).

    H+(q) · p  =  q ⊗ p   (Hamilton product).

    Parameters
    ----------
    q : MX (4,)  – quaternion [qw, qx, qy, qz].

    Returns
    -------
    H : MX (4,4)
    """
    return vertcat(
        horzcat(q[0], -q[1], -q[2], -q[3]),
        horzcat(q[1],  q[0], -q[3],  q[2]),
        horzcat(q[2],  q[3],  q[0], -q[1]),
        horzcat(q[3], -q[2],  q[1],  q[0]),
    )


def quat_product_casadi(p, q):
    """Hamilton product  p ⊗ q  using H+ matrix (CasADi symbolic).

    Parameters
    ----------
    p, q : MX (4,)  – quaternions [qw, qx, qy, qz].
    """
    return H_plus_matrix(p) @ q


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Dual quaternion operations
# ══════════════════════════════════════════════════════════════════════════════

def dq_from_pose_casadi(quat, trans):
    """Build a unit dual quaternion from rotation quaternion + translation.

    Q = q_rot + ε (1/2) t * q_rot

    Parameters
    ----------
    quat  : MX (4,)  – rotation quaternion [qw, qx, qy, qz].
    trans : MX (3,)  – translation [tx, ty, tz].

    Returns
    -------
    dq : MX (8,)  – dual quaternion [qr(4), qd(4)].
    """
    t_pure = vertcat(MX(0), trans)                       # pure quaternion [0, tx, ty, tz]
    q_r = quat
    q_d = 0.5 * quat_product_casadi(t_pure, q_r)        # (1/2) t ⊗ q_r
    return vertcat(q_r, q_d)


def dq_product_casadi(dq1, dq2):
    """Dual quaternion product  dq1 * dq2  (CasADi symbolic).

    Parameters
    ----------
    dq1, dq2 : MX (8,)  – dual quaternions [qr(4), qd(4)].
    """
    q1r, q1d = dq1[0:4], dq1[4:8]
    q2r, q2d = dq2[0:4], dq2[4:8]
    real = quat_product_casadi(q1r, q2r)
    dual = quat_product_casadi(q1r, q2d) + quat_product_casadi(q1d, q2r)
    return vertcat(real, dual)


def dq_conjugate_casadi(dq):
    """Dual quaternion conjugate  Q* = conj(Qr) + ε conj(Qd).

    Parameters
    ----------
    dq : MX (8,)  – dual quaternion [qr(4), qd(4)].
    """
    return vertcat(
        dq[0], -dq[1], -dq[2], -dq[3],
        dq[4], -dq[5], -dq[6], -dq[7],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Dual quaternion kinematics
# ══════════════════════════════════════════════════════════════════════════════

def dq_kinematics_casadi(dq, twist):
    """Dual quaternion time-derivative:  Q̇ = (1/2) Q ⊗ Ξ  + K·correction.

    The twist Ξ is a dual quaternion formed from the body-frame velocities:
        Ξ = [0, ωx, ωy, ωz, 0, vx, vy, vz]
    where ω is the body angular velocity and v is the body-frame linear velocity.

    A unit-norm correction term is added to prevent numerical drift.

    Parameters
    ----------
    dq    : MX (8,)  – dual quaternion [qr(4), qd(4)].
    twist : MX (6,)  – body twist [ωx, ωy, ωz, vx, vy, vz].

    Returns
    -------
    dq_dot : MX (8,)  – time derivative of the dual quaternion.
    """
    qr = dq[0:4]
    qd = dq[4:8]

    # Unit-norm correction (prevents drift)
    K_quat = 10.0
    norm_r = norm_2(qr)
    quat_error = 1.0 - norm_r
    correction = vertcat(qr * (K_quat * quat_error),
                         qd * MX(0))  # only correct real part

    # Build H+ matrices for dual quaternion product
    H_r = H_plus_matrix(qr)
    H_d = H_plus_matrix(qd)
    zeros = DM.zeros(4, 4)
    H_dual = vertcat(
        horzcat(H_r, zeros),
        horzcat(H_d, H_r),
    )

    # Twist as 8-vector: [0, ωx, ωy, ωz, 0, vx, vy, vz]
    omega_dq = vertcat(
        MX(0), twist[0], twist[1], twist[2],
        MX(0), twist[3], twist[4], twist[5],
    )

    dq_dot = 0.5 * (H_dual @ omega_dq) + correction
    return dq_dot


def dq_acceleration_casadi(dq, twist, u, L):
    """Twist acceleration for the quadrotor in dual-quaternion form.

    Computes  Ξ̇ = [α_body; a_body]  where:
        α = J⁻¹(τ − ω × Jω)       (Euler's equation)
        a = v × ω − R⁻¹ g e₃ + (T/m) e₃    (translational dynamics in body)

    Parameters
    ----------
    dq    : MX (8,)  – dual quaternion state.
    twist : MX (6,)  – body twist [ωx, ωy, ωz, vx, vy, vz].
    u     : MX (4,)  – control [T, τx, τy, τz].
    L     : list     – system parameters [m, Jxx, Jyy, Jzz, g].

    Returns
    -------
    twist_dot : MX (6,)  – [α_x, α_y, α_z, a_x, a_y, a_z].
    """
    force   = u[0]
    torques = u[1:4]

    m   = L[0]
    J   = DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_inv = ca.inv(J)
    g_val = L[4]
    e3 = vertcat(MX(0), MX(0), MX(1))

    w = twist[0:3]
    v = twist[3:6]

    # Extract quaternion for inverse rotation
    quat = dq[0:4]

    # Angular acceleration (Euler's equation in body frame)
    alpha = J_inv @ (torques - ca.cross(w, J @ w))

    # Linear acceleration in body frame
    # gravity in body: R⁻¹ g e₃  =  rotation_inverse(q, g*e3)
    g_body = rotation_inverse_expr(quat, g_val * e3)
    a_body = ca.cross(v, w) - g_body + (force / m) * e3

    return vertcat(alpha, a_body)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Extraction functions
# ══════════════════════════════════════════════════════════════════════════════

def dq_get_translation_casadi(dq):
    """Extract translation [0, tx, ty, tz] from a unit dual quaternion.

    t = 2 · Qd ⊗ Qr*

    Parameters
    ----------
    dq : MX (8,)  – dual quaternion.

    Returns
    -------
    t : MX (4,)  – pure quaternion of translation [0, tx, ty, tz].
    """
    qr = dq[0:4]
    qd = dq[4:8]
    qr_c = vertcat(qr[0], -qr[1], -qr[2], -qr[3])
    return 2.0 * quat_product_casadi(qd, qr_c)


def dq_get_position_casadi(dq):
    """Extract position [tx, ty, tz] from a unit dual quaternion.

    Parameters
    ----------
    dq : MX (8,)  – dual quaternion.

    Returns
    -------
    pos : MX (3,)  – position [tx, ty, tz].
    """
    t = dq_get_translation_casadi(dq)
    return t[1:4]


def dq_get_quaternion_casadi(dq):
    """Extract rotation quaternion [qw, qx, qy, qz] from a dual quaternion.

    Parameters
    ----------
    dq : MX (8,)  – dual quaternion.

    Returns
    -------
    q : MX (4,)  – rotation quaternion.
    """
    return dq[0:4]


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Rotation functions (body ↔ inertial)
# ══════════════════════════════════════════════════════════════════════════════

def rotation_expr(quat, vector):
    """Rotate vector from body to inertial:  v' = q ⊗ [0,v] ⊗ q*.

    Parameters
    ----------
    quat   : MX (4,)  – rotation quaternion.
    vector : MX (3,)  – 3D vector in body frame.

    Returns
    -------
    v_inertial : MX (3,)  – rotated vector.
    """
    v_pure = vertcat(MX(0), vector)
    qc = vertcat(quat[0], -quat[1], -quat[2], -quat[3])
    tmp = quat_product_casadi(quat, v_pure)
    result = quat_product_casadi(tmp, qc)
    return result[1:4]


def rotation_inverse_expr(quat, vector):
    """Rotate vector from inertial to body:  v' = q* ⊗ [0,v] ⊗ q.

    Parameters
    ----------
    quat   : MX (4,)  – rotation quaternion.
    vector : MX (3,)  – 3D vector in inertial frame.

    Returns
    -------
    v_body : MX (3,)  – rotated vector in body frame.
    """
    v_pure = vertcat(MX(0), vector)
    qc = vertcat(quat[0], -quat[1], -quat[2], -quat[3])
    tmp = quat_product_casadi(qc, v_pure)
    result = quat_product_casadi(tmp, quat)
    return result[1:4]


def build_rotation_functions():
    """Build CasADi Function objects for body↔inertial rotation.

    Returns
    -------
    f_rot     : Function(quat, vec) → vec_inertial
    f_rot_inv : Function(quat, vec) → vec_body
    """
    q_sym = MX.sym('q', 4, 1)
    v_sym = MX.sym('v', 3, 1)

    f_rot = Function('f_rot', [q_sym, v_sym],
                     [rotation_expr(q_sym, v_sym)])
    f_rot_inv = Function('f_rot_inv', [q_sym, v_sym],
                         [rotation_inverse_expr(q_sym, v_sym)])
    return f_rot, f_rot_inv


# ══════════════════════════════════════════════════════════════════════════════
#  6.  Logarithmic map  (dual quaternion error → ℝ⁶)
# ══════════════════════════════════════════════════════════════════════════════

def dq_error_casadi(dq_desired, dq_actual):
    """Dual quaternion error:  Q_err = Q_d* ⊗ Q.

    Parameters
    ----------
    dq_desired : MX (8,)  – desired dual quaternion.
    dq_actual  : MX (8,)  – actual dual quaternion.

    Returns
    -------
    dq_err : MX (8,)  – error dual quaternion.
    """
    dq_d_conj = dq_conjugate_casadi(dq_desired)
    # Build H+ matrices for dual quaternion product
    qr_c = dq_d_conj[0:4]
    qd_c = dq_d_conj[4:8]

    H_r = H_plus_matrix(qr_c)
    H_d = H_plus_matrix(qd_c)
    zeros = DM.zeros(4, 4)
    H_dual = vertcat(
        horzcat(H_r, zeros),
        horzcat(H_d, H_r),
    )
    return H_dual @ dq_actual


def left_jacobian_SO3(phi):
    """Left Jacobian of SO(3) for rotation vector φ.

    J_l(φ) = I₃ + (1 − cos‖φ‖)/‖φ‖² · [φ]×
                 + (‖φ‖ − sin‖φ‖)/‖φ‖³ · [φ]×²

    Parameters
    ----------
    phi : MX (3,)  – rotation vector.

    Returns
    -------
    J_l : MX (3,3)
    """
    norm_phi = norm_2(phi)
    eps = 1e-8

    # Skew-symmetric matrix [φ]×
    phi_x = vertcat(
        horzcat(MX(0),  -phi[2],  phi[1]),
        horzcat(phi[2],  MX(0),  -phi[0]),
        horzcat(-phi[1], phi[0],  MX(0)),
    )
    phi_x2 = phi_x @ phi_x

    I3 = DM.eye(3)
    c = (1 - ca.cos(norm_phi)) / (norm_phi**2 + eps)
    d = (norm_phi - ca.sin(norm_phi)) / (norm_phi**3 + eps)
    J_l = I3 + c * phi_x + d * phi_x2

    # Small-angle approximation: J_l ≈ I + 0.5·[φ]×
    J_l_approx = I3 + 0.5 * phi_x
    small = if_else(norm_phi < eps, 1, 0)
    J_l = if_else(small, J_l_approx, J_l)

    return J_l


def left_jacobian_SO3_inv(phi):
    """Analytical inverse of the left Jacobian of SO(3).

    J_l(φ)⁻¹ = (θ/2)·cot(θ/2)·I₃
              − (θ/2)·[n̂]×
              + (1 − (θ/2)·cot(θ/2))·n̂·n̂ᵀ

    where θ = ‖φ‖ and n̂ = φ/θ is the unit rotation axis.

    This closed-form (Solà 2018) avoids the heavier expression graph
    that ca.inv would introduce, yielding cleaner AD and faster
    code generation inside acados.

    For θ < 1e-7 the expression reduces to J_l⁻¹ ≈ I₃ − 0.5·[φ]×.

    Parameters
    ----------
    phi : MX (3,)  – rotation vector.

    Returns
    -------
    J_l_inv : MX (3,3)
    """
    theta = norm_2(phi)
    eps = 1e-7

    I3 = DM.eye(3)

    # Skew-symmetric matrix [φ]×
    phi_x = vertcat(
        horzcat(MX(0),  -phi[2],  phi[1]),
        horzcat(phi[2],  MX(0),  -phi[0]),
        horzcat(-phi[1], phi[0],  MX(0)),
    )

    # Unit axis  n̂ = φ / θ  (safe division)
    n_hat = phi / (theta + 1e-14)

    # n̂ · n̂ᵀ  (3×3 outer product)
    n_hat_col = ca.reshape(n_hat, 3, 1)
    nnT = n_hat_col @ n_hat_col.T

    # Skew of n̂  (= [φ]× / θ)
    n_hat_x = phi_x / (theta + 1e-14)

    # half_theta · cot(half_theta)  =  (θ/2) · cos(θ/2) / sin(θ/2)
    half_theta = 0.5 * theta
    alpha = half_theta * ca.cos(half_theta) / (ca.sin(half_theta) + 1e-14)

    # J_l⁻¹ = α·I₃  −  (θ/2)·[n̂]×  +  (1 − α)·n̂·n̂ᵀ
    J_l_inv = alpha * I3 - half_theta * n_hat_x + (1 - alpha) * nnT

    # Small-angle approximation: J_l⁻¹ ≈ I₃ − 0.5·[φ]×
    J_l_inv_approx = I3 - 0.5 * phi_x
    J_l_inv = if_else(theta < eps, J_l_inv_approx, J_l_inv)

    return J_l_inv


def ln_dual_casadi(dq_error):
    """Logarithmic map of a unit dual quaternion error → ℝ⁶.

    log(Q_err) = [φ; ρ] ∈ se(3)

    where:
      φ = 2·atan2(‖q_v‖, q_w) · q_v / ‖q_v‖     (rotation vector, ℝ³)
      ρ = J_l(φ)⁻¹ · t_err                         (translational error, ℝ³)

    The raw translation is extracted as  t = 2 · Qd_err ⊗ Qr_err*,
    which gives R_d^T (p − p_d) directly. This is then corrected by
    the inverse left Jacobian of SO(3) to properly account for the
    rotation-translation coupling in se(3).

    Parameters
    ----------
    dq_error : MX (8,)  – error dual quaternion.

    Returns
    -------
    log_vec : MX (6,)  – [φ_x, φ_y, φ_z, ρ_x, ρ_y, ρ_z].
    """
    q_real = dq_error[0:4]

    # Enforce positive scalar part (double cover)
    q_real = if_else(q_real[0] < 0, -q_real, q_real)

    q_real_c = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3])
    q_dual = dq_error[4:8]

    # ── Rotational part (φ) ──
    eps = 1e-10
    norm_v = norm_2(q_real[1:4] + eps)
    angle = 2.0 * atan2(norm_v, q_real[0])
    phi = vertcat(
        0.5 * angle * q_real[1] / norm_v,
        0.5 * angle * q_real[2] / norm_v,
        0.5 * angle * q_real[3] / norm_v,
    )

    # ── Translational part (ρ) ──
    # Raw translation: t_err = 2 · Qd_err ⊗ Qr_err*  →  R_d^T (p − p_d)
    t_err = 2.0 * quat_product_casadi(q_dual, q_real_c)
    t_vec = vertcat(t_err[1], t_err[2], t_err[3])

    # Apply J_l(φ)⁻¹ correction:  ρ = J_l(φ)⁻¹ · t_vec
    J_l_inv = left_jacobian_SO3_inv(phi)
    rho = J_l_inv @ t_vec

    return vertcat(phi, rho)


# ══════════════════════════════════════════════════════════════════════════════
#  7.  CasADi Function builders (for use outside the OCP)
# ══════════════════════════════════════════════════════════════════════════════

def build_dq_extraction_functions():
    """Build CasADi Function objects for translation/quaternion extraction.

    Returns
    -------
    f_get_trans : Function(dq) → [0, tx, ty, tz]
    f_get_pos   : Function(dq) → [tx, ty, tz]
    f_get_quat  : Function(dq) → [qw, qx, qy, qz]
    """
    dq_sym = MX.sym('dq', 8, 1)

    f_get_trans = Function('get_trans', [dq_sym],
                           [dq_get_translation_casadi(dq_sym)])
    f_get_pos = Function('get_pos', [dq_sym],
                         [dq_get_position_casadi(dq_sym)])
    f_get_quat = Function('get_quat', [dq_sym],
                          [dq_get_quaternion_casadi(dq_sym)])
    return f_get_trans, f_get_pos, f_get_quat


def build_dq_from_pose_function():
    """Build CasADi Function for pose → dual quaternion.

    Returns
    -------
    f_from_pose : Function(qw, qx, qy, qz, tx, ty, tz) → dq (8,)
    """
    qw = MX.sym('qw'); qx = MX.sym('qx')
    qy = MX.sym('qy'); qz = MX.sym('qz')
    tx = MX.sym('tx'); ty = MX.sym('ty'); tz = MX.sym('tz')

    quat = vertcat(qw, qx, qy, qz)
    trans = vertcat(tx, ty, tz)
    dq = dq_from_pose_casadi(quat, trans)

    return Function('dq_from_pose', [qw, qx, qy, qz, tx, ty, tz], [dq])


# ══════════════════════════════════════════════════════════════════════════════
#  8.  MPCC-specific functions  (lag / contouring decomposition)
# ══════════════════════════════════════════════════════════════════════════════

def rotate_tangent_to_desired_frame(tangent_world, quat_desired):
    """Rotate an inertial-frame tangent vector into the desired body frame.

    t_body = R_d^T · t_world

    Parameters
    ----------
    tangent_world : MX (3,)  – unit tangent in the inertial frame.
    quat_desired  : MX (4,)  – desired rotation quaternion.

    Returns
    -------
    tangent_body : MX (3,)  – tangent expressed in the desired body frame.
    """
    return rotation_inverse_expr(quat_desired, tangent_world)


def lag_contouring_decomposition(rho, tangent_body):
    """Decompose the translational error ρ into lag (‖) and contouring (⊥).

    Given ρ ∈ ℝ³ (translation error in the desired body frame) and the
    unit tangent t_body (also in the desired body frame):

        ρ_lag  = (t_body · ρ) · t_body     (projection along the path)
        ρ_cont = ρ − ρ_lag                  (orthogonal complement)

    Parameters
    ----------
    rho          : MX (3,)  – translational component of the se(3) error.
    tangent_body : MX (3,)  – unit tangent in the desired body frame.

    Returns
    -------
    rho_lag  : MX (3,)  – lag error vector  (‖ to the path tangent).
    rho_cont : MX (3,)  – contouring error vector  (⊥ to the path tangent).
    """
    proj_scalar = ca.dot(tangent_body, rho)            # scalar projection
    rho_lag  = proj_scalar * tangent_body              # lag  (along tangent)
    rho_cont = rho - rho_lag                           # cont (orthogonal)
    return rho_lag, rho_cont
