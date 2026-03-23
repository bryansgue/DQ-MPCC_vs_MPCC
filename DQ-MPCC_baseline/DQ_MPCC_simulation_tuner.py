"""
DQ_MPCC_simulation_tuner.py  –  Headless DQ-MPCC simulation for bilevel gain tuning.

This is a COPY of DQ_MPCC_baseline.py modified to:
  1. Accept a weights dict as input
  2. Use the tunable controller (gains as runtime parameters → no recompilation)
  3. Run WITHOUT real-time sleep (as fast as possible)
  4. Return a metrics dict instead of generating plots
  5. Be importable as a module: `from DQ_MPCC_simulation_tuner import run_simulation`

Original file: DQ_MPCC_baseline.py  (UNTOUCHED)
"""

import os
import sys
import ctypes
import numpy as np
import time as _time

# ── C-level stdout/stderr suppression (silences acados QP warnings) ──────────
_libc = ctypes.CDLL(None)
_c_stdout = ctypes.c_void_p.in_dll(_libc, 'stdout')
_c_stderr = ctypes.c_void_p.in_dll(_libc, 'stderr')

def _suppress_c_output():
    """Redirect C-level stdout/stderr to /dev/null."""
    sys.stdout.flush(); sys.stderr.flush()
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _old_stdout_fd = os.dup(1)
    _old_stderr_fd = os.dup(2)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    return _old_stdout_fd, _old_stderr_fd

def _restore_c_output(old_stdout_fd, old_stderr_fd):
    """Restore C-level stdout/stderr."""
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)

# ── Shared tuning configuration ──────────────────────────────────────────────
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)
from tuning_config import (
    T_FINAL as _CFG_T_FINAL,
    FREC as _CFG_FREC,
    T_PREDICTION as _CFG_T_PREDICTION,
    N_WAYPOINTS as _CFG_N_WAYPOINTS,
    S_MAX_MANUAL as _CFG_S_MAX_MANUAL,
)
from experiment_config import trayectoria as _trayectoria

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import (
    euler_to_quaternion,
    build_arc_length_parameterisation,
    build_waypoints,
    mpcc_errors,
)
from utils.dq_numpy_utils import (
    dq_from_pose_numpy,
    dq_get_position_numpy,
    dq_get_quaternion_numpy,
    dq_normalize,
    dq_hemisphere_correction,
    quat_rotate_numpy,
    rk4_step_dq_mpcc,
    state15_to_standard13,
    dq_error_numpy,
    ln_dual_numpy,
    rotate_tangent_to_desired_frame_numpy,
    lag_contouring_decomposition_numpy,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi as create_casadi_position_interpolator,
    create_tangent_interpolator_casadi  as create_casadi_tangent_interpolator,
    create_quat_interpolator_casadi     as create_casadi_quat_interpolator,
)
from ocp.dq_mpcc_controller_tuner import (
    build_dq_mpcc_solver_tunable,
    weights_to_param_vector,
    N_PARAMS,
    DEFAULT_VTHETA_MAX,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory — from experiment_config.py (single source of truth)
#  _trayectoria is imported above
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Precomputed infrastructure (built ONCE, reused across all evaluations)
# ═══════════════════════════════════════════════════════════════════════════════

_INFRA = None          # will hold the precomputed trajectory + solver


def _quat_interp_by_arc(s: float, s_wp: np.ndarray, quat_wp: np.ndarray) -> np.ndarray:
    """Piecewise-linear quaternion interpolation at arc-length s (NumPy).

    Uses NLERP (normalised linear interpolation), accurate enough for
    small waypoint spacing.
    """
    s = np.clip(s, s_wp[0], s_wp[-1])
    idx = np.searchsorted(s_wp, s, side='right') - 1
    idx = np.clip(idx, 0, len(s_wp) - 2)
    alpha = (s - s_wp[idx]) / (s_wp[idx + 1] - s_wp[idx] + 1e-12)
    q = (1 - alpha) * quat_wp[:, idx] + alpha * quat_wp[:, idx + 1]
    # Ensure shortest-path interpolation
    if np.dot(quat_wp[:, idx], quat_wp[:, idx + 1]) < 0:
        q = (1 - alpha) * quat_wp[:, idx] - alpha * quat_wp[:, idx + 1]
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else q


def _build_infrastructure():
    """Build trajectory, interpolation and solver ONCE.

    All timing / trajectory parameters come from tuning_config.py.
    """
    t_final      = _CFG_T_FINAL
    frec         = _CFG_FREC
    t_s          = 1 / frec
    t_prediction = _CFG_T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))

    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    xd, yd, zd, xd_p, yd_p, zd_p = _trayectoria(t)

    t_finer = np.linspace(0, t_final, len(t))
    arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, \
        s_max_full = build_arc_length_parameterisation(
            xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    # Optional manual arc-length limit from tuning_config
    if _CFG_S_MAX_MANUAL is not None and _CFG_S_MAX_MANUAL < s_max_full:
        s_max = float(_CFG_S_MAX_MANUAL)
    else:
        s_max = s_max_full

    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        s_max, _CFG_N_WAYPOINTS, position_by_arc_length, tangent_by_arc_length,
        euler_to_quat_fn=euler_to_quaternion,
    )

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)

    # Initial state — start ON the path with identity quaternion
    p0 = position_by_arc_length(0.0)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    dq0 = dq_from_pose_numpy(q0, p0)

    x0 = np.zeros(15)
    x0[0:8] = dq0              # dual quaternion
    x0[8:11] = [0, 0, 0]      # angular velocity (body)
    x0[11:14] = [0, 0, 0]     # linear velocity  (body)
    x0[14] = 0.0               # arc-length progress

    print("[DQ-TUNER] Building tunable DQ-MPCC solver (compiles ONCE) ...")
    solver, ocp, model, f = build_dq_mpcc_solver_tunable(
        x0, N_prediction, t_prediction, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )
    print("[DQ-TUNER] Solver ready.")

    return {
        't': t, 't_s': t_s, 't_final': t_final,
        'N_sim': N_sim, 'N_prediction': N_prediction,
        's_max': s_max,
        'position_by_arc_length': position_by_arc_length,
        'tangent_by_arc_length': tangent_by_arc_length,
        's_wp': s_wp, 'quat_wp': quat_wp,
        'solver': solver, 'model': model, 'f': f,
        'x0': x0,
    }


def _get_infra():
    """Return cached infrastructure (build on first call)."""
    global _INFRA
    if _INFRA is None:
        _INFRA = _build_infrastructure()
    return _INFRA


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(weights: dict | None = None, verbose: bool = False,
                   vtheta_max: float | None = None) -> dict:
    """Run a full DQ-MPCC simulation with the given weights.

    Parameters
    ----------
    weights : dict or None
        Cost weight overrides.  Keys:
            'Q_phi'   – [3] orientation error  (so(3))
            'Q_ec'    – [3] contouring error
            'Q_el'    – [3] lag error
            'U_mat'   – [4] control effort
            'Q_omega' – [3] angular velocity
            'Q_s'     – float  progress speed
    verbose : bool
        If True, print per-step info.
    vtheta_max : float or None
        Override for the maximum progress velocity v_θ used in both the
        cost function  Q_s·(v_max − v_θ)²  and the control constraint
        u[4] ≤ v_max.  If None, DEFAULT_VTHETA_MAX is used.

    Returns
    -------
    dict with keys:
        'rmse_contorno'  – RMSE of contouring error  (scalar)
        'rmse_lag'       – RMSE of lag error          (scalar)
        'rmse_total'     – RMSE of total position error
        'mean_effort'    – mean control effort ‖u‖²   (scalar)
        'path_completed' – fraction of path completed  [0, 1]
        'mean_vtheta'    – mean progress speed v_θ
        'mean_solver_ms' – mean solver time [ms]
        'mean_mpcc_cost' – mean DQ-MPCC stage cost (same formula as solver)
        'total_mpcc_cost'– total accumulated DQ-MPCC cost
        'N_steps'        – number of simulation steps
        'x'              – state trajectory   (15, N+1)
        'u'              – control trajectory (5, N)
        'e_contorno'     – contouring error   (3, N)
        'e_lag'          – lag error           (3, N)
        'theta_history'  – arc-length state   (1, N+1)
        'success'        – bool: solver didn't crash
    """
    infra = _get_infra()

    t_s            = infra['t_s']
    N_sim          = infra['N_sim']
    N_prediction   = infra['N_prediction']
    s_max          = infra['s_max']
    pos_by_arc     = infra['position_by_arc_length']
    tang_by_arc    = infra['tangent_by_arc_length']
    s_wp           = infra['s_wp']
    quat_wp        = infra['quat_wp']
    solver         = infra['solver']
    model          = infra['model']
    f              = infra['f']
    x0             = infra['x0'].copy()

    nx = model.x.size()[0]   # 15
    nu = model.u.size()[0]   # 5

    # ── Resolve effective v_theta_max ────────────────────────────────────
    v_theta_max = float(vtheta_max) if vtheta_max is not None else DEFAULT_VTHETA_MAX

    # ── Set runtime parameters on ALL stages ─────────────────────────────
    p_vec = weights_to_param_vector(weights)
    p_vec[17] = v_theta_max        # inject v_theta_max into parameter vector
    for stage in range(N_prediction + 1):
        solver.set(stage, "p", p_vec)

    # ── Update v_theta upper bound constraint to match ───────────────────
    from ocp.dq_mpcc_controller_tuner import (
        DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX, DEFAULT_TAUZ_MAX,
    )
    ubu = np.array([DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX,
                     DEFAULT_TAUZ_MAX, v_theta_max])
    for stage in range(N_prediction):
        solver.constraints_set(stage, "ubu", ubu)

    # ── Extract weight values for numerical DQ-MPCC cost computation ─────
    w = weights or {}
    Q_phi_vec   = np.array(w.get('Q_phi',   p_vec[0:3]))
    Q_ec_vec    = np.array(w.get('Q_ec',    p_vec[3:6]))
    Q_el_vec    = np.array(w.get('Q_el',    p_vec[6:9]))
    U_mat_vec   = np.array(w.get('U_mat',   p_vec[9:13]))
    Q_omega_vec = np.array(w.get('Q_omega', p_vec[13:16]))
    Q_s_val     = float(w.get('Q_s',        p_vec[16]))

    # ── Storage ──────────────────────────────────────────────────────────
    x            = np.zeros((nx, N_sim + 1))
    u_control    = np.zeros((nu, N_sim))
    e_contorno   = np.zeros((3, N_sim))
    e_arrastre   = np.zeros((3, N_sim))
    e_total      = np.zeros((3, N_sim))
    vel_progres  = np.zeros((1, N_sim))
    theta_hist   = np.zeros((1, N_sim + 1))
    t_solver     = np.zeros(N_sim)
    mpcc_cost    = np.zeros(N_sim)          # per-step DQ-MPCC stage cost

    x[:, 0] = x0
    theta_hist[0, 0] = 0.0

    # ── FULL solver reset (critical: prevents state leakage between trials) ─
    for stage in range(N_prediction + 1):
        solver.set(stage, "x", x0)
        solver.set(stage, "p", p_vec)
    for stage in range(N_prediction):
        solver.set(stage, "u", np.zeros(nu))
    # Reset dual variables (multipliers) to prevent contamination
    try:
        for stage in range(N_prediction + 1):
            solver.set(stage, "lam", np.zeros(nx))
        for stage in range(N_prediction):
            solver.set(stage, "sl", np.zeros(1))
            solver.set(stage, "su", np.zeros(1))
    except Exception:
        pass  # some solver interfaces may not expose these

    # Keep track of previous DQ for hemisphere correction
    dq_prev = x[0:8, 0].copy()

    # ── Control loop (no sleep — fast as possible) ───────────────────────
    actual_steps = N_sim
    success = True
    consecutive_fails = 0
    MAX_CONSECUTIVE_FAILS = 50   # abort early if solver is stuck

    for k in range(N_sim):
        # Stop when path is complete
        if x[14, k] >= s_max - 0.01:
            actual_steps = k
            break

        # ── NaN / Inf guard: abort if state has diverged ─────────────
        if not np.all(np.isfinite(x[:, k])):
            if verbose:
                print(f"[k={k}] NaN/Inf in state — aborting trial")
            actual_steps = k
            success = False
            break

        # Set initial state
        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])

        # Solve (suppress C-level QP warnings)
        tic = _time.time()
        _old_out, _old_err = _suppress_c_output()
        try:
            status = solver.solve()
        finally:
            _restore_c_output(_old_out, _old_err)
        t_solver[k] = _time.time() - tic

        if status != 0 and status != 2:
            # status 2 = max iter reached but still usable
            consecutive_fails += 1
            if verbose:
                print(f"[k={k}] Solver failed with status {status}")
            if k > 0:
                u_control[:, k] = u_control[:, k-1]
            else:
                u_control[:, k] = np.zeros(nu)
            # Abort early if solver is hopelessly stuck
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                if verbose:
                    print(f"[k={k}] {MAX_CONSECUTIVE_FAILS} consecutive failures — aborting")
                actual_steps = k + 1
                success = False
                break
        else:
            u_control[:, k] = solver.get(0, "u")
            consecutive_fails = 0  # reset on success

        vel_progres[0, k]  = u_control[4, k]
        theta_hist[0, k]   = x[14, k]

        # System evolution (RK4, 15 states)
        x[:, k + 1] = rk4_step_dq_mpcc(x[:, k], u_control[:, k], t_s, f)

        # Post-integration normalization
        x[0:8, k + 1] = dq_normalize(x[0:8, k + 1])
        x[0:8, k + 1] = dq_hemisphere_correction(x[0:8, k + 1], dq_prev)
        dq_prev = x[0:8, k + 1].copy()

        # Clamp θ
        x[14, k + 1] = np.clip(x[14, k + 1], 0.0, s_max)
        theta_hist[0, k + 1] = x[14, k + 1]

        # Compute classic MPCC errors (for metrics)
        theta_k = x[14, k]
        sd_k    = pos_by_arc(theta_k)
        tang_k  = tang_by_arc(theta_k)
        pos_k   = dq_get_position_numpy(x[0:8, k])
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(pos_k, tang_k, sd_k)

        # ── Compute DQ-MPCC stage cost (same formula as solver) ──────
        # Desired quaternion at θ_k
        qd_k = _quat_interp_by_arc(theta_k, s_wp, quat_wp)
        sd_k_pos = pos_by_arc(theta_k)

        # Build desired DQ
        dq_desired_k = dq_from_pose_numpy(qd_k, sd_k_pos)

        # DQ error + Log
        dq_err_k = dq_error_numpy(dq_desired_k, x[0:8, k])
        log_err_k = ln_dual_numpy(dq_err_k)    # [φ(3); ρ(3)]

        phi_k = log_err_k[0:3]
        rho_k = log_err_k[3:6]

        # Lag-contouring decomposition in the desired body frame
        tang_body_k = rotate_tangent_to_desired_frame_numpy(tang_k, qd_k)
        rho_lag_k, rho_cont_k = lag_contouring_decomposition_numpy(rho_k, tang_body_k)

        omega_k  = x[8:11, k]                     # angular velocity
        u_k      = u_control[0:4, k]              # T, τx, τy, τz
        vtheta_k = u_control[4, k]

        # J_k = φ'Q_φ·φ + ρ_cont'Q_ec·ρ_cont + ρ_lag'Q_el·ρ_lag
        #      + u'U·u + ω'Q_ω·ω + Q_s*(v_max - v_θ)²
        mpcc_cost[k] = (
            np.dot(phi_k**2,      Q_phi_vec)
            + np.dot(rho_cont_k**2, Q_ec_vec)
            + np.dot(rho_lag_k**2,  Q_el_vec)
            + np.dot(u_k**2,        U_mat_vec)
            + np.dot(omega_k**2,    Q_omega_vec)
            + Q_s_val * (v_theta_max - vtheta_k)**2
        )

        if verbose and k % 200 == 0:
            print(f"  [k={k:04d}]  θ={x[14,k]:7.2f}/{s_max:.0f}  "
                  f"v_θ={vel_progres[0,k]:5.2f}  "
                  f"solver={t_solver[k]*1e3:5.2f} ms")

    # ── Trim to actual length ────────────────────────────────────────────
    N = actual_steps
    x           = x[:, :N + 1]
    u_control   = u_control[:, :N]
    e_contorno  = e_contorno[:, :N]
    e_arrastre  = e_arrastre[:, :N]
    e_total     = e_total[:, :N]
    vel_progres = vel_progres[:, :N]
    theta_hist  = theta_hist[:, :N + 1]
    t_solver    = t_solver[:N]
    mpcc_cost   = mpcc_cost[:N]

    # ── Compute metrics ──────────────────────────────────────────────────
    rmse_c = np.sqrt(np.mean(np.sum(e_contorno**2, axis=0))) if N > 0 else 0.0
    rmse_l = np.sqrt(np.mean(np.sum(e_arrastre**2, axis=0))) if N > 0 else 0.0
    rmse_t = np.sqrt(np.mean(np.sum(e_total**2, axis=0)))    if N > 0 else 0.0

    # Control effort: normalised by hover thrust
    T_hover = 9.81  # MASS=1.0 * G
    u_norm = u_control.copy()
    if N > 0:
        u_norm[0, :] -= T_hover
    mean_effort = np.mean(np.sum(u_norm**2, axis=0)) if N > 0 else 0.0

    # DQ-MPCC cost (same formula as the solver's cost function)
    total_mpcc_cost = np.sum(mpcc_cost) if N > 0 else 0.0
    mean_mpcc_cost  = np.mean(mpcc_cost) if N > 0 else 0.0

    path_completed = x[14, N] / s_max if N > 0 else 0.0
    mean_vtheta    = np.mean(vel_progres) if N > 0 else 0.0
    mean_solver_ms = np.mean(t_solver) * 1e3 if N > 0 else 0.0

    return {
        'rmse_contorno':  rmse_c,
        'rmse_lag':       rmse_l,
        'rmse_total':     rmse_t,
        'mean_effort':    mean_effort,
        'path_completed': path_completed,
        'mean_vtheta':    mean_vtheta,
        'mean_solver_ms': mean_solver_ms,
        'mean_mpcc_cost': mean_mpcc_cost,
        'total_mpcc_cost': total_mpcc_cost,
        'N_steps':        N,
        'x':              x,
        'u':              u_control,
        'e_contorno':     e_contorno,
        'e_lag':          e_arrastre,
        'theta_history':  theta_hist,
        'success':        success,
        's_max':          s_max,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("  DQ-MPCC Simulation — standalone test with DEFAULT gains")
    print("="*60)

    result = run_simulation(weights=None, verbose=True)

    print(f"\n{'─'*60}")
    print(f"  RMSE contorno : {result['rmse_contorno']:.4f} m")
    print(f"  RMSE lag      : {result['rmse_lag']:.4f} m")
    print(f"  RMSE total    : {result['rmse_total']:.4f} m")
    print(f"  Mean effort   : {result['mean_effort']:.4f}")
    print(f"  Path completed: {result['path_completed']*100:.1f} %")
    print(f"  Mean v_θ      : {result['mean_vtheta']:.3f} m/s")
    print(f"  Mean solver   : {result['mean_solver_ms']:.2f} ms")
    print(f"  Mean DQ-MPCC cost: {result['mean_mpcc_cost']:.4f}")
    print(f"  Total DQ-MPCC cost:{result['total_mpcc_cost']:.2f}")
    print(f"  Steps         : {result['N_steps']}")
    print(f"{'─'*60}")
