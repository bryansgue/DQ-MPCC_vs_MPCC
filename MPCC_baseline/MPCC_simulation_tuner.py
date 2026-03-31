"""
MPCC_simulation_tuner.py  –  Headless MPCC simulation for bilevel gain tuning.

This is a COPY of MPCC_baseline.py modified to:
  1. Accept a weights dict as input
  2. Use the tunable controller (gains as runtime parameters → no recompilation)
  3. Run WITHOUT real-time sleep (as fast as possible)
  4. Return a metrics dict instead of generating plots
  5. Be importable as a module: `from MPCC_simulation_tuner import run_simulation`

Original file: MPCC_baseline.py  (UNTOUCHED)
"""

import os
import sys
import ctypes
import numpy as np
import time as _time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# ── C-level stdout/stderr suppression (silences acados QP warnings) ──────────
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
    T_FINAL,
    FREC,
    T_PREDICTION,
    N_WAYPOINTS,
    S_MAX_MANUAL,
)
from experiment_config import (
    P0,
    Q0,
    V0,
    W0,
    THETA0,
    TRAJECTORY_T_FINAL,
    VTHETA_MAX,
    ATHETA_MIN, ATHETA_MAX,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    T_MAX, T_MIN,
    TAUX_MAX, TAUY_MAX, TAUZ_MAX,
    VTHETA_MIN,
    trayectoria,
)

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import (
    euler_to_quaternion,
    build_arc_length_parameterisation,
    build_terminally_extended_path,
    build_waypoints,
    mpcc_errors,
    rk4_step_mpcc,
    quat_error_numpy,
    quat_log_numpy,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi,
    create_tangent_interpolator_casadi,
    create_quat_interpolator_casadi,
)
from ocp.mpcc_controller_tuner import (
    build_mpcc_solver_tunable,
    weights_to_param_vector,
    DEFAULT_VTHETA_MAX,
    apply_input_bounds,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory — from experiment_config.py (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_trajectory_functions(t):
    """Support both trayectoria() and legacy trayectoria(t) signatures."""
    try:
        return trayectoria()
    except TypeError:
        return trayectoria(t)


# ═══════════════════════════════════════════════════════════════════════════════
#  Precomputed infrastructure (built ONCE, reused across all evaluations)
# ═══════════════════════════════════════════════════════════════════════════════

_INFRA = None          # will hold the precomputed trajectory + solver
_INFRA_WEIGHTS = None  # last weights used (to detect if p must be updated)


def _quat_interp_by_arc(s: float, s_wp: np.ndarray, quat_wp: np.ndarray) -> np.ndarray:
    """Piecewise-linear quaternion interpolation at arc-length s (NumPy).

    Uses SLERP approximation via normalised linear interpolation (NLERP),
    which is accurate enough for small waypoint spacing.
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
    t_final      = T_FINAL
    t_path_final = TRAJECTORY_T_FINAL
    frec         = FREC
    t_s          = 1 / frec
    t_prediction = T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))

    t = np.arange(0, t_final + t_s, t_s)
    t_path = np.linspace(0.0, t_path_final, len(t))
    N_sim = t.shape[0] - N_prediction

    xd, yd, zd, xd_p, yd_p, zd_p = _get_trajectory_functions(t_path)

    t_finer = np.linspace(0.0, t_path_final, len(t))
    arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, \
        s_max_full = build_arc_length_parameterisation(
            xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    # Optional manual arc-length limit from tuning_config
    if S_MAX_MANUAL is not None and S_MAX_MANUAL < s_max_full:
        s_max = float(S_MAX_MANUAL)
    else:
        s_max = s_max_full

    delta_s_terminal = 1.10 * VTHETA_MAX * t_prediction
    s_max_solver = s_max + delta_s_terminal
    pos_by_arc_solver, tang_by_arc_solver = build_terminally_extended_path(
        position_by_arc_length,
        tangent_by_arc_length,
        s_max,
        s_max_solver,
    )

    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        s_max_solver, N_WAYPOINTS, pos_by_arc_solver, tang_by_arc_solver,
        euler_to_quat_fn=euler_to_quaternion,
        reference_speed=ATTITUDE_REF_SPEED,
        max_tilt_deg=ATTITUDE_REF_MAX_TILT_DEG,
    )

    gamma_pos  = create_position_interpolator_casadi(s_wp, pos_wp)
    gamma_vel  = create_tangent_interpolator_casadi(s_wp, tang_wp)
    gamma_quat = create_quat_interpolator_casadi(s_wp, quat_wp)

    # Initial state — use the same experiment_config initial condition as production
    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)
    x0 = np.array([
        P0[0], P0[1], P0[2],
        V0[0], V0[1], V0[2],
        q0_normed[0], q0_normed[1], q0_normed[2], q0_normed[3],
        W0[0], W0[1], W0[2],
        THETA0, VTHETA_MIN,
    ], dtype=float)

    print("[TUNER] Building tunable MPCC solver (compiles ONCE) ...")
    solver, ocp, model, f = build_mpcc_solver_tunable(
        x0, N_prediction, t_prediction, s_max=s_max_solver,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )
    print("[TUNER] Solver ready.")

    return {
        't': t, 't_s': t_s, 't_final': t_final, 't_path_final': t_path_final,
        'N_sim': N_sim, 'N_prediction': N_prediction,
        's_max': s_max,
        's_max_solver': s_max_solver,
        'delta_s_terminal': delta_s_terminal,
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
    """Run a full MPCC simulation with the given weights.

    Parameters
    ----------
    weights : dict or None
        Cost weight overrides.  Keys:
            'Q_ec'    – [3] contouring error
            'Q_el'    – [3] lag error
            'Q_q'     – [3] quaternion error
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
        'mean_mpcc_cost' – mean MPCC stage cost (same formula as solver)
        'total_mpcc_cost'– total accumulated MPCC cost
        'N_steps'        – number of simulation steps
        'x'              – state trajectory   (14, N+1)
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

    # ── Update runtime numeric bounds (lbu + ubu) ────────────────────────
    apply_input_bounds(solver, N_prediction, s_max, vtheta_max=v_theta_max)

    # ── Extract weight matrices for numerical MPCC cost computation ──────
    w = weights or {}
    Q_ec_vec    = np.array(w.get('Q_ec',    p_vec[0:3]))
    Q_el_vec    = np.array(w.get('Q_el',    p_vec[3:6]))
    Q_q_vec     = np.array(w.get('Q_q',     p_vec[6:9]))
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
    mpcc_cost    = np.zeros(N_sim)          # per-step MPCC stage cost
    log_q_hist   = np.zeros((3, N_sim))
    omega_hist   = np.zeros((3, N_sim))
    sat_ratio    = np.zeros((5, N_sim))
    delta_u_hist = np.zeros((nu, N_sim))
    solve_status = np.zeros(N_sim, dtype=int)
    vpath_ratio_err = np.zeros(N_sim)
    vpath_ratio = np.zeros(N_sim)

    x[:, 0] = x0
    theta_hist[0, 0] = 0.0

    # ── FULL solver reset (prevents state leakage between trials) ──────
    for stage in range(N_prediction + 1):
        solver.set(stage, "x", x0)
        solver.set(stage, "p", p_vec)
    for stage in range(N_prediction):
        solver.set(stage, "u", np.zeros(nu))
    # Reset dual variables (multipliers) to prevent contamination
    try:
        for stage in range(N_prediction + 1):
            solver.set(stage, "lam", np.zeros(nx))
    except Exception:
        pass

    # ── Control loop (no sleep — fast as possible) ───────────────────────
    actual_steps = N_sim
    success = True
    consecutive_fails = 0
    MAX_CONSECUTIVE_FAILS = 50

    for k in range(N_sim):
        # Stop when path is complete
        if x[13, k] >= s_max - 0.01:
            actual_steps = k
            break

        # ── NaN / Inf guard ──────────────────────────────────────────
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
        solve_status[k] = status

        if status != 0 and status != 2:
            # status 2 = max iter reached but still usable
            consecutive_fails += 1
            if verbose:
                print(f"[k={k}] Solver failed with status {status}")
            if k > 0:
                u_control[:, k] = u_control[:, k-1]
            else:
                u_control[:, k] = np.zeros(nu)
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                if verbose:
                    print(f"[k={k}] {MAX_CONSECUTIVE_FAILS} consecutive failures — aborting")
                actual_steps = k + 1
                success = False
                break
        else:
            u_control[:, k] = solver.get(0, "u")
            consecutive_fails = 0

        vel_progres[0, k]  = x[14, k]
        theta_hist[0, k]   = x[13, k]

        # System evolution (RK4)
        x[:, k + 1] = rk4_step_mpcc(x[:, k], u_control[:, k], t_s, f)
        x[13, k + 1] = np.clip(x[13, k + 1], 0.0, s_max)
        x[14, k + 1] = np.clip(x[14, k + 1], VTHETA_MIN, v_theta_max)
        theta_hist[0, k + 1] = x[13, k + 1]

        if x[13, k + 1] >= s_max:
            x[13, k + 1] = s_max
            theta_hist[0, k + 1] = s_max
            actual_steps = k + 1
            break

        # Compute errors
        theta_k = min(x[13, k], s_max)
        sd_k    = pos_by_arc(theta_k)
        tang_k  = tang_by_arc(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(x[0:3, k], tang_k, sd_k)

        # ── Compute MPCC stage cost (same formula as solver) ─────────
        # Quaternion error
        qd_k     = _quat_interp_by_arc(theta_k, s_wp, quat_wp)
        q_err_k  = quat_error_numpy(x[6:10, k], qd_k)
        log_q_k  = quat_log_numpy(q_err_k)        # (3,)
        log_q_hist[:, k] = log_q_k

        ec_k     = e_contorno[:, k]                # (3,)
        el_k     = e_arrastre[:, k]                # (3,)
        omega_k  = x[10:13, k]                     # (3,)
        omega_hist[:, k] = omega_k
        u_k      = u_control[0:4, k]               # (4,)  T, τx, τy, τz
        vtheta_k = x[14, k]
        v_path_k = float(np.dot(x[3:6, k], tang_k))

        if k > 0:
            delta_u_hist[:, k] = u_control[:, k] - u_control[:, k - 1]

        sat_ratio[0, k] = float(
            abs(u_control[0, k] - T_MAX) <= 1e-3
            or abs(u_control[0, k] - T_MIN) <= 1e-3
        )
        sat_ratio[1, k] = float(abs(abs(u_control[1, k]) - TAUX_MAX) <= 1e-3)
        sat_ratio[2, k] = float(abs(abs(u_control[2, k]) - TAUY_MAX) <= 1e-3)
        sat_ratio[3, k] = float(abs(abs(u_control[3, k]) - TAUZ_MAX) <= 1e-3)
        sat_ratio[4, k] = float(
            abs(u_control[4, k] - ATHETA_MAX) <= 1e-3
            or abs(u_control[4, k] - ATHETA_MIN) <= 1e-3
        )
        vpath_ratio_err[k] = ((v_path_k - vtheta_k) ** 2) / (vtheta_k ** 2 + 1e-3)
        vpath_ratio[k] = v_path_k / (vtheta_k + 1e-3)

        # J_k = ec'Q_ec·ec + el'Q_el·el + logq'Q_q·logq
        #      + u'U·u + ω'Q_ω·ω - Q_s*v_θ
        mpcc_cost[k] = (
            np.dot(ec_k**2,    Q_ec_vec)
            + np.dot(el_k**2,    Q_el_vec)
            + np.dot(log_q_k**2, Q_q_vec)
            + np.dot(u_k**2,     U_mat_vec)
            + np.dot(omega_k**2, Q_omega_vec)
            - Q_s_val * vtheta_k
        )

        if verbose and k % 200 == 0:
            print(f"  [k={k:04d}]  θ={x[13,k]:7.2f}/{s_max:.0f}  "
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
    log_q_hist  = log_q_hist[:, :N]
    omega_hist  = omega_hist[:, :N]
    sat_ratio   = sat_ratio[:, :N]
    delta_u_hist = delta_u_hist[:, :N]
    solve_status = solve_status[:N]
    vpath_ratio_err = vpath_ratio_err[:N]
    vpath_ratio = vpath_ratio[:N]

    # ── Compute metrics ──────────────────────────────────────────────────
    rmse_c = np.sqrt(np.mean(np.sum(e_contorno**2, axis=0)))
    rmse_l = np.sqrt(np.mean(np.sum(e_arrastre**2, axis=0)))
    rmse_t = np.sqrt(np.mean(np.sum(e_total**2, axis=0)))

    # Control effort: normalised by hover thrust
    T_hover = 9.81
    u_norm = u_control.copy()
    u_norm[0, :] -= T_hover  # penalise deviation from hover
    mean_effort = np.mean(np.sum(u_norm**2, axis=0))

    # MPCC cost (same formula as the solver's cost function)
    total_mpcc_cost = np.sum(mpcc_cost) if N > 0 else 0.0
    mean_mpcc_cost  = np.mean(mpcc_cost) if N > 0 else 0.0

    path_completed = x[13, N] / s_max
    mean_vtheta    = np.mean(vel_progres) if N > 0 else 0.0
    mean_solver_ms = np.mean(t_solver) * 1e3 if N > 0 else 0.0
    rmse_att       = np.sqrt(np.mean(np.sum(log_q_hist**2, axis=0))) if N > 0 else 0.0
    mean_omega_sq  = np.mean(np.sum(omega_hist**2, axis=0)) if N > 0 else 0.0
    mean_du_sq     = np.mean(np.sum(delta_u_hist**2, axis=0)) if N > 0 else 0.0
    saturation_ratio = float(np.mean(sat_ratio)) if N > 0 else 0.0
    sat_ratio_per_channel = np.mean(sat_ratio, axis=1) if N > 0 else np.zeros(5)
    total_err_norm = np.sqrt(np.sum(e_total**2, axis=0)) if N > 0 else np.array([])
    peak_total_error = float(np.max(total_err_norm)) if N > 0 else 0.0
    p95_total_error  = float(np.percentile(total_err_norm, 95)) if N > 0 else 0.0
    solver_fail_ratio = float(np.mean((solve_status != 0) & (solve_status != 2))) if N > 0 else 0.0
    mean_vpath_error = float(np.mean(vpath_ratio_err)) if N > 0 else 0.0
    mean_vpath_ratio = float(np.mean(vpath_ratio)) if N > 0 else 0.0

    return {
        'rmse_contorno':  rmse_c,
        'rmse_lag':       rmse_l,
        'rmse_total':     rmse_t,
        'rmse_attitude':  rmse_att,
        'mean_effort':    mean_effort,
        'mean_omega_sq':  mean_omega_sq,
        'mean_du_sq':     mean_du_sq,
        'saturation_ratio': saturation_ratio,
        'sat_ratio_per_channel': sat_ratio_per_channel,
        'peak_total_error': peak_total_error,
        'p95_total_error':  p95_total_error,
        'path_completed': path_completed,
        'mean_vtheta':    mean_vtheta,
        'mean_vpath_error': mean_vpath_error,
        'mean_vpath_ratio': mean_vpath_ratio,
        'mean_solver_ms': mean_solver_ms,
        'solver_fail_ratio': solver_fail_ratio,
        'mean_mpcc_cost': mean_mpcc_cost,
        'total_mpcc_cost': total_mpcc_cost,
        'N_steps':        N,
        'x':              x,
        'u':              u_control,
        'e_contorno':     e_contorno,
        'e_lag':          e_arrastre,
        'theta_history':  theta_hist,
        'log_q':          log_q_hist,
        'omega':          omega_hist,
        'solve_status':   solve_status,
        'success':        success,
        's_max':          s_max,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("  MPCC Simulation — standalone test with DEFAULT gains")
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
    print(f"  Mean MPCC cost: {result['mean_mpcc_cost']:.4f}")
    print(f"  Total MPCC cost:{result['total_mpcc_cost']:.2f}")
    print(f"  Steps         : {result['N_steps']}")
    print(f"{'─'*60}")
