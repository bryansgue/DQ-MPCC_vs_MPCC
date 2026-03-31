"""mpcc_mujoco_tuner_runner.py — Inner runner for MPCC-rates bilevel tuning.

Architecture
────────────
  ROS2 + MuJoCo interface : initialised ONCE at first call (module singleton)
  Acados solver            : compiled ONCE, reused across all trials
  Between trials           : sim.reset() + PD hold to P0 + re-read state

This file is imported by tuning/mpcc_rate_mujoco_tuner.py (Optuna outer loop).
It can also be run standalone for a single evaluation:

    cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline
    python3 MPCC_baseline_rates/mpcc_mujoco_tuner_runner.py

Prerequisites
─────────────
  1. MuJoCo simulator running:
         mujoco_launch.sh scene:=motors
  2. path_cache.npz already built:
         python3 MPCC_baseline_rates/precompute_path.py
"""

import os
import sys
import time
import threading

import numpy as np

# ── ROS2 ─────────────────────────────────────────────────────────────────────
import rclpy

_WORKSPACE_ROOT   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT     = os.path.dirname(_WORKSPACE_ROOT)
_SHARED_MPCC_ROOT = os.path.join(_PROJECT_ROOT, "MPCC_baseline")
for _path in (_PROJECT_ROOT, _SHARED_MPCC_ROOT, _WORKSPACE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.numpy_utils import (
    euler_to_quaternion,
    mpcc_errors,
    quat_error_numpy,
    quat_log_numpy,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi as create_casadi_position_interpolator,
    create_quat_interpolator_casadi     as create_casadi_quat_interpolator,
    create_tangent_interpolator_casadi  as create_casadi_tangent_interpolator,
)
from MPCC_baseline_rates.config.experiment_config import (
    FREC, G, MASS_MUJOCO,
    P0, Q0, V0, W0, THETA0,
    T_MAX, W_MAX, VTHETA_MIN,
    T_PREDICTION,
)
from MPCC_baseline_rates.path_loader import load_path
from MPCC_baseline_rates.ocp.mpcc_controller_rate_mujoco_tuner import (
    build_mpcc_rate_solver_mujoco_tuner,
    weights_to_param_vector,
    N_PARAMS,
    DEFAULT_VTHETA_MAX,
    DEFAULT_T_MIN,
    DEFAULT_T_MAX,
    DEFAULT_W_MAX,
    DEFAULT_VTHETA_MIN,
)
from MPCC_baseline_rates.ros2_interface.mujoco_interface import (
    MujocoInterface,
    wait_for_connection,
)
from MPCC_baseline_rates.ros2_interface.reset_sim import SimControl
from MPCC_baseline_rates.tuning.tuning_config import (
    PD_SETTLE_DIST,
    PD_SETTLE_TIMEOUT,
    S_MAX_TUNER,
    T_FINAL_TUNER,
    CRASH_MIN_Z,
    CRASH_MAX_TILT_COS,
    CRASH_STILL_VEL,
    CRASH_STILL_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level singletons (ROS2 + MuJoCo + solver, built ONCE)
# ══════════════════════════════════════════════════════════════════════════════

_ros2_ready  = False
_muj: MujocoInterface | None = None
_sim: SimControl | None = None
_spin_thread: threading.Thread | None = None

_INFRA = None   # path + solver, built on first call


# ──────────────────────────────────────────────────────────────────────────────

def _init_ros2_once():
    """Initialise ROS2 and MuJoCo interface (idempotent)."""
    global _ros2_ready, _muj, _sim, _spin_thread

    if _ros2_ready:
        return

    rclpy.init()
    _muj = MujocoInterface(node_name='mpcc_rate_tuner_controller')
    _spin_thread = threading.Thread(
        target=rclpy.spin, args=(_muj,), daemon=True)
    _spin_thread.start()
    _sim = SimControl(node=_muj)

    if not wait_for_connection(_muj, timeout=15.0):
        raise RuntimeError(
            "[TUNER] No /quadrotor/odom received. "
            "Launch the simulator before starting the tuner.")
    _ros2_ready = True
    print("[TUNER] ROS2 + MuJoCo interface ready.")


def _build_infrastructure():
    """Load path and compile solver ONCE."""
    t_s          = 1.0 / FREC
    N_prediction = int(round(T_PREDICTION / t_s))

    print("[TUNER] Loading path cache ...")
    (s_wp, pos_wp, tang_wp, quat_wp,
     s_max_full, s_max_solver, s_max,
     arc_lengths, pos_ref,
     position_by_arc_length, tangent_by_arc_length) = load_path()
    print(f"[TUNER] s_max = {s_max:.3f} m  |  s_max_solver = {s_max_solver:.3f} m")

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)

    # Initial state for solver compilation (path start, not from MuJoCo)
    x0_build = np.concatenate([P0, V0, Q0, W0, [THETA0]])

    print("[TUNER] Building tunable MPCC rate solver (compiles ONCE) ...")
    solver, ocp, model, _ = build_mpcc_rate_solver_mujoco_tuner(
        x0_build, N_prediction, T_PREDICTION, s_max_solver,
        gamma_pos, gamma_vel, gamma_quat,
        use_cython=False,
    )
    print("[TUNER] Solver ready.")

    return {
        't_s':           t_s,
        'N_prediction':  N_prediction,
        's_max':         s_max,
        's_max_solver':  s_max_solver,
        's_wp':          s_wp,
        'quat_wp':       quat_wp,
        'pos_ref':       pos_ref,
        'arc_lengths':   arc_lengths,
        'position_by_arc_length':  position_by_arc_length,
        'tangent_by_arc_length':   tangent_by_arc_length,
        'solver':  solver,
        'model':   model,
        'nx':      model.x.size()[0],   # 14
        'nu':      model.u.size()[0],   # 5
    }


def _get_infra():
    """Return cached infrastructure (build on first call)."""
    global _INFRA
    if _INFRA is None:
        _INFRA = _build_infrastructure()
    return _INFRA


# ──────────────────────────────────────────────────────────────────────────────
#  PD convergence helper
# ──────────────────────────────────────────────────────────────────────────────

def _wait_for_pd_convergence():
    """Block until drone reaches P0 within PD_SETTLE_DIST, or timeout."""
    t0 = time.time()
    while time.time() - t0 < PD_SETTLE_TIMEOUT:
        pos, _, _, _ = _muj.get_state()
        dist = float(np.linalg.norm(pos - P0))
        if dist < PD_SETTLE_DIST:
            elapsed = time.time() - t0
            print(f"[PD]  Converged to P0 in {elapsed:.1f}s  "
                  f"(dist={dist:.3f} m < {PD_SETTLE_DIST} m)")
            return True
        time.sleep(0.1)
    pos, _, _, _ = _muj.get_state()
    dist = float(np.linalg.norm(pos - P0))
    print(f"[PD]  Timeout after {PD_SETTLE_TIMEOUT:.0f}s  (dist={dist:.3f} m — proceeding anyway)")
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Crash detection
# ──────────────────────────────────────────────────────────────────────────────

def _is_crashed(pos: np.ndarray, quat: np.ndarray) -> tuple[bool, str]:
    """Return (crashed: bool, reason: str).

    Crash indicators:
      1. Altitude below CRASH_MIN_Z  (drone on floor)
      2. Tilt > ~81°  (drone flipped or tipped over)
         R[2,2] = 1 - 2*(qx²+qy²)  < CRASH_MAX_TILT_COS
    """
    if pos[2] < CRASH_MIN_Z:
        return True, f"altitude z={pos[2]:.3f} m < {CRASH_MIN_Z} m"

    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    z_body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    if z_body_up < CRASH_MAX_TILT_COS:
        tilt_deg = float(np.degrees(np.arccos(np.clip(z_body_up, -1, 1))))
        return True, f"tilt={tilt_deg:.0f}° (R[2,2]={z_body_up:.3f} < {CRASH_MAX_TILT_COS})"

    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
#  Quaternion interpolation helper (NumPy, same as MiL tuner)
# ──────────────────────────────────────────────────────────────────────────────

def _quat_interp_by_arc(s: float,
                         s_wp: np.ndarray,
                         quat_wp: np.ndarray) -> np.ndarray:
    """NLERP quaternion interpolation at arc-length s."""
    s = np.clip(s, s_wp[0], s_wp[-1])
    idx = np.searchsorted(s_wp, s, side='right') - 1
    idx = np.clip(idx, 0, len(s_wp) - 2)
    alpha = (s - s_wp[idx]) / (s_wp[idx + 1] - s_wp[idx] + 1e-12)
    q_a = quat_wp[:, idx]
    q_b = quat_wp[:, idx + 1]
    if np.dot(q_a, q_b) < 0:
        q_b = -q_b
    q = (1 - alpha) * q_a + alpha * q_b
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else q


# ══════════════════════════════════════════════════════════════════════════════
#  Main simulation runner (called by Optuna objective for each trial)
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_mujoco(
    weights: dict | None = None,
    verbose: bool = False,
    vtheta_max: float | None = None,
) -> dict:
    """Run a full MPCC SiL simulation with the given weights and return metrics.

    The ROS2/MuJoCo infrastructure and solver are reused across calls.
    Between calls the simulation is reset via sim.reset() + PD hold.

    Parameters
    ----------
    weights : dict or None
        Cost weight overrides.  Keys:
            'Q_ec', 'Q_el', 'Q_q', 'U_mat', 'Q_omega', 'Q_s'
    verbose : bool
        Print per-step diagnostics every 50 steps.
    vtheta_max : float or None
        Override for max progress velocity.  None → uses DEFAULT_VTHETA_MAX.

    Returns
    -------
    dict  (same keys as MPCC_simulation_tuner.run_simulation)
    """
    _init_ros2_once()
    infra = _get_infra()

    t_s          = infra['t_s']
    N_prediction = infra['N_prediction']
    s_max        = infra['s_max']          # full production path length
    s_max_solver = infra['s_max_solver']
    solver       = infra['solver']
    nx           = infra['nx']
    nu           = infra['nu']
    s_wp         = infra['s_wp']
    quat_wp      = infra['quat_wp']
    pos_by_arc   = infra['position_by_arc_length']
    tang_by_arc  = infra['tangent_by_arc_length']

    # Truncate trial to S_MAX_TUNER so each evaluation is faster than production.
    # The solver and path cache cover the full path — we just stop earlier.
    s_max_trial = float(S_MAX_TUNER) if S_MAX_TUNER is not None else s_max
    s_max_trial = min(s_max_trial, s_max)

    v_theta_max = float(vtheta_max) if vtheta_max is not None else DEFAULT_VTHETA_MAX

    print(f"[TRIAL]  s_max_trial={s_max_trial:.1f} m  "
          f"(S_MAX_TUNER={S_MAX_TUNER} m  |  production s_max={s_max:.1f} m)  "
          f"|  v_θ_max={v_theta_max:.1f} m/s")

    # ── Build parameter vector ─────────────────────────────────────────────
    p_vec = weights_to_param_vector(weights)
    p_vec[14] = v_theta_max   # p[14] = vtheta_max  (N_PARAMS=15, Q_omega removed)

    # Extract weight arrays for numerical cost computation
    Q_ec_vec  = p_vec[0:3].copy()
    Q_el_vec  = p_vec[3:6].copy()
    Q_q_vec   = p_vec[6:9].copy()
    U_mat_vec = p_vec[9:13].copy()
    Q_s_val   = float(p_vec[13])

    # ── Update v_theta upper-bound constraint to match current vtheta_max ─
    ubu = np.array([DEFAULT_T_MAX, DEFAULT_W_MAX, DEFAULT_W_MAX,
                    DEFAULT_W_MAX, v_theta_max], dtype=np.double)
    for stage in range(N_prediction):
        solver.constraints_set(stage, "ubu", ubu)

    # ── Stop any lingering PD hold from the previous trial ────────────────
    # CRITICAL: must stop before reset — otherwise the old PD thread keeps
    # sending commands during the reset, causing a velocity spike on restart.
    _muj.stop_pd_hold()

    # ── Reset simulation ───────────────────────────────────────────────────
    print(f"[TRIAL]  Resetting simulation  (v_θ_max={v_theta_max:.1f} m/s) ...")
    _sim.reset()

    # ── Wait for odom to confirm drone is at rest after reset ──────────────
    # The reset service is synchronous, but physics still need a few cycles.
    # We poll odom until velocity is near zero (< 0.3 m/s), with 3s timeout.
    _t_rest = time.time()
    while time.time() - _t_rest < 3.0:
        _, vel_check, _, _ = _muj.get_state()
        if float(np.linalg.norm(vel_check)) < 0.3:
            break
        time.sleep(0.05)
    else:
        print("[TRIAL]  WARNING: drone velocity did not settle after reset — proceeding anyway")

    # ── PD hold: fly to P0 and wait until drone converges ─────────────────
    _muj.start_pd_hold(target=P0, mass=MASS_MUJOCO, g=G)
    _wait_for_pd_convergence()

    # ── Fresh state + solver initialisation ───────────────────────────────
    pos0, vel0, quat0, omega0 = _muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)
    x0 = np.concatenate([pos0, vel0, quat0, omega0, [THETA0]])

    u_hover = np.array([MASS_MUJOCO * G, 0., 0., 0., 0.], dtype=np.double)
    for stage in range(N_prediction + 1):
        solver.set(stage, "x", x0)
        solver.set(stage, "p", p_vec)
    for stage in range(N_prediction):
        solver.set(stage, "u", u_hover)

    # ── Stop PD hold: MPCC takes over ─────────────────────────────────────
    _muj.stop_pd_hold()

    # Re-read state immediately after PD stops (fresh before loop)
    pos0, vel0, quat0, omega0 = _muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)
    x0 = np.concatenate([pos0, vel0, quat0, omega0, [THETA0]])
    for stage in range(N_prediction + 1):
        solver.set(stage, "x", x0)

    # ── Allocate storage ──────────────────────────────────────────────────
    t_arr = np.arange(0, T_FINAL_TUNER + t_s, t_s)
    N_sim = t_arr.shape[0] - N_prediction

    x           = np.zeros((nx, N_sim + 1))
    u_control   = np.zeros((nu, N_sim))
    e_contorno  = np.zeros((3, N_sim))
    e_arrastre  = np.zeros((3, N_sim))
    vel_progres = np.zeros((1, N_sim))
    t_solver    = np.zeros(N_sim)
    mpcc_cost   = np.zeros(N_sim)

    x[:, 0] = x0

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop  (real-time: sleep to hit FREC)
    # ══════════════════════════════════════════════════════════════════════
    actual_steps      = N_sim
    success           = True
    crashed           = False
    consecutive_fails = 0
    MAX_FAILS         = 50
    still_counter     = 0    # counts consecutive steps with |v| < threshold

    for k in range(N_sim):
        tic = time.time()

        # ── Stop when tuner path segment is complete ──────────────────────
        if x[13, k] >= s_max_trial:
            actual_steps = k
            break

        # ── NaN / Inf guard ────────────────────────────────────────────────
        if not np.all(np.isfinite(x[:, k])):
            if verbose:
                print(f"[k={k}]  NaN/Inf in state — aborting trial")
            actual_steps = k
            success = False
            crashed = True
            break

        # ── Crash detection ────────────────────────────────────────────────
        has_crashed, crash_reason = _is_crashed(x[0:3, k], x[6:10, k])
        if has_crashed:
            print(f"[k={k:04d}]  CRASH detected: {crash_reason} — aborting trial")
            actual_steps = k
            success = False
            crashed = True
            break

        # ── Stuck detection: drone not moving for too long ─────────────────
        vel_norm = float(np.linalg.norm(x[3:6, k]))
        if vel_norm < CRASH_STILL_VEL and k > 20:
            still_counter += 1
            if still_counter >= CRASH_STILL_STEPS:
                print(f"[k={k:04d}]  STUCK: |v|={vel_norm:.3f} m/s for "
                      f"{CRASH_STILL_STEPS} steps — aborting trial")
                actual_steps = k
                success = False
                crashed = True
                break
        else:
            still_counter = 0

        # ── Fix initial state ──────────────────────────────────────────────
        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])

        # ── Per-stage braking: set p[14]=0 when θ_pred >= s_max ───────────
        dt_stage    = T_PREDICTION / N_prediction
        theta_k_cur = x[13, k]
        vtheta_prev = u_control[4, max(k - 1, 0)]
        for stage in range(N_prediction + 1):
            theta_pred = theta_k_cur + stage * dt_stage * vtheta_prev
            p_stage    = p_vec.copy()
            p_stage[14] = 0.0 if theta_pred >= s_max_trial else v_theta_max
            solver.set(stage, "p", p_stage)

        # ── Solve ──────────────────────────────────────────────────────────
        tic_solver  = time.time()
        status      = solver.solve()
        t_solver[k] = time.time() - tic_solver

        if status not in (0, 2):
            consecutive_fails += 1
            if k > 0:
                u_control[:, k] = u_control[:, k - 1]
            else:
                u_control[:, k] = u_hover
            if consecutive_fails >= MAX_FAILS:
                if verbose:
                    print(f"[k={k}]  {MAX_FAILS} consecutive solver failures — aborting")
                actual_steps = k + 1
                success = False
                break
        else:
            u_control[:, k] = solver.get(0, "u")
            consecutive_fails = 0

        vel_progres[0, k] = u_control[4, k]

        # ── Send command to MuJoCo ────────────────────────────────────────
        T_send = np.clip(u_control[0, k], 0.0, 80.0)
        _muj.send_cmd(T_send,
                      u_control[1, k], u_control[2, k], u_control[3, k])

        # ── Real-time rate control ─────────────────────────────────────────
        elapsed   = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

        # ── Read next state from MuJoCo ────────────────────────────────────
        pos_new, vel_new, quat_new, omega_new = _muj.get_state()
        quat_new /= (np.linalg.norm(quat_new) + 1e-12)

        x[0:3,   k + 1] = pos_new
        x[3:6,   k + 1] = vel_new
        x[6:10,  k + 1] = quat_new
        x[10:13, k + 1] = omega_new

        # ── Euler θ integration ────────────────────────────────────────────
        theta_new       = x[13, k] + u_control[4, k] * t_s
        x[13, k + 1]    = np.clip(theta_new, 0.0, s_max_solver)

        # ── MPCC errors ────────────────────────────────────────────────────
        theta_k = x[13, k]
        sd_k    = pos_by_arc(theta_k)
        tang_k  = tang_by_arc(theta_k)
        e_contorno[:, k], e_arrastre[:, k], _ = mpcc_errors(
            x[0:3, k], tang_k, sd_k)

        # ── Numerical MPCC stage cost (same formula as solver) ─────────────
        qd_k     = _quat_interp_by_arc(theta_k, s_wp, quat_wp)
        q_err_k  = quat_error_numpy(x[6:10, k], qd_k)
        log_q_k  = quat_log_numpy(q_err_k)               # (3,)

        ec_k       = e_contorno[:, k]
        el_k       = e_arrastre[:, k]
        thrust_err = u_control[0, k] - MASS_MUJOCO * G
        rates_k    = u_control[1:4, k]
        u_cost_k   = (U_mat_vec[0] * thrust_err**2
                      + np.dot(rates_k**2, U_mat_vec[1:4]))

        mpcc_cost[k] = (
            np.dot(ec_k**2,    Q_ec_vec)
            + np.dot(el_k**2,   Q_el_vec)
            + np.dot(log_q_k**2, Q_q_vec)
            + u_cost_k
            + Q_s_val * (v_theta_max - vel_progres[0, k])**2
        )

        # Progress print every 200 steps (~2 s at 100 Hz) — always visible
        if k % 200 == 0:
            e_c_norm = float(np.linalg.norm(e_contorno[:, k]))
            pct = x[13, k] / s_max_trial * 100.0
            bar = int(pct / 5)
            print(f"  θ={x[13, k]:5.1f}/{s_max_trial:.0f}m "
                  f"[{'█'*bar+'░'*(20-bar)}] {pct:4.0f}%  "
                  f"v_θ={vel_progres[0, k]:4.1f}m/s  "
                  f"ec={e_c_norm:.3f}m  "
                  f"z={x[2, k]:.2f}m  "
                  f"solver={t_solver[k]*1e3:.1f}ms",
                  flush=True)
        elif verbose and k % 50 == 0:
            e_c_norm = float(np.linalg.norm(e_contorno[:, k]))
            print(f"  [k={k:04d}]  θ={x[13, k]:7.2f}/{s_max_trial:.0f}  "
                  f"v_θ={vel_progres[0, k]:5.2f}  ec={e_c_norm:.3f}  "
                  f"solver={t_solver[k]*1e3:5.2f} ms  "
                  f"status={status}")

    # ── Safety hover after loop ────────────────────────────────────────────
    pos_final = _muj.get_state()[0]
    _muj.start_pd_hold(target=pos_final, mass=MASS_MUJOCO, g=G)

    # ── Trim arrays to actual length ───────────────────────────────────────
    N           = actual_steps
    x           = x[:, :N + 1]
    u_control   = u_control[:, :N]
    e_contorno  = e_contorno[:, :N]
    e_arrastre  = e_arrastre[:, :N]
    vel_progres = vel_progres[:, :N]
    t_solver    = t_solver[:N]
    mpcc_cost   = mpcc_cost[:N]

    # ── Compute metrics ────────────────────────────────────────────────────
    if N > 0:
        rmse_c = float(np.sqrt(np.mean(np.sum(e_contorno**2, axis=0))))
        rmse_l = float(np.sqrt(np.mean(np.sum(e_arrastre**2, axis=0))))
        rmse_t = float(np.sqrt(np.mean(
            np.sum((e_contorno + e_arrastre)**2, axis=0))))
        path_completed  = float(x[13, N] / s_max_trial)   # fraction of TUNER segment
        mean_vtheta     = float(np.mean(vel_progres))
        mean_solver_ms  = float(np.mean(t_solver)) * 1e3
        total_mpcc_cost = float(np.sum(mpcc_cost))
        mean_mpcc_cost  = float(np.mean(mpcc_cost))
    else:
        rmse_c = rmse_l = rmse_t = np.inf
        path_completed  = 0.0
        mean_vtheta     = 0.0
        mean_solver_ms  = 0.0
        total_mpcc_cost = 0.0
        mean_mpcc_cost  = 0.0

    return {
        'rmse_contorno':   rmse_c,
        'rmse_lag':        rmse_l,
        'rmse_total':      rmse_t,
        'path_completed':  path_completed,
        'mean_vtheta':     mean_vtheta,
        'mean_solver_ms':  mean_solver_ms,
        'mean_mpcc_cost':  mean_mpcc_cost,
        'total_mpcc_cost': total_mpcc_cost,
        'N_steps':         N,
        'x':               x,
        'u':               u_control,
        'e_contorno':      e_contorno,
        'e_lag':           e_arrastre,
        'theta_history':   x[13:14, :],   # (1, N+1)
        'success':         success,
        'crashed':         crashed,
        's_max':           s_max_trial,    # tuner segment length (not full production path)
        's_max_full':      s_max,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  MPCC Rate MuJoCo Runner — standalone test (default gains)")
    print("=" * 60)

    result = run_simulation_mujoco(weights=None, verbose=True)

    print(f"\n{'─'*60}")
    print(f"  RMSE contorno  : {result['rmse_contorno']:.4f} m")
    print(f"  RMSE lag       : {result['rmse_lag']:.4f} m")
    print(f"  Path completed : {result['path_completed']*100:.1f} %")
    print(f"  Mean v_θ       : {result['mean_vtheta']:.3f} m/s")
    print(f"  Mean MPCC cost : {result['mean_mpcc_cost']:.4f}")
    print(f"  Steps          : {result['N_steps']}")
    print(f"  Success        : {result['success']}")
    print(f"{'─'*60}")

    try:
        rclpy.shutdown()
    except Exception:
        pass
