#!/usr/bin/env python3
"""
run_experiment3.py — Experiment 3: Monte Carlo with Random Poses.

Runs N_RUNS simulations at a FIXED velocity with random initial
position and orientation.  For each run it records:

  • RMSE position, orientation, contouring (ec), lag (el)
  • Lap time, mean progress velocity
  • Control effort  (∑‖Δu‖²)
  • Solver timing   (mean, max, std per step)
  • Initial distance to path  (measures "difficulty")
  • Success / failure flag

Saves everything to
    results/experiment3/data/montecarlo_data.mat

Usage:
    python results/experiment3/scripts/run_experiment3.py
"""

import os, sys, time, math
import numpy as np
from scipy.io import savemat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))  # workspace root
_DQ_DIR     = os.path.join(_ROOT, "DQ-MPCC_baseline")
_MPCC_DIR   = os.path.join(_ROOT, "MPCC_baseline")

# ── Import experiment configs ────────────────────────────────────────────────
sys.path.insert(0, _ROOT)
from config.experiment_config import (
    P0, Q0, THETA0, VTHETA_MIN, trayectoria,
    T_MAX, TAUX_MAX, TAUY_MAX, TAUZ_MAX,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
)
from config.tuning_registry import get_active_weight_summary, flatten_weight_summary
from config.result_paths import experiment_dirs
from config.montecarlo_config import (
    VELOCITY, N_RUNS, SIGMA_P, SIGMA_Q, SEED,
    MAX_POS_OFFSET_NORM,
    S_MAX, T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS,
    COMPLETION_RATIO, POS_DIVERGENCE_THRESHOLD,
)

_EXP3_DIRS  = experiment_dirs("experiment3")
_OUT_DIR    = str(_EXP3_DIRS["data"])


# ═════════════════════════════════════════════════════════════════════════════
#  Perturbation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _perturb_position(p0, sigma, max_norm, rng):
    delta = rng.normal(0, sigma, size=3)
    n = np.linalg.norm(delta)
    if max_norm is not None and n > max_norm and n > 1e-12:
        delta = delta * (max_norm / n)
    return p0 + delta


def _perturb_quaternion(q0, sigma, rng):
    """Yaw-only perturbation around the nominal hover attitude."""
    angle = rng.uniform(-sigma, sigma)
    ha = angle / 2.0
    dq = np.array([math.cos(ha), 0.0, 0.0, math.sin(ha)])
    # Hamilton product  q_out = dq ⊗ q0
    w0, x0, y0, z0 = dq
    w1, x1, y1, z1 = q0
    out = np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])
    out /= np.linalg.norm(out) + 1e-12
    return out


def _quat_log_norm(q):
    """||Log(q)|| for unit quaternion [qw,qx,qy,qz]."""
    qw = q[0]
    qv = q[1:4]
    s = np.sign(qw) if qw != 0 else 1.0
    qw, qv = qw * s, qv * s
    return abs(2.0 * math.atan2(np.linalg.norm(qv), qw))


def _quat_mult(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ═════════════════════════════════════════════════════════════════════════════
#  Trajectory builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_trajectory():
    t_s = 1.0 / FREC
    t = np.arange(0, T_FINAL + t_s, t_s)
    t_finer = np.linspace(0, T_FINAL, len(t))
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
    return t, t_finer, xd, yd, zd, xd_p, yd_p, zd_p


# ═════════════════════════════════════════════════════════════════════════════
#  MPCC Baseline — single run  (with timing + control effort)
# ═════════════════════════════════════════════════════════════════════════════

def _run_mpcc_baseline(p0, q0, vtheta_max, solver, f_sys,
                       position_by_arc, tangent_by_arc, s_max,
                       s_wp, quat_wp, quat_interp_fn, mpcc_errors_fn,
                       rk4_fn, N_sim, t_s, N_prediction):
    """Run one MPCC simulation.

    Returns dict with all metrics, or None on failure.
    """
    nx, nu = 15, 5
    x = np.zeros((nx, N_sim + 1))
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    x[:, 0] = [p0[0], p0[1], p0[2],
               0, 0, 0,
               q0n[0], q0n[1], q0n[2], q0n[3],
               0, 0, 0,
               THETA0, VTHETA_MIN]

    # warm start
    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    pos_errs, ori_errs, ec_errs, el_errs = [], [], [], []
    solver_times = []
    u_prev = np.zeros(nu)
    control_effort = 0.0
    diverged = False
    k_final = 0

    for k in range(N_sim):
        if x[13, k] >= s_max - 0.01:
            k_final = k
            break

        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])

        t0_solve = time.perf_counter()
        status = solver.solve()
        dt_solve = time.perf_counter() - t0_solve
        solver_times.append(dt_solve)
        if status != 0:
            diverged = True
            break

        u_k = solver.get(0, "u")
        # Control effort (Δu squared)
        du = u_k - u_prev
        control_effort += np.dot(du, du)
        u_prev = u_k.copy()

        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[13, k+1] = np.clip(x[13, k+1], 0.0, s_max)

        # position error
        theta_k = x[13, k]
        sd_k = position_by_arc(theta_k)
        e_t = sd_k - x[0:3, k]
        e_pos = np.linalg.norm(e_t)
        pos_errs.append(e_pos)

        # lag / contouring
        tang_k = tangent_by_arc(theta_k)
        tang_k = tang_k / (np.linalg.norm(tang_k) + 1e-12)
        e_lag_scalar = np.dot(tang_k, e_t)
        e_lag_vec = e_lag_scalar * tang_k
        e_cont_vec = e_t - e_lag_vec
        ec_errs.append(np.linalg.norm(e_cont_vec))
        el_errs.append(abs(e_lag_scalar))

        # orientation error against the same shared quaternion reference
        # used by both controllers in the experiment.
        qd = np.array(quat_interp_fn(min(theta_k, s_max), s_wp, quat_wp))
        q_err = _quat_mult(_quat_conj(qd), x[6:10, k])
        ori_errs.append(_quat_log_norm(q_err))

        if e_pos > POS_DIVERGENCE_THRESHOLD:
            diverged = True
            break
        k_final = k

    end_idx = min(k_final + 1, N_sim) if len(pos_errs) > 0 else 0
    progress_k = x[13, k_final] if len(pos_errs) > 0 else 0.0
    progress_k1 = (
        x[13, k_final + 1]
        if (len(pos_errs) > 0 and (k_final + 1) <= N_sim and x[13, k_final + 1] > progress_k)
        else progress_k
    )
    end_idx = k_final + 1 if progress_k1 > progress_k else k_final
    theta_final = progress_k1 if len(pos_errs) > 0 else 0.0
    completed = (
        (theta_final >= COMPLETION_RATIO * s_max)
        and not diverged
    )

    if len(pos_errs) < 10 or not completed:
        return None

    t_lap = k_final * t_s
    return dict(
        rmse_pos    = math.sqrt(np.mean(np.array(pos_errs)**2)),
        rmse_ori    = math.sqrt(np.mean(np.array(ori_errs)**2)),
        rmse_ec     = math.sqrt(np.mean(np.array(ec_errs)**2)),
        rmse_el     = math.sqrt(np.mean(np.array(el_errs)**2)),
        theta_final = theta_final,
        t_lap       = t_lap,
        mean_vtheta = theta_final / t_lap if t_lap > 0 else 0.0,
        ctrl_effort = control_effort,
        solve_mean  = np.mean(solver_times),
        solve_max   = np.max(solver_times),
        solve_std   = np.std(solver_times),
        xyz         = x[0:3, :end_idx+1].copy(),   # (3, K) trajectory
    )


# ═════════════════════════════════════════════════════════════════════════════
#  DQ-MPCC — single run  (with timing + control effort)
# ═════════════════════════════════════════════════════════════════════════════

def _run_dq_mpcc(p0, q0, vtheta_max, solver, f_sys,
                 position_by_arc, tangent_by_arc, s_max,
                 s_wp, quat_wp, quat_interp_fn,
                 dq_from_pose_fn, dq_get_pos_fn, dq_get_quat_fn,
                 dq_normalize_fn, dq_hemi_fn, rk4_fn, quat_rotate_fn,
                 N_sim, t_s, N_prediction):
    """Run one DQ-MPCC simulation.

    Returns dict with all metrics, or None on failure.
    """
    nx, nu = 16, 5
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    dq0 = dq_from_pose_fn(q0n, p0)

    x = np.zeros((nx, N_sim + 1))
    x[0:8, 0] = dq0
    x[14, 0] = THETA0
    x[15, 0] = VTHETA_MIN

    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    dq_prev = x[0:8, 0].copy()
    pos_errs, ori_errs, ec_errs, el_errs = [], [], [], []
    pos_log = [np.array(dq_get_pos_fn(x[0:8, 0])).flatten()]
    solver_times = []
    u_prev = np.zeros(nu)
    control_effort = 0.0
    diverged = False
    k_final = 0

    for k in range(N_sim):
        if x[14, k] >= s_max - 0.01:
            k_final = k
            break

        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])

        t0_solve = time.perf_counter()
        status = solver.solve()
        dt_solve = time.perf_counter() - t0_solve
        solver_times.append(dt_solve)
        if status != 0:
            diverged = True
            break

        u_k = solver.get(0, "u")
        du = u_k - u_prev
        control_effort += np.dot(du, du)
        u_prev = u_k.copy()

        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[0:8, k+1] = dq_normalize_fn(x[0:8, k+1])
        x[0:8, k+1] = dq_hemi_fn(x[0:8, k+1], dq_prev)
        dq_prev = x[0:8, k+1].copy()
        x[14, k+1] = np.clip(x[14, k+1], 0.0, s_max)

        # position error
        theta_k = x[14, k]
        sd_k = position_by_arc(theta_k)
        pos_k = np.array(dq_get_pos_fn(x[0:8, k])).flatten()
        pos_log.append(pos_k)
        e_t = sd_k - pos_k
        e_pos = np.linalg.norm(e_t)
        pos_errs.append(e_pos)

        # lag / contouring
        tang_k = tangent_by_arc(theta_k)
        tang_k = tang_k / (np.linalg.norm(tang_k) + 1e-12)
        e_lag_scalar = np.dot(tang_k, e_t)
        e_lag_vec = e_lag_scalar * tang_k
        e_cont_vec = e_t - e_lag_vec
        ec_errs.append(np.linalg.norm(e_cont_vec))
        el_errs.append(abs(e_lag_scalar))

        # orientation error against the same shared quaternion reference
        # used by both controllers in the experiment.
        qd = np.array(quat_interp_fn(min(theta_k, s_max), s_wp, quat_wp))
        quat_k = dq_get_quat_fn(x[0:8, k])
        q_err = _quat_mult(_quat_conj(qd), quat_k)
        ori_errs.append(_quat_log_norm(q_err))

        if e_pos > POS_DIVERGENCE_THRESHOLD:
            diverged = True
            break
        k_final = k

    end_idx = min(k_final + 1, N_sim) if len(pos_errs) > 0 else 0
    progress_k = x[14, k_final] if len(pos_errs) > 0 else 0.0
    progress_k1 = (
        x[14, k_final + 1]
        if (len(pos_errs) > 0 and (k_final + 1) <= N_sim and x[14, k_final + 1] > progress_k)
        else progress_k
    )
    end_idx = k_final + 1 if progress_k1 > progress_k else k_final
    theta_final = progress_k1 if len(pos_errs) > 0 else 0.0
    completed = (
        (theta_final >= COMPLETION_RATIO * s_max)
        and not diverged
    )

    if len(pos_errs) < 10 or not completed:
        return None

    t_lap = k_final * t_s
    return dict(
        rmse_pos    = math.sqrt(np.mean(np.array(pos_errs)**2)),
        rmse_ori    = math.sqrt(np.mean(np.array(ori_errs)**2)),
        rmse_ec     = math.sqrt(np.mean(np.array(ec_errs)**2)),
        rmse_el     = math.sqrt(np.mean(np.array(el_errs)**2)),
        theta_final = theta_final,
        t_lap       = t_lap,
        mean_vtheta = theta_final / t_lap if t_lap > 0 else 0.0,
        ctrl_effort = control_effort,
        solve_mean  = np.mean(solver_times),
        solve_max   = np.max(solver_times),
        solve_std   = np.std(solver_times),
        xyz         = np.array(pos_log[:end_idx+1]).T,   # (3, K) trajectory
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)
    t_s = 1.0 / FREC
    N_prediction = int(round(T_PREDICTION / t_s))
    s_max = S_MAX
    vtheta_max = VELOCITY

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 3 — Monte Carlo with Random Poses")
    print(f"  Velocity  = {VELOCITY} m/s")
    print(f"  N_runs    = {N_RUNS}")
    print(f"  σ_p       = {SIGMA_P} m   (position perturbation)")
    print(f"  σ_q       = {SIGMA_Q} rad (orientation perturbation)")
    print(f"  s_max     = {s_max} m")
    print(f"  T_final   = {T_FINAL} s")
    print(f"{'='*70}")

    # ── Build trajectory ─────────────────────────────────────────────────
    t_vec, t_finer, xd, yd, zd, xd_p, yd_p, zd_p = _build_trajectory()
    N_sim_max = t_vec.shape[0] - N_prediction

    # ── Import helpers (same isolation trick as experiment 2) ─────────────
    _CONFLICTING = ('utils', 'models', 'ocp')

    def _save_and_clear_modules():
        backup = {}
        for key in list(sys.modules):
            for prefix in _CONFLICTING:
                if key == prefix or key.startswith(prefix + '.'):
                    backup[key] = sys.modules.pop(key)
        return backup

    def _restore_modules(backup):
        for key in list(sys.modules):
            for prefix in _CONFLICTING:
                if key == prefix or key.startswith(prefix + '.'):
                    del sys.modules[key]
        sys.modules.update(backup)

    # ── MPCC Baseline imports ────────────────────────────────────────────
    _save_and_clear_modules()
    sys.path.insert(0, _MPCC_DIR)

    from utils.numpy_utils import (
        euler_to_quaternion as mpcc_euler2quat,
        build_arc_length_parameterisation as mpcc_build_arc,
        build_terminally_extended_path as mpcc_build_terminal_path,
        build_waypoints as mpcc_build_wp,
        quat_interp_by_arc as mpcc_quat_interp_by_arc,
        mpcc_errors as mpcc_errors_fn,
        rk4_step_mpcc as mpcc_rk4,
    )
    from utils.casadi_utils import (
        create_position_interpolator_casadi as mpcc_pos_interp,
        create_tangent_interpolator_casadi  as mpcc_tang_interp,
        create_quat_interpolator_casadi     as mpcc_quat_interp,
    )
    import ocp.mpcc_controller as mpcc_ocp_module
    build_mpcc_solver = mpcc_ocp_module.build_mpcc_solver
    mpcc_weights_to_param_vector = mpcc_ocp_module.weights_to_param_vector

    sys.path.remove(_MPCC_DIR)
    _save_and_clear_modules()

    # ── DQ-MPCC imports ──────────────────────────────────────────────────
    sys.path.insert(0, _DQ_DIR)

    from utils.numpy_utils import (
        build_arc_length_parameterisation as dq_build_arc,
        build_terminally_extended_path as dq_build_terminal_path,
        build_waypoints as dq_build_wp,
        quat_interp_by_arc as dq_quat_interp_by_arc,
    )
    from utils.dq_numpy_utils import (
        dq_from_pose_numpy, dq_get_position_numpy, dq_get_quaternion_numpy,
        dq_normalize, dq_hemisphere_correction,
        rk4_step_dq_mpcc as dq_rk4, quat_rotate_numpy,
    )
    from utils.casadi_utils import (
        create_position_interpolator_casadi as dq_pos_interp,
        create_tangent_interpolator_casadi  as dq_tang_interp,
        create_quat_interpolator_casadi     as dq_quat_interp,
    )
    import ocp.dq_mpcc_controller as dq_ocp_module
    build_dq_mpcc_solver = dq_ocp_module.build_dq_mpcc_solver
    dq_weights_to_param_vector = dq_ocp_module.weights_to_param_vector

    sys.path.remove(_DQ_DIR)

    # ── Build arc-length parameterisation ────────────────────────────────
    arc_lengths, pos_ref, position_by_arc, tangent_by_arc, s_max_full = \
        mpcc_build_arc(xd, yd, zd, xd_p, yd_p, zd_p, t_finer)
    s_max = min(float(S_MAX), float(s_max_full)) if S_MAX is not None else float(s_max_full)

    delta_s_terminal = 1.10 * vtheta_max * T_PREDICTION
    s_max_solver = s_max + delta_s_terminal
    pos_by_arc_solver, tang_by_arc_solver = mpcc_build_terminal_path(
        position_by_arc, tangent_by_arc, s_max, s_max_solver
    )

    s_wp, pos_wp, tang_wp, quat_wp = mpcc_build_wp(
        s_max_solver, N_WAYPOINTS, pos_by_arc_solver, tang_by_arc_solver,
        euler_to_quat_fn=mpcc_euler2quat,
        reference_speed=ATTITUDE_REF_SPEED,
        max_tilt_deg=ATTITUDE_REF_MAX_TILT_DEG,
    )
    gamma_pos  = mpcc_pos_interp(s_wp, pos_wp)
    gamma_vel  = mpcc_tang_interp(s_wp, tang_wp)
    gamma_quat = mpcc_quat_interp(s_wp, quat_wp)

    dq_gamma_pos  = dq_pos_interp(s_wp, pos_wp)
    dq_gamma_vel  = dq_tang_interp(s_wp, tang_wp)
    dq_gamma_quat = dq_quat_interp(s_wp, quat_wp)

    # ── Pre-generate all initial conditions ──────────────────────────────
    p0_nom = P0.copy()
    q0_nom = Q0 / (np.linalg.norm(Q0) + 1e-12)

    ics = []
    ref_xyz_samples = np.column_stack((
        np.array([xd(ti) for ti in t_finer]),
        np.array([yd(ti) for ti in t_finer]),
        np.array([zd(ti) for ti in t_finer]),
    ))

    init_dists = []   # distance from perturbed p0 to nearest sampled path point
    init_angles = []  # angular perturbation magnitude [rad]
    for _ in range(N_RUNS):
        p0_pert = _perturb_position(p0_nom, SIGMA_P, MAX_POS_OFFSET_NORM, rng)
        q0_pert = _perturb_quaternion(q0_nom, SIGMA_Q, rng)
        ics.append((p0_pert, q0_pert))
        init_dists.append(float(np.min(np.linalg.norm(ref_xyz_samples - p0_pert, axis=1))))
        q_delta = _quat_mult(_quat_conj(q0_nom), q0_pert)
        init_angles.append(_quat_log_norm(q_delta))

    init_dists = np.array(init_dists)
    init_angles = np.array(init_angles)

    print(f"\n  Initial condition statistics:")
    print(f"    Position distance to path: "
          f"mean={np.mean(init_dists):.2f}  std={np.std(init_dists):.2f}  "
          f"max={np.max(init_dists):.2f} m")
    print(f"    Angular perturbation:      "
          f"mean={np.mean(init_angles):.2f}  std={np.std(init_angles):.2f}  "
          f"max={np.max(init_angles):.2f} rad")

    # ══════════════════════════════════════════════════════════════════════
    #  BUILD SOLVERS ONCE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  Building solvers with v_theta_max = {vtheta_max} m/s ...")

    # -- MPCC solver --
    mpcc_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max
    x0_mpcc = np.zeros(15)
    x0_mpcc[0:3] = p0_nom
    x0_mpcc[6:10] = q0_nom
    x0_mpcc[13] = THETA0
    x0_mpcc[14] = VTHETA_MIN

    print(f"  [MPCC] Generating + compiling solver ...")
    os.chdir(_MPCC_DIR)
    mpcc_solver, _, mpcc_model, mpcc_f = build_mpcc_solver(
        x0_mpcc, N_prediction, T_PREDICTION, s_max=s_max_solver,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=False,
    )
    print(f"  [MPCC] Solver ready.")

    # -- DQ solver --
    dq_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max
    x0_dq = np.zeros(16)
    x0_dq[0:8] = dq_from_pose_numpy(q0_nom, p0_nom)
    x0_dq[14] = THETA0
    x0_dq[15] = VTHETA_MIN

    for key in list(sys.modules):
        if 'acados_ocp_solver_pyx' in key:
            del sys.modules[key]

    print(f"  [DQ]   Generating + compiling solver ...")
    os.chdir(_DQ_DIR)
    dq_solver, _, dq_model, dq_f = build_dq_mpcc_solver(
        x0_dq, N_prediction, T_PREDICTION, s_max=s_max_solver,
        gamma_pos=dq_gamma_pos, gamma_vel=dq_gamma_vel,
        gamma_quat=dq_gamma_quat,
        use_cython=False,
    )
    print(f"  [DQ]   Solver ready.")
    os.chdir(_ROOT)

    # ── Set v_theta_max online ───────────────────────────────────────────
    mpcc_ubu_template = np.array([T_MAX, TAUX_MAX, TAUY_MAX, TAUZ_MAX, 0.0])
    dq_ubu_template = np.array([T_MAX, TAUX_MAX, TAUY_MAX, TAUZ_MAX, 0.0])

    def _set_vtheta_mpcc(solver, vtheta_max, N_pred):
        p_val = mpcc_weights_to_param_vector({"vtheta_max": vtheta_max})
        ubu = mpcc_ubu_template.copy()
        ubu[4] = vtheta_max
        for stage in range(N_pred + 1):
            solver.set(stage, "p", p_val)
        for stage in range(N_pred):
            solver.constraints_set(stage, "ubu", ubu)
        mpcc_ocp_module.apply_input_bounds(solver, N_pred, s_max_solver, vtheta_max=vtheta_max)

    def _set_vtheta_dq(solver, vtheta_max, N_pred):
        p_val = dq_weights_to_param_vector({"vtheta_max": vtheta_max})
        ubu = dq_ubu_template.copy()
        ubu[4] = vtheta_max
        for stage in range(N_pred + 1):
            solver.set(stage, "p", p_val)
        for stage in range(N_pred):
            solver.constraints_set(stage, "ubu", ubu)
        dq_ocp_module.apply_input_bounds(solver, N_pred, s_max_solver, vtheta_max=vtheta_max)

    _set_vtheta_mpcc(mpcc_solver, vtheta_max, N_prediction)
    _set_vtheta_dq(dq_solver, vtheta_max, N_prediction)

    # ── Results storage ──────────────────────────────────────────────────
    METRIC_KEYS = [
        'rmse_pos', 'rmse_ori', 'rmse_ec', 'rmse_el',
        't_lap', 'mean_vtheta', 'ctrl_effort',
        'solve_mean', 'solve_max', 'solve_std',
    ]
    results = {ctrl: {k: [] for k in METRIC_KEYS} for ctrl in ['dq', 'base']}
    success  = {ctrl: [] for ctrl in ['dq', 'base']}  # bool per run
    trajectories = {ctrl: [] for ctrl in ['dq', 'base']}  # list of (3,K) arrays

    total_runs = 2 * N_RUNS
    run_count = 0
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  MONTE CARLO LOOP
    # ══════════════════════════════════════════════════════════════════════
    for run_i, (p0_i, q0_i) in enumerate(ics):
        # ── Baseline MPCC ────────────────────────────────────────────────
        run_count += 1
        try:
            res = _run_mpcc_baseline(
                p0_i, q0_i, vtheta_max, mpcc_solver, mpcc_f,
                position_by_arc, tangent_by_arc, s_max,
                s_wp, quat_wp, mpcc_quat_interp_by_arc, mpcc_errors_fn, mpcc_rk4,
                N_sim_max, t_s, N_prediction)
            if res is not None:
                for k in METRIC_KEYS:
                    results['base'][k].append(res[k])
                success['base'].append(True)
                trajectories['base'].append(res['xyz'])
            else:
                success['base'].append(False)
        except Exception as e:
            success['base'].append(False)
            if run_i < 5:
                print(f"\n    [MPCC] Run {run_i+1} EXCEPTION: {e}")

        # ── DQ-MPCC ─────────────────────────────────────────────────────
        run_count += 1
        try:
            res = _run_dq_mpcc(
                p0_i, q0_i, vtheta_max, dq_solver, dq_f,
                position_by_arc, tangent_by_arc, s_max,
                s_wp, quat_wp, dq_quat_interp_by_arc,
                dq_from_pose_numpy, dq_get_position_numpy,
                dq_get_quaternion_numpy, dq_normalize,
                dq_hemisphere_correction, dq_rk4, quat_rotate_numpy,
                N_sim_max, t_s, N_prediction)
            if res is not None:
                for k in METRIC_KEYS:
                    results['dq'][k].append(res[k])
                success['dq'].append(True)
                trajectories['dq'].append(res['xyz'])
            else:
                success['dq'].append(False)
        except Exception as e:
            success['dq'].append(False)
            if run_i < 5:
                print(f"\n    [DQ]   Run {run_i+1} EXCEPTION: {e}")

        # Progress
        elapsed = time.time() - t_start
        pct = run_count / total_runs * 100
        eta = elapsed / run_count * (total_runs - run_count) if run_count > 0 else 0
        if (run_i + 1) % 10 == 0 or run_i < 5:
            n_ok_dq   = sum(success['dq'])
            n_ok_base = sum(success['base'])
            print(f"  [{run_count:4d}/{total_runs}] "
                  f"run={run_i+1:3d}/{N_RUNS}  "
                  f"DQ_ok={n_ok_dq} Base_ok={n_ok_base}  "
                  f"({pct:5.1f}%)  ETA {eta/60:.1f} min")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESULTS — v = {VELOCITY} m/s,  {N_RUNS} runs")
    print(f"{'='*70}")

    for ctrl, tag in [('dq', 'DQ-MPCC'), ('base', 'Baseline')]:
        n_ok = sum(success[ctrl])
        n_fail = N_RUNS - n_ok
        conv_rate = n_ok / N_RUNS * 100
        print(f"\n  [{tag}]")
        print(f"    Convergence rate: {n_ok}/{N_RUNS} = {conv_rate:.1f}%")
        if n_ok > 0:
            for k in METRIC_KEYS:
                arr = np.array(results[ctrl][k])
                print(f"    {k:15s}: "
                      f"mean={np.mean(arr):.4f}  "
                      f"std={np.std(arr):.4f}  "
                      f"med={np.median(arr):.4f}  "
                      f"p95={np.percentile(arr,95):.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    mat_dict = {
        'velocity': VELOCITY,
        'N_runs': N_RUNS,
        'sigma_p': SIGMA_P,
        'sigma_q': SIGMA_Q,
        'seed': SEED,
        'completion_ratio': COMPLETION_RATIO,
        's_max': s_max,
        's_max_full': float(s_max_full),
        't_final': T_FINAL,
        'init_dists': init_dists,
        'init_angles': init_angles,
        # Reference path (dense sample for plotting)
        'ref_x': np.array([xd(ti) for ti in t_finer]),
        'ref_y': np.array([yd(ti) for ti in t_finer]),
        'ref_z': np.array([zd(ti) for ti in t_finer]),
    }
    mat_dict.update(flatten_weight_summary("dq", get_active_weight_summary("dq")))
    mat_dict.update(flatten_weight_summary("base", get_active_weight_summary("mpcc")))
    for ctrl in ['dq', 'base']:
        mat_dict[f'{ctrl}_success'] = np.array(success[ctrl], dtype=np.float64)
        for k in METRIC_KEYS:
            arr = results[ctrl][k]
            mat_dict[f'{ctrl}_{k}'] = np.array(arr) if arr else np.array([])
        # Trajectories as object array (variable-length per run)
        n_traj = len(trajectories[ctrl])
        traj_obj = np.empty(n_traj, dtype=object)
        for i, t in enumerate(trajectories[ctrl]):
            traj_obj[i] = t
        mat_dict[f'{ctrl}_trajectories'] = traj_obj

    out_path = os.path.join(_OUT_DIR, 'montecarlo_data.mat')
    savemat(out_path, mat_dict, do_compression=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE — {run_count} runs in {elapsed_total/60:.1f} min")
    print(f"  Results saved to {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
