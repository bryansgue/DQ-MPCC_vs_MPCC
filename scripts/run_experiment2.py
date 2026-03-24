#!/usr/bin/env python3
"""
run_experiment2.py — Experiment 2: Velocity Sweep (Monte Carlo).

Runs N_RUNS simulations per speed per controller, computing RMSE_pos,
RMSE_ori, RMSE_ec (contouring), and RMSE_el (lag) for each run.
Saves everything to
  results/experiment2/velocity_sweep_data.mat

Usage:
    python scripts/run_experiment2.py
"""

import os, sys, time, math
import numpy as np
from scipy.io import savemat

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)                # workspace root
_DQ_DIR     = os.path.join(_ROOT, "DQ-MPCC_baseline")
_MPCC_DIR   = os.path.join(_ROOT, "MPCC_baseline")
_OUT_DIR    = os.path.join(_ROOT, "results", "experiment2")
os.makedirs(_OUT_DIR, exist_ok=True)

# ── Import experiment configs ────────────────────────────────────────────────
sys.path.insert(0, _ROOT)
from config.experiment_config import P0, Q0, VALUE, trayectoria
from config.sweep_config import (
    VELOCITIES, N_RUNS, SIGMA_P, SIGMA_Q, SEED,
    S_MAX, T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Perturbation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _perturb_position(p0, sigma, rng):
    return p0 + rng.normal(0, sigma, size=3)

def _perturb_quaternion(q0, sigma, rng):
    """Small random rotation: axis-angle with ||angle|| ~ U(0, sigma)."""
    axis = rng.normal(0, 1, size=3)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return q0.copy()
    axis /= n
    angle = rng.uniform(0, sigma)
    ha = angle / 2.0
    dq = np.array([math.cos(ha),
                    math.sin(ha) * axis[0],
                    math.sin(ha) * axis[1],
                    math.sin(ha) * axis[2]])
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
    w0,x0,y0,z0 = a
    w1,x1,y1,z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])

def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ═════════════════════════════════════════════════════════════════════════════
#  Trajectory builder (same Lissajous as both baselines)
# ═════════════════════════════════════════════════════════════════════════════

def _build_trajectory():
    """Build arc-length parameterised Lissajous and return helpers."""
    t_s = 1.0 / FREC
    t = np.arange(0, T_FINAL + t_s, t_s)
    t_finer = np.linspace(0, T_FINAL, len(t))

    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()

    return t, t_finer, xd, yd, zd, xd_p, yd_p, zd_p


# ═════════════════════════════════════════════════════════════════════════════
#  MPCC Baseline — single run
# ═════════════════════════════════════════════════════════════════════════════

# ── Completion / divergence thresholds ────────────────────────────────────
COMPLETION_RATIO = 0.95        # θ must reach ≥ 95 % of s_max to count
POS_DIVERGENCE_THRESHOLD = 50.0  # [m] abort if position error exceeds this


def _run_mpcc_baseline(p0, q0, vtheta_max, solver, f_sys,
                       position_by_arc, tangent_by_arc, s_max,
                       euler_to_quaternion_fn, mpcc_errors_fn,
                       rk4_fn, N_sim, t_s, N_prediction):
    """Run one MPCC baseline simulation.

    Returns (rmse_pos, rmse_ori, rmse_ec, rmse_el,
             theta_final, t_lap, mean_vtheta, success).
      rmse_ec    : contouring RMSE (inertial frame)
      rmse_el    : lag RMSE (inertial frame)
      t_lap      : wall-clock arc-length time  [s]  = k_final * t_s
      mean_vtheta: mean virtual speed  [m/s]  = theta_final / t_lap
    success = True  iff  θ_final ≥ COMPLETION_RATIO * s_max  AND  no divergence.
    """
    nx, nu = 14, 5
    x = np.zeros((nx, N_sim + 1))
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    x[:, 0] = [p0[0], p0[1], p0[2],
               0, 0, 0,
               q0n[0], q0n[1], q0n[2], q0n[3],
               0, 0, 0,
               0.0]

    # warm start
    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    pos_errs = []
    ori_errs = []
    ec_errs  = []
    el_errs  = []
    diverged = False
    k_final  = 0

    for k in range(N_sim):
        if x[13, k] >= s_max - 0.01:
            k_final = k
            break

        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])
        status = solver.solve()

        u_k = solver.get(0, "u")
        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[13, k+1] = np.clip(x[13, k+1], 0.0, s_max)

        # position error
        theta_k = x[13, k]
        sd_k = position_by_arc(theta_k)
        e_t = sd_k - x[0:3, k]
        e_pos = np.linalg.norm(e_t)
        pos_errs.append(e_pos)

        # lag / contouring (inertial frame)
        tang_k = tangent_by_arc(theta_k)
        tang_k = tang_k / (np.linalg.norm(tang_k) + 1e-12)
        e_lag_scalar = np.dot(tang_k, e_t)
        e_lag_vec = e_lag_scalar * tang_k
        e_cont_vec = e_t - e_lag_vec
        ec_errs.append(np.linalg.norm(e_cont_vec))
        el_errs.append(abs(e_lag_scalar))

        # orientation error
        psi_d = math.atan2(tang_k[1], tang_k[0])
        qd = np.array(euler_to_quaternion_fn(0, 0, psi_d))
        q_err = _quat_mult(_quat_conj(qd), x[6:10, k])
        ori_errs.append(_quat_log_norm(q_err))

        # Early abort on divergence
        if e_pos > POS_DIVERGENCE_THRESHOLD:
            diverged = True
            break

        k_final = k

    theta_final = x[13, k_final] if len(pos_errs) > 0 else 0.0
    completed = (theta_final >= COMPLETION_RATIO * s_max) and not diverged

    if len(pos_errs) < 10 or not completed:
        return np.nan, np.nan, np.nan, np.nan, theta_final, np.nan, np.nan, False

    rmse_pos   = math.sqrt(np.mean(np.array(pos_errs)**2))
    rmse_ori   = math.sqrt(np.mean(np.array(ori_errs)**2))
    rmse_ec    = math.sqrt(np.mean(np.array(ec_errs)**2))
    rmse_el    = math.sqrt(np.mean(np.array(el_errs)**2))
    t_lap      = k_final * t_s
    mean_vtheta = theta_final / t_lap if t_lap > 0 else 0.0
    return rmse_pos, rmse_ori, rmse_ec, rmse_el, theta_final, t_lap, mean_vtheta, True


# ═════════════════════════════════════════════════════════════════════════════
#  DQ-MPCC — single run
# ═════════════════════════════════════════════════════════════════════════════

def _run_dq_mpcc(p0, q0, vtheta_max, solver, f_sys,
                 position_by_arc, tangent_by_arc, s_max,
                 euler_to_quaternion_fn,
                 dq_from_pose_fn, dq_get_pos_fn, dq_get_quat_fn,
                 dq_normalize_fn, dq_hemi_fn, rk4_fn, quat_rotate_fn,
                 N_sim, t_s, N_prediction):
    """Run one DQ-MPCC simulation.

    Returns (rmse_pos, rmse_ori, rmse_ec, rmse_el,
             theta_final, t_lap, mean_vtheta, success).
      rmse_ec    : contouring RMSE (inertial frame)
      rmse_el    : lag RMSE (inertial frame)
      t_lap      : wall-clock arc-length time  [s]  = k_final * t_s
      mean_vtheta: mean virtual speed  [m/s]  = theta_final / t_lap
    success = True  iff  θ_final ≥ COMPLETION_RATIO * s_max  AND  no divergence.
    """
    nx, nu = 15, 5
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    dq0 = dq_from_pose_fn(q0n, p0)

    x = np.zeros((nx, N_sim + 1))
    x[0:8, 0] = dq0
    x[14, 0] = 0.0

    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    dq_prev = x[0:8, 0].copy()
    pos_errs = []
    ori_errs = []
    ec_errs  = []
    el_errs  = []
    diverged = False
    k_final  = 0

    for k in range(N_sim):
        if x[14, k] >= s_max - 0.01:
            k_final = k
            break

        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])
        status = solver.solve()

        u_k = solver.get(0, "u")
        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[0:8, k+1] = dq_normalize_fn(x[0:8, k+1])
        x[0:8, k+1] = dq_hemi_fn(x[0:8, k+1], dq_prev)
        dq_prev = x[0:8, k+1].copy()
        x[14, k+1] = np.clip(x[14, k+1], 0.0, s_max)

        # position error
        theta_k = x[14, k]
        sd_k = position_by_arc(theta_k)
        pos_k = dq_get_pos_fn(x[0:8, k])
        e_t = sd_k - pos_k
        e_pos = np.linalg.norm(e_t)
        pos_errs.append(e_pos)

        # lag / contouring (inertial frame — same metric for both controllers)
        tang_k = tangent_by_arc(theta_k)
        tang_k = tang_k / (np.linalg.norm(tang_k) + 1e-12)
        e_lag_scalar = np.dot(tang_k, e_t)
        e_lag_vec = e_lag_scalar * tang_k
        e_cont_vec = e_t - e_lag_vec
        ec_errs.append(np.linalg.norm(e_cont_vec))
        el_errs.append(abs(e_lag_scalar))

        # orientation error
        psi_d = math.atan2(tang_k[1], tang_k[0])
        qd = np.array(euler_to_quaternion_fn(0, 0, psi_d))
        quat_k = dq_get_quat_fn(x[0:8, k])
        q_err = _quat_mult(_quat_conj(qd), quat_k)
        ori_errs.append(_quat_log_norm(q_err))

        # Early abort on divergence
        if e_pos > POS_DIVERGENCE_THRESHOLD:
            diverged = True
            break

        k_final = k

    theta_final = x[14, k_final] if len(pos_errs) > 0 else 0.0
    completed = (theta_final >= COMPLETION_RATIO * s_max) and not diverged

    if len(pos_errs) < 10 or not completed:
        return np.nan, np.nan, np.nan, np.nan, theta_final, np.nan, np.nan, False

    rmse_pos    = math.sqrt(np.mean(np.array(pos_errs)**2))
    rmse_ori    = math.sqrt(np.mean(np.array(ori_errs)**2))
    rmse_ec     = math.sqrt(np.mean(np.array(ec_errs)**2))
    rmse_el     = math.sqrt(np.mean(np.array(el_errs)**2))
    t_lap       = k_final * t_s
    mean_vtheta = theta_final / t_lap if t_lap > 0 else 0.0
    return rmse_pos, rmse_ori, rmse_ec, rmse_el, theta_final, t_lap, mean_vtheta, True


# ═════════════════════════════════════════════════════════════════════════════
#  Main sweep
# ═════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)
    t_s = 1.0 / FREC
    N_prediction = int(round(T_PREDICTION / t_s))
    s_max = S_MAX

    # ── Build trajectory (shared) ────────────────────────────────────────
    t_vec, t_finer, xd, yd, zd, xd_p, yd_p, zd_p = _build_trajectory()
    N_sim_max = t_vec.shape[0] - N_prediction

    # ── Import helpers ─────────────────────────────────────────────────────
    # Both sub-projects have `utils/`, `models/`, `ocp/` packages.
    # To avoid namespace collisions we load each group in isolation,
    # temporarily setting sys.path and clearing cached sub-packages.

    _CONFLICTING = ('utils', 'models', 'ocp')

    def _save_and_clear_modules():
        """Remove conflicting packages from sys.modules, return backup."""
        backup = {}
        for key in list(sys.modules):
            for prefix in _CONFLICTING:
                if key == prefix or key.startswith(prefix + '.'):
                    backup[key] = sys.modules.pop(key)
        return backup

    def _restore_modules(backup):
        """Put backed-up modules back into sys.modules."""
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
        build_waypoints as mpcc_build_wp,
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

    sys.path.remove(_MPCC_DIR)
    _save_and_clear_modules()

    # ── DQ-MPCC imports ──────────────────────────────────────────────────
    sys.path.insert(0, _DQ_DIR)

    from utils.numpy_utils import (
        euler_to_quaternion as dq_euler2quat,
        build_arc_length_parameterisation as dq_build_arc,
        build_waypoints as dq_build_wp,
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

    sys.path.remove(_DQ_DIR)

    # ── Build arc-length parameterisation (same for both) ────────────────
    arc_lengths, pos_ref, position_by_arc, tangent_by_arc, s_max_full = \
        mpcc_build_arc(xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    s_wp, pos_wp, tang_wp, quat_wp = mpcc_build_wp(
        s_max, N_WAYPOINTS, position_by_arc, tangent_by_arc,
        euler_to_quat_fn=mpcc_euler2quat,
    )
    gamma_pos  = mpcc_pos_interp(s_wp, pos_wp)
    gamma_vel  = mpcc_tang_interp(s_wp, tang_wp)
    gamma_quat = mpcc_quat_interp(s_wp, quat_wp)

    # For DQ we need the same interpolators (rebuilt with DQ casadi_utils)
    dq_gamma_pos  = dq_pos_interp(s_wp, pos_wp)
    dq_gamma_vel  = dq_tang_interp(s_wp, tang_wp)
    dq_gamma_quat = dq_quat_interp(s_wp, quat_wp)

    # ── Pre-generate ICs ─────────────────────────────────────────────────
    p0_nom = P0.copy()
    q0_nom = Q0 / (np.linalg.norm(Q0) + 1e-12)
    ics = []
    for _ in range(N_RUNS):
        p0_pert = _perturb_position(p0_nom, SIGMA_P, rng)
        q0_pert = _perturb_quaternion(q0_nom, SIGMA_Q, rng)
        ics.append((p0_pert, q0_pert))

    # ══════════════════════════════════════════════════════════════════════
    #  BUILD SOLVERS ONCE — use largest v_theta_max for constraints
    # ══════════════════════════════════════════════════════════════════════
    vtheta_max_build = max(VELOCITIES)
    print(f"\n  Building solvers once with v_theta_max = {vtheta_max_build} m/s ...")

    # -- MPCC solver --
    mpcc_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max_build
    x0_mpcc = np.zeros(14)
    x0_mpcc[0:3] = p0_nom
    x0_mpcc[6:10] = q0_nom

    print(f"  [MPCC] Generating + compiling solver ...")
    os.chdir(_MPCC_DIR)
    mpcc_solver, _, mpcc_model, mpcc_f = build_mpcc_solver(
        x0_mpcc, N_prediction, T_PREDICTION, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=False,
    )
    print(f"  [MPCC] Solver ready.")

    # -- DQ solver --
    dq_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max_build
    x0_dq = np.zeros(15)
    x0_dq[0:8] = dq_from_pose_numpy(q0_nom, p0_nom)

    # Flush Cython cache before DQ build
    for key in list(sys.modules):
        if 'acados_ocp_solver_pyx' in key:
            del sys.modules[key]

    print(f"  [DQ]   Generating + compiling solver ...")
    os.chdir(_DQ_DIR)
    dq_solver, _, dq_model, dq_f = build_dq_mpcc_solver(
        x0_dq, N_prediction, T_PREDICTION, s_max=s_max,
        gamma_pos=dq_gamma_pos, gamma_vel=dq_gamma_vel,
        gamma_quat=dq_gamma_quat,
        use_cython=False,
    )
    print(f"  [DQ]   Solver ready.")
    os.chdir(_ROOT)

    # ── Helper: update v_theta_max online (parameter + ubu constraint) ──
    #    MPCC:  ubu = [T_max, taux, tauy, tauz, vtheta_max]
    #    DQ:    same layout
    mpcc_ubu_template = np.array([
        mpcc_ocp_module.DEFAULT_T_MAX,
        mpcc_ocp_module.DEFAULT_TAUX_MAX,
        mpcc_ocp_module.DEFAULT_TAUY_MAX,
        mpcc_ocp_module.DEFAULT_TAUZ_MAX,
        0.0,  # placeholder — filled per velocity
    ])
    dq_ubu_template = np.array([
        dq_ocp_module.DEFAULT_T_MAX,
        dq_ocp_module.DEFAULT_TAUX_MAX,
        dq_ocp_module.DEFAULT_TAUY_MAX,
        dq_ocp_module.DEFAULT_TAUZ_MAX,
        0.0,  # placeholder
    ])

    def _set_vtheta_max_mpcc(solver, vtheta_max, N_pred):
        """Update p (cost param) and ubu (constraint) for all stages."""
        p_val = np.array([vtheta_max])
        ubu = mpcc_ubu_template.copy()
        ubu[4] = vtheta_max
        for stage in range(N_pred + 1):
            solver.set(stage, "p", p_val)
        for stage in range(N_pred):
            solver.constraints_set(stage, "ubu", ubu)

    def _set_vtheta_max_dq(solver, vtheta_max, N_pred):
        """Update p (cost param) and ubu (constraint) for all stages."""
        p_val = np.array([vtheta_max])
        ubu = dq_ubu_template.copy()
        ubu[4] = vtheta_max
        for stage in range(N_pred + 1):
            solver.set(stage, "p", p_val)
        for stage in range(N_pred):
            solver.constraints_set(stage, "ubu", ubu)

    # ── Results storage ──────────────────────────────────────────────────
    results = {ctrl: {v: {'rmse_pos': [], 'rmse_ori': [],
                          'rmse_ec': [], 'rmse_el': [],
                          't_lap': [], 'mean_vtheta': []}
                      for v in VELOCITIES}
               for ctrl in ['dq', 'base']}
    failures = {ctrl: {v: 0 for v in VELOCITIES}
                for ctrl in ['dq', 'base']}

    total_runs = 2 * len(VELOCITIES) * N_RUNS
    run_count = 0
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  SWEEP — no recompilation, just online parameter updates
    # ══════════════════════════════════════════════════════════════════════
    for vi, vtheta_max in enumerate(VELOCITIES):
        print(f"\n{'='*70}")
        print(f"  VELOCITY {vi+1}/{len(VELOCITIES)}:  v_theta_max = {vtheta_max} m/s")
        print(f"{'='*70}")

        # Update solvers online (no recompilation!)
        _set_vtheta_max_mpcc(mpcc_solver, vtheta_max, N_prediction)
        _set_vtheta_max_dq(dq_solver, vtheta_max, N_prediction)

        # ── Run Monte Carlo for this velocity ────────────────────────────
        for run_i, (p0_i, q0_i) in enumerate(ics):
            # ── Baseline MPCC ────────────────────────────────────────────
            run_count += 1
            try:
                rp, ro, rec, rel, theta_f, tlap, mvtheta, ok = _run_mpcc_baseline(
                    p0_i, q0_i, vtheta_max, mpcc_solver, mpcc_f,
                    position_by_arc, tangent_by_arc, s_max,
                    mpcc_euler2quat, mpcc_errors_fn, mpcc_rk4,
                    N_sim_max, t_s, N_prediction)
                if ok:
                    results['base'][vtheta_max]['rmse_pos'].append(rp)
                    results['base'][vtheta_max]['rmse_ori'].append(ro)
                    results['base'][vtheta_max]['rmse_ec'].append(rec)
                    results['base'][vtheta_max]['rmse_el'].append(rel)
                    results['base'][vtheta_max]['t_lap'].append(tlap)
                    results['base'][vtheta_max]['mean_vtheta'].append(mvtheta)
                else:
                    failures['base'][vtheta_max] += 1
                    pct_done = theta_f / s_max * 100 if s_max > 0 else 0
                    print(f"\n    [MPCC] Run {run_i+1} INCOMPLETE: "
                          f"theta={theta_f:.1f}/{s_max:.1f} ({pct_done:.0f}%)")
            except Exception as e:
                failures['base'][vtheta_max] += 1
                print(f"\n    [MPCC] Run {run_i+1} EXCEPTION: {e}")

            # ── DQ-MPCC ─────────────────────────────────────────────────
            run_count += 1
            try:
                rp, ro, rec, rel, theta_f, tlap, mvtheta, ok = _run_dq_mpcc(
                    p0_i, q0_i, vtheta_max, dq_solver, dq_f,
                    position_by_arc, tangent_by_arc, s_max,
                    dq_euler2quat,
                    dq_from_pose_numpy, dq_get_position_numpy,
                    dq_get_quaternion_numpy, dq_normalize,
                    dq_hemisphere_correction, dq_rk4, quat_rotate_numpy,
                    N_sim_max, t_s, N_prediction)
                if ok:
                    results['dq'][vtheta_max]['rmse_pos'].append(rp)
                    results['dq'][vtheta_max]['rmse_ori'].append(ro)
                    results['dq'][vtheta_max]['rmse_ec'].append(rec)
                    results['dq'][vtheta_max]['rmse_el'].append(rel)
                    results['dq'][vtheta_max]['t_lap'].append(tlap)
                    results['dq'][vtheta_max]['mean_vtheta'].append(mvtheta)
                else:
                    failures['dq'][vtheta_max] += 1
                    pct_done = theta_f / s_max * 100 if s_max > 0 else 0
                    print(f"\n    [DQ]   Run {run_i+1} INCOMPLETE: "
                          f"theta={theta_f:.1f}/{s_max:.1f} ({pct_done:.0f}%)")
            except Exception as e:
                failures['dq'][vtheta_max] += 1
                print(f"\n    [DQ]   Run {run_i+1} EXCEPTION: {e}")

            # Progress
            elapsed = time.time() - t_start
            pct = run_count / total_runs * 100
            eta = elapsed / run_count * (total_runs - run_count)
            print(f"  [{run_count:4d}/{total_runs}] "
                  f"v={vtheta_max}  run={run_i+1:3d}/{N_RUNS}  "
                  f"({pct:5.1f}%)  ETA {eta/60:.1f} min", end='\r')

        print()
        for ctrl in ['dq', 'base']:
            n_ok = len(results[ctrl][vtheta_max]['rmse_pos'])
            n_fail = failures[ctrl][vtheta_max]
            tag = "DQ-MPCC" if ctrl == 'dq' else "Baseline"
            if n_ok > 0:
                rp = np.array(results[ctrl][vtheta_max]['rmse_pos'])
                ro = np.array(results[ctrl][vtheta_max]['rmse_ori'])
                rec = np.array(results[ctrl][vtheta_max]['rmse_ec'])
                rel = np.array(results[ctrl][vtheta_max]['rmse_el'])
                tl = np.array(results[ctrl][vtheta_max]['t_lap'])
                mv = np.array(results[ctrl][vtheta_max]['mean_vtheta'])
                print(f"  [{tag:10s}] v={vtheta_max}  OK={n_ok}  FAIL={n_fail}  "
                      f"RMSE_pos={np.median(rp):.4f}  RMSE_ori={np.median(ro):.4f}  "
                      f"RMSE_ec={np.median(rec):.4f}  RMSE_el={np.median(rel):.4f}  "
                      f"t_lap={np.median(tl):.2f}s  vθ_mean={np.median(mv):.2f}m/s")
            else:
                print(f"  [{tag:10s}] v={vtheta_max}  ALL FAILED ({n_fail})")
        print(f"  (completion criterion: theta >= {COMPLETION_RATIO*100:.0f}% of s_max={s_max})")

    # ── Save results ─────────────────────────────────────────────────────
    mat_dict = {
        'velocities': np.array(VELOCITIES),
        'N_runs': N_RUNS,
        'sigma_p': SIGMA_P,
        'sigma_q': SIGMA_Q,
        'completion_ratio': COMPLETION_RATIO,
        's_max': s_max,
    }
    for ctrl in ['dq', 'base']:
        for v in VELOCITIES:
            vstr = f"{v:.2f}".replace('.', 'p')
            rp  = results[ctrl][v]['rmse_pos']
            ro  = results[ctrl][v]['rmse_ori']
            rec = results[ctrl][v]['rmse_ec']
            rel = results[ctrl][v]['rmse_el']
            tl  = results[ctrl][v]['t_lap']
            mv  = results[ctrl][v]['mean_vtheta']
            mat_dict[f'{ctrl}_v{vstr}_rmse_pos']    = np.array(rp) if rp else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_rmse_ori']    = np.array(ro) if ro else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_rmse_ec']     = np.array(rec) if rec else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_rmse_el']     = np.array(rel) if rel else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_t_lap']       = np.array(tl) if tl else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_mean_vtheta'] = np.array(mv) if mv else np.array([])
            mat_dict[f'{ctrl}_v{vstr}_failures']    = failures[ctrl][v]

    out_path = os.path.join(_OUT_DIR, 'velocity_sweep_data.mat')
    savemat(out_path, mat_dict, do_compression=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE — {run_count} runs in {elapsed_total/60:.1f} min")
    print(f"  Results saved to {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
