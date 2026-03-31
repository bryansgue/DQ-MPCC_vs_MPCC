"""
DQ_MPCC_simulation_tuner.py  –  Headless DQ-MPCC simulation for bilevel tuning.
"""

import os
import sys
import numpy as np
import time as _time


def _suppress_c_output():
    """Redirect C-level stdout/stderr to /dev/null."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    return old_stdout_fd, old_stderr_fd


def _restore_c_output(old_stdout_fd, old_stderr_fd):
    """Restore C-level stdout/stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)


_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from tuning_config import T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS, S_MAX_MANUAL
from experiment_config import (
    P0,
    Q0,
    V0,
    W0,
    THETA0,
    TRAJECTORY_T_FINAL,
    VTHETA_MAX,
    VTHETA_MIN,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    trayectoria,
)

from utils.numpy_utils import (
    euler_to_quaternion,
    build_arc_length_parameterisation,
    build_terminally_extended_path,
    build_waypoints,
    quat_interp_by_arc,
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
)
from utils.dq_numpy_utils import (
    dq_error_numpy,
    ln_dual_numpy,
    rotate_tangent_to_desired_frame_numpy,
    lag_contouring_decomposition_numpy,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi,
    create_tangent_interpolator_casadi,
    create_quat_interpolator_casadi,
)
from ocp.dq_mpcc_controller_tuner import (
    build_dq_mpcc_solver_tunable,
    weights_to_param_vector,
    DEFAULT_VTHETA_MAX,
    apply_input_bounds,
)


def _get_trajectory_functions(t):
    """Support both trayectoria() and legacy trayectoria(t) signatures."""
    try:
        return trayectoria()
    except TypeError:
        return trayectoria(t)


_INFRA = None


def _build_infrastructure():
    """Build trajectory, interpolation and solver ONCE."""
    t_final = T_FINAL
    t_path_final = TRAJECTORY_T_FINAL
    frec = FREC
    t_s = 1 / frec
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

    gamma_pos = create_position_interpolator_casadi(s_wp, pos_wp)
    gamma_vel = create_tangent_interpolator_casadi(s_wp, tang_wp)
    gamma_quat = create_quat_interpolator_casadi(s_wp, quat_wp)

    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)
    dq0 = dq_from_pose_numpy(q0_normed, P0)
    q0_inv = np.array([q0_normed[0], -q0_normed[1], -q0_normed[2], -q0_normed[3]])
    v_body0 = quat_rotate_numpy(q0_inv, V0)
    x0 = np.array([
        dq0[0], dq0[1], dq0[2], dq0[3], dq0[4], dq0[5], dq0[6], dq0[7],
        W0[0], W0[1], W0[2], v_body0[0], v_body0[1], v_body0[2],
        THETA0, VTHETA_MIN,
    ], dtype=float)

    print("[DQ-TUNER] Building tunable DQ-MPCC solver (compiles ONCE) ...")
    solver, ocp, model, f = build_dq_mpcc_solver_tunable(
        x0, N_prediction, t_prediction, s_max=s_max_solver,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )
    print("[DQ-TUNER] Solver ready.")

    return {
        "t_s": t_s,
        "N_sim": N_sim,
        "N_prediction": N_prediction,
        "s_max": s_max,
        "position_by_arc_length": position_by_arc_length,
        "tangent_by_arc_length": tangent_by_arc_length,
        "s_wp": s_wp,
        "quat_wp": quat_wp,
        "solver": solver,
        "model": model,
        "f": f,
        "x0": x0,
    }


def _get_infra():
    global _INFRA
    if _INFRA is None:
        _INFRA = _build_infrastructure()
    return _INFRA


def run_simulation(weights: dict | None = None, verbose: bool = False,
                   vtheta_max: float | None = None) -> dict:
    """Run a full DQ-MPCC simulation with the given weights."""
    infra = _get_infra()

    t_s = infra["t_s"]
    N_sim = infra["N_sim"]
    N_prediction = infra["N_prediction"]
    s_max = infra["s_max"]
    pos_by_arc = infra["position_by_arc_length"]
    tang_by_arc = infra["tangent_by_arc_length"]
    s_wp = infra["s_wp"]
    quat_wp = infra["quat_wp"]
    solver = infra["solver"]
    model = infra["model"]
    f = infra["f"]
    x0 = infra["x0"].copy()

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    v_theta_max = float(vtheta_max) if vtheta_max is not None else DEFAULT_VTHETA_MAX

    p_vec = weights_to_param_vector(weights)
    p_vec[17] = v_theta_max
    for stage in range(N_prediction + 1):
        solver.set(stage, "p", p_vec)

    apply_input_bounds(solver, N_prediction, s_max, vtheta_max=v_theta_max)

    w = weights or {}
    Q_phi_vec = np.array(w.get("Q_phi", p_vec[0:3]))
    Q_ec_vec = np.array(w.get("Q_ec", p_vec[3:6]))
    Q_el_vec = np.array(w.get("Q_el", p_vec[6:9]))
    U_mat_vec = np.array(w.get("U_mat", p_vec[9:13]))
    Q_omega_vec = np.array(w.get("Q_omega", p_vec[13:16]))
    Q_s_val = float(w.get("Q_s", p_vec[16]))

    x = np.zeros((nx, N_sim + 1))
    u_control = np.zeros((nu, N_sim))
    e_contorno = np.zeros((3, N_sim))
    e_arrastre = np.zeros((3, N_sim))
    e_total = np.zeros((3, N_sim))
    vel_progres = np.zeros((1, N_sim))
    vel_real = np.zeros((1, N_sim))
    theta_hist = np.zeros((1, N_sim + 1))
    t_solver = np.zeros(N_sim)
    mpcc_cost = np.zeros(N_sim)

    x[:, 0] = x0
    theta_hist[0, 0] = x0[14]

    for stage in range(N_prediction + 1):
        solver.set(stage, "x", x0)
        solver.set(stage, "p", p_vec)
    for stage in range(N_prediction):
        solver.set(stage, "u", np.zeros(nu))

    dq_prev = x[0:8, 0].copy()
    actual_steps = N_sim
    success = True
    consecutive_fails = 0
    max_consecutive_fails = 50
    solver_fail_count = 0

    for k in range(N_sim):
        if x[14, k] >= s_max:
            actual_steps = k
            break

        if not np.all(np.isfinite(x[:, k])):
            actual_steps = k
            success = False
            break

        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])

        tic = _time.time()
        old_out, old_err = _suppress_c_output()
        try:
            status = solver.solve()
        finally:
            _restore_c_output(old_out, old_err)
        t_solver[k] = _time.time() - tic

        if status not in (0, 2):
            solver_fail_count += 1
            consecutive_fails += 1
            if k > 0:
                u_control[:, k] = u_control[:, k - 1]
            else:
                u_control[:, k] = np.zeros(nu)
            if consecutive_fails >= max_consecutive_fails:
                actual_steps = k + 1
                success = False
                break
        else:
            u_control[:, k] = solver.get(0, "u")
            consecutive_fails = 0

        vel_progres[0, k] = x[15, k]
        theta_hist[0, k] = x[14, k]

        quat_k = dq_get_quaternion_numpy(x[0:8, k])
        v_body_k = x[11:14, k]
        tang_k_now = tang_by_arc(min(x[14, k], s_max))
        v_inertial_k = quat_rotate_numpy(quat_k, v_body_k)
        vel_real[0, k] = np.dot(tang_k_now, v_inertial_k)

        x[:, k + 1] = rk4_step_dq_mpcc(x[:, k], u_control[:, k], t_s, f)
        x[0:8, k + 1] = dq_normalize(x[0:8, k + 1])
        x[0:8, k + 1] = dq_hemisphere_correction(x[0:8, k + 1], dq_prev)
        dq_prev = x[0:8, k + 1].copy()
        x[14, k + 1] = np.clip(x[14, k + 1], 0.0, s_max)
        x[15, k + 1] = np.clip(x[15, k + 1], VTHETA_MIN, v_theta_max)
        theta_hist[0, k + 1] = x[14, k + 1]

        if x[14, k + 1] >= s_max:
            x[14, k + 1] = s_max
            theta_hist[0, k + 1] = s_max
            actual_steps = k + 1
            break

        theta_k = min(x[14, k], s_max)
        sd_k = pos_by_arc(theta_k)
        tang_k = tang_by_arc(theta_k)
        pos_k = dq_get_position_numpy(x[0:8, k])
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(pos_k, tang_k, sd_k)

        qd_k = quat_interp_by_arc(theta_k, s_wp, quat_wp)
        dq_desired_k = dq_from_pose_numpy(qd_k, sd_k)
        dq_err_k = dq_error_numpy(dq_desired_k, x[0:8, k])
        log_err_k = ln_dual_numpy(dq_err_k)
        phi_k = log_err_k[0:3]
        rho_k = log_err_k[3:6]
        tang_body_k = rotate_tangent_to_desired_frame_numpy(tang_k, qd_k)
        rho_lag_k, rho_cont_k = lag_contouring_decomposition_numpy(rho_k, tang_body_k)
        omega_k = x[8:11, k]
        u_k = u_control[0:4, k]
        a_theta_k = u_control[4, k]

        from experiment_config import DQ_Q_ATHETA
        mpcc_cost[k] = (
            np.dot(phi_k**2, Q_phi_vec)
            + np.dot(rho_cont_k**2, Q_ec_vec)
            + np.dot(rho_lag_k**2, Q_el_vec)
            + np.dot(u_k**2, U_mat_vec)
            + np.dot(omega_k**2, Q_omega_vec)
            + DQ_Q_ATHETA * (a_theta_k ** 2)
            - Q_s_val * x[15, k]
        )

        if verbose and k % 200 == 0:
            print(f"  [k={k:04d}]  θ={x[14,k]:7.2f}/{s_max:.0f}  "
                  f"v_θ={x[15,k]:5.2f}  solver={t_solver[k]*1e3:5.2f} ms")

    N = actual_steps
    x = x[:, :N + 1]
    u_control = u_control[:, :N]
    e_contorno = e_contorno[:, :N]
    e_arrastre = e_arrastre[:, :N]
    e_total = e_total[:, :N]
    vel_progres = vel_progres[:, :N]
    vel_real = vel_real[:, :N]
    theta_hist = theta_hist[:, :N + 1]
    t_solver = t_solver[:N]
    mpcc_cost = mpcc_cost[:N]

    rmse_c = np.sqrt(np.mean(np.sum(e_contorno**2, axis=0))) if N > 0 else 0.0
    rmse_l = np.sqrt(np.mean(np.sum(e_arrastre**2, axis=0))) if N > 0 else 0.0
    rmse_t = np.sqrt(np.mean(np.sum(e_total**2, axis=0))) if N > 0 else 0.0
    mean_effort = np.mean(np.sum(u_control[0:4, :]**2, axis=0)) if N > 0 else 0.0
    total_mpcc_cost = np.sum(mpcc_cost) if N > 0 else 0.0
    mean_mpcc_cost = np.mean(mpcc_cost) if N > 0 else 0.0
    path_completed = x[14, N] / s_max if N > 0 else 0.0
    mean_vtheta = np.mean(vel_progres) if N > 0 else 0.0
    mean_solver_ms = np.mean(t_solver) * 1e3 if N > 0 else 0.0
    mean_vreal = np.mean(vel_real) if N > 0 else 0.0
    mean_vpath_ratio = mean_vreal / (mean_vtheta + 1e-9) if N > 0 else 0.0
    solver_fail_ratio = solver_fail_count / max(N, 1)

    return {
        "rmse_contorno": rmse_c,
        "rmse_lag": rmse_l,
        "rmse_total": rmse_t,
        "mean_effort": mean_effort,
        "path_completed": path_completed,
        "mean_vtheta": mean_vtheta,
        "mean_vreal": mean_vreal,
        "mean_vpath_ratio": mean_vpath_ratio,
        "mean_solver_ms": mean_solver_ms,
        "solver_fail_ratio": solver_fail_ratio,
        "mean_mpcc_cost": mean_mpcc_cost,
        "total_mpcc_cost": total_mpcc_cost,
        "N_steps": N,
        "x": x,
        "u": u_control,
        "e_contorno": e_contorno,
        "e_lag": e_arrastre,
        "theta_history": theta_hist,
        "success": success,
        "s_max": s_max,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  DQ-MPCC Simulation — standalone test with DEFAULT gains")
    print("=" * 60)
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
