"""
DQ_MPCC_baseline.py  –  Dual-Quaternion MPCC for quadrotor trajectory tracking.

Strict DQ-MPCC formulation:
  • θ (arc-length progress) is an optimisation STATE   (x[14])
  • v_θ (progress velocity) is an optimisation STATE   (x[15])
  • a_θ (progress acceleration) is an optimisation CONTROL (u[4])
  • Dynamics:  θ̇ = v_θ,  v̇_θ = a_θ
  • Pose error via exact se(3) logarithmic map:  [φ; ρ] = Log(Q_d* ⊗ Q)
  • Lag-contouring decomposition in the desired body frame
"""

import numpy as np
import time
import time as time_module
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_config import (
    P0, Q0, V0, W0, THETA0,
    T_FINAL, TRAJECTORY_T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS, S_MAX_MANUAL,
    VTHETA_MAX, VTHETA_MIN,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    USE_TUNED_WEIGHTS_DQ, DQ_WEIGHTS_PATH, DQ_WEIGHTS,
    trayectoria,
)
from config.tuning_registry import get_active_weight_summary, flatten_weight_summary

from utils.numpy_utils import (
    euler_to_quaternion,
    quat_interp_by_arc,
    mpcc_errors,
    compute_curvature,
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
)
from utils.casadi_utils import (
    create_position_interpolator_casadi,
    create_tangent_interpolator_casadi,
    create_quat_interpolator_casadi,
)
from ocp.dq_mpcc_controller import (
    build_dq_mpcc_solver,
    weights_to_param_vector,
    apply_input_bounds,
)
from utils.graficas import (
    plot_pose, plot_error, plot_time, plot_control,
    plot_vel_lineal, plot_vel_angular,
    plot_progress_velocity, plot_velocity_analysis,
    plot_3d_trajectory, plot_timing,
)
from path_loader import load_or_build_path_cache


FORCE_REBUILD_OCP = True


def _plot_quaternion_tracking(t_plot, quat_real, quat_desired):
    """Plot desired vs realised quaternion components over time."""
    labels = ["q_w", "q_x", "q_y", "q_z"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_plot, quat_real[i, :], label="real", linewidth=1.6)
        ax.plot(t_plot, quat_desired[i, :], "--", label="desired", linewidth=1.4)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Quaternion Tracking")
    fig.tight_layout()
    return fig


def _get_trajectory_functions(t):
    """Support both trayectoria() and legacy trayectoria(t) signatures."""
    try:
        return trayectoria()
    except TypeError:
        return trayectoria(t)


def _check_solver_staleness(cache_path, solver_json_path):
    """Warn if the compiled solver is older than the path cache."""
    import os
    if os.path.isfile(cache_path) and os.path.isfile(solver_json_path):
        cache_mtime = os.path.getmtime(cache_path)
        solver_mtime = os.path.getmtime(solver_json_path)
        if cache_mtime > solver_mtime:
            print("⚠" * 30)
            print("  WARNING: path_cache.npz is NEWER than the compiled solver!")
            print("  The solver may contain stale bsplines.")
            print("  Run:  python precompile_paths.py")
            print("  Or set FORCE_REBUILD_OCP = True")
            print("⚠" * 30)


def main():
    t_final = T_FINAL
    t_path_final = TRAJECTORY_T_FINAL
    frec = FREC
    t_s = 1 / frec
    t_prediction = T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")
    print(f"[PATH]    trajectory_t_final={t_path_final} s  |  simulation_t_final={t_final} s")

    t = np.arange(0, t_final + t_s, t_s)
    t_path = np.linspace(0.0, t_path_final, len(t))
    N_sim = t.shape[0] - N_prediction

    # ── Trajectory / gamma cache ─────────────────────────────────────────
    (
        arc_lengths,
        pos_ref,
        position_by_arc_length,
        tangent_by_arc_length,
        s_wp,
        pos_wp,
        tang_wp,
        quat_wp,
        s_max_full,
        s_max,
        S_MAX_SOLVER,
        delta_s_terminal,
    ) = load_or_build_path_cache(verbose=True)

    s_max_solver = S_MAX_SOLVER

    if S_MAX_MANUAL is not None and S_MAX_MANUAL < s_max_full:
        print(f"[ARC]  Total arc length = {s_max_full:.3f} m  →  "
              f"LIMITED to s_max = {s_max:.3f} m")
    else:
        print(f"[ARC]  Total arc length = {s_max:.3f} m")

    print(f"[PATH]   available_length = {s_max_full:.3f} m")
    print(f"[PATH]   active_s_max     = {s_max:.3f} m")
    print(f"[SIM]    time_limit       = {t_final:.3f} s")

    # ── Staleness check ──────────────────────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cache_path = os.path.join(_script_dir, "path_cache.npz")
    _solver_json = os.path.join(_script_dir, "acados_ocp_DQ_Drone_MPCC_runtime_accel.json")
    _check_solver_staleness(_cache_path, _solver_json)

    delta_t = np.zeros((1, N_sim), dtype=np.double)
    e_contorno = np.zeros((3, N_sim), dtype=np.double)
    e_arrastre = np.zeros((3, N_sim), dtype=np.double)
    e_total = np.zeros((3, N_sim), dtype=np.double)
    vel_progres = np.zeros((1, N_sim), dtype=np.double)
    vel_real = np.zeros((1, N_sim), dtype=np.double)
    vel_tangent = np.zeros((1, N_sim), dtype=np.double)
    theta_history = np.zeros((1, N_sim + 1), dtype=np.double)
    quat_d_theta = np.zeros((4, N_sim), dtype=np.double)
    t_solver = np.zeros((1, N_sim), dtype=np.double)
    t_loop = np.zeros((1, N_sim), dtype=np.double)

    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)
    dq0 = dq_from_pose_numpy(q0_normed, P0)
    q0_inv = np.array([q0_normed[0], -q0_normed[1], -q0_normed[2], -q0_normed[3]])
    v_body0 = quat_rotate_numpy(q0_inv, V0)

    x = np.zeros((16, N_sim + 1), dtype=np.double)
    x[0:8, 0] = dq0
    x[8:11, 0] = W0
    x[11:14, 0] = v_body0
    x[14, 0] = THETA0
    x[15, 0] = VTHETA_MIN
    theta_history[0, 0] = x[14, 0]
    print(f"[IC]  p0 = {P0}  |  q0 = {np.round(q0_normed,4)}  |  θ₀ = {THETA0}")

    x_std = np.zeros((13, N_sim + 1), dtype=np.double)
    x_std[:, 0] = state15_to_standard13(x[:, 0])

    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :] = pos_ref[0, :]
    xref[1, :] = pos_ref[1, :]
    xref[2, :] = pos_ref[2, :]
    xref[3, :] = dp_ds[0, :]
    xref[4, :] = dp_ds[1, :]
    xref[5, :] = dp_ds[2, :]

    u_control = np.zeros((5, N_sim), dtype=np.double)

    gamma_pos = create_position_interpolator_casadi(s_wp, pos_wp)
    gamma_vel = create_tangent_interpolator_casadi(s_wp, tang_wp)
    gamma_quat = create_quat_interpolator_casadi(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints "
          f"(s_max_solver={s_max_solver:.2f})")
    print(f"[PATH]   terminal_extension = {delta_s_terminal:.3f} m")
    print(f"[ATT]    ref_speed={ATTITUDE_REF_SPEED:.2f} m/s  |  max_tilt={ATTITUDE_REF_MAX_TILT_DEG:.1f} deg")

    for i, sv in enumerate(arc_lengths):
        qd_i = quat_interp_by_arc(min(sv, s_max_solver), s_wp, quat_wp)
        xref[6, i] = qd_i[0]
        xref[7, i] = qd_i[1]
        xref[8, i] = qd_i[2]
        xref[9, i] = qd_i[3]

    if USE_TUNED_WEIGHTS_DQ and DQ_WEIGHTS_PATH.is_file():
        print(f"[WEIGHTS] Using tuned DQ-MPCC weights from {DQ_WEIGHTS_PATH}")
    else:
        print("[WEIGHTS] Using manual DQ-MPCC weights from experiment_config.py")

    acados_ocp_solver, ocp, model, f = build_dq_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=s_max_solver,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True, force_rebuild=FORCE_REBUILD_OCP,
    )

    p_base = weights_to_param_vector(DQ_WEIGHTS)

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
        acados_ocp_solver.set(stage, "p", p_base)
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    apply_input_bounds(acados_ocp_solver, N_prediction, s_max_solver, vtheta_max=p_base[17])

    print("Initializing simulation...")
    print("Ready!!!")
    time_all = time.time()
    t_lap = np.nan
    dq_prev = x[0:8, 0].copy()

    for k in range(N_sim):
        tic = time.time()

        if x[14, k] >= s_max:
            if np.isnan(t_lap):
                t_lap = k * t_s
            print(f"[k={k:04d}]  Path complete at θ={x[14,k]:.3f} m  "
                  f"→  t_lap = {t_lap:.3f} s")
            N_sim = k
            break

        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        u_control[:, k] = acados_ocp_solver.get(0, "u")

        vel_progres[:, k] = x[15, k]
        theta_history[:, k] = x[14, k]

        theta_k_now = np.clip(x[14, k], 0.0, s_max)
        tang_k_now = tangent_by_arc_length(theta_k_now)
        quat_k = dq_get_quaternion_numpy(x[0:8, k])
        v_body_k = x[11:14, k]
        v_inertial_k = quat_rotate_numpy(quat_k, v_body_k)
        vel_real[:, k] = np.dot(tang_k_now, v_inertial_k)
        vel_tangent[:, k] = np.linalg.norm(v_inertial_k)

        x[:, k + 1] = rk4_step_dq_mpcc(x[:, k], u_control[:, k], t_s, f)
        x[0:8, k + 1] = dq_normalize(x[0:8, k + 1])
        x[0:8, k + 1] = dq_hemisphere_correction(x[0:8, k + 1], dq_prev)
        dq_prev = x[0:8, k + 1].copy()
        x[14, k + 1] = np.clip(x[14, k + 1], 0.0, s_max_solver)
        x[15, k + 1] = np.clip(x[15, k + 1], VTHETA_MIN, p_base[17])
        theta_history[:, k + 1] = x[14, k + 1]

        if x[14, k + 1] >= s_max:
            x[14, k + 1] = s_max
            theta_history[:, k + 1] = s_max
            if np.isnan(t_lap):
                t_lap = (k + 1) * t_s
            N_sim = k + 1
            x_std[:, k + 1] = state15_to_standard13(x[:, k + 1])
            print(f"[k={k+1:04d}]  Path complete at θ={x[14,k+1]:.3f} m  "
                  f"→  t_lap = {t_lap:.3f} s")
            break

        x_std[:, k + 1] = state15_to_standard13(x[:, k + 1])

        theta_k = min(x[14, k], s_max)
        sd_k = position_by_arc_length(theta_k)
        tang_k = tangent_by_arc_length(theta_k)
        pos_k = dq_get_position_numpy(x[0:8, k])
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(pos_k, tang_k, sd_k)

        quat_d_theta[:, k] = quat_interp_by_arc(theta_k, s_wp, quat_wp)

        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        if abs(vel_progres[0, k]) > 0.1:
            ratio_vtheta = vel_real[0, k] / vel_progres[0, k]
            ratio_str = f"{ratio_vtheta:5.2f}"
        else:
            ratio_str = " n/a "
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  v_real={vel_real[0,k]:5.2f}  "
              f"ratio={ratio_str}  |  "
              f"θ={x[14,k]:7.2f}/{s_max:.0f} m  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    total_time = time.time() - time_all
    print(f"\nTotal execution time: {total_time:.4f} seconds")
    print(f"Final θ = {x[14, N_sim]:.3f} / {s_max:.3f} m  "
          f"({x[14, N_sim]/s_max*100:.1f}% of path)")

    x = x[:, :N_sim + 1]
    x_std = x_std[:, :N_sim + 1]
    u_control = u_control[:, :N_sim]
    e_contorno = e_contorno[:, :N_sim]
    e_arrastre = e_arrastre[:, :N_sim]
    e_total = e_total[:, :N_sim]
    vel_progres = vel_progres[:, :N_sim]
    vel_real = vel_real[:, :N_sim]
    vel_tangent = vel_tangent[:, :N_sim]
    theta_history = theta_history[:, :N_sim + 1]
    quat_d_theta = quat_d_theta[:, :N_sim]
    t_solver = t_solver[:, :N_sim]
    t_loop = t_loop[:, :N_sim]
    t_plot = t[:N_sim + 1]

    print("Generating figures...")

    xref_theta = np.zeros((17, N_sim + 1))
    for i in range(N_sim + 1):
        s_i = theta_history[0, i]
        pos_i = position_by_arc_length(s_i)
        tang_i = tangent_by_arc_length(s_i)
        xref_theta[0:3, i] = pos_i
        xref_theta[3:6, i] = tang_i

    curvature = compute_curvature(position_by_arc_length, s_max, N_samples=500)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig1 = plot_pose(x_std, xref_theta, t_plot)
    fig1.savefig(os.path.join(figures_dir, "1_pose.png"))

    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig(os.path.join(figures_dir, "2_control_actions.png"))

    fig3 = plot_vel_lineal(x_std[3:6, :], t_plot)
    fig3.savefig(os.path.join(figures_dir, "3_vel_lineal.png"))

    fig4 = plot_vel_angular(x_std[10:13, :], t_plot)
    fig4.savefig(os.path.join(figures_dir, "4_vel_angular.png"))

    fig5 = plot_velocity_analysis(
        vel_progres, vel_real, vel_tangent,
        curvature, theta_history, s_max, t_plot[:N_sim])
    fig5.savefig(os.path.join(figures_dir, "5_velocity_analysis.png"), dpi=150)

    fig6 = plot_3d_trajectory(
        x_std, pos_ref, s_max=s_max,
        position_by_arc=position_by_arc_length, N_plot=600)
    fig6.savefig(os.path.join(figures_dir, "6_trajectory_3d.png"), dpi=150)

    fig7 = _plot_quaternion_tracking(t_plot[:N_sim], x_std[6:10, :N_sim], quat_d_theta)
    fig7.savefig(os.path.join(figures_dir, "7_quaternion_tracking.png"), dpi=150)

    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig(os.path.join(figures_dir, "8_progress_velocity.png"), dpi=150)

    s_ms = t_solver[0, :] * 1e3
    l_ms = t_loop[0, :] * 1e3
    ts_ms = t_s * 1e3
    n_overrun = int(np.sum(l_ms > ts_ms * 1.05))

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║              DQ-MPCC TIMING STATISTICS                         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Nominal t_s = {ts_ms:5.2f} ms  ({frec:.0f} Hz)                              ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  [Solver]  mean={np.mean(s_ms):5.2f}  max={np.max(s_ms):5.2f}  std={np.std(s_ms):4.2f}  ms     ║")
    print(f"║  [Loop  ]  mean={np.mean(l_ms):5.2f}  max={np.max(l_ms):5.2f}  std={np.std(l_ms):4.2f}  ms     ║")
    print(f"║  Freq real : {1000/np.mean(l_ms):5.1f} Hz                                  ║")
    print(f"║  Overruns  : {n_overrun:4d} / {len(l_ms)} iters ({n_overrun/len(l_ms)*100:.1f} %)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    print(f"[v_θ]  mean={np.mean(vel_progres):6.3f}  max={np.max(vel_progres):6.3f}  "
          f"min={np.min(vel_progres):6.3f}")
    print(f"[v_r]  mean={np.mean(vel_real):6.3f}  max={np.max(vel_real):6.3f}  "
          f"min={np.min(vel_real):6.3f}")
    mask = vel_progres[0, :] > 0.1
    if np.any(mask):
        ratio_mean = np.mean(vel_real[0, mask] / vel_progres[0, mask])
        print(f"[ratio v_real/v_θ]  mean={ratio_mean:5.3f}  (only where v_θ>0.1)")
    print(f"[θ  ]  final={x[14, N_sim]:8.3f}  /  {s_max:.3f} m  "
          f"({x[14, N_sim]/s_max*100:.1f}%)\n")

    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Results_DQ_MPCC"
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"Path {pwd} does not exist. Using local directory.")
        pwd = os.path.dirname(os.path.abspath(__file__))

    experiment_number = 1
    weight_summary = get_active_weight_summary("dq")
    classic_name_file = f"Results_DQ_MPCC_baseline_{experiment_number}.mat"
    tagged_name_file = f"Results_DQ_MPCC_baseline_{weight_summary['label']}.mat"

    mat_payload = {
        "states": x,
        "states_std": x_std,
        "T_control": u_control,
        "time": t_plot,
        "ref": xref[:, :N_sim + 1],
        "e_total": e_total,
        "e_contorno": e_contorno,
        "e_arrastre": e_arrastre,
        "vel_progres": vel_progres,
        "vel_real": vel_real,
        "vel_tangent": vel_tangent,
        "theta_history": theta_history,
        "quat_d_theta": quat_d_theta,
        "s_max": s_max,
        "t_lap": t_lap,
        "t_solver_mean": np.mean(t_solver) * 1e3,
        "t_solver_max": np.max(t_solver) * 1e3,
        "t_solver_std": np.std(t_solver) * 1e3,
        "controller_name": "DQ-MPCC",
        "gain_label": weight_summary["label"],
        "gain_source": weight_summary["source"],
    }
    mat_payload.update(flatten_weight_summary("dq", weight_summary))

    classic_path = os.path.join(pwd, classic_name_file)
    tagged_path = os.path.join(pwd, tagged_name_file)
    savemat(classic_path, mat_payload)
    savemat(tagged_path, mat_payload)
    print(f"✓ Results saved to {classic_path}")
    print(f"✓ Tagged results saved to {tagged_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
    else:
        print("Complete Execution")
