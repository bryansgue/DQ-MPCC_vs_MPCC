"""Standalone strict MPCC baseline with rate-control dynamics (MiL)."""

import os
import sys
import time
import time as time_module
from scipy.io import savemat

import numpy as np

_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_WORKSPACE_ROOT)
_SHARED_MPCC_ROOT = os.path.join(_WORKSPACE_ROOT, "MPCC_baseline")
for _path in (_WORKSPACE_ROOT, _SHARED_MPCC_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.casadi_utils import (
    create_position_interpolator_casadi as create_casadi_position_interpolator,
    create_quat_interpolator_casadi as create_casadi_quat_interpolator,
    create_tangent_interpolator_casadi as create_casadi_tangent_interpolator,
)
from utils.graficas import (
    plot_3d_trajectory,
    plot_control,
    plot_error,
    plot_pose,
    plot_progress_velocity,
    plot_timing,
    plot_vel_angular,
    plot_vel_lineal,
    plot_velocity_analysis,
)
from utils.numpy_utils import (
    compute_curvature,
    euler_to_quaternion,
    mpcc_errors,
    rk4_step_mpcc,
)
from MPCC_baseline_rates.config.experiment_config import (
    FREC,
    MPCC_Q_EC,
    MPCC_Q_EL,
    MPCC_Q_OMEGA,
    MPCC_Q_Q,
    MPCC_Q_S,
    MPCC_RATE_U_MAT,
    N_WAYPOINTS,
    P0,
    Q0,
    S_MAX_MANUAL,
    T_MAX,
    T_FINAL,
    THETA0,
    T_PREDICTION,
    V0,
    VTHETA_MAX,
    W_MAX,
    W0,
)
from MPCC_baseline_rates.ocp.mpcc_controller_rate import build_mpcc_rate_solver
from MPCC_baseline_rates.path_loader import load_path


def main():
    t_s = 1.0 / FREC
    N_prediction = int(round(T_PREDICTION / t_s))
    print(
        f"[CONFIG]  frec={FREC} Hz  |  t_s={t_s*1e3:.2f} ms  "
        f"|  t_prediction={T_PREDICTION} s  |  N_prediction={N_prediction} steps"
    )
    print(
        f"[WEIGHTS]  Qec={np.round(MPCC_Q_EC, 3)}  |  Qel={np.round(MPCC_Q_EL, 3)}  "
        f"|  Qq={np.round(MPCC_Q_Q, 3)}  |  Qw={np.round(MPCC_Q_OMEGA, 3)}  "
        f"|  Qs={MPCC_Q_S:.3f}  |  U={np.round(MPCC_RATE_U_MAT, 3)}"
    )

    t = np.arange(0, T_FINAL + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Arc-length parameterisation (load from cache or build) ────────────
    (s_wp, pos_wp, tang_wp, quat_wp,
     s_max_full, s_max_solver, s_max,
     arc_lengths, pos_ref,
     position_by_arc_length, tangent_by_arc_length) = load_path()
    print(f"[ARC]  s_max = {s_max:.3f} m  |  s_max_full = {s_max_full:.3f} m")

    delta_t = np.zeros((1, N_sim), dtype=np.double)
    e_contorno = np.zeros((3, N_sim), dtype=np.double)
    e_arrastre = np.zeros((3, N_sim), dtype=np.double)
    e_total = np.zeros((3, N_sim), dtype=np.double)
    e_ref_norm = np.zeros((1, N_sim), dtype=np.double)
    e_path_nearest = np.zeros((1, N_sim), dtype=np.double)
    vel_progres = np.zeros((1, N_sim), dtype=np.double)
    vel_real = np.zeros((1, N_sim), dtype=np.double)
    vel_tangent = np.zeros((1, N_sim), dtype=np.double)
    theta_history = np.zeros((1, N_sim + 1), dtype=np.double)
    quat_d_theta = np.zeros((4, N_sim), dtype=np.double)
    t_solver = np.zeros((1, N_sim), dtype=np.double)
    t_loop = np.zeros((1, N_sim), dtype=np.double)

    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)
    x = np.zeros((14, N_sim + 1), dtype=np.double)
    x[:, 0] = [
        P0[0], P0[1], P0[2],
        V0[0], V0[1], V0[2],
        q0_normed[0], q0_normed[1], q0_normed[2], q0_normed[3],
        W0[0], W0[1], W0[2],
        THETA0,
    ]
    theta_history[0, 0] = THETA0
    print(f"[IC]  p0 = {P0}  |  q0 = {np.round(q0_normed, 4)}  |  theta0 = {THETA0}")

    # Reference for plotting (arc-length dense array, truncated to N_sim+1)
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    norms = np.linalg.norm(dp_ds, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    dp_ds /= norms
    _n_ref = min(pos_ref.shape[1], N_sim + 1)
    xref = np.zeros((17, _n_ref))
    xref[0:3, :] = pos_ref[:, :_n_ref]
    xref[3:6, :] = dp_ds[:, :_n_ref]

    u_control = np.zeros((5, N_sim), dtype=np.double)

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)
    print(
        f"[INTERP] CasADi interpolation with {len(s_wp)} waypoints "
        f"(s_max_solver={s_max_solver:.2f})"
    )
    print(
        f"[LIMITS]  T=[0,{T_MAX:.2f}] N  "
        f"|  W_max={W_MAX:.2f} rad/s  "
        f"|  v_theta_max={VTHETA_MAX:.2f} m/s"
    )

    acados_ocp_solver, ocp, model, f = build_mpcc_rate_solver(
        x[:, 0],
        N_prediction,
        T_PREDICTION,
        s_max_solver,
        gamma_pos,
        gamma_vel,
        gamma_quat,
        use_cython=False,
    )

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))
    u_hover = np.array([9.81, 0.0, 0.0, 0.0, 0.0], dtype=np.double)
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", u_hover)

    print("Ready!!!")
    time_all = time.time()
    t_lap = np.nan

    for k in range(N_sim):
        tic = time.time()

        if x[13, k] >= s_max:
            t_lap = k * t_s
            print(f"[k={k:04d}]  Path complete at theta={x[13, k]:.3f} m  ->  t_lap = {t_lap:.3f} s")
            N_sim = k
            break

        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        dt_stage = T_PREDICTION / N_prediction
        theta_k_cur = x[13, k]
        vtheta_cur = u_control[4, max(k - 1, 0)]
        for stage in range(N_prediction + 1):
            theta_pred = theta_k_cur + stage * dt_stage * vtheta_cur
            acados_ocp_solver.set(
                stage, "p", np.array([0.0 if theta_pred >= s_max else VTHETA_MAX], dtype=np.double)
            )

        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        u_control[:, k] = acados_ocp_solver.get(0, "u")
        vel_progres[:, k] = u_control[4, k]
        theta_history[:, k] = x[13, k]

        theta_k_now = np.clip(x[13, k], 0.0, s_max)
        tang_k_now = tangent_by_arc_length(theta_k_now)
        vel_real[:, k] = np.dot(tang_k_now, x[3:6, k])
        vel_tangent[:, k] = np.linalg.norm(x[3:6, k])

        x[:, k + 1] = rk4_step_mpcc(x[:, k], u_control[:, k], t_s, f)
        x[6:10, k + 1] /= np.linalg.norm(x[6:10, k + 1]) + 1e-12
        x[13, k + 1] = np.clip(x[13, k + 1], 0.0, s_max_solver)
        theta_history[:, k + 1] = x[13, k + 1]

        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        theta_k = x[13, k]
        sd_k = position_by_arc_length(theta_k)
        tang_k = tangent_by_arc_length(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = mpcc_errors(x[0:3, k], tang_k, sd_k)
        quat_d_theta[:, k] = euler_to_quaternion(0.0, 0.0, np.arctan2(tang_k[1], tang_k[0]))
        e_l_scalar = float(np.dot(tang_k, sd_k - x[0:3, k]))
        e_c_norm = float(np.linalg.norm(e_contorno[:, k]))
        e_ref = float(np.linalg.norm(sd_k - x[0:3, k]))
        d_path_nearest = float(np.min(np.linalg.norm(pos_ref.T - x[0:3, k], axis=1)))
        e_ref_norm[:, k] = e_ref
        e_path_nearest[:, k] = d_path_nearest
        theta_step = float(x[13, k + 1] - x[13, k])

        overrun = " OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0, k] / max(vel_progres[0, k], 1e-8)
        print(
            f"[k={k:04d}]  solver={t_solver[0, k]*1e3:5.2f} ms  "
            f"|  status={status:2d}  "
            f"|  v_theta={vel_progres[0, k]:5.2f}  v_real={vel_real[0, k]:5.2f}  "
            f"ratio={ratio_vtheta:4.2f}  |  "
            f"theta={x[13, k]:6.2f}->{x[13, k + 1]:6.2f}/{s_max:.0f}  "
            f"dth={theta_step:4.2f}  |  "
            f"ec={e_c_norm:5.3f}  el={e_l_scalar:+6.3f}  "
            f"|  eref={e_ref:5.3f}  dmin={d_path_nearest:5.3f}  |  "
            f"T={u_control[0, k]:5.1f}N  w=[{u_control[1, k]:+.2f},{u_control[2, k]:+.2f},{u_control[3, k]:+.2f}]  |  "
            f"{1 / max(t_loop[0, k], 1e-9):5.1f} Hz{overrun}"
        )

    total_time = time.time() - time_all
    print(f"\nTiempo total de ejecucion: {total_time:.4f} segundos")
    print(f"Final theta = {x[13, N_sim]:.3f} / {s_max:.3f} m  ({x[13, N_sim] / s_max * 100:.1f}% of path)")

    x = x[:, :N_sim + 1]
    u_control = u_control[:, :N_sim]
    e_contorno = e_contorno[:, :N_sim]
    e_arrastre = e_arrastre[:, :N_sim]
    e_total = e_total[:, :N_sim]
    e_ref_norm = e_ref_norm[:, :N_sim]
    e_path_nearest = e_path_nearest[:, :N_sim]
    vel_progres = vel_progres[:, :N_sim]
    vel_real = vel_real[:, :N_sim]
    vel_tangent = vel_tangent[:, :N_sim]
    theta_history = theta_history[:, :N_sim + 1]
    quat_d_theta = quat_d_theta[:, :N_sim]
    t_solver = t_solver[:, :N_sim]
    t_loop = t_loop[:, :N_sim]
    t_plot = t[:N_sim + 1]
    t_sample = t_s * np.ones((1, N_sim), dtype=np.double)

    xref_theta = np.zeros((17, N_sim + 1))
    for i in range(N_sim + 1):
        s_i = theta_history[0, i]
        xref_theta[0:3, i] = position_by_arc_length(s_i)
        xref_theta[3:6, i] = tangent_by_arc_length(s_i)

    curvature = compute_curvature(position_by_arc_length, s_max, N_samples=500)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    def _save(fig, name):
        fig.savefig(os.path.join(results_dir, name), dpi=150)
        print(f"Saved {name}")

    print("Generating figures...")
    print(
        f"[SUMMARY]  ec_rms={np.sqrt(np.mean(np.sum(e_contorno**2, axis=0))):.4f} m  "
        f"|  eref_rms={np.sqrt(np.mean(e_ref_norm**2)):.4f} m  "
        f"|  dmin_rms={np.sqrt(np.mean(e_path_nearest**2)):.4f} m  "
        f"|  dmin_max={np.max(e_path_nearest):.4f} m"
    )
    _save(plot_pose(x[:13, :], xref_theta, t_plot), "1_pose.png")
    _save(plot_control(u_control[:4, :], t_plot[:N_sim]), "2_control_actions.png")
    _save(plot_vel_lineal(x[3:6, :], t_plot), "3_vel_lineal.png")
    _save(plot_vel_angular(x[10:13, :], t_plot), "4_vel_angular.png")
    _save(
        plot_velocity_analysis(vel_progres, vel_real, vel_tangent, curvature, theta_history, s_max, t_plot[:N_sim]),
        "5_velocity_analysis.png",
    )
    _save(plot_3d_trajectory(x[0:3, :], xref_theta[0:3, :]), "6_trajectory_3d.png")
    _save(plot_error(e_total, t_plot[:N_sim]), "7_error.png")
    _save(plot_progress_velocity(vel_progres, vel_real, theta_history[:, :N_sim], t_plot[:N_sim]), "8_progress_velocity.png")
    _save(plot_timing(t_solver, t_loop, t_sample, t_plot[:N_sim]), "9_timing.png")

    savemat(
        os.path.join(results_dir, "Results_MPCC_baseline_rates.mat"),
        {
            "x": x,
            "xref_theta": xref_theta,
            "u_control": u_control,
            "theta_history": theta_history,
            "e_contorno": e_contorno,
            "e_arrastre": e_arrastre,
            "e_total": e_total,
            "e_ref_norm": e_ref_norm,
            "e_path_nearest": e_path_nearest,
            "vel_progres": vel_progres,
            "vel_real": vel_real,
            "vel_tangent": vel_tangent,
            "t_solver": t_solver,
            "t_loop": t_loop,
            "t_plot": t_plot,
            "t_lap": t_lap,
            "s_max": s_max,
            "quat_d_theta": quat_d_theta,
        },
    )
    print("Saved Results_MPCC_baseline_rates.mat")


if __name__ == "__main__":
    main()
