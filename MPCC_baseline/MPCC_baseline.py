"""
MPCC_baseline.py  –  Strict MPCC for quadrotor trajectory tracking.

Strict MPCC formulation:
  • θ (arc-length progress) is an optimisation STATE   (x[13])
  • v_θ (progress velocity) is an optimisation STATE   (x[14])
  • a_θ (progress acceleration) is an optimisation CONTROL (u[4])
  • Dynamics:  θ̇ = v_θ,  v̇_θ = a_θ
  • The solver maximises v_θ while smoothing it through a_θ.
  • Reference points are evaluated at predicted θ values (per shooting node).

Uses the modular project structure:
    utils/   → quaternion & rotation helpers
    models/  → quadrotor dynamics (CasADi / acados)
    ocp/     → MPCC formulation & solver build
"""

import numpy as np
import time
import time as time_module
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat

# ── Add parent directory to path for shared config ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_config import (
    P0, Q0, V0, W0, THETA0,
    T_FINAL, TRAJECTORY_T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS, S_MAX_MANUAL,
    VTHETA_MAX, VTHETA_MIN,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    USE_TUNED_WEIGHTS_MPCC, MPCC_WEIGHTS_PATH, MPCC_WEIGHTS,
    trayectoria,
)
from config.tuning_registry import get_active_weight_summary, flatten_weight_summary

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import (
    euler_to_quaternion,
    quat_interp_by_arc,
    mpcc_errors,
    rk4_step_mpcc,
    compute_curvature,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi,
    create_tangent_interpolator_casadi,
    create_quat_interpolator_casadi,
)
from ocp.mpcc_controller import (
    build_mpcc_solver,
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants — imported from experiment_config.py
#  (S_MAX_MANUAL, T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS,
#   P0, Q0, V0, W0, THETA0, trayectoria are all set in experiment_config.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Rebuild control for the MPCC OCP.
#
# False:
#   Reuse the existing compiled solver in `MPCC_baseline/c_generated_code_mpcc_runtime/`.
#   Use this when you only want to change NUMERIC runtime parameters such as:
#     - `Q_ec`, `Q_el`, `Q_q`, `U_mat`, `Q_omega`, `Q_s`
#     - input limits set with `constraints_set`
#
# True:
#   Force regeneration and recompilation of the OCP.
#   Set this after changing any STRUCTURAL quantity,
#   especially:
#     - trajectory definition in `trayectoria()`
#     - `TRAJECTORY_T_FINAL`, `FREC`, `T_PREDICTION`, `N_WAYPOINTS`, `S_MAX_MANUAL`
#     - `T_MAX`, `T_MIN`, `TAUX_MAX`, `TAUY_MAX`, `TAUZ_MAX`
#     - state/control dimensions or symbolic cost structure
#
# In the current design, weights are runtime parameters in `p ∈ R^18`.
# The trajectory/interpolation graph is still compiled.
FORCE_REBUILD_OCP = True


def _check_solver_staleness(cache_path, solver_json_path):
    """Warn if the compiled solver is older than the path cache."""
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




# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Timing configuration (from experiment_config.py) ─────────────────
    t_final = T_FINAL
    t_path_final = TRAJECTORY_T_FINAL
    frec    = FREC
    t_s     = 1 / frec
    t_prediction = T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")
    print(f"[PATH]    trajectory_t_final={t_path_final} s  |  simulation_t_final={t_final} s")

    # ── Time vector ──────────────────────────────────────────────────────
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

    if S_MAX_MANUAL is not None and S_MAX_MANUAL < s_max_full:
        print(f"[ARC]  Total arc length = {s_max_full:.3f} m  →  "
              f"LIMITED to s_max = {s_max:.3f} m")
    else:
        print(f"[ARC]  Total arc length = {s_max:.3f} m")

    print(f"[PATH]   available_length = {s_max_full:.3f} m")
    print(f"[PATH]   active_s_max     = {s_max:.3f} m")
    print(f"[SIM]    time_limit       = {t_final:.3f} s")

    # ── Storage vectors ──────────────────────────────────────────────────
    delta_t        = np.zeros((1, N_sim), dtype=np.double)
    e_contorno     = np.zeros((3, N_sim), dtype=np.double)
    e_arrastre     = np.zeros((3, N_sim), dtype=np.double)
    e_total        = np.zeros((3, N_sim), dtype=np.double)
    vel_progres    = np.zeros((1, N_sim), dtype=np.double)     # v_θ state
    vel_real       = np.zeros((1, N_sim), dtype=np.double)     # real progress speed (dot(tangent, v))
    vel_tangent    = np.zeros((1, N_sim), dtype=np.double)     # ‖v(t)‖ actual speed
    theta_history  = np.zeros((1, N_sim + 1), dtype=np.double) # θ state
    quat_d_theta   = np.zeros((4, N_sim), dtype=np.double)     # desired quaternion at θ_k (θ-indexed)
    t_solver       = np.zeros((1, N_sim), dtype=np.double)
    t_loop         = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state (15-dim: [p, v, q, ω, θ₀, v_θ0]) ──────────────────
    #    All initial conditions come from experiment_config.py
    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)    # ensure unit quaternion

    x = np.zeros((15, N_sim + 1), dtype=np.double)
    x[:, 0] = [P0[0], P0[1], P0[2],                   # position  ℝ³
               V0[0], V0[1], V0[2],                    # velocity  ℝ³
               q0_normed[0], q0_normed[1],              # quaternion ℍ
               q0_normed[2], q0_normed[3],
               W0[0], W0[1], W0[2],                    # angular velocity ℝ³
               THETA0, VTHETA_MIN]                      # progress states
    theta_history[0, 0] = x[13, 0]
    print(f"[IC]  p0 = {P0}  |  q0 = {np.round(q0_normed,4)}  |  θ₀ = {THETA0}")

    # ── Reference for plotting (17-dim, time-indexed) ────────────────────
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :]  = pos_ref[0, :]   # px_d
    xref[1, :]  = pos_ref[1, :]   # py_d
    xref[2, :]  = pos_ref[2, :]   # pz_d
    xref[3, :]  = dp_ds[0, :]     # tangent_x
    xref[4, :]  = dp_ds[1, :]     # tangent_y
    xref[5, :]  = dp_ds[2, :]     # tangent_z
    # ── Control storage (5-dim: T, τx, τy, τz, a_θ) ─────────────────────
    u_control = np.zeros((5, N_sim), dtype=np.double)

    gamma_pos  = create_position_interpolator_casadi(s_wp, pos_wp)
    gamma_vel  = create_tangent_interpolator_casadi(s_wp, tang_wp)
    gamma_quat = create_quat_interpolator_casadi(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints "
          f"(s_max_solver={S_MAX_SOLVER:.2f})")
    print(f"[PATH]   terminal_extension = {delta_s_terminal:.3f} m")
    print(f"[ATT]    ref_speed={ATTITUDE_REF_SPEED:.2f} m/s  |  max_tilt={ATTITUDE_REF_MAX_TILT_DEG:.1f} deg")

    # ── Staleness check ──────────────────────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cache_path = os.path.join(_script_dir, "path_cache.npz")
    _solver_json = os.path.join(_script_dir, "acados_ocp_Drone_ode_complete_runtime_bspline.json")
    _check_solver_staleness(_cache_path, _solver_json)

    for i, sv in enumerate(arc_lengths):
        qd_i = quat_interp_by_arc(min(sv, S_MAX_SOLVER), s_wp, quat_wp)
        xref[6, i] = qd_i[0]
        xref[7, i] = qd_i[1]
        xref[8, i] = qd_i[2]
        xref[9, i] = qd_i[3]

    if USE_TUNED_WEIGHTS_MPCC and MPCC_WEIGHTS_PATH.is_file():
        print(f"[WEIGHTS] Using tuned MPCC weights from {MPCC_WEIGHTS_PATH}")
    else:
        print("[WEIGHTS] Using manual MPCC weights from experiment_config.py")

    # ── Build solver ─────────────────────────────────────────────────────────
    #    Compile once, then reuse.
    #    Numeric weights are injected later through p ∈ R^18.
    acados_ocp_solver, ocp, model, f = build_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=S_MAX_SOLVER,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True, force_rebuild=FORCE_REBUILD_OCP,
    )

    p_base = weights_to_param_vector(MPCC_WEIGHTS)

    nx = model.x.size()[0]   # 15
    nu = model.u.size()[0]   # 5

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start + runtime numeric parameters
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
        acados_ocp_solver.set(stage, "p", p_base)
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    apply_input_bounds(acados_ocp_solver, N_prediction, S_MAX_SOLVER, vtheta_max=p_base[17])

    print("Initializing simulation...")

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop  (strict MPCC: reference computed from predicted θ)
    # ══════════════════════════════════════════════════════════════════════
    print("Ready!!!")
    time_all = time.time()
    t_lap    = np.nan          # time of first lap completion [s]
    stop_tol = 1e-6

    for k in range(N_sim):
        tic = time.time()

        # ── Stop when the requested active path length is reached ────────
        if x[13, k] >= s_max - stop_tol:
            if np.isnan(t_lap):
                t_lap = k * t_s
            print(f"[k={k:04d}]  Active path complete at θ={x[13,k]:.3f} m  "
                  f"→  t_lap = {t_lap:.3f} s")
            N_sim = k
            break

        # ── Set initial state (15-dim) ───────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Solve ────────────────────────────────────────────────────────
        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        # ── Read predicted trajectory (for next iteration warm-start) ────
        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        # ── Get optimal control (5-dim) ──────────────────────────────────
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        # ── Store v_θ and θ ──────────────────────────────────────────────
        vel_progres[:, k]       = x[14, k]           # v_θ state
        theta_history[:, k]     = x[13, k]            # current θ

        # Real progress speed = projection of UAV velocity onto tangent
        theta_k_now = np.clip(x[13, k], 0.0, s_max)
        tang_k_now  = tangent_by_arc_length(theta_k_now)
        vel_real[:, k]    = np.dot(tang_k_now, x[3:6, k])
        vel_tangent[:, k] = np.linalg.norm(x[3:6, k])       # ‖v‖ actual speed

        # ── System evolution (augmented RK4, 15 states) ──────────────────
        x[:, k + 1] = rk4_step_mpcc(x[:, k], u_control[:, k], t_s, f)

        # Clamp progress states to the admissible range.
        x[13, k + 1] = np.clip(x[13, k + 1], 0.0, S_MAX_SOLVER)
        x[14, k + 1] = np.clip(x[14, k + 1], VTHETA_MIN, p_base[17])
        theta_history[:, k + 1] = x[13, k + 1]

        # Stop immediately when the trajectory crosses the active path limit.
        # The solver still saw the extended path over the horizon, so this
        # does not force early braking; it only truncates the nominal rollout
        # at the crossing event for plotting and metrics.
        if x[13, k + 1] >= s_max - stop_tol:
            alpha_term = 1.0
            theta_prev = x[13, k]
            theta_next = x[13, k + 1]
            if theta_next > theta_prev + 1e-12:
                alpha_term = (s_max - theta_prev) / (theta_next - theta_prev)
                alpha_term = float(np.clip(alpha_term, 0.0, 1.0))
                x[:, k + 1] = x[:, k] + alpha_term * (x[:, k + 1] - x[:, k])
                q_interp = x[6:10, k + 1]
                q_norm = np.linalg.norm(q_interp)
                if q_norm > 1e-12:
                    x[6:10, k + 1] = q_interp / q_norm
            x[13, k + 1] = s_max
            theta_history[:, k + 1] = s_max
            if np.isnan(t_lap):
                t_lap = (k + alpha_term) * t_s
            N_sim = k + 1
            print(f"[k={k+1:04d}]  Crossed active s_max at θ={x[13,k+1]:.3f} m  "
                  f"→  t_lap = {t_lap:.3f} s")
            break

        # ── Rate control — run in real time ──────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        # ── Compute MPCC errors (using current θ-based reference) ────────
        theta_k = min(x[13, k], s_max)
        sd_k    = position_by_arc_length(theta_k)
        tang_k  = tangent_by_arc_length(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(x[0:3, k], tang_k, sd_k)

        # ── Desired quaternion at θ_k (for θ-indexed orientation error) ──
        quat_d_theta[:, k] = quat_interp_by_arc(theta_k, s_wp, quat_wp)

        # ── Print progress ───────────────────────────────────────────────
        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0,k] / (vel_progres[0,k] + 1e-8)
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  a_θ={u_control[4,k]:6.2f}  v_real={vel_real[0,k]:5.2f}  "
              f"ratio={ratio_vtheta:4.2f}  |  "
              f"θ={x[13,k]:7.2f}/{s_max:.0f} m  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing  (trim all arrays to the actual run length N_sim)
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - time_all
    print(f"\nTiempo total de ejecución: {total_time:.4f} segundos")
    print(f"Final θ = {x[13, N_sim]:.3f} / {s_max:.3f} m  "
          f"({x[13, N_sim]/s_max*100:.1f}% of path)")

    # Trim all storage arrays to actual run length
    x            = x[:, :N_sim + 1]
    u_control    = u_control[:, :N_sim]
    e_contorno   = e_contorno[:, :N_sim]
    e_arrastre   = e_arrastre[:, :N_sim]
    e_total      = e_total[:, :N_sim]
    vel_progres  = vel_progres[:, :N_sim]
    vel_real     = vel_real[:, :N_sim]
    vel_tangent  = vel_tangent[:, :N_sim]
    theta_history= theta_history[:, :N_sim + 1]
    quat_d_theta = quat_d_theta[:, :N_sim]
    t_solver     = t_solver[:, :N_sim]
    t_loop       = t_loop[:, :N_sim]
    t_plot       = t[:N_sim + 1]

    print("Generating figures...")

    # Build reference from θ history for plotting
    xref_theta = np.zeros((17, N_sim + 1))
    for i in range(N_sim + 1):
        s_i = theta_history[0, i]
        pos_i  = position_by_arc_length(s_i)
        tang_i = tangent_by_arc_length(s_i)
        xref_theta[0:3, i]  = pos_i
        xref_theta[3:6, i]  = tang_i

    # ── Compute path curvature for analysis plot ─────────────────────────
    curvature = compute_curvature(position_by_arc_length, s_max, N_samples=500)

    # ── Save all figures inside MPCC_baseline/figures/ ───────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _figures_dir = os.path.join(_script_dir, "figures")
    os.makedirs(_figures_dir, exist_ok=True)

    fig1 = plot_pose(x[:13, :], xref_theta, t_plot)
    fig1.savefig(os.path.join(_figures_dir, "1_pose.png"))
    print("✓ Saved figures/1_pose.png")

    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig(os.path.join(_figures_dir, "2_control_actions.png"))
    print("✓ Saved figures/2_control_actions.png")

    fig3 = plot_vel_lineal(x[3:6, :], t_plot)
    fig3.savefig(os.path.join(_figures_dir, "3_vel_lineal.png"))
    print("✓ Saved figures/3_vel_lineal.png")

    fig4 = plot_vel_angular(x[10:13, :], t_plot)
    fig4.savefig(os.path.join(_figures_dir, "4_vel_angular.png"))
    print("✓ Saved figures/4_vel_angular.png")

    # ── Velocity analysis: v_θ, v_real, ‖v‖, curvature ──────────────────
    fig5 = plot_velocity_analysis(
        vel_progres, vel_real, vel_tangent,
        curvature, theta_history, s_max, t_plot[:N_sim])
    fig5.savefig(os.path.join(_figures_dir, "5_velocity_analysis.png"), dpi=150)
    print("✓ Saved figures/5_velocity_analysis.png")

    # ── 3D trajectory ────────────────────────────────────────────────────
    fig6 = plot_3d_trajectory(
        x, pos_ref, s_max=s_max,
        position_by_arc=position_by_arc_length, N_plot=600,
        theta_history=theta_history,
        s_max_solver=S_MAX_SOLVER,
        position_by_arc_solver=position_by_arc_length,
        theta_anchor=float(theta_history[0, -1]),
        theta_future_end=float(min(theta_history[0, -1] + delta_s_terminal, S_MAX_SOLVER)))
    fig6.savefig(os.path.join(_figures_dir, "6_trajectory_3d.png"), dpi=150)
    print("✓ Saved figures/6_trajectory_3d.png")

    fig7 = _plot_quaternion_tracking(t_plot[:N_sim], x[6:10, :N_sim], quat_d_theta)
    fig7.savefig(os.path.join(_figures_dir, "7_quaternion_tracking.png"), dpi=150)
    print("✓ Saved figures/7_quaternion_tracking.png")

    # ── Progress velocity (simple version) ───────────────────────────────
    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig(os.path.join(_figures_dir, "8_progress_velocity.png"), dpi=150)
    print("✓ Saved figures/8_progress_velocity.png")

    # ── Timing statistics ────────────────────────────────────────────────
    s_ms  = t_solver[0, :] * 1e3
    l_ms  = t_loop[0, :]   * 1e3
    ts_ms = t_s * 1e3
    n_overrun = int(np.sum(l_ms > ts_ms * 1.05))

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(  "║                     TIMING STATISTICS                          ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Nominal t_s = {ts_ms:5.2f} ms  ({frec:.0f} Hz)                              ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  [Solver]  mean={np.mean(s_ms):5.2f}  max={np.max(s_ms):5.2f}  std={np.std(s_ms):4.2f}  ms     ║")
    print(f"║  [Loop  ]  mean={np.mean(l_ms):5.2f}  max={np.max(l_ms):5.2f}  std={np.std(l_ms):4.2f}  ms     ║")
    print(f"║  Freq real : {1000/np.mean(l_ms):5.1f} Hz                                  ║")
    print(f"║  Overruns  : {n_overrun:4d} / {len(l_ms)} iters ({n_overrun/len(l_ms)*100:.1f} %)                 ║")
    print(  "╚══════════════════════════════════════════════════════════════════╝\n")

    # ── v_θ and θ statistics ─────────────────────────────────────────────
    print(f"[v_θ]  mean={np.mean(vel_progres):6.3f}  max={np.max(vel_progres):6.3f}  "
          f"min={np.min(vel_progres):6.3f}")
    print(f"[v_r]  mean={np.mean(vel_real):6.3f}  max={np.max(vel_real):6.3f}  "
          f"min={np.min(vel_real):6.3f}")
    # Only compute ratio where v_θ > 0 to avoid divide-by-near-zero noise
    mask = vel_progres[0, :] > 0.1
    if np.any(mask):
        ratio_mean = np.mean(vel_real[0, mask] / vel_progres[0, mask])
        print(f"[ratio v_real/v_θ]  mean={ratio_mean:5.3f}  (only where v_θ>0.1)")
    print(f"[θ  ]  final={x[13, N_sim]:8.3f}  /  {s_max:.3f} m  "
          f"({x[13, N_sim]/s_max*100:.1f}%)\n")

    # ── Save results ─────────────────────────────────────────────────────
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Results_MPCC"
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Usando directorio local.")
        pwd = os.path.dirname(os.path.abspath(__file__))

    experiment_number = 1
    weight_summary = get_active_weight_summary("mpcc")
    classic_name_file = f"Results_MPCC_baseline_{experiment_number}.mat"
    tagged_name_file = f"Results_MPCC_baseline_{weight_summary['label']}.mat"

    mat_payload = {
        'states': x,
        'T_control': u_control,
        'time': t_plot,
        'ref': xref_theta,                   # θ-indexed reference (aligned with actual trajectory)
        'e_total': e_total,
        'e_contorno': e_contorno,
        'e_arrastre': e_arrastre,
        'vel_progres': vel_progres,
        'vel_real': vel_real,
        'vel_tangent': vel_tangent,
        'theta_history': theta_history,
        'quat_d_theta': quat_d_theta,        # desired quaternion at θ_k (θ-indexed, 4×N)
        's_max': s_max,
        't_lap': t_lap,                      # [s] time to complete one full lap (KPI)
        't_solver_mean': np.mean(t_solver) * 1e3,   # [ms]
        't_solver_max':  np.max(t_solver)  * 1e3,   # [ms]
        't_solver_std':  np.std(t_solver)  * 1e3,   # [ms]
        'controller_name': 'Baseline MPCC',
        'gain_label': weight_summary['label'],
        'gain_source': weight_summary['source'],
    }
    mat_payload.update(flatten_weight_summary("mpcc", weight_summary))

    classic_path = os.path.join(pwd, classic_name_file)
    tagged_path = os.path.join(pwd, tagged_name_file)
    savemat(classic_path, mat_payload)
    savemat(tagged_path, mat_payload)
    print(f"✓ Results saved to {classic_path}")
    print(f"✓ Tagged results saved to {tagged_path}")


if __name__ == '__main__':
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
