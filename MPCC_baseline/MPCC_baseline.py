"""
MPCC_baseline.py  –  Strict MPCC for quadrotor trajectory tracking.

Strict MPCC formulation:
  • θ (arc-length progress) is an optimisation STATE   (x[13])
  • v_θ (progress velocity) is an optimisation CONTROL (u[4])
  • Dynamics:  θ̇ = v_θ
  • The solver maximises v_θ to push the UAV along the path.
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
from scipy.io import savemat

# ── Add parent directory to path for shared config ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_config import (
    P0, Q0, V0, W0, THETA0,
    VALUE, T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS, S_MAX_MANUAL,
)

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import (
    euler_to_quaternion,
    build_arc_length_parameterisation,
    build_waypoints,
    mpcc_errors,
    rk4_step_mpcc,
    compute_curvature,
)
from utils.casadi_utils import (
    create_position_interpolator_casadi as create_casadi_position_interpolator,
    create_tangent_interpolator_casadi  as create_casadi_tangent_interpolator,
    create_quat_interpolator_casadi     as create_casadi_quat_interpolator,
)
from ocp.mpcc_controller import build_mpcc_solver
from utils.graficas import (
    plot_pose, plot_error, plot_time, plot_control,
    plot_vel_lineal, plot_vel_angular,
    plot_progress_velocity, plot_velocity_analysis,
    plot_3d_trajectory, plot_timing,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants — imported from experiment_config.py
#  (VALUE, S_MAX_MANUAL, T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS,
#   P0, Q0, V0, W0, THETA0 are all set in experiment_config.py)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory generators
# ═══════════════════════════════════════════════════════════════════════════════

def trayectoria(t):
    v = VALUE
    xd   = lambda t: 7 * np.sin(v * 0.04 * t) + 3
    yd   = lambda t: 7 * np.sin(v * 0.08 * t)
    zd   = lambda t: 1.5 * np.sin(v * 0.08 * t) + 6
    xd_p = lambda t: 7 * v * 0.04 * np.cos(v * 0.04 * t)
    yd_p = lambda t: 7 * v * 0.08 * np.cos(v * 0.08 * t)
    zd_p = lambda t: 1.5 * v * 0.08 * np.cos(v * 0.08 * t)
    return xd, yd, zd, xd_p, yd_p, zd_p


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Timing configuration (from experiment_config.py) ─────────────────
    t_final = T_FINAL
    frec    = FREC
    t_s     = 1 / frec
    t_prediction = T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")

    # ── Time vector ──────────────────────────────────────────────────────
    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Trajectory A (UAV path) ──────────────────────────────────────────
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)

    # Arc-length parameterisation — delegates to utils.numpy_utils
    t_finer = np.linspace(0, t_final, len(t))
    arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, \
        s_max_full = build_arc_length_parameterisation(
            xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    # ── Optional manual arc-length limit ─────────────────────────────────
    if S_MAX_MANUAL is not None and S_MAX_MANUAL < s_max_full:
        s_max = float(S_MAX_MANUAL)
        print(f"[ARC]  Total arc length = {s_max_full:.3f} m  →  "
              f"LIMITED to s_max = {s_max:.3f} m")
    else:
        s_max = s_max_full
        print(f"[ARC]  Total arc length = {s_max:.3f} m")

    # ── Storage vectors ──────────────────────────────────────────────────
    delta_t        = np.zeros((1, N_sim), dtype=np.double)
    e_contorno     = np.zeros((3, N_sim), dtype=np.double)
    e_arrastre     = np.zeros((3, N_sim), dtype=np.double)
    e_total        = np.zeros((3, N_sim), dtype=np.double)
    vel_progres    = np.zeros((1, N_sim), dtype=np.double)     # v_θ (from solver)
    vel_real       = np.zeros((1, N_sim), dtype=np.double)     # real progress speed (dot(tangent, v))
    vel_tangent    = np.zeros((1, N_sim), dtype=np.double)     # ‖v(t)‖ actual speed
    theta_history  = np.zeros((1, N_sim + 1), dtype=np.double) # θ state
    quat_d_theta   = np.zeros((4, N_sim), dtype=np.double)     # desired quaternion at θ_k (θ-indexed)
    t_solver       = np.zeros((1, N_sim), dtype=np.double)
    t_loop         = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state (14-dim: [p, v, q, ω, θ₀]) ───────────────────────
    #    All initial conditions come from experiment_config.py
    q0_normed = Q0 / (np.linalg.norm(Q0) + 1e-12)    # ensure unit quaternion

    x = np.zeros((14, N_sim + 1), dtype=np.double)
    x[:, 0] = [P0[0], P0[1], P0[2],                   # position  ℝ³
               V0[0], V0[1], V0[2],                    # velocity  ℝ³
               q0_normed[0], q0_normed[1],              # quaternion ℍ
               q0_normed[2], q0_normed[3],
               W0[0], W0[1], W0[2],                    # angular velocity ℝ³
               THETA0]                                  # arc-length progress
    theta_history[0, 0] = x[13, 0]
    print(f"[IC]  p0 = {P0}  |  q0 = {np.round(q0_normed,4)}  |  θ₀ = {THETA0}")

    # ── Desired yaw from tangent (precomputed for reference quaternions) ─
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)
    psid = np.arctan2(yd_p_vals, xd_p_vals)

    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    # ── Reference for plotting (17-dim, time-indexed) ────────────────────
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :]  = pos_ref[0, :]   # px_d
    xref[1, :]  = pos_ref[1, :]   # py_d
    xref[2, :]  = pos_ref[2, :]   # pz_d
    xref[3, :]  = dp_ds[0, :]     # tangent_x
    xref[4, :]  = dp_ds[1, :]     # tangent_y
    xref[5, :]  = dp_ds[2, :]     # tangent_z
    xref[6, :]  = quatd[0, :]     # qw_d
    xref[7, :]  = quatd[1, :]     # qx_d
    xref[8, :]  = quatd[2, :]     # qy_d
    xref[9, :]  = quatd[3, :]     # qz_d

    # ── Control storage (5-dim: T, τx, τy, τz, v_θ) ─────────────────────
    u_control = np.zeros((5, N_sim), dtype=np.double)

    # ── Create CasADi trajectory interpolation  (θ → reference) ──────────
    #    Delegates to utils.numpy_utils.build_waypoints and
    #    utils.casadi_utils.create_*_interpolator_casadi.
    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        s_max, N_WAYPOINTS, position_by_arc_length, tangent_by_arc_length,
        euler_to_quat_fn=euler_to_quaternion,
    )

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints")

    # ── Build MPCC solver ────────────────────────────────────────────────
    acados_ocp_solver, ocp, model, f = build_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )

    nx = model.x.size()[0]   # 14
    nu = model.u.size()[0]   # 5

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    print("Initializing simulation...")

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop  (strict MPCC: reference computed from predicted θ)
    # ══════════════════════════════════════════════════════════════════════
    print("Ready!!!")
    time_all = time.time()
    t_lap    = np.nan          # lap time: filled when θ reaches s_max

    for k in range(N_sim):
        tic = time.time()

        # ── Stop when path is complete ────────────────────────────────────
        if x[13, k] >= s_max - 0.01:
            t_lap = k * t_s        # wall time when lap completes [s]
            print(f"[k={k:04d}]  Path complete at θ={x[13,k]:.3f} m  →  t_lap = {t_lap:.3f} s")
            N_sim = k   # trim storage arrays to actual run length
            break

        # ── Set initial state (14-dim) ───────────────────────────────────
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
        vel_progres[:, k]       = u_control[4, k]    # v_θ (solver input)
        theta_history[:, k]     = x[13, k]            # current θ

        # Real progress speed = projection of UAV velocity onto tangent
        theta_k_now = np.clip(x[13, k], 0.0, s_max)
        tang_k_now  = tangent_by_arc_length(theta_k_now)
        vel_real[:, k]    = np.dot(tang_k_now, x[3:6, k])
        vel_tangent[:, k] = np.linalg.norm(x[3:6, k])       # ‖v‖ actual speed

        # ── System evolution (augmented RK4, 14 states) ──────────────────
        x[:, k + 1] = rk4_step_mpcc(x[:, k], u_control[:, k], t_s, f)

        # Clamp θ to valid range
        x[13, k + 1] = np.clip(x[13, k + 1], 0.0, s_max)
        theta_history[:, k + 1] = x[13, k + 1]

        # ── Rate control — run in real time ──────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        # ── Compute MPCC errors (using current θ-based reference) ────────
        theta_k = x[13, k]
        sd_k    = position_by_arc_length(theta_k)
        tang_k  = tangent_by_arc_length(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(x[0:3, k], tang_k, sd_k)

        # ── Desired quaternion at θ_k (for θ-indexed orientation error) ──
        tang_k_xy = tang_k[:2]
        psi_d_k   = np.arctan2(tang_k_xy[1], tang_k_xy[0])
        quat_d_theta[:, k] = euler_to_quaternion(0.0, 0.0, psi_d_k)

        # ── Print progress ───────────────────────────────────────────────
        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0,k] / (vel_progres[0,k] + 1e-8)
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  v_real={vel_real[0,k]:5.2f}  "
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

    # ── Output directory: always the folder where THIS script lives ──────
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    fig1 = plot_pose(x[:13, :], xref_theta, t_plot)
    fig1.savefig(os.path.join(_script_dir, "1_pose.png"))
    print("✓ Saved 1_pose.png")

    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig(os.path.join(_script_dir, "2_control_actions.png"))
    print("✓ Saved 2_control_actions.png")

    fig3 = plot_vel_lineal(x[3:6, :], t_plot)
    fig3.savefig(os.path.join(_script_dir, "3_vel_lineal.png"))
    print("✓ Saved 3_vel_lineal.png")

    fig4 = plot_vel_angular(x[10:13, :], t_plot)
    fig4.savefig(os.path.join(_script_dir, "4_vel_angular.png"))
    print("✓ Saved 4_vel_angular.png")

    # ── Velocity analysis: v_θ, v_real, ‖v‖, curvature ──────────────────
    fig5 = plot_velocity_analysis(
        vel_progres, vel_real, vel_tangent,
        curvature, theta_history, s_max, t_plot[:N_sim])
    fig5.savefig(os.path.join(_script_dir, "5_velocity_analysis.png"), dpi=150)
    print("✓ Saved 5_velocity_analysis.png")

    # ── 3D trajectory ────────────────────────────────────────────────────
    fig6 = plot_3d_trajectory(
        x, pos_ref, s_max=s_max,
        position_by_arc=position_by_arc_length, N_plot=600)
    fig6.savefig(os.path.join(_script_dir, "6_trajectory_3d.png"), dpi=150)
    print("✓ Saved 6_trajectory_3d.png")

    # ── Progress velocity (simple version) ───────────────────────────────
    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig(os.path.join(_script_dir, "8_progress_velocity.png"), dpi=150)
    print("✓ Saved 8_progress_velocity.png")

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
    name_file = f"Results_MPCC_baseline_{experiment_number}.mat"

    savemat(os.path.join(pwd, name_file), {
        'states': x,
        'T_control': u_control,
        'time': t_plot,
        'ref': xref[:, :N_sim + 1],
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
    })
    print(f"✓ Results saved to {os.path.join(pwd, name_file)}")


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
