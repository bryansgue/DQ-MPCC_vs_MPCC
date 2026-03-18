"""
DQ_MPCC_baseline.py  –  Dual-Quaternion MPCC for quadrotor trajectory tracking.

Lie-invariant DQ-MPCC formulation:
  • State  x ∈ ℝ¹⁵ = [dq(8), twist(6), θ]
  • Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]
  • Pose error via exact se(3) logarithmic map:  [φ; ρ] = Log(Q_d* ⊗ Q)
  • Lag-contouring decomposition in the desired body frame
  • θ̇ = v_θ  (augmented arc-length dynamics)

Uses the modular project structure:
    utils/   → dq_casadi_utils, dq_numpy_utils, casadi_utils, numpy_utils
    models/  → dq_quadrotor_mpcc_model (15-state DQ + θ)
    ocp/     → dq_mpcc_controller (Lie-invariant OCP)
"""

import numpy as np
import time
import time as time_module
import os
from scipy.io import savemat

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import (
    euler_to_quaternion,
    build_arc_length_parameterisation,
    build_waypoints,
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
    create_position_interpolator_casadi as create_casadi_position_interpolator,
    create_tangent_interpolator_casadi  as create_casadi_tangent_interpolator,
    create_quat_interpolator_casadi     as create_casadi_quat_interpolator,
)
from ocp.dq_mpcc_controller import build_dq_mpcc_solver
from utils.graficas import (
    plot_pose, plot_error, plot_time, plot_control,
    plot_vel_lineal, plot_vel_angular,
    plot_progress_velocity, plot_velocity_analysis,
    plot_3d_trajectory, plot_timing,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

VALUE  = 5

# Manual arc-length limit (set to None to use the full curve)
S_MAX_MANUAL = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory generators (same as MPCC_baseline)
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
    # ── Timing configuration ─────────────────────────────────────────────
    t_final = 30                                       # [s]
    frec    = 100                                       # [Hz]
    t_s     = 1 / frec                                 # [s]
    t_prediction = 0.3                                 # [s]
    N_prediction = int(round(t_prediction / t_s))      # auto
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")

    # ── Time vector ──────────────────────────────────────────────────────
    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Trajectory ───────────────────────────────────────────────────────
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)

    # Arc-length parameterisation
    t_finer = np.linspace(0, t_final, len(t))
    arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, \
        s_max_full = build_arc_length_parameterisation(
            xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    # Optional manual arc-length limit
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
    vel_progres    = np.zeros((1, N_sim), dtype=np.double)
    vel_real       = np.zeros((1, N_sim), dtype=np.double)
    vel_tangent    = np.zeros((1, N_sim), dtype=np.double)
    theta_history  = np.zeros((1, N_sim + 1), dtype=np.double)
    t_solver       = np.zeros((1, N_sim), dtype=np.double)
    t_loop         = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state (15-dim: [dq(8), twist(6), θ₀=0]) ─────────────────
    p0 = position_by_arc_length(0.0)
    q0 = np.array(euler_to_quaternion(0, 0, 0))   # identity
    dq0 = dq_from_pose_numpy(q0, p0)

    x = np.zeros((15, N_sim + 1), dtype=np.double)
    x[0:8, 0]  = dq0                              # dual quaternion
    x[8:14, 0] = np.zeros(6)                      # twist = 0
    x[14, 0]   = 0.0                              # θ₀ = 0
    theta_history[0, 0] = 0.0

    # Also store in standard 13-dim format for plotting
    x_std = np.zeros((13, N_sim + 1), dtype=np.double)
    x_std[:, 0] = state15_to_standard13(x[:, 0])

    # ── Desired yaw from tangent (for reference quaternions) ─────────────
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)
    psid = np.arctan2(yd_p_vals, xd_p_vals)

    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    # ── Reference for plotting (17-dim, time-indexed) ────────────────────
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :] = pos_ref[0, :]
    xref[1, :] = pos_ref[1, :]
    xref[2, :] = pos_ref[2, :]
    xref[3, :] = dp_ds[0, :]
    xref[4, :] = dp_ds[1, :]
    xref[5, :] = dp_ds[2, :]
    xref[6, :] = quatd[0, :]
    xref[7, :] = quatd[1, :]
    xref[8, :] = quatd[2, :]
    xref[9, :] = quatd[3, :]

    # ── Control storage (5-dim) ──────────────────────────────────────────
    u_control = np.zeros((5, N_sim), dtype=np.double)

    # ── Create CasADi trajectory interpolation  (θ → reference) ──────────
    N_WAYPOINTS = 30
    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        s_max, N_WAYPOINTS, position_by_arc_length, tangent_by_arc_length,
        euler_to_quat_fn=euler_to_quaternion,
    )

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints")

    # ── Build DQ-MPCC solver ─────────────────────────────────────────────
    acados_ocp_solver, ocp, model, f = build_dq_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )

    nx = model.x.size()[0]   # 15
    nu = model.u.size()[0]   # 5

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    print("Initializing simulation...")
    print("Ready!!!")
    time_all = time.time()

    # Keep track of previous DQ for hemisphere correction
    dq_prev = x[0:8, 0].copy()

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop
    # ══════════════════════════════════════════════════════════════════════
    for k in range(N_sim):
        tic = time.time()

        # ── Stop when path is complete ────────────────────────────────────
        if x[14, k] >= s_max - 0.01:
            print(f"[k={k:04d}]  Path complete at θ={x[14,k]:.3f} m. Stopping.")
            N_sim = k
            break

        # ── Set initial state (15-dim) ───────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Solve ────────────────────────────────────────────────────────
        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        # ── Read predicted trajectory ────────────────────────────────────
        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        # ── Get optimal control ──────────────────────────────────────────
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        # ── Store v_θ and θ ──────────────────────────────────────────────
        vel_progres[:, k]   = u_control[4, k]
        theta_history[:, k] = x[14, k]

        # Real progress speed = projection of inertial velocity onto tangent
        theta_k_now = np.clip(x[14, k], 0.0, s_max)
        tang_k_now  = tangent_by_arc_length(theta_k_now)
        # Get inertial velocity from body velocity
        quat_k = dq_get_quaternion_numpy(x[0:8, k])
        v_body_k = x[11:14, k]  # [vx, vy, vz] body frame
        v_inertial_k = quat_rotate_numpy(quat_k, v_body_k)
        vel_real[:, k]    = np.dot(tang_k_now, v_inertial_k)
        vel_tangent[:, k] = np.linalg.norm(v_inertial_k)

        # ── System evolution (RK4, 15 states) ────────────────────────────
        x[:, k + 1] = rk4_step_dq_mpcc(x[:, k], u_control[:, k], t_s, f)

        # ── Post-integration normalization ───────────────────────────────
        # Normalize dual quaternion
        x[0:8, k + 1] = dq_normalize(x[0:8, k + 1])

        # Hemisphere correction (prevent sign flips)
        x[0:8, k + 1] = dq_hemisphere_correction(x[0:8, k + 1], dq_prev)
        dq_prev = x[0:8, k + 1].copy()

        # Clamp θ to valid range
        x[14, k + 1] = np.clip(x[14, k + 1], 0.0, s_max)
        theta_history[:, k + 1] = x[14, k + 1]

        # Convert to standard 13-dim for plotting
        x_std[:, k + 1] = state15_to_standard13(x[:, k + 1])

        # ── Compute MPCC errors (for logging) ────────────────────────────
        theta_k = x[14, k]
        sd_k    = position_by_arc_length(theta_k)
        tang_k  = tangent_by_arc_length(theta_k)
        pos_k   = dq_get_position_numpy(x[0:8, k])
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            mpcc_errors(pos_k, tang_k, sd_k)

        # ── Rate control ─────────────────────────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        # ── Print progress ───────────────────────────────────────────────
        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0, k] / (vel_progres[0, k] + 1e-8)
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  v_real={vel_real[0,k]:5.2f}  "
              f"ratio={ratio_vtheta:4.2f}  |  "
              f"θ={x[14,k]:7.2f}/{s_max:.0f} m  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - time_all
    print(f"\nTotal execution time: {total_time:.4f} seconds")
    print(f"Final θ = {x[14, N_sim]:.3f} / {s_max:.3f} m  "
          f"({x[14, N_sim]/s_max*100:.1f}% of path)")

    # Trim arrays
    x            = x[:, :N_sim + 1]
    x_std        = x_std[:, :N_sim + 1]
    u_control    = u_control[:, :N_sim]
    e_contorno   = e_contorno[:, :N_sim]
    e_arrastre   = e_arrastre[:, :N_sim]
    e_total      = e_total[:, :N_sim]
    vel_progres  = vel_progres[:, :N_sim]
    vel_real     = vel_real[:, :N_sim]
    vel_tangent  = vel_tangent[:, :N_sim]
    theta_history= theta_history[:, :N_sim + 1]
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
        xref_theta[0:3, i] = pos_i
        xref_theta[3:6, i] = tang_i

    # Curvature for analysis plot
    curvature = compute_curvature(position_by_arc_length, s_max, N_samples=500)

    # ── Plots (use standard 13-dim for compatibility with plot functions)
    fig1 = plot_pose(x_std, xref_theta, t_plot)
    fig1.savefig("1_pose.png");   print("✓ Saved 1_pose.png")

    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig("2_control_actions.png"); print("✓ Saved 2_control_actions.png")

    fig3 = plot_vel_lineal(x_std[3:6, :], t_plot)
    fig3.savefig("3_vel_lineal.png");  print("✓ Saved 3_vel_lineal.png")

    fig4 = plot_vel_angular(x_std[10:13, :], t_plot)
    fig4.savefig("4_vel_angular.png"); print("✓ Saved 4_vel_angular.png")

    fig5 = plot_velocity_analysis(
        vel_progres, vel_real, vel_tangent,
        curvature, theta_history, s_max, t_plot[:N_sim])
    fig5.savefig("5_velocity_analysis.png", dpi=150)
    print("✓ Saved 5_velocity_analysis.png")

    fig6 = plot_3d_trajectory(
        x_std, pos_ref, s_max=s_max,
        position_by_arc=position_by_arc_length, N_plot=600)
    fig6.savefig("6_trajectory_3d.png", dpi=150)
    print("✓ Saved 6_trajectory_3d.png")

    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig("8_progress_velocity.png", dpi=150)
    print("✓ Saved 8_progress_velocity.png")

    # ── Timing statistics ────────────────────────────────────────────────
    s_ms  = t_solver[0, :] * 1e3
    l_ms  = t_loop[0, :]   * 1e3
    ts_ms = t_s * 1e3
    n_overrun = int(np.sum(l_ms > ts_ms * 1.05))

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(  "║              DQ-MPCC TIMING STATISTICS                         ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Nominal t_s = {ts_ms:5.2f} ms  ({frec:.0f} Hz)                              ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  [Solver]  mean={np.mean(s_ms):5.2f}  max={np.max(s_ms):5.2f}  std={np.std(s_ms):4.2f}  ms     ║")
    print(f"║  [Loop  ]  mean={np.mean(l_ms):5.2f}  max={np.max(l_ms):5.2f}  std={np.std(l_ms):4.2f}  ms     ║")
    print(f"║  Freq real : {1000/np.mean(l_ms):5.1f} Hz                                  ║")
    print(f"║  Overruns  : {n_overrun:4d} / {len(l_ms)} iters ({n_overrun/len(l_ms)*100:.1f} %)                 ║")
    print(  "╚══════════════════════════════════════════════════════════════════╝\n")

    # v_θ and θ statistics
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

    # ── Save results ─────────────────────────────────────────────────────
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Results_DQ_MPCC"
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"Path {pwd} does not exist. Using local directory.")
        pwd = os.getcwd()

    experiment_number = 1
    name_file = f"Results_DQ_MPCC_baseline_{experiment_number}.mat"

    savemat(os.path.join(pwd, name_file), {
        'states': x,
        'states_std': x_std,
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
        's_max': s_max,
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
