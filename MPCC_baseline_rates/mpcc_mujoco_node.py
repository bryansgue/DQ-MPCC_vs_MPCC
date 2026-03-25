"""MPCC baseline with rate-control dynamics — MuJoCo SiL.

Same MPCC formulation as MPCC_baseline_rates.py, but instead of:
  - RK4 integration  -> reads state from /quadrotor/odom (MuJoCo simulator)
  - storing controls  -> publishes to /quadrotor/trpy_cmd (thrust + omega_desired)

The MPCC solver outputs u = [T, wx_cmd, wy_cmd, wz_cmd, v_theta].
Only [T, wx, wy, wz] is sent to MuJoCo; v_theta is used to integrate
the virtual arc-length state theta via Euler: theta_{k+1} = theta_k + v_theta * t_s.

Usage
-----
    # Terminal 1: Launch MuJoCo
    source ~/uav_ws/install/setup.bash
    mujoco_launch.sh scene:=motors

    # Terminal 2: Run MPCC controller
    source ~/uav_ws/install/setup.bash
    cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline
    python3 MPCC_baseline_rates/mpcc_mujoco_node.py
"""

import os
import sys
import time
import time as time_module
import threading
from scipy.io import savemat

import numpy as np

# ── ROS 2 ─────────────────────────────────────────────────────────────────────
import rclpy

_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WORKSPACE_ROOT)
_SHARED_MPCC_ROOT = os.path.join(_PROJECT_ROOT, "MPCC_baseline")
for _path in (_PROJECT_ROOT, _SHARED_MPCC_ROOT, _WORKSPACE_ROOT):
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
)
from MPCC_baseline_rates.config.experiment_config import (
    FREC,
    G,
    MASS_MUJOCO,
    MPCC_Q_EC,
    MPCC_Q_EL,
    MPCC_Q_OMEGA,
    MPCC_Q_Q,
    MPCC_Q_S,
    MPCC_RATE_U_MAT,
    N_WAYPOINTS,
    P0,
    S_MAX_MANUAL,
    T_MAX,
    T_FINAL,
    THETA0,
    T_PREDICTION,
    VTHETA_MAX,
    W_MAX,
)
from MPCC_baseline_rates.path_loader import load_path
from MPCC_baseline_rates.ocp.mpcc_controller_rate_mujoco import build_mpcc_rate_solver_mujoco

# ── ROS 2 interface (reusable) ───────────────────────────────────────────────
from MPCC_baseline_rates.ros2_interface.mujoco_interface import (
    MujocoInterface,
    wait_for_connection,
)
from MPCC_baseline_rates.ros2_interface.reset_sim import SimControl


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── ROS 2 init & connect to MuJoCo ─────────────────────────────────────
    rclpy.init()
    muj = MujocoInterface(node_name='mpcc_rate_mujoco_controller')

    spin_thread = threading.Thread(target=rclpy.spin, args=(muj,), daemon=True)
    spin_thread.start()

    sim = SimControl(node=muj)
    print("[SIM]  Resetting simulation...")
    sim.reset()

    if not wait_for_connection(muj):
        rclpy.shutdown()
        return

    # ── PD hold: fly to P0 and hold until solver is ready ────────────────
    muj.start_pd_hold(target=P0, mass=MASS_MUJOCO, g=G)

    # ── Timing configuration ──────────────────────────────────────────────
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
    print(f"[CONFIG]  MASS_MUJOCO={MASS_MUJOCO} kg")

    # ── Time vector (upper bound only — loop exits early when θ >= s_max) ─
    t = np.arange(0, T_FINAL + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Arc-length parameterisation (load from cache or build) ────────────
    (s_wp, pos_wp, tang_wp, quat_wp,
     s_max_full, s_max_solver, s_max,
     arc_lengths, pos_ref,
     position_by_arc_length, tangent_by_arc_length) = load_path()
    print(f"[ARC]  s_max = {s_max:.3f} m  |  s_max_full = {s_max_full:.3f} m")

    # ── Storage vectors ──────────────────────────────────────────────────
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

    # ── Initial state from MuJoCo (13 physical states + theta) ───────────
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)

    x = np.zeros((14, N_sim + 1), dtype=np.double)
    x[:, 0] = np.concatenate([pos0, vel0, quat0, omega0, [THETA0]])
    theta_history[0, 0] = THETA0
    print(f"[IC]  p0 = {pos0}  |  q0 = {np.round(quat0, 4)}  |  theta0 = {THETA0}")

    # ── Reference for plotting (arc-length indexed, not time) ────────────
    # xref is only used for post-hoc plots, built from waypoints
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    norms = np.linalg.norm(dp_ds, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    dp_ds /= norms
    # Truncate pos_ref to match N_sim+1 if needed (pos_ref may be much longer)
    _n_ref = min(pos_ref.shape[1], N_sim + 1)
    xref = np.zeros((17, _n_ref))
    xref[0:3, :] = pos_ref[:, :_n_ref]
    xref[3:6, :] = dp_ds[:, :_n_ref]

    # ── Control storage (5-dim: T, wx, wy, wz, v_theta) ─────────────────
    u_control = np.zeros((5, N_sim), dtype=np.double)

    # ── Build MPCC solver ────────────────────────────────────────────────
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

    acados_ocp_solver, ocp, model, f = build_mpcc_rate_solver_mujoco(
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
    u_hover = np.array([MASS_MUJOCO * G, 0.0, 0.0, 0.0, 0.0], dtype=np.double)
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", u_hover)

    # ── Stop PD hold – MPCC takes over ──────────────────────────────────
    muj.stop_pd_hold()

    # ── Re-read state right before loop (fresh after hover) ──────────────
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)
    x[:, 0] = np.concatenate([pos0, vel0, quat0, omega0, [THETA0]])
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop
    # ══════════════════════════════════════════════════════════════════════
    print("Ready!!!")
    time_all = time.time()
    t_lap = np.nan

    for k in range(N_sim):
        tic = time.time()

        # ── Stop when path is complete ────────────────────────────────────
        if x[13, k] >= s_max:
            t_lap = k * t_s
            print(f"[k={k:04d}]  Path complete at theta={x[13, k]:.3f} m  ->  t_lap = {t_lap:.3f} s")
            N_sim = k
            break

        # ── Set initial state (14-dim) ───────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Per-stage p_vtheta_max: brake after s_max ────────────────────
        dt_stage = T_PREDICTION / N_prediction
        theta_k_cur = x[13, k]
        vtheta_cur = u_control[4, max(k - 1, 0)]
        for stage in range(N_prediction + 1):
            theta_pred = theta_k_cur + stage * dt_stage * vtheta_cur
            acados_ocp_solver.set(
                stage, "p", np.array([0.0 if theta_pred >= s_max else VTHETA_MAX], dtype=np.double)
            )

        # ── Solve ────────────────────────────────────────────────────────
        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        # ── Read predicted trajectory ────────────────────────────────────
        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        # ── Get optimal control (5-dim) ──────────────────────────────────
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        vel_progres[:, k] = u_control[4, k]
        theta_history[:, k] = x[13, k]

        # ── Send command to MuJoCo ───────────────────────────────────────
        #    MPCC output:  u = [T, wx_cmd, wy_cmd, wz_cmd, v_theta]
        #    MuJoCo input: (thrust [N], wx, wy, wz [rad/s])
        T_send = np.clip(u_control[0, k], 0.0, 80.0)
        muj.send_cmd(T_send, u_control[1, k], u_control[2, k], u_control[3, k])

        # ── Rate control ─────────────────────────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        # ── Read next state from MuJoCo odom ─────────────────────────────
        pos_new, vel_new, quat_new, omega_new = muj.get_state()
        quat_new /= (np.linalg.norm(quat_new) + 1e-12)

        x[0:3,   k + 1] = pos_new
        x[3:6,   k + 1] = vel_new
        x[6:10,  k + 1] = quat_new
        x[10:13, k + 1] = omega_new

        # ── Integrate theta (virtual state, Euler) ───────────────────────
        theta_new = x[13, k] + u_control[4, k] * t_s
        x[13, k + 1] = np.clip(theta_new, 0.0, s_max_solver)
        theta_history[:, k + 1] = x[13, k + 1]

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        # ── Real progress speed ──────────────────────────────────────────
        theta_k_now = np.clip(x[13, k], 0.0, s_max)
        tang_k_now = tangent_by_arc_length(theta_k_now)
        vel_real[:, k] = np.dot(tang_k_now, x[3:6, k])
        vel_tangent[:, k] = np.linalg.norm(x[3:6, k])

        # ── Compute MPCC errors ──────────────────────────────────────────
        theta_k = x[13, k]
        sd_k = position_by_arc_length(theta_k)
        tang_k = tangent_by_arc_length(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = mpcc_errors(x[0:3, k], tang_k, sd_k)
        quat_d_theta[:, k] = euler_to_quaternion(0.0, 0.0, np.arctan2(tang_k[1], tang_k[0]))
        e_ref = float(np.linalg.norm(sd_k - x[0:3, k]))
        d_path_nearest = float(np.min(np.linalg.norm(pos_ref.T - x[0:3, k], axis=1)))
        e_ref_norm[:, k] = e_ref
        e_path_nearest[:, k] = d_path_nearest
        e_l_scalar = float(np.dot(tang_k, sd_k - x[0:3, k]))
        e_c_norm = float(np.linalg.norm(e_contorno[:, k]))
        theta_step = float(x[13, k + 1] - x[13, k])

        # ── Print progress ───────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════
    #  Safety: hover after loop ends
    # ══════════════════════════════════════════════════════════════════════
    pos_final = muj.get_state()[0]
    muj.start_pd_hold(target=pos_final, mass=MASS_MUJOCO, g=G)
    print(f"[HOVER]  Holding at {np.round(pos_final, 2)} — press Ctrl+C to stop")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing
    # ══════════════════════════════════════════════════════════════════════
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
    results_dir = os.path.join(script_dir, "results_mujoco")
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
    _save(plot_pose(x[:13, :], xref_theta, t_plot), "1_pose_mujoco.png")
    _save(plot_control(u_control[:4, :], t_plot[:N_sim]), "2_control_mujoco.png")
    _save(plot_vel_lineal(x[3:6, :], t_plot), "3_vel_lineal_mujoco.png")
    _save(plot_vel_angular(x[10:13, :], t_plot), "4_vel_angular_mujoco.png")
    _save(
        plot_velocity_analysis(vel_progres, vel_real, vel_tangent, curvature, theta_history, s_max, t_plot[:N_sim]),
        "5_velocity_analysis_mujoco.png",
    )
    _save(plot_3d_trajectory(x[0:3, :], xref_theta[0:3, :]), "6_trajectory_3d_mujoco.png")
    _save(plot_error(e_total, t_plot[:N_sim]), "7_error_mujoco.png")
    _save(plot_progress_velocity(vel_progres, vel_real, theta_history[:, :N_sim], t_plot[:N_sim]), "8_progress_velocity_mujoco.png")
    _save(plot_timing(t_solver, t_loop, t_sample, t_plot[:N_sim]), "9_timing_mujoco.png")

    # ── Timing statistics ────────────────────────────────────────────────
    s_ms = t_solver[0, :] * 1e3
    l_ms = t_loop[0, :] * 1e3
    ts_ms = t_s * 1e3
    n_overrun = int(np.sum(l_ms > ts_ms * 1.05))

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(  "║                     TIMING STATISTICS                          ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Nominal t_s = {ts_ms:5.2f} ms  ({FREC:.0f} Hz)                              ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  [Solver]  mean={np.mean(s_ms):5.2f}  max={np.max(s_ms):5.2f}  std={np.std(s_ms):4.2f}  ms     ║")
    print(f"║  [Loop  ]  mean={np.mean(l_ms):5.2f}  max={np.max(l_ms):5.2f}  std={np.std(l_ms):4.2f}  ms     ║")
    print(f"║  Freq real : {1000/np.mean(l_ms):5.1f} Hz                                  ║")
    print(f"║  Overruns  : {n_overrun:4d} / {len(l_ms)} iters ({n_overrun/len(l_ms)*100:.1f} %)                 ║")
    print(  "╚══════════════════════════════════════════════════════════════════╝\n")

    # ── Save results ─────────────────────────────────────────────────────
    savemat(
        os.path.join(results_dir, "Results_MPCC_rates_mujoco.mat"),
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
    print("Saved Results_MPCC_rates_mujoco.mat")

    # ── Cleanup ──────────────────────────────────────────────────────────
    muj.stop_pd_hold()
    muj.stop()
    muj.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        try:
            rclpy.shutdown()
        except Exception:
            pass
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    else:
        print("Complete Execution")
