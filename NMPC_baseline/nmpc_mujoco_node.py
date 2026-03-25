"""
nmpc_mujoco_node.py  –  NMPC for quadrotor trajectory tracking (MuJoCo SiL).

Same NMPC formulation as NMPC_baseline.py, but instead of:
  - RK4 integration  -> reads state from /quadrotor/odom (MuJoCo simulator)
  - storing controls -> publishes to /quadrotor/trpy_cmd (thrust + omega_desired)

The NMPC solver outputs u = [T, tau_x, tau_y, tau_z].  MuJoCo's AcroMode
expects (thrust, omega_desired).  We extract the predicted angular velocity
at stage 1 of the MPC horizon: this is the omega the model expects after
applying u[0], so the AcroMode's rate controller tracks it directly.

Thrust is scaled by MASS_MUJOCO / MASS to compensate model mismatch.

Usage
-----
    # Terminal 1: Launch MuJoCo
    source ~/uav_ws/install/setup.bash
    ros2 launch ...  (your MuJoCo launch)

    # Terminal 2: Run NMPC controller
    source ~/uav_ws/install/setup.bash
    cd ~/dev/ros2/NMPC_baseline
    python3 nmpc_mujoco_node.py
"""

import numpy as np
import time
import time as time_module
import os
import threading

# ── ROS 2 ─────────────────────────────────────────────────────────────────────
import rclpy

# ── Configuration (single source of truth) ───────────────────────────────────
from config.experiment_config import (
    P0, X0,
    T_FINAL, FREC, T_S, T_PREDICTION, N_PREDICTION,
    G, MASS_MUJOCO,
    trayectoria,
)

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import euler_to_quaternion, rk4_step_quadrotor
from ocp.nmpc_controller_rate import build_ocp_solver
from utils.graficas import (
    plot_pose, plot_control_rate, plot_omega_cmd_vs_actual,
    plot_vel_lineal, plot_vel_angular, plot_timing,
)

# ── ROS 2 interface (reusable) ───────────────────────────────────────────────
from ros2_interface.mujoco_interface import (
    MujocoInterface,
    wait_for_connection,
)
from ros2_interface.reset_sim import SimControl


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── ROS 2 init & connect to MuJoCo ─────────────────────────────────────
    rclpy.init()
    muj = MujocoInterface(node_name='nmpc_mujoco_controller')

    # Spin ROS 2 in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(muj,), daemon=True)
    spin_thread.start()

    # ── SimControl: reset/reload the simulation ──────────────────────────
    sim = SimControl(node=muj)
    print("[SIM]  Resetting simulation...")
    sim.reset()

    # Wait for odom
    if not wait_for_connection(muj):
        rclpy.shutdown()
        return

    # ── PD hold: fly to P0 and hold until solver is ready ────────────────
    muj.start_pd_hold(target=P0, mass=MASS_MUJOCO, g=G)

    # ── Timing configuration ──────────────────────────────────────────────
    t_final      = T_FINAL
    frec         = FREC
    t_s          = T_S
    t_prediction = T_PREDICTION
    N_prediction = N_PREDICTION
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")
    print(f"[CONFIG]  MASS_MUJOCO={MASS_MUJOCO} kg  (model uses real mass, no scaling)")

    # ── Time vector ──────────────────────────────────────────────────────
    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Storage vectors ──────────────────────────────────────────────────
    t_sample  = t_s * np.ones((1, N_sim), dtype=np.double)
    t_solver  = np.zeros((1, N_sim), dtype=np.double)
    t_loop    = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state from MuJoCo (NOT from config) ──────────────────────
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)

    x = np.zeros((13, N_sim + 1), dtype=np.double)
    x[:, 0] = np.concatenate([pos0, vel0, quat0, omega0])
    print(f"[IC]  p0 = {pos0}  |  q0 = {np.round(quat0, 4)}")

    # ── Desired trajectory (from config) ─────────────────────────────────
    xd, yd, zd, xdp, ydp, zdp = trayectoria()

    hxd  = xd(t);   hyd  = yd(t);   hzd  = zd(t)
    hxdp = xdp(t);  hydp = ydp(t);  hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)

    # Desired quaternion from yaw
    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    # ── Reference signal (17-dim: 13 states + 4 controls) ───────────────
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :]  = hxd
    xref[1, :]  = hyd
    xref[2, :]  = hzd
    xref[6, :]  = quatd[0, :]
    xref[7, :]  = quatd[1, :]
    xref[8, :]  = quatd[2, :]
    xref[9, :]  = quatd[3, :]

    # ── Control storage ──────────────────────────────────────────────────
    u_control   = np.zeros((4, N_sim), dtype=np.double)
    T_sent_hist = np.zeros(N_sim, dtype=np.double)   # thrust after scaling/clip

    # ── Build solver ─────────────────────────────────────────────────────
    acados_ocp_solver, ocp, model, f = build_ocp_solver(
        x[:, 0], N_prediction, t_prediction, use_cython=True
    )

    nx = model.x.size()[0]   # 13
    nu = model.u.size()[0]   # 4

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # ── Stop PD hold – NMPC takes over ──────────────────────────────────
    muj.stop_pd_hold()

    # ── Re-read state right before loop (fresh after hover) ──────────────
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)
    x[:, 0] = np.concatenate([pos0, vel0, quat0, omega0])
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop
    # ══════════════════════════════════════════════════════════════════════
    print("Ready!!!")

    for k in range(N_sim):
        tic = time.time()

        # ── Set initial state ────────────────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Set references ───────────────────────────────────────────────
        for j in range(N_prediction):
            acados_ocp_solver.set(j, "p", xref[:, k + j])
        acados_ocp_solver.set(N_prediction, "p", xref[:, k + N_prediction])

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

        # ── Send command to MuJoCo ───────────────────────────────────────
        #    NMPC output:  u = [T, wx_cmd, wy_cmd, wz_cmd]
        #    MuJoCo input: (thrust [N], wx, wy, wz [rad/s])
        #    Model uses MASS_MUJOCO → T_cmd already correct, no scaling needed.
        T_send = np.clip(u_control[0, k], 0.0, 80.0)
        T_sent_hist[k] = T_send
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

        t_loop[:, k] = time.time() - tic

        overrun = " OVERRUN" if elapsed > t_s else ""
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  loop={t_loop[0,k]*1e3:6.2f} ms  "
              f"|  T={T_send:5.1f}N  "
              f"w=[{u_control[1,k]:+.2f},{u_control[2,k]:+.2f},{u_control[3,k]:+.2f}]  "
              f"|  {1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Safety: hover after loop ends
    # ══════════════════════════════════════════════════════════════════════
    pos_final = muj.get_state()[0]
    muj.start_pd_hold(target=pos_final, mass=MASS_MUJOCO, g=G)
    print(f"[HOVER]  Holding at {np.round(pos_final, 2)} — press Ctrl+C to stop")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing
    # ══════════════════════════════════════════════════════════════════════
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _results_dir = os.path.join(_script_dir, "results_sim")
    os.makedirs(_results_dir, exist_ok=True)

    def _save(fig, name):
        path = os.path.join(_results_dir, name)
        fig.savefig(path, dpi=150)
        print(f"Saved {name}")

    print("Generating figures...")

    fig1 = plot_pose(x, xref, t)
    _save(fig1, "1_pose_mujoco.png")

    fig2 = plot_control_rate(u_control, t, T_sent=T_sent_hist)
    _save(fig2, "2_control_mujoco.png")

    fig3 = plot_vel_lineal(x[3:6, :], t)
    _save(fig3, "3_vel_lineal_mujoco.png")

    fig4 = plot_vel_angular(x[10:13, :], t)
    _save(fig4, "4_vel_angular_mujoco.png")

    fig5 = plot_omega_cmd_vs_actual(u_control[1:4, :], x[10:13, :], t)
    _save(fig5, "5_omega_cmd_vs_actual_mujoco.png")

    fig6 = plot_timing(t_solver, t_loop, t_sample, t)
    _save(fig6, "6_timing_mujoco.png")

    # ── Timing statistics ────────────────────────────────────────────────
    s_ms = t_solver[0, :] * 1e3
    l_ms = t_loop[0, :]   * 1e3
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

    # ── Cleanup ──────────────────────────────────────────────────────────
    muj.stop_pd_hold()
    muj.stop()
    muj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
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
