"""
mpcc_mujoco_node.py  –  MPCC for quadrotor trajectory tracking (MuJoCo SiL).

IDENTICAL structure to MPCC_baseline.py, but instead of:
  • RK4 integration  → reads state from /quadrotor/odom (MuJoCo simulator)
  • storing controls → publishes to /quadrotor/trpy_cmd (thrust + ω_desired)

The MPCC solver outputs u = [T, τx, τy, τz, v_θ].  MuJoCo's AcroMode
expects (thrust, ω_desired).  We extract the predicted angular velocity
at stage 1 of the MPC horizon: this is the ω the model expects after
applying u[0], so the AcroMode's rate controller tracks it directly.

Thrust is scaled by MASS_MUJOCO / MASS to compensate model mismatch.

Usage
-----
    # Terminal 1: Launch MuJoCo
    source ~/uav_ws/install/setup.bash
    mujoco_launch.sh scene:=motors

    # Terminal 2: Run MPCC controller
    source ~/uav_ws/install/setup.bash
    cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline
    python3 MPCC_baseline/mpcc_mujoco_node.py
"""

import numpy as np
import time
import time as time_module
import os
import sys
import math
import threading
from scipy.io import savemat

# ── ROS 2 ─────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TRPYCommand

# ── Add parent directory to path for shared config ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_config import (
    P0, Q0, V0, W0, THETA0,
    T_FINAL, FREC, T_PREDICTION, N_WAYPOINTS, S_MAX_MANUAL,
    VTHETA_MAX, G, MASS,
    trayectoria,
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
#  MuJoCo drone parameters
# ═══════════════════════════════════════════════════════════════════════════════
MASS_MUJOCO = 1.08          # [kg] (from MuJoCo model)
MASS_RATIO  = MASS_MUJOCO / MASS   # thrust scaling factor


# ═══════════════════════════════════════════════════════════════════════════════
#  ROS 2 helper — lightweight node for odom + cmd
# ═══════════════════════════════════════════════════════════════════════════════

class MujocoInterface(Node):
    """Minimal ROS 2 node: subscribes to odom, publishes trpy_cmd."""

    def __init__(self):
        super().__init__('mpcc_mujoco_controller')

        # State from odom
        self.pos   = np.zeros(3)
        self.vel   = np.zeros(3)                         # world frame
        self.quat  = np.array([1., 0., 0., 0.])          # [qw, qx, qy, qz]
        self.omega = np.zeros(3)                          # body frame
        self.connected = False
        self._lock = threading.Lock()

        # Pub / Sub
        self.cmd_pub = self.create_publisher(
            TRPYCommand, '/quadrotor/trpy_cmd', 10)
        self.create_subscription(
            Odometry, '/quadrotor/odom', self._odom_cb, 10)

    def _odom_cb(self, msg):
        with self._lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear       # world frame (flg_local=0)
            w = msg.twist.twist.angular      # body frame  (R^T · w_world)
            self.pos[:]   = [p.x, p.y, p.z]
            self.vel[:]   = [v.x, v.y, v.z]
            self.quat[:]  = [q.w, q.x, q.y, q.z]
            self.omega[:] = [w.x, w.y, w.z]
            self.connected = True

    def get_state(self):
        """Return (pos, vel, quat, omega) thread-safe snapshot."""
        with self._lock:
            return (self.pos.copy(), self.vel.copy(),
                    self.quat.copy(), self.omega.copy())

    def send_cmd(self, thrust, wx=0., wy=0., wz=0.):
        msg = TRPYCommand()
        msg.thrust = float(thrust)
        msg.angular_velocity.x = float(wx)
        msg.angular_velocity.y = float(wy)
        msg.angular_velocity.z = float(wz)
        self.cmd_pub.publish(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  Takeoff helper — PD to reach P0 before engaging MPCC
# ═══════════════════════════════════════════════════════════════════════════════

def takeoff_to_P0(muj, target=P0, tol_pos=0.20, tol_vel=0.40, timeout=15.0):
    """Block until the drone reaches *target* using a simple PD controller."""
    KP_XY, KD_XY = 4.0, 2.5
    KP_Z,  KD_Z  = 8.0, 4.0
    KP_ATT       = 6.0
    KP_YAW       = 2.0
    dt = 0.01   # 100 Hz

    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(muj, timeout_sec=0.001)
        pos, vel, quat, _ = muj.get_state()
        err = target - pos

        if np.linalg.norm(err) < tol_pos and np.linalg.norm(vel) < tol_vel:
            print(f"[TAKEOFF]  Reached P0 (err={np.linalg.norm(err):.3f} m)")
            return True

        # Desired acceleration (world frame)
        a_des = np.array([
            KP_XY * err[0] - KD_XY * vel[0],
            KP_XY * err[1] - KD_XY * vel[1],
            KP_Z  * err[2] - KD_Z  * vel[2] + G,
        ])

        # Rotation matrix from quaternion
        w, x, y, z = quat
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

        thrust = MASS_MUJOCO * np.dot(a_des, R[:, 2])
        thrust = np.clip(thrust, 0., 60.)

        psi = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        az = max(a_des[2], 0.1)
        pitch_des = np.clip(
            (a_des[0]*math.cos(psi) + a_des[1]*math.sin(psi)) / az, -0.5, 0.5)
        roll_des = np.clip(
            (a_des[0]*math.sin(psi) - a_des[1]*math.cos(psi)) / az, -0.5, 0.5)

        pitch_cur = math.asin(np.clip(R[0, 2], -1, 1))
        roll_cur  = math.atan2(-R[1, 2], R[2, 2])
        yaw_err   = (0.0 - psi + math.pi) % (2*math.pi) - math.pi

        wx = KP_ATT * (roll_des - roll_cur)
        wy = KP_ATT * (pitch_des - pitch_cur)
        wz = KP_YAW * yaw_err

        muj.send_cmd(thrust, wx, wy, wz)
        time_module.sleep(dt)

    print("[TAKEOFF]  Timeout — proceeding anyway")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── ROS 2 init & connect to MuJoCo ─────────────────────────────────────
    rclpy.init()
    muj = MujocoInterface()

    # Spin ROS 2 in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(muj,), daemon=True)
    spin_thread.start()

    # Wait for odom
    print("\nEsperando conexion con MuJoCo (/quadrotor/odom)...")
    for _ in range(100):
        if muj.connected:
            break
        time.sleep(0.1)
    if not muj.connected:
        print("No se recibio /quadrotor/odom. Lanza primero el simulador.")
        rclpy.shutdown()
        return
    print(f"Conectado!  Pos: {muj.pos}")

    # ── Takeoff to P0 ──────────────────────────────────────────────────────
    print(f"[TAKEOFF]  Subiendo a P0 = {P0} ...")
    takeoff_to_P0(muj, target=P0)

    # ── Timing configuration (from experiment_config.py) ─────────────────
    t_final = T_FINAL
    frec    = FREC
    t_s     = 1 / frec
    t_prediction = T_PREDICTION
    N_prediction = int(round(t_prediction / t_s))
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")
    print(f"[CONFIG]  MASS_MPCC={MASS} kg  |  MASS_MUJOCO={MASS_MUJOCO} kg  "
          f"|  ratio={MASS_RATIO:.3f}")

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
    #    Read initial state from MuJoCo odom (NOT from experiment_config)
    pos0, vel0, quat0, omega0 = muj.get_state()
    quat0 /= (np.linalg.norm(quat0) + 1e-12)

    x = np.zeros((14, N_sim + 1), dtype=np.double)
    x[:, 0] = [pos0[0], pos0[1], pos0[2],           # position  ℝ³
               vel0[0], vel0[1], vel0[2],            # velocity  ℝ³
               quat0[0], quat0[1],                   # quaternion ℍ
               quat0[2], quat0[3],
               omega0[0], omega0[1], omega0[2],      # angular velocity ℝ³
               THETA0]                                # arc-length progress
    theta_history[0, 0] = x[13, 0]
    print(f"[IC]  p0 = {pos0}  |  q0 = {np.round(quat0,4)}  |  θ₀ = {THETA0}")

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

    # ── Build MPCC solver ────────────────────────────────────────────────
    # Pass s_max * 1.2 so the solver's θ upper-bound is extended beyond the
    # finish line — prevents the predictive horizon from braking early.
    # Clamp to s_max_full so the interpolation never goes out of range.
    S_MAX_SOLVER = min(s_max * 1.2, s_max_full)

    # Build waypoints up to S_MAX_SOLVER so the CasADi interpolation
    # covers the entire range the solver can explore.
    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        S_MAX_SOLVER, N_WAYPOINTS, position_by_arc_length, tangent_by_arc_length,
        euler_to_quat_fn=euler_to_quaternion,
    )

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints "
          f"(s_max_solver={S_MAX_SOLVER:.2f})")
    acados_ocp_solver, ocp, model, f = build_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=S_MAX_SOLVER,
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
        if x[13, k] >= s_max:               # cruzó la línea de meta
            t_lap = k * t_s        # wall time when lap completes [s]
            print(f"[k={k:04d}]  Path complete at θ={x[13,k]:.3f} m  →  t_lap = {t_lap:.3f} s")
            N_sim = k   # trim storage arrays to actual run length
            break

        # ── Set initial state (14-dim) ───────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Per-stage p_vtheta_max: brake after s_max ────────────────────
        #    Estimate predicted θ at each horizon stage.  For stages where
        #    the predicted θ exceeds s_max, set v_ref = 0 so the cost
        #    Q_s*(0 − v_θ)² drives the solver to brake.
        dt_stage = t_prediction / N_prediction
        theta_k_cur = x[13, k]
        vtheta_cur  = u_control[4, max(k - 1, 0)]  # best guess of current v_θ
        for stage in range(N_prediction + 1):
            theta_pred = theta_k_cur + stage * dt_stage * vtheta_cur
            if theta_pred >= s_max:
                acados_ocp_solver.set(stage, "p", np.array([0.0]))
            else:
                acados_ocp_solver.set(stage, "p", np.array([VTHETA_MAX]))

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

        # ── Send command to MuJoCo ─────────────────────────────────────
        #    Thrust: scaled by mass ratio to compensate model mismatch
        #    Angular velocity: predicted ω at stage 1 of the horizon
        #      (the ω the MPCC model predicts after applying u[0])
        T_send    = u_control[0, k] * MASS_RATIO
        T_send    = np.clip(T_send, 0.0, 80.0)
        omega_cmd = simX[10:13, 1]   # predicted ω at next step
        muj.send_cmd(T_send, omega_cmd[0], omega_cmd[1], omega_cmd[2])

        # ── System evolution: read next state from MuJoCo odom ─────────
        #    (replaces rk4_step_mpcc in the MiL version)
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        pos_new, vel_new, quat_new, omega_new = muj.get_state()
        quat_new /= (np.linalg.norm(quat_new) + 1e-12)

        x[0:3,   k + 1] = pos_new                   # position  (world)
        x[3:6,   k + 1] = vel_new                   # velocity  (world)
        x[6:10,  k + 1] = quat_new                  # quaternion [qw,qx,qy,qz]
        x[10:13, k + 1] = omega_new                 # angular vel (body)
        # θ is a virtual state — take it from the solver's prediction
        x[13, k + 1] = np.clip(simX[13, 1], 0.0, S_MAX_SOLVER)
        theta_history[:, k + 1] = x[13, k + 1]

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
        overrun = " OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0,k] / (vel_progres[0,k] + 1e-8)
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  v_real={vel_real[0,k]:5.2f}  "
              f"ratio={ratio_vtheta:4.2f}  |  "
              f"θ={x[13,k]:7.2f}/{s_max:.0f} m  |  "
              f"T={T_send:5.1f}N  ω=[{omega_cmd[0]:+.2f},{omega_cmd[1]:+.2f},{omega_cmd[2]:+.2f}]  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Safety: hover after loop ends
    # ══════════════════════════════════════════════════════════════════════
    muj.send_cmd(MASS_MUJOCO * G)   # hover thrust

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
    print("Saved 1_pose.png")

    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig(os.path.join(_script_dir, "2_control_actions.png"))
    print("Saved 2_control_actions.png")

    fig3 = plot_vel_lineal(x[3:6, :], t_plot)
    fig3.savefig(os.path.join(_script_dir, "3_vel_lineal.png"))
    print("Saved 3_vel_lineal.png")

    fig4 = plot_vel_angular(x[10:13, :], t_plot)
    fig4.savefig(os.path.join(_script_dir, "4_vel_angular.png"))
    print("Saved 4_vel_angular.png")

    # ── Velocity analysis: v_θ, v_real, ‖v‖, curvature ──────────────────
    fig5 = plot_velocity_analysis(
        vel_progres, vel_real, vel_tangent,
        curvature, theta_history, s_max, t_plot[:N_sim])
    fig5.savefig(os.path.join(_script_dir, "5_velocity_analysis.png"), dpi=150)
    print("Saved 5_velocity_analysis.png")

    # ── 3D trajectory ────────────────────────────────────────────────────
    fig6 = plot_3d_trajectory(
        x, pos_ref, s_max=s_max,
        position_by_arc=position_by_arc_length, N_plot=600)
    fig6.savefig(os.path.join(_script_dir, "6_trajectory_3d.png"), dpi=150)
    print("Saved 6_trajectory_3d.png")

    # ── Progress velocity (simple version) ───────────────────────────────
    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig(os.path.join(_script_dir, "8_progress_velocity.png"), dpi=150)
    print("Saved 8_progress_velocity.png")

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
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    name_file = "Results_MPCC_mujoco.mat"

    savemat(os.path.join(_script_dir, name_file), {
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
    })
    print(f"Results saved to {os.path.join(_script_dir, name_file)}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    muj.send_cmd(0.0)
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
