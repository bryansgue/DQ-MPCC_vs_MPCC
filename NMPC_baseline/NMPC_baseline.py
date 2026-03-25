"""
NMPC_baseline.py  –  NMPC baseline for quadrotor trajectory tracking.

Modules:
    config/  → centralized experiment parameters
    utils/   → quaternion & rotation helpers
    models/  → quadrotor dynamics (CasADi / acados)
    ocp/     → NMPC formulation & solver build
"""

import numpy as np
import time
import time as time_module

# ── Configuration (single source of truth) ───────────────────────────────────
from config.experiment_config import (
    X0, T_FINAL, FREC, T_S, T_PREDICTION, N_PREDICTION,
    trayectoria,
)

# ── Project modules ──────────────────────────────────────────────────────────
from utils.numpy_utils import euler_to_quaternion, euler_dot as Euler_p, rk4_step_quadrotor
from models.quadrotor_model import f_system_model
from ocp.nmpc_controller import build_ocp_solver
from utils.graficas import (
    plot_pose, plot_error, plot_time, plot_control,
    plot_vel_lineal, plot_vel_angular, plot_timing,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions (I/O placeholders for ROS / hardware)
# ═══════════════════════════════════════════════════════════════════════════════

def send_velocity_control(u, vel_pub=None, vel_msg=None):
    """Send control command (placeholder for ROS / hardware interface)."""
    return None


def send_full_state_to_sim(state_vector):
    """Publish full state to simulator (placeholder)."""
    pass


def publish_matrix(matrix_data, topic_name='/Prediction'):
    """Publish prediction matrix (placeholder)."""
    pass


def print_state_vector(state_vector):
    """Pretty-print the 13-element state vector."""
    headers = ["px", "py", "pz", "vx", "vy", "vz",
               "qw", "qx", "qy", "qz", "wx", "wy", "wz"]
    if len(state_vector) != len(headers):
        raise ValueError(
            f"State has {len(state_vector)} elements, expected {len(headers)}."
        )
    max_len = max(len(h) for h in headers)
    for h, v in zip(headers, state_vector):
        print(f"{h.ljust(max_len)}: {v:.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Timing configuration (from config) ────────────────────────────────
    t_final      = T_FINAL
    frec         = FREC
    t_s          = T_S
    t_prediction = T_PREDICTION
    N_prediction = N_PREDICTION
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")

    # ── Time vector ──────────────────────────────────────────────────────
    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Storage vectors ──────────────────────────────────────────────────
    delta_t   = np.zeros((1, N_sim), dtype=np.double)
    t_sample  = t_s * np.ones((1, N_sim), dtype=np.double)
    t_solver  = np.zeros((1, N_sim), dtype=np.double)
    t_loop    = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state (from config) ──────────────────────────────────────
    x = np.zeros((13, N_sim + 1), dtype=np.double)
    x[:, 0] = X0

    # ── Desired trajectory (from config) ─────────────────────────────────
    xd, yd, zd, xdp, ydp, zdp = trayectoria()

    hxd  = xd(t);   hyd  = yd(t);   hzd  = zd(t)
    hxdp = xdp(t);  hydp = ydp(t);  hzdp = zdp(t)

    psid  = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

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
    u_control = np.zeros((4, N_sim), dtype=np.double)

    # ── Build solver ─────────────────────────────────────────────────────
    acados_ocp_solver, ocp, model, f = build_ocp_solver(
        x[:, 0], N_prediction, t_prediction, use_cython=True
    )

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

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

        # ── Read predicted trajectory ────────────────────────────────────
        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')
        publish_matrix(xref[0:3, 0:500:5], '/task_desired')

        u_control[:, k] = simU[:, 0]

        # ── Solve ────────────────────────────────────────────────────────
        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        # ── Get optimal control ──────────────────────────────────────────
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        send_velocity_control(u_control[:, k])

        # ── System evolution ─────────────────────────────────────────────
        x[:, k + 1] = rk4_step_quadrotor(x[:, k], u_control[:, k], t_s, f)
        send_full_state_to_sim(x[:, k + 1])

        # ── Rate control ─────────────────────────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]
        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  loop={t_loop[0,k]*1e3:6.2f} ms  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing
    # ══════════════════════════════════════════════════════════════════════
    send_velocity_control([0, 0, 0, 0])

    print("Generating figures...")
    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png");   print("✓ Saved 1_pose.png")

    fig3 = plot_vel_lineal(x[3:6, :], t)
    fig3.savefig("3_vel_lineal.png");  print("✓ Saved 3_vel_lineal.png")

    fig4 = plot_vel_angular(x[10:13, :], t)
    fig4.savefig("4_vel_angular.png"); print("✓ Saved 4_vel_angular.png")

    fig2 = plot_control(u_control, t)
    fig2.savefig("2_control_actions.png"); print("✓ Saved 2_control_actions.png")

    fig6 = plot_timing(t_solver, t_loop, t_sample, t)
    fig6.savefig("6_timing.png"); print("✓ Saved 6_timing.png")

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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        send_velocity_control([0, 0, 0, 0])
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
    else:
        print("Complete Execution")
