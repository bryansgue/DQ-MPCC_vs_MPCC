"""
test_pd_mujoco.py – Diagnostic: PD position control + live plots.

Sends thrust + angular velocity to MuJoCo via /quadrotor/trpy_cmd,
reads state from /quadrotor/odom, logs everything, and plots at the end.

Purpose: verify that the commands we send match what the drone actually does.
"""

import numpy as np
import math
import time
import threading

import rclpy
from ros2_interface.mujoco_interface import MujocoInterface, wait_for_connection
from config.experiment_config import G, MASS_MUJOCO

# ── Target position ──────────────────────────────────────────────────────────
TARGET = np.array([2.5, 0.0, 1.2])
TARGET_YAW = 0.0

# ── PD gains ─────────────────────────────────────────────────────────────────
KP_XY, KD_XY = 4.0, 2.5
KP_Z,  KD_Z  = 8.0, 4.0
KP_ATT       = 6.0
KP_YAW       = 2.0

# ── Experiment ───────────────────────────────────────────────────────────────
DURATION = 15.0   # seconds
DT       = 0.01   # 100 Hz


def main():
    rclpy.init()
    muj = MujocoInterface(node_name='test_pd_diag')
    spin_thread = threading.Thread(target=rclpy.spin, args=(muj,), daemon=True)
    spin_thread.start()

    if not wait_for_connection(muj):
        rclpy.shutdown()
        return

    N = int(DURATION / DT)

    # ── Storage ──────────────────────────────────────────────────────────────
    log_t       = np.zeros(N)
    log_pos     = np.zeros((N, 3))
    log_vel     = np.zeros((N, 3))
    log_quat    = np.zeros((N, 4))
    log_omega   = np.zeros((N, 3))
    log_T_cmd   = np.zeros(N)
    log_w_cmd   = np.zeros((N, 3))

    print(f"[TEST PD]  target={TARGET}  yaw={math.degrees(TARGET_YAW):.1f} deg")
    print(f"[TEST PD]  Running {DURATION}s at {1/DT:.0f} Hz ...")

    t0 = time.time()
    for k in range(N):
        tic = time.time()

        pos, vel, quat, omega = muj.get_state()
        err = TARGET - pos

        # ── PD control ───────────────────────────────────────────────────
        a_des = np.array([
            KP_XY * err[0] - KD_XY * vel[0],
            KP_XY * err[1] - KD_XY * vel[1],
            KP_Z  * err[2] - KD_Z  * vel[2] + G,
        ])

        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

        thrust = MASS_MUJOCO * np.dot(a_des, R[:, 2])
        thrust = np.clip(thrust, 0.0, 60.0)

        psi = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        az  = max(a_des[2], 0.1)
        pitch_des = np.clip(
            (a_des[0]*math.cos(psi) + a_des[1]*math.sin(psi)) / az, -0.5, 0.5)
        roll_des = np.clip(
            (a_des[0]*math.sin(psi) - a_des[1]*math.cos(psi)) / az, -0.5, 0.5)

        pitch_cur = math.asin(np.clip(R[0, 2], -1, 1))
        roll_cur  = math.atan2(-R[1, 2], R[2, 2])
        yaw_err   = (TARGET_YAW - psi + math.pi) % (2*math.pi) - math.pi

        wx_cmd = KP_ATT * (roll_des - roll_cur)
        wy_cmd = KP_ATT * (pitch_des - pitch_cur)
        wz_cmd = KP_YAW * yaw_err

        # ── Send ─────────────────────────────────────────────────────────
        muj.send_cmd(thrust, wx_cmd, wy_cmd, wz_cmd)

        # ── Log ──────────────────────────────────────────────────────────
        log_t[k]       = time.time() - t0
        log_pos[k]     = pos
        log_vel[k]     = vel
        log_quat[k]    = quat
        log_omega[k]   = omega
        log_T_cmd[k]   = thrust
        log_w_cmd[k]   = [wx_cmd, wy_cmd, wz_cmd]

        # ── Print every 100 steps ────────────────────────────────────────
        if k % 100 == 0:
            err_norm = np.linalg.norm(err)
            print(f"  [k={k:04d}]  pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]  "
                  f"err={err_norm:.3f}m  T={thrust:.1f}N  "
                  f"w_cmd=[{wx_cmd:+.2f},{wy_cmd:+.2f},{wz_cmd:+.2f}]  "
                  f"w_real=[{omega[0]:+.2f},{omega[1]:+.2f},{omega[2]:+.2f}]")

        # ── Rate control ─────────────────────────────────────────────────
        elapsed = time.time() - tic
        remaining = DT - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ── Hover at end ─────────────────────────────────────────────────────────
    muj.send_cmd(MASS_MUJOCO * G, 0.0, 0.0, 0.0)
    print("\n[TEST PD]  Done. Generating plots...")

    # ══════════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('PD Position Control – Diagnostic', fontsize=14)

    # 1. Position vs target
    ax = axes[0]
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.plot(log_t, log_pos[:, i], colors[i], label=f'{labels[i]} real')
        ax.axhline(TARGET[i], color=colors[i], linestyle='--', alpha=0.5, label=f'{labels[i]} target')
    ax.set_ylabel('Position [m]')
    ax.legend(ncol=3, fontsize=8)
    ax.set_title('Position: real vs target')
    ax.grid(True, alpha=0.3)

    # 2. Thrust command
    ax = axes[1]
    ax.plot(log_t, log_T_cmd, 'k', label='T cmd')
    ax.axhline(MASS_MUJOCO * G, color='gray', linestyle='--', alpha=0.5, label='hover thrust')
    ax.set_ylabel('Thrust [N]')
    ax.legend(fontsize=8)
    ax.set_title('Thrust command')
    ax.grid(True, alpha=0.3)

    # 3. Angular velocity: command vs real
    ax = axes[2]
    w_labels = ['wx', 'wy', 'wz']
    for i in range(3):
        ax.plot(log_t, log_w_cmd[:, i], colors[i], label=f'{w_labels[i]} cmd')
        ax.plot(log_t, log_omega[:, i], colors[i], linestyle='--', alpha=0.6, label=f'{w_labels[i]} real')
    ax.set_ylabel('Angular vel [rad/s]')
    ax.legend(ncol=3, fontsize=8)
    ax.set_title('Angular velocity: cmd vs real (should match!)')
    ax.grid(True, alpha=0.3)

    # 4. Euler angles (from quaternion)
    euler = np.zeros((N, 3))
    for k in range(N):
        w, x, y, z = log_quat[k]
        euler[k, 0] = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))       # roll
        euler[k, 1] = math.asin(np.clip(2*(w*y - z*x), -1, 1))            # pitch
        euler[k, 2] = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))       # yaw
    euler_deg = np.degrees(euler)

    ax = axes[3]
    e_labels = ['roll', 'pitch', 'yaw']
    for i in range(3):
        ax.plot(log_t, euler_deg[:, i], colors[i], label=e_labels[i])
    ax.axhline(math.degrees(TARGET_YAW), color='b', linestyle='--', alpha=0.5, label='yaw target')
    ax.set_ylabel('Angle [deg]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=8)
    ax.set_title('Euler angles')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_pd_diagnostic.png', dpi=150)
    print("[TEST PD]  Saved test_pd_diagnostic.png")
    plt.show()

    muj.stop()
    muj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        try:
            rclpy.shutdown()
        except Exception:
            pass
