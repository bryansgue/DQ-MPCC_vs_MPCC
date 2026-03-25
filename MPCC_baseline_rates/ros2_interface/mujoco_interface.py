"""
mujoco_interface.py – Reusable ROS 2 interface for MuJoCo quadrotor simulator.

Subscribes to /quadrotor/odom and publishes to /quadrotor/trpy_cmd.
Controller-agnostic: works with NMPC, MPCC, PID, or any other controller.

Usage
-----
    from ros2_interface.mujoco_interface import MujocoInterface, wait_for_connection

    rclpy.init()
    muj = MujocoInterface()
    spin_thread = threading.Thread(target=rclpy.spin, args=(muj,), daemon=True)
    spin_thread.start()
    wait_for_connection(muj)

    # Fly to position and hold (runs PD in background thread)
    muj.start_pd_hold(target=[1, 1, 1], mass=1.08)

    # ... compile solver, load model, etc ...

    # Stop PD, main controller takes over
    muj.stop_pd_hold()

    # Read state
    pos, vel, quat, omega = muj.get_state()

    # Send command (AcroMode: thrust + desired angular velocity)
    muj.send_cmd(thrust=9.81, wx=0.0, wy=0.0, wz=0.0)
"""

import numpy as np
import time
import math
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TRPYCommand


class MujocoInterface(Node):
    """Minimal ROS 2 node: subscribes to odom, publishes trpy_cmd.

    Thread-safe: state is protected by a lock so the control loop
    (running in the main thread) can read while the ROS 2 spinner
    writes from callbacks.

    Parameters
    ----------
    node_name  : str   – ROS 2 node name (default: 'mujoco_controller')
    odom_topic : str   – odometry topic (default: '/quadrotor/odom')
    cmd_topic  : str   – command topic  (default: '/quadrotor/trpy_cmd')
    """

    def __init__(
        self,
        node_name: str = 'mujoco_controller',
        odom_topic: str = '/quadrotor/odom',
        cmd_topic: str = '/quadrotor/trpy_cmd',
    ):
        super().__init__(node_name)

        # State from odom
        self.pos   = np.zeros(3)
        self.vel   = np.zeros(3)                          # world frame
        self.quat  = np.array([1., 0., 0., 0.])           # [qw, qx, qy, qz]
        self.omega = np.zeros(3)                           # body frame
        self.connected = False
        self._lock = threading.Lock()

        # PD hold thread state
        self._pd_active = False
        self._pd_thread = None

        # Publisher / Subscriber
        self.cmd_pub = self.create_publisher(
            TRPYCommand, cmd_topic, 10)
        self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10)

    def _odom_cb(self, msg: Odometry):
        """Callback: extract state from nav_msgs/Odometry."""
        with self._lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear        # world frame
            w = msg.twist.twist.angular       # body frame

            self.pos[:]   = [p.x, p.y, p.z]
            self.vel[:]   = [v.x, v.y, v.z]
            self.quat[:]  = [q.w, q.x, q.y, q.z]
            self.omega[:] = [w.x, w.y, w.z]
            self.connected = True

    def get_state(self):
        """Return (pos, vel, quat, omega) as a thread-safe snapshot.

        Returns
        -------
        pos   : ndarray (3,)  – position [x, y, z] (world)
        vel   : ndarray (3,)  – velocity [vx, vy, vz] (world)
        quat  : ndarray (4,)  – quaternion [qw, qx, qy, qz]
        omega : ndarray (3,)  – angular velocity [wx, wy, wz] (body)
        """
        with self._lock:
            return (self.pos.copy(), self.vel.copy(),
                    self.quat.copy(), self.omega.copy())

    def get_state_vector(self):
        """Return full 13-element state vector [p, v, q, w].

        Returns
        -------
        x : ndarray (13,)
        """
        pos, vel, quat, omega = self.get_state()
        return np.concatenate([pos, vel, quat, omega])

    def send_cmd(self, thrust: float, wx: float = 0., wy: float = 0., wz: float = 0.):
        """Publish TRPYCommand (AcroMode: thrust + desired angular velocity).

        Parameters
        ----------
        thrust : float  – total thrust [N]
        wx, wy, wz : float  – desired body angular velocity [rad/s]
        """
        msg = TRPYCommand()
        msg.thrust = float(thrust)
        msg.angular_velocity.x = float(wx)
        msg.angular_velocity.y = float(wy)
        msg.angular_velocity.z = float(wz)
        self.cmd_pub.publish(msg)

    # ── PD position hold (takeoff + hold in one) ────────────────────────────

    def start_pd_hold(self, target, mass: float, g: float = 9.81,
                      kp_xy: float = 4.0, kd_xy: float = 2.5,
                      kp_z: float = 8.0, kd_z: float = 4.0,
                      kp_att: float = 6.0, kp_yaw: float = 2.0):
        """Fly to target and hold position via PD control (background thread).

        Runs at ~100 Hz until stop_pd_hold() is called.  Handles both the
        initial flight to the target AND holding once there — no separate
        takeoff function needed.

        Parameters
        ----------
        target  : array-like (3,)  – desired [x, y, z]
        mass    : float            – drone mass [kg]
        g       : float            – gravity [m/s²]
        kp_xy, kd_xy : float      – PD gains for X-Y position
        kp_z, kd_z   : float      – PD gains for Z position
        kp_att        : float      – P gain for attitude (roll/pitch)
        kp_yaw        : float      – P gain for yaw
        """
        target = np.asarray(target, dtype=np.float64)
        self._pd_active = True

        def _pd_loop():
            print(f"[PD HOLD]  Flying to target = {target} and holding...")
            while self._pd_active:
                pos, vel, quat, _ = self.get_state()
                err = target - pos

                # Desired acceleration (world frame)
                a_des = np.array([
                    kp_xy * err[0] - kd_xy * vel[0],
                    kp_xy * err[1] - kd_xy * vel[1],
                    kp_z  * err[2] - kd_z  * vel[2] + g,
                ])

                # Rotation matrix from quaternion
                w, x, y, z = quat
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
                    [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
                ])

                thrust = mass * np.dot(a_des, R[:, 2])
                thrust = np.clip(thrust, 0.0, 60.0)

                psi = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                az  = max(a_des[2], 0.1)
                pitch_des = np.clip(
                    (a_des[0]*math.cos(psi) + a_des[1]*math.sin(psi)) / az, -0.5, 0.5)
                roll_des = np.clip(
                    (a_des[0]*math.sin(psi) - a_des[1]*math.cos(psi)) / az, -0.5, 0.5)

                pitch_cur = math.asin(np.clip(R[0, 2], -1, 1))
                roll_cur  = math.atan2(-R[1, 2], R[2, 2])
                yaw_err   = (0.0 - psi + math.pi) % (2*math.pi) - math.pi

                wx = kp_att * (roll_des - roll_cur)
                wy = kp_att * (pitch_des - pitch_cur)
                wz = kp_yaw * yaw_err

                self.send_cmd(thrust, wx, wy, wz)
                time.sleep(0.01)

            print("[PD HOLD]  Stopped — main controller takes over")

        self._pd_thread = threading.Thread(target=_pd_loop, daemon=True)
        self._pd_thread.start()

    def stop_pd_hold(self):
        """Stop the PD hold thread. Call right before the main control loop."""
        self._pd_active = False
        if self._pd_thread is not None:
            self._pd_thread.join(timeout=0.5)
            self._pd_thread = None

    def stop(self):
        """Send zero thrust."""
        self.send_cmd(thrust=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone helpers
# ══════════════════════════════════════════════════════════════════════════════

def wait_for_connection(muj: MujocoInterface, timeout: float = 10.0) -> bool:
    """Block until the first odom message arrives.

    Parameters
    ----------
    muj     : MujocoInterface
    timeout : float  – max wait time [s]

    Returns
    -------
    connected : bool
    """
    print("Waiting for MuJoCo connection (/quadrotor/odom)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if muj.connected:
            print(f"Connected!  pos = {muj.pos}")
            return True
        time.sleep(0.1)
    print("ERROR: No /quadrotor/odom received. Launch the simulator first.")
    return False
