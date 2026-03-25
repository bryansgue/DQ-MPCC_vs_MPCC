"""Local configuration for the standalone MPCC rate-control baseline (MiL)."""

import numpy as np


TRAJ_VALUE = 10    # frequency scaling factor

def trayectoria():
    """Return 6 lambdas: (xd, yd, zd, xdp, ydp, zdp).

    Lissajous figure-8 trajectory within MuJoCo walls.
    X: cx=2.5 ± 3.0  → [−0.5, 5.5]   (margen 2.5m a paredes)
    Y: cy=0.0 ± 1.5  → [−1.5, 1.5]   (margen 0.5m a paredes)
    Z: cz=1.2 ± 0.5  → [0.7, 1.7]    (dentro de paredes h=3m)
    Relación freq X:Y = 1:2 → forma de 8 en plano XY
    """
    v = TRAJ_VALUE
    xd  = lambda t: 3.0 * np.sin(v * 0.04 * t) + 2.5
    yd  = lambda t: 1.5 * np.sin(v * 0.08 * t)
    zd  = lambda t: 0.5 * np.sin(v * 0.04 * t) + 1.2
    xdp = lambda t: 3.0 * v * 0.04 * np.cos(v * 0.04 * t)
    ydp = lambda t: 1.5 * v * 0.08 * np.cos(v * 0.08 * t)
    zdp = lambda t: 0.5 * v * 0.04 * np.cos(v * 0.04 * t)
    return xd, yd, zd, xdp, ydp, zdp


# Initial conditions
# For the standalone MiL rate baseline we start directly on the path,
# tangent-aligned, so the diagnosis is about MPCC-rate itself instead of
# an avoidable takeoff/transient mismatch.
_xd0, _yd0, _zd0, _xdp0, _ydp0, _zdp0 = trayectoria()
_yaw0 = np.arctan2(_ydp0(0.0), _xdp0(0.0))
P0 = np.array([_xd0(0.0), _yd0(0.0), _zd0(0.0)], dtype=np.double)
Q0 = np.array([np.cos(_yaw0 / 2.0), 0.0, 0.0, np.sin(_yaw0 / 2.0)], dtype=np.double)
V0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
W0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
THETA0 = 0.0



# Timing
T_FINAL = 60.0
FREC = 100
T_PREDICTION = 0.5
N_WAYPOINTS = 30
S_MAX_MANUAL = 80.0


# Limits
# Mirrored from NMPC_baseline/config/experiment_config.py for the
# rate-controlled action space.
G = 9.81
T_MIN = 0.0
T_MAX = 3.0 * G
W_MAX = 3.0
VTHETA_MIN = 0.0
VTHETA_MAX = 15


# Physical parameters
MASS = 1.0
TAU_RC = 0.03


# MPCC-rate weights
# These are intentionally local to this pipeline. The torque-based MPCC tuned
# weights from MPCC_baseline are not appropriate for a body-rate-controlled
# plant with first-order rate dynamics.
MPCC_Q_EC = [45.0, 45.0, 55.0]
MPCC_Q_EL = [18.0, 18.0, 22.0]
MPCC_Q_Q = [3.0, 3.0, 1.5]
MPCC_Q_OMEGA = [0.12, 0.12, 0.08]
MPCC_Q_S = 0.35

# Dedicated rate-control effort weights.
# Keep rates relatively cheap so the optimiser can actually steer the vehicle
# onto the path; thrust is still penalised around hover in the OCP.
MPCC_RATE_U_MAT = [1.0, 1.5, 1.5, 1.0]
