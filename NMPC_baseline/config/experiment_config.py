"""
experiment_config.py – Single source of truth for ALL NMPC baseline parameters.

Sections
--------
1. Initial conditions
2. Trajectory definition
3. Timing & MPC horizon
4. Control limits  (torque model + rate-control model)
5. Physical parameters
6. NMPC cost weights (torque model + rate-control model)
7. MuJoCo simulator parameters
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Initial conditions
# ══════════════════════════════════════════════════════════════════════════════

P0 = [2.0, 0.0, 1.0]            # position  [x, y, z]  [m]
V0 = [0.0, 0.0, 0.0]            # velocity  [vx, vy, vz]  [m/s]
Q0 = [1.0, 0.0, 0.0, 0.0]      # quaternion [qw, qx, qy, qz]
W0 = [0.0, 0.0, 0.0]            # angular velocity [wx, wy, wz]  [rad/s]

X0 = np.array(P0 + V0 + Q0 + W0, dtype=np.double)   # 13-state vector


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Trajectory definition  (Lissajous)
# ══════════════════════════════════════════════════════════════════════════════

TRAJ_VALUE = 15    # frequency scaling factor

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


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Timing & MPC horizon
# ══════════════════════════════════════════════════════════════════════════════

T_FINAL      = 30       # total simulation time [s]
FREC         = 100      # control frequency [Hz]
T_S          = 1 / FREC # sampling time [s]
T_PREDICTION = 0.5      # prediction horizon [s]
N_PREDICTION = int(round(T_PREDICTION / T_S))


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Control limits
# ══════════════════════════════════════════════════════════════════════════════

G = 9.81              # gravitational acceleration [m/s²]

T_MAX    = 3 * G      # max thrust [N]
T_MIN    = 0.0        # min thrust [N]

# -- Torque model (MiL) --
TAU_MAX  = 0.05       # max torque [N·m]

# -- Rate-control model (MuJoCo / real) --
W_MAX    = 3.0        # max angular velocity command [rad/s]


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Physical parameters
# ══════════════════════════════════════════════════════════════════════════════

MASS = 1.0            # [kg]

# -- Torque model (MiL) --
JXX  = 0.00305587     # [kg·m²]
JYY  = 0.00159695     # [kg·m²]
JZZ  = 0.00159687     # [kg·m²]

# -- Rate-control model (MuJoCo / real) --
TAU_RC = 0.03         # rate controller time constant [s]


# ══════════════════════════════════════════════════════════════════════════════
#  6.  NMPC cost weights
# ══════════════════════════════════════════════════════════════════════════════

Q_POSITION    = [25, 25, 30]                # position error  [x, y, z]
Q_ORIENTATION = [12, 12, 12]                # orientation (SO(3) log-map)

# -- Torque model (MiL) --
R_CONTROL_TORQUE = [1.0, 800, 800, 800]     # [T, τx, τy, τz]

# -- Rate-control model (MuJoCo / real) --
R_CONTROL_RATE   = [1.0, 5.0, 5.0, 5.0]    # [T, wx, wy, wz]


# ══════════════════════════════════════════════════════════════════════════════
#  7.  MuJoCo simulator parameters
# ══════════════════════════════════════════════════════════════════════════════

MASS_MUJOCO = 1.08            # [kg] drone mass in MuJoCo model
MASS_RATIO  = MASS_MUJOCO / MASS   # thrust scaling (compensate model mismatch)
