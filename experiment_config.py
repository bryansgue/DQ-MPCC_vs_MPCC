"""
experiment_config.py — Shared experimental parameters for DQ-MPCC vs MPCC.

ALL initial conditions and protocol settings live here.
Both simulation scripts import from this file so the experiment is
guaranteed to be identical.

Modify ONLY this file to change the experiment setup.
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Initial conditions  (SAME for both controllers)
# ═════════════════════════════════════════════════════════════════════════════

# Position [x, y, z] in ℝ³  (arbitrary, does NOT need to be on the path)
P0 = np.array([3.0, 0.0, 6.0])

# Orientation as unit quaternion [qw, qx, qy, qz] (scalar-first Hamilton)
# Examples:
#   Identity (yaw=0):           [1, 0, 0, 0]
#   Yaw = 90°:                  [0.7071, 0, 0, 0.7071]
#   Yaw = 63.43° (path-aligned at θ=0): [0.8507, 0, 0, 0.5257]
Q0 = np.array([1.0, 0.0, 0.0, 0.0])

# Initial linear velocity [vx, vy, vz] in m/s  (inertial frame for MPCC,
#                                                 body frame for DQ-MPCC)
V0 = np.array([0.0, 0.0, 0.0])

# Initial angular velocity [ωx, ωy, ωz] in rad/s
W0 = np.array([0.0, 0.0, 0.0])

# Initial arc-length progress θ₀ [m]
THETA0 = 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Trajectory & timing
# ═════════════════════════════════════════════════════════════════════════════

VALUE        = 5        # trajectory speed scaling
T_FINAL      = 60       # [s] safety time budget (never the binding constraint)
FREC         = 100      # [Hz] control frequency
T_PREDICTION = 0.3      # [s] MPC prediction horizon
N_WAYPOINTS  = 30       # CasADi interpolation waypoints

# Arc-length limit [m].  None → full curve (~126.8 m).
S_MAX_MANUAL = 100.0
