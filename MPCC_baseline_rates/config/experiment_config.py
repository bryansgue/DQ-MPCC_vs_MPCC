"""Local configuration for the standalone MPCC rate-control baseline (MiL)."""

import numpy as np


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
    xd  = lambda t: 2.50 * np.sin(v * 0.04 * t) + 2.5
    yd  = lambda t: 1.5 * np.sin(v * 0.08 * t)
    zd  = lambda t: 0.5 * np.sin(v * 0.04 * t) + 1.2
    xdp = lambda t: 2.5 * v * 0.04 * np.cos(v * 0.04 * t)
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
# ── High-speed config (round 3 tuning: 5–20 m/s) ─────────────────────────────
# At 20 m/s one figure-8 loop (≈18m) takes <1s → need multiple loops.
# S_MAX = 60m ≈ 3 loops — enough to evaluate sustained high-speed tracking.
# T_TRAJ_BUILD must give arc ≥ S_MAX*1.2 = 72m.  At ~1.67 m/s → need ≥43s.
T_FINAL = 30          # upper-bound [s] — (60m @ 5 m/s = 12s; @ 20 m/s = 3s)
T_TRAJ_BUILD = 50     # [s] → ~83m arc > 72m ✓
FREC = 100
T_PREDICTION = 0.2
N_WAYPOINTS = 40      # more waypoints for 72m of path (was 20 for 24m)
S_MAX_MANUAL = 60.0   # [m] — ~3 figure-8 loops for high-speed evaluation


# Limits — acrobatic regime for ≥15 m/s flight
G = 9.81
T_MIN = 0.0
T_MAX = 5.0 * G       # [N]      5g max thrust — acrobatic headroom
W_MAX = 20.0           # [rad/s]  20 rad/s ≈ full acrobatic angular rate
VTHETA_MIN = 0.0
VTHETA_MAX = 10      # [m/s]    headroom above 15 m/s target


# Physical parameters
MASS = 1.0
MASS_MUJOCO = 1.08
TAU_RC = 0.03


# MPCC-rate weights
# These are intentionally local to this pipeline. The torque-based MPCC tuned
# weights from MPCC_baseline are not appropriate for a body-rate-controlled
# plant with first-order rate dynamics.
# Round-2 bilevel tuning result — trial #96, J_multi=32.24
# v̄θ ≈ 3.3–4.0 m/s, RMSE_c ≈ 12–14 cm, path 100%
MPCC_Q_EC       = [50.85580056256094, 50.813767399692967, 274.37361520466055]
MPCC_Q_EL       = [183.17397384681195, 80.9436746714794, 78.0862549670832]
MPCC_Q_Q        = [0.10976784589628846, 0.10754075088677592, 0.9804324408870685]
MPCC_Q_OMEGA = [0.12, 0.12, 0.08]   # not tuned (removed from bilevel)
MPCC_Q_S   = 1.853716595168155

# Rate-control effort weights (tuned round 2)
MPCC_RATE_U_MAT = [0.15266200900271537, 0.20681449138135174, 0.2664526664190134, 0.4855109861396445]
