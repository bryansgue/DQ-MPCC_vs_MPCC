"""
experiment_config.py — SINGLE SOURCE OF TRUTH for the entire experiment.

ALL parameters that define the experiment live here:
  • Initial conditions (position, orientation, velocities)
  • Trajectory definition (symbolic lambdas)
  • Timing / MPC horizon
  • Control limits (thrust, torques, progress velocity)
  • Quadrotor physical parameters

Both baselines (DQ-MPCC, MPCC), both tuners, both plot_from_mat scripts,
and both controller OCP files import ONLY from here.

>>> Change parameters ONCE here and they propagate everywhere. <<<
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  1. Initial conditions  (SAME for both controllers)
# ═════════════════════════════════════════════════════════════════════════════

P0     = np.array([0.0, 0.0, 6.0])          # position  [x, y, z]
Q0     = np.array([1.0, 0.0, 0.0, 0.0])     # quaternion [qw, qx, qy, qz]
V0     = np.array([0.0, 0.0, 0.0])          # linear velocity  [m/s]
W0     = np.array([0.0, 0.0, 0.0])          # angular velocity [rad/s]
THETA0 = 0.0                                 # initial arc-length progress [m]


# ═════════════════════════════════════════════════════════════════════════════
#  2. Trajectory definition
#
#     ALL trajectory functions are defined ONLY here.
#     Every file calls:   xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
#
#     VALUE is used internally as speed-scaling factor.
# ═════════════════════════════════════════════════════════════════════════════

VALUE = 5

def trayectoria(t=None):
    """Return the six trajectory lambdas (position + derivatives).

    Returns
    -------
    xd, yd, zd       : callable(t) → float   position components
    xd_p, yd_p, zd_p : callable(t) → float   d/dt of each component
    """
    v = VALUE
    xd   = lambda t: 7.0 * np.sin(v * 0.04 * t) + 3
    yd   = lambda t: 3.5 * np.sin(v * 0.08 * t)
    zd   = lambda t: 1.5 * np.sin(v * 0.08 * t) + 6

    xd_p = lambda t: 7.0 * v * 0.04 * np.cos(v * 0.04 * t)
    yd_p = lambda t: 3.5 * v * 0.08 * np.cos(v * 0.08 * t)
    zd_p = lambda t: 1.5 * v * 0.08 * np.cos(v * 0.08 * t)
    return xd, yd, zd, xd_p, yd_p, zd_p


# ═════════════════════════════════════════════════════════════════════════════
#  3. Timing & MPC horizon
# ═════════════════════════════════════════════════════════════════════════════

T_FINAL      = 60       # [s]  maximum simulation time
FREC         = 100      # [Hz] control frequency
T_PREDICTION = 0.3      # [s]  MPC prediction horizon
N_WAYPOINTS  = 30       # CasADi arc-length interpolation waypoints

# Arc-length limit [m].  None → full curve.
S_MAX_MANUAL = 80


# ═════════════════════════════════════════════════════════════════════════════
#  4. Control limits  (SHARED by BOTH controllers — production AND tuners)
# ═════════════════════════════════════════════════════════════════════════════

G = 9.81                # gravitational acceleration [m/s²]

# Thrust  [N]
T_MAX       = 10 * G    # ~98.1 N
T_MIN       = 0.0

# Torques  [N·m]
TAUX_MAX    = 0.5
TAUY_MAX    = 0.5
TAUZ_MAX    = 0.5

# Progress velocity  [m/s]
VTHETA_MIN  = 0.0
VTHETA_MAX  = 15


# ═════════════════════════════════════════════════════════════════════════════
#  5. Quadrotor physical parameters
# ═════════════════════════════════════════════════════════════════════════════

MASS = 1.0              # [kg]
JXX  = 0.00305587       # [kg·m²]
JYY  = 0.00159695
JZZ  = 0.00159687


# ═════════════════════════════════════════════════════════════════════════════
#  6. Cost weights — DQ-MPCC
#
#     USE_TUNED_WEIGHTS_DQ = True  → load from best_weights.json
#     USE_TUNED_WEIGHTS_DQ = False → use MANUAL values below
# ═════════════════════════════════════════════════════════════════════════════

import os as _os, json as _json

# Workspace root = parent of config/
_WORKSPACE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

USE_TUNED_WEIGHTS_DQ = True

_dq_weights_path = _os.path.join(
    _WORKSPACE_ROOT,
    'DQ-MPCC_baseline', 'tuning', 'best_weights.json',
)

# ── Manual fallback (edit here when USE_TUNED_WEIGHTS_DQ = False) ────────
_DQ_MANUAL_WEIGHTS = dict(
    Q_phi      = [2.043287184439576, 13.372364599755707, 0.6571587126901631],
    Q_ec       = [12.598105468469319, 3.777969151847432, 3.3410980401536166],
    Q_el       = [2.32336746505148, 16.75639691546697, 1.0119359868077427],
    U_mat      = [0.00883036003668909, 24.124142732497422, 72.86548918704212, 32.67238988214993],
    Q_omega    = [0.03951996239989822, 0.032459290238729205, 0.045319866820168864],
    Q_s        = 0.503816052969886
)

if USE_TUNED_WEIGHTS_DQ and _os.path.isfile(_dq_weights_path):
    with open(_dq_weights_path) as _f:
        _dq_w = _json.load(_f)['weights']
    DQ_Q_PHI   = _dq_w['Q_phi']
    DQ_Q_EC    = _dq_w['Q_ec']
    DQ_Q_EL    = _dq_w['Q_el']
    DQ_U_MAT   = _dq_w['U_mat']
    DQ_Q_OMEGA = _dq_w['Q_omega']
    DQ_Q_S     = _dq_w['Q_s']
    print(f"[experiment_config] ✓ DQ-MPCC weights loaded from {_dq_weights_path}")
else:
    if USE_TUNED_WEIGHTS_DQ:
        print(f"[experiment_config] ⚠ {_dq_weights_path} not found → manual defaults")
    DQ_Q_PHI   = _DQ_MANUAL_WEIGHTS['Q_phi']
    DQ_Q_EC    = _DQ_MANUAL_WEIGHTS['Q_ec']
    DQ_Q_EL    = _DQ_MANUAL_WEIGHTS['Q_el']
    DQ_U_MAT   = _DQ_MANUAL_WEIGHTS['U_mat']
    DQ_Q_OMEGA = _DQ_MANUAL_WEIGHTS['Q_omega']
    DQ_Q_S     = _DQ_MANUAL_WEIGHTS['Q_s']


# ═════════════════════════════════════════════════════════════════════════════
#  7. Cost weights — MPCC
#
#     USE_TUNED_WEIGHTS_MPCC = True  → load from best_weights.json
#     USE_TUNED_WEIGHTS_MPCC = False → use MANUAL values below
# ═════════════════════════════════════════════════════════════════════════════

USE_TUNED_WEIGHTS_MPCC = True

_mpcc_weights_path = _os.path.join(
    _WORKSPACE_ROOT,
    'MPCC_baseline', 'tuning', 'best_weights.json',
)

# ── Manual fallback (edit here when USE_TUNED_WEIGHTS_MPCC = False) ──────
_MPCC_MANUAL_WEIGHTS = dict(
    Q_ec    = [1.4093957442618654, 27.817042827694642, 24.48548392738109],
    Q_el    = [5.9582011063084055, 23.795100612706815, 4.161494597550919],
    Q_q     = [0.1343138656640675, 0.432728076186103, 2.1883004979219733],
    U_mat   = [0.012397089263446425, 247.4781149655814, 271.0477082160289, 420.10767213246106],
    Q_omega = [0.10042795453551734, 0.08553884748528592, 0.05216153572943215],
    Q_s     = 0.5004891759540515,
)

if USE_TUNED_WEIGHTS_MPCC and _os.path.isfile(_mpcc_weights_path):
    with open(_mpcc_weights_path) as _f:
        _mpcc_w = _json.load(_f)['weights']
    MPCC_Q_EC    = _mpcc_w['Q_ec']
    MPCC_Q_EL    = _mpcc_w['Q_el']
    MPCC_Q_Q     = _mpcc_w['Q_q']
    MPCC_U_MAT   = _mpcc_w['U_mat']
    MPCC_Q_OMEGA = _mpcc_w['Q_omega']
    MPCC_Q_S     = _mpcc_w['Q_s']
    print(f"[experiment_config] ✓ MPCC weights loaded from {_mpcc_weights_path}")
else:
    if USE_TUNED_WEIGHTS_MPCC:
        print(f"[experiment_config] ⚠ {_mpcc_weights_path} not found → manual defaults")
    MPCC_Q_EC    = _MPCC_MANUAL_WEIGHTS['Q_ec']
    MPCC_Q_EL    = _MPCC_MANUAL_WEIGHTS['Q_el']
    MPCC_Q_Q     = _MPCC_MANUAL_WEIGHTS['Q_q']
    MPCC_U_MAT   = _MPCC_MANUAL_WEIGHTS['U_mat']
    MPCC_Q_OMEGA = _MPCC_MANUAL_WEIGHTS['Q_omega']
    MPCC_Q_S     = _MPCC_MANUAL_WEIGHTS['Q_s']


# ═════════════════════════════════════════════════════════════════════════════
#  8. Tuning configuration
#
#     Everything the bilevel tuners need: timing overrides, Optuna settings,
#     objective penalties, and search-space bounds.
#     Previously in tuning_config.py — now centralised here.
# ═════════════════════════════════════════════════════════════════════════════

# ── Timing overrides for tuning (shorter than production) ────────────────
TUNING_T_FINAL      = 30       # [s]  shorter sim per evaluation
TUNING_FREC         = 100      # [Hz] (same as production, but explicit)
TUNING_T_PREDICTION = 0.3      # [s]
TUNING_N_WAYPOINTS  = 30
TUNING_S_MAX_MANUAL = None     # None → use full curve length

# ── Multi-velocity tuning ────────────────────────────────────────────────
#    Representative velocities at which EACH Optuna trial is evaluated.
#    The tuner's objective is the MEAN over all velocities, producing a
#    single robust weight set that generalises across the velocity sweep.
#    Pick ≥ 3 velocities spanning the intended sweep range.
TUNING_VELOCITIES = [5, 10, 16]   # [m/s] — low, mid, high

# ── Optuna / optimiser settings ──────────────────────────────────────────
DEFAULT_N_TRIALS    = 50       # Optuna trials per run
DEFAULT_SAMPLER     = 'tpe'    # 'tpe' or 'cmaes'
N_STARTUP_TRIALS    = 10       # TPE random warmup trials
OPTUNA_SEED         = 42       # reproducibility

# ── Objective function penalty weights ───────────────────────────────────

# Incomplete-path penalty (same for both baselines)
W_INCOMPLETE        = 1000.0

# Speed-awareness penalties (prevent "velocity starvation")
#   Term A: W_TIME · (t_lap / T_ref)         penalises long lap times
#   Term B: W_VEL  · (v_max − v̄_θ) / v_max  penalises low mean speed
#   T_ref = s_max / VTHETA_MAX
W_TIME              = 0.5
W_VEL               = 2.0

# Isotropy penalty (prevent axis-neglect)
#   W_ISOTROPY · (max(rmse_xyz) / min(rmse_xyz) − 1)
W_ISOTROPY          = 0.3

# Contouring-error penalty (force tight path tracking)
#   W_CONTOUR · rmse_contorno
W_CONTOUR           = 3.0

# ── Search-space bounds ──────────────────────────────────────────────────
#    Each entry: (low, high, log_scale)

# Contouring error  Q_ec  [3]
Q_EC_RANGE    = (1.0,  30.0,  True)
# Lag error  Q_el  [3]
Q_EL_RANGE    = (0.5,  30.0,  True)
# Control effort  U_mat
U_T_RANGE     = (0.005, 0.5,  True)    # thrust weight
U_TAU_RANGE   = (20,   800,   True)    # torque weights (τx, τy, τz)
# Angular velocity  Q_omega  [3]
Q_OMEGA_RANGE = (0.01,  2.0,  True)
# Progress speed  Q_s
Q_S_RANGE     = (0.5,   20,   True)
# Rotation error (log-map):  Q_q (MPCC) / Q_phi (DQ-MPCC)
Q_ROT_RANGE   = (0.1,  20.0,  True)
