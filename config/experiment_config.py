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

import json
from pathlib import Path
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  1. Initial conditions  (SAME for both controllers)
# ═════════════════════════════════════════════════════════════════════════════

P0     = np.array([4, 0.0, 1])          # position  [x, y, z]
Q0     = np.array([1.0, 0.0, 0.0, 0.0])     # quaternion [qw, qx, qy, qz]
V0     = np.array([0.0, 0.0, 0.0])          # linear velocity  [m/s]
W0     = np.array([0.0, 0.0, 0.0])          # angular velocity [rad/s]
THETA0 = 0.94                                # initial arc-length progress [m] — arc-length of path point closest to P0=[4,0,1]


# ═════════════════════════════════════════════════════════════════════════════
#  2. Trajectory definition
#
#     ALL trajectory functions are defined ONLY here.
#     Every file calls:   xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
#
#     VALUE is used internally as speed-scaling factor.
# ═════════════════════════════════════════════════════════════════════════════



TRAJ_VALUE = 15    # frequency scaling factor

def trayectoria(t=None):
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
    zd  = lambda t: 0.5 * np.sin(v * 0.04 * t) + 1.5
    xdp = lambda t: 2.5 * v * 0.04 * np.cos(v * 0.04 * t)
    ydp = lambda t: 1.5 * v * 0.08 * np.cos(v * 0.08 * t)
    zdp = lambda t: 0.5 * v * 0.04 * np.cos(v * 0.04 * t)
    return xd, yd, zd, xdp, ydp, zdp



# ═════════════════════════════════════════════════════════════════════════════
#  3. Timing & MPC horizon
# ═════════════════════════════════════════════════════════════════════════════

T_FINAL      = 85      # [s]  maximum simulation time
TRAJECTORY_T_FINAL = 70 # [s]  parameter interval used to define the path geometry 63
FREC         = 100      # [Hz] control frequency
T_PREDICTION = 0.3      # [s]  MPC prediction horizon
N_WAYPOINTS  = 160      # CasADi arc-length interpolation waypoints

# Important:
# - `T_FINAL` only changes how long the simulation is allowed to run.
# - `TRAJECTORY_T_FINAL` changes the geometric path used to build the
#   arc-length parameterisation. Changing it does change the OCP trajectory,
#   so it requires rebuilding the compiled OCP once.

# Arc-length limit [m].  None → full curve.
S_MAX_MANUAL = 100

# ═════════════════════════════════════════════════════════════════════════════
#  4. Control limits  (SHARED by BOTH controllers — production AND tuners)
# ═════════════════════════════════════════════════════════════════════════════
#
# MPCC OCP note:
# These limits are NUMERIC runtime parameters for the current MPCC pipeline.
# They are used to set the OCP input bounds when the solver is initialised and,
# if needed, can be updated without changing the symbolic cost structure.
#
# The associated progress-speed reference also enters the MPCC parameter vector
# through:
#   p[17] = vtheta_max
# in `MPCC_baseline/ocp/mpcc_controller.py`.

G = 9.81                # gravitational acceleration [m/s²]

# Thrust  [N]
T_MAX       = 10 * G    # ~98.1 N | editable without rebuilding the MPCC OCP
T_MIN       = 0.0       # editable without rebuilding the MPCC OCP

# Torques  [N·m]
TAUX_MAX    = 0.5       # editable without rebuilding the MPCC OCP
TAUY_MAX    = 0.5       # editable without rebuilding the MPCC OCP
TAUZ_MAX    = 0.5       # editable without rebuilding the MPCC OCP

# Progress velocity  [m/s]
VTHETA_MIN  = 0.0       # editable without rebuilding the MPCC OCP
VTHETA_MAX  = 20         # editable without rebuilding the MPCC OCP | mapped to p[17]

# Progress acceleration  [m/s²]
ATHETA_MIN  = -25.0
ATHETA_MAX  = 25.0

# Attitude-reference construction for MPCC.
# The desired quaternion is built from:
#   - path tangent (heading)
#   - path curvature + nominal speed (lateral acceleration demand)
# instead of assuming roll = pitch = 0 everywhere.
ATTITUDE_REF_SPEED = 15       # [m/s] nominal speed used for attitude reference
ATTITUDE_REF_MAX_TILT_DEG = 60.0  # [deg] keep reference physically reasonable


# ═════════════════════════════════════════════════════════════════════════════
#  5. Quadrotor physical parameters
# ═════════════════════════════════════════════════════════════════════════════

MASS = 1.0380             # [kg]
JXX  = 3.2086e-03       # [kg·m²]
JYY  = 1.6731e-03
JZZ  = 1.6730e-03


# Repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ═════════════════════════════════════════════════════════════════════════════
#  6. Cost weights — DQ-MPCC
#
#     Simple rule:
#       - `USE_TUNED_WEIGHTS_DQ = True`  -> load JSON from tuning/
#       - `USE_TUNED_WEIGHTS_DQ = False` -> use the manual values below
# ═════════════════════════════════════════════════════════════════════════════

USE_TUNED_WEIGHTS_DQ = True

# Active DQ set for experiments:
# keep the refined translational/progress structure, but slightly increase the
# rotation penalty to probe whether the external orientation RMSE of
# Experiment 2 improves under the shared quaternion reference.
DQ_WEIGHTS_PATH = PROJECT_ROOT / "DQ-MPCC_baseline" / "tuning" / "final_refined_oriented_weights.json"

# Used when `USE_TUNED_WEIGHTS_DQ = False`, or if the JSON file is missing.
DQ_MANUAL_WEIGHTS = dict(
    Q_ec       = [40, 40, 40],
    Q_el       = [3, 3, 3],
    Q_phi      = [0.65, 0.65, 1.9],
    U_mat      = [0.05, 70, 70, 70],
    Q_omega    = [0.02, 0.02, 0.10],
    Q_s        = 11
)

if USE_TUNED_WEIGHTS_DQ and DQ_WEIGHTS_PATH.is_file():
    with DQ_WEIGHTS_PATH.open(encoding="utf-8") as f:
        DQ_WEIGHTS = json.load(f)["weights"]
    print(f"[experiment_config] ✓ DQ-MPCC weights loaded from {DQ_WEIGHTS_PATH}")
else:
    if USE_TUNED_WEIGHTS_DQ:
        print(f"[experiment_config] ⚠ {DQ_WEIGHTS_PATH} not found -> manual defaults")
    DQ_WEIGHTS = DQ_MANUAL_WEIGHTS

DQ_Q_PHI   = DQ_WEIGHTS["Q_phi"]
DQ_Q_EC    = DQ_WEIGHTS["Q_ec"]
DQ_Q_EL    = DQ_WEIGHTS["Q_el"]
DQ_U_MAT   = DQ_WEIGHTS["U_mat"]
DQ_Q_OMEGA = DQ_WEIGHTS["Q_omega"]
DQ_Q_S     = DQ_WEIGHTS["Q_s"]
DQ_Q_ATHETA = 0.20


# ═════════════════════════════════════════════════════════════════════════════
#  7. Cost weights — MPCC
#
#     Simple rule:
#       - `USE_TUNED_WEIGHTS_MPCC = True`  -> load JSON from tuning/
#       - `USE_TUNED_WEIGHTS_MPCC = False` -> use the manual values below
#
#     These weights are runtime parameters of the canonical MPCC OCP:
#       p[0:3]   = MPCC_Q_EC
#       p[3:6]   = MPCC_Q_EL
#       p[6:9]   = MPCC_Q_Q
#       p[9:13]  = MPCC_U_MAT
#       p[13:16] = MPCC_Q_OMEGA
#       p[16]    = MPCC_Q_S
#
#     Changing these values does NOT require rebuilding the MPCC OCP.
#     They are injected at runtime through `solver.set(stage, "p", p_vec)`.
# ═════════════════════════════════════════════════════════════════════════════

USE_TUNED_WEIGHTS_MPCC = False
# Active MPCC set for experiments:
# keep the local-refined tracking/orientation structure, but slightly relax the
# high-speed progress incentive so the baseline does not plateau too early in
# the velocity sweep.
MPCC_WEIGHTS_PATH = PROJECT_ROOT / "MPCC_baseline" / "tuning" / "final_refined_relaxed_weights.json"

# Used when `USE_TUNED_WEIGHTS_MPCC = False`, or if the JSON file is missing.
# Main manual reference for the current MPCC baseline experiments.
# Previous MPCC manual weights kept here for quick rollback if needed.
MPCC_MANUAL_WEIGHTS_PREV = dict(
    Q_ec       = [25, 25, 25],
    Q_el       = [3, 3, 3],
    Q_q        = [0.5, 0.5, 1.9],
    U_mat      = [0.05, 70, 70, 70],
    Q_omega    = [0.02, 0.02, 0.10],
    Q_s        = 15
)

# Temporary MPCC manual weights matched to the current DQ tuning so you can
# inspect both baselines under the same gain pattern.
MPCC_MANUAL_WEIGHTS = dict(
    Q_ec       = [40, 40, 40],
    Q_el       = [3, 3, 3],
    Q_q        = [0.65, 0.65, 1.9],
    U_mat      = [0.05, 70, 70, 70],
    Q_omega    = [0.02, 0.02, 0.10],
    Q_s        = 11
)

# Fixed penalty for the new progress-acceleration input a_theta.
MPCC_Q_ATHETA = 0.20

if USE_TUNED_WEIGHTS_MPCC and MPCC_WEIGHTS_PATH.is_file():
    with MPCC_WEIGHTS_PATH.open(encoding="utf-8") as f:
        MPCC_WEIGHTS = json.load(f)["weights"]
    print(f"[experiment_config] ✓ MPCC weights loaded from {MPCC_WEIGHTS_PATH}")
else:
    if USE_TUNED_WEIGHTS_MPCC:
        print(f"[experiment_config] ⚠ {MPCC_WEIGHTS_PATH} not found -> manual defaults")
    MPCC_WEIGHTS = MPCC_MANUAL_WEIGHTS

MPCC_Q_EC    = MPCC_WEIGHTS["Q_ec"]      # p[0:3]   | runtime-editable, no OCP rebuild
MPCC_Q_EL    = MPCC_WEIGHTS["Q_el"]      # p[3:6]   | runtime-editable, no OCP rebuild
MPCC_Q_Q     = MPCC_WEIGHTS["Q_q"]       # p[6:9]   | runtime-editable, no OCP rebuild
MPCC_U_MAT   = MPCC_WEIGHTS["U_mat"]     # p[9:13]  | runtime-editable, no OCP rebuild
MPCC_Q_OMEGA = MPCC_WEIGHTS["Q_omega"]   # p[13:16] | runtime-editable, no OCP rebuild
MPCC_Q_S     = MPCC_WEIGHTS["Q_s"]       # p[16]    | runtime-editable, no OCP rebuild


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
TUNING_VELOCITIES = [5, 10, 16]   # [m/s] — multi-velocity (paper Eq.11)

# ── Optuna / optimiser settings ──────────────────────────────────────────
DEFAULT_N_TRIALS    = 200       # Optuna trials per run
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
W_VEL               = 1.0

# Isotropy penalty (prevent axis-neglect)
#   W_ISOTROPY · (max(rmse_xyz) / min(rmse_xyz) − 1)
W_ISOTROPY          = 0.3

# Contouring-error penalty (force tight path tracking)
#   W_CONTOUR · rmse_contorno
W_CONTOUR           = 3.0

# Additional behaviour penalties for production-aligned tuning
W_LAG               = 3.0
W_ATT               = 1.5
W_OMEGA             = 0.05
W_EFFORT            = 0.1
W_DU                = 0.2
W_SAT               = 25.0
W_PEAK              = 4.0
W_FAIL              = 500.0
W_VPATH             = 1.5
W_VPATH_HARD        = 400.0
VPATH_RATIO_MIN     = 0.75
W_SAT_HARD          = 400.0
SAT_RATIO_MAX       = 0.03

# ── Search-space bounds ──────────────────────────────────────────────────
#    Each entry: (low, high, log_scale)

# Contouring error  Q_ec  [3]
Q_EC_RANGE    = (1.0,  30.0,  True)
# Lag error  Q_el  [3]
Q_EL_RANGE    = (0.5,  30.0,  True)
# Control effort  U_mat
U_T_RANGE     = (0.02, 1.0,   True)    # thrust weight
U_TAU_RANGE   = (50,   1200,  True)    # torque weights (τx, τy, τz)
# Angular velocity  Q_omega  [3]
Q_OMEGA_RANGE = (0.01,  2.0,  True)
# Progress speed  Q_s
Q_S_RANGE     = (0.5,   20,   True)
# Rotation error (log-map):  Q_q (MPCC) / Q_phi (DQ-MPCC)
Q_ROT_RANGE   = (0.1,  20.0,  True)
