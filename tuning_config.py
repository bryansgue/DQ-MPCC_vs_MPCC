"""
tuning_config.py — Shared configuration for bilevel gain tuning.

Both MPCC and DQ-MPCC tuners import from this file so the tuning
experiment is guaranteed to run under IDENTICAL conditions.

Modify ONLY this file to change the tuning setup.

Sections
────────
  1. Simulation timing & trajectory
  2. Control limits (SHARED between both baselines)
  3. Optuna / optimiser settings
  4. Search-space bounds
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  1.  Simulation timing & trajectory
# ═════════════════════════════════════════════════════════════════════════════

VALUE        = 5        # trajectory speed-scaling factor
T_FINAL      = 30       # [s]  simulation time budget for each evaluation
FREC         = 100      # [Hz] control frequency
T_PREDICTION = 0.3      # [s]  MPC prediction horizon
N_WAYPOINTS  = 30       # CasADi arc-length interpolation waypoints

# Arc-length limit.  None → use the full curve length.
S_MAX_MANUAL = None     # [m]  (None = auto)


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Control limits  (SHARED — both controllers use the same envelope)
# ═════════════════════════════════════════════════════════════════════════════

G = 9.81

# Thrust  [N]
T_MAX       = 10 * G        # max thrust
T_MIN       = 0.0           # min thrust  (no negative thrust)

# Torques  [N·m]
# NOTE: best MPCC weights (J=32.71) were tuned with τ_max=0.5.
#       Production controller MUST use the same value.
TAUX_MAX    = 0.5            # roll  torque limit
TAUY_MAX    = 0.5            # pitch torque limit
TAUZ_MAX    = 0.5            # yaw   torque limit

# Progress velocity  [m/s]
VTHETA_MIN  = 0.0
VTHETA_MAX  = 15.0


# ═════════════════════════════════════════════════════════════════════════════
#  3.  Optuna / optimiser settings
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_N_TRIALS      = 50       # Optuna trials per run
DEFAULT_SAMPLER       = 'tpe'    # 'tpe' or 'cmaes'
N_STARTUP_TRIALS      = 10       # TPE random warmup trials
OPTUNA_SEED           = 42       # reproducibility

# Incomplete-path penalty  (same for both baselines)
W_INCOMPLETE          = 1000.0


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Search-space bounds
#      Shared ranges so both tuners explore the same hyper-volumes.
#      Each entry: (name, low, high, log_scale)
# ═════════════════════════════════════════════════════════════════════════════

# ── Common search ranges ─────────────────────────────────────────────────────
# Contouring error Q_ec  [3]
Q_EC_RANGE   = (1.0,  30.0,  True)    # (low, high, log)
# Lag error Q_el  [3]
Q_EL_RANGE   = (0.5,  30.0,  True)
# Control effort  U_mat
U_T_RANGE    = (0.005, 1.0,  True)    # thrust weight
U_TAU_RANGE  = (20,  500.0, True)    # torque weights (τx, τy, τz)
# Angular velocity  Q_omega  [3]
Q_OMEGA_RANGE = (0.01, 2.0, True)
# Progress speed  Q_s
# NOTE: low=0.1 — DQ-MPCC works well with Q_s~0.3; values above ~3 cause stalls
Q_S_RANGE    = (0.1,  3.0,  True)

# ── Rotation-error weight (log-map of rotation) ─────────────────────────────
# Same concept in both baselines:
#   MPCC    → Q_q   (quaternion log error)
#   DQ-MPCC → Q_phi (dual-quaternion rotation log error)
Q_ROT_RANGE  = (0.1,  20.0,  True)


