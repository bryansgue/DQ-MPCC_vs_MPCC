"""
tuning_config.py — COMPATIBILITY SHIM.

All tuning parameters now live in experiment_config.py (sections 4 & 8).
This file re-exports them so existing imports keep working.

>>> Do NOT add new parameters here. Edit experiment_config.py instead. <<<
"""

# ── Re-export EVERYTHING from experiment_config ──────────────────────────────
from experiment_config import (
    # Control limits (§4)
    G,
    T_MAX, T_MIN,
    TAUX_MAX, TAUY_MAX, TAUZ_MAX,
    VTHETA_MIN, VTHETA_MAX,

    # Tuning timing overrides (§8)
    TUNING_T_FINAL      as T_FINAL,
    TUNING_FREC         as FREC,
    TUNING_T_PREDICTION as T_PREDICTION,
    TUNING_N_WAYPOINTS  as N_WAYPOINTS,
    TUNING_S_MAX_MANUAL as S_MAX_MANUAL,

    # Multi-velocity tuning (§8)
    TUNING_VELOCITIES,

    # Optuna settings (§8)
    DEFAULT_N_TRIALS,
    DEFAULT_SAMPLER,
    N_STARTUP_TRIALS,
    OPTUNA_SEED,

    # Objective penalties (§8)
    W_INCOMPLETE,
    W_TIME,
    W_VEL,
    W_ISOTROPY,
    W_CONTOUR,

    # Search-space bounds (§8)
    Q_EC_RANGE,
    Q_EL_RANGE,
    U_T_RANGE,
    U_TAU_RANGE,
    Q_OMEGA_RANGE,
    Q_S_RANGE,
    Q_ROT_RANGE,
)


