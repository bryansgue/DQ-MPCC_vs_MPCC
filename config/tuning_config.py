"""
tuning_config.py — COMPATIBILITY SHIM.

Tuning parameters now live in config/experiment_config.py (primarily section 8).
This file re-exports them so existing imports keep working.

Prefer importing physical controller limits directly from `config.experiment_config`
instead of from this shim.

>>> Do NOT add new parameters here. Edit config/experiment_config.py instead. <<<
"""

# ── Re-export EVERYTHING from experiment_config ──────────────────────────────
from config.experiment_config import (
    # Legacy re-exports of controller limits (§4).
    # Prefer importing these directly from `config.experiment_config`.
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
    W_LAG,
    W_ATT,
    W_OMEGA,
    W_EFFORT,
    W_DU,
    W_SAT,
    W_PEAK,
    W_FAIL,
    W_VPATH,
    W_VPATH_HARD,
    VPATH_RATIO_MIN,
    W_SAT_HARD,
    SAT_RATIO_MAX,

    # Search-space bounds (§8)
    Q_EC_RANGE,
    Q_EL_RANGE,
    U_T_RANGE,
    U_TAU_RANGE,
    Q_OMEGA_RANGE,
    Q_S_RANGE,
    Q_ROT_RANGE,
)
