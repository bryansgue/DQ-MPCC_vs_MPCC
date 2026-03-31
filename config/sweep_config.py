"""
sweep_config.py — Configuration for Experiment 2: Velocity Sweep.

Imports shared parameters from config.experiment_config (single source of truth)
and only overrides what is specific to this experiment.

>>> Change shared params in config/experiment_config.py — they propagate here. <<<
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Shared parameters from experiment_config (single source of truth)
# ═════════════════════════════════════════════════════════════════════════════

from config.experiment_config import (
    S_MAX_MANUAL,        # arc-length limit [m]
    T_FINAL,             # simulation time budget [s]
    FREC,                # control frequency [Hz]
    T_PREDICTION,        # MPC prediction horizon [s]
    N_WAYPOINTS,         # CasADi interpolation waypoints
    VTHETA_MAX,          # nominal progress velocity [m/s]
)

# Re-export S_MAX with the name consumers expect
S_MAX = S_MAX_MANUAL if S_MAX_MANUAL is not None else 80


# ═════════════════════════════════════════════════════════════════════════════
#  Sweep parameters  (specific to Experiment 2)
# ═════════════════════════════════════════════════════════════════════════════

# Maximum virtual progress speeds to test [m/s]
VELOCITIES = [4, 8, 12, 16, 20, 24]

# Number of Monte Carlo runs per speed per controller
# ── Final sweep setting for the current Experiment 2 report ──
N_RUNS = 5


# ═════════════════════════════════════════════════════════════════════════════
#  Initial condition perturbations (applied to nominal IC from experiment_config)
# ═════════════════════════════════════════════════════════════════════════════

# Position perturbation std [m] per axis — delta_p ~ N(0, sigma_p^2)
SIGMA_P = 0.05

# Orientation perturbation max angle [rad] — ||Log(delta_q)|| < sigma_q
SIGMA_Q = 0.05

# Random seed for reproducibility
SEED = 42


# Timing is now inherited directly from experiment_config.py so the sweep uses
# the same trajectory-length budget and stopping horizon as the main baselines.
