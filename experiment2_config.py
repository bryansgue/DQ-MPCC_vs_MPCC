"""
experiment2_config.py — Configuration for Experiment 2: Velocity Sweep.

Imports shared parameters from experiment_config.py (single source of truth)
and only overrides what is specific to this experiment.

>>> Change shared params in experiment_config.py — they propagate here. <<<
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Shared parameters from experiment_config (single source of truth)
# ═════════════════════════════════════════════════════════════════════════════

from experiment_config import (
    S_MAX_MANUAL,        # arc-length limit [m]
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
VELOCITIES = [4, 6, 8, 10, 12, 15, 16, 18]

# Number of Monte Carlo runs per speed per controller
# ── Quick test: 2 runs.  Final experiment: set to 50 ──
N_RUNS = 10


# ═════════════════════════════════════════════════════════════════════════════
#  Initial condition perturbations (applied to nominal IC from experiment_config)
# ═════════════════════════════════════════════════════════════════════════════

# Position perturbation std [m] per axis — delta_p ~ N(0, sigma_p^2)
SIGMA_P = 0.05

# Orientation perturbation max angle [rad] — ||Log(delta_q)|| < sigma_q
SIGMA_Q = 0.05

# Random seed for reproducibility
SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
#  Timing override  (Experiment 2 only)
#
#  At high speeds the drone finishes 80 m much faster, so we use a shorter
#  safety budget than production (60 s).  The slowest speed (4 m/s) needs
#  ≈ 80/4 = 20 s theoretical, so 30 s gives plenty of margin.
# ═════════════════════════════════════════════════════════════════════════════

T_FINAL = 30   # [s]  (production uses 60 s)
