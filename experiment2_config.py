"""
experiment2_config.py — Configuration for Experiment 2: Velocity Sweep.

Defines the sweep velocities, number of Monte Carlo runs per speed,
and initial condition perturbation magnitudes.

Does NOT modify experiment_config.py (Experiment 1).
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Sweep parameters
# ═════════════════════════════════════════════════════════════════════════════

# Maximum virtual progress speeds to test [m/s]
# ── Quick test: 3 speeds.  Full sweep: add more as needed ──
VELOCITIES = [8, 12, 15]

# Number of Monte Carlo runs per speed per controller
# ── Quick test: 2 runs.  Final experiment: set to 50 ──
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

# ═════════════════════════════════════════════════════════════════════════════
#  Simulation settings (override experiment_config for this experiment)
# ═════════════════════════════════════════════════════════════════════════════

# Fixed arc-length for all runs [m]
S_MAX = 100.0

# Safety time budget [s]
T_FINAL = 30

# Control frequency [Hz]
FREC = 100

# Prediction horizon [s]
T_PREDICTION = 0.3

# CasADi interpolation waypoints
N_WAYPOINTS = 30
