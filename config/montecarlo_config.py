"""
montecarlo_config.py — Configuration for Experiment 3: Monte Carlo with Random Poses.

Fixed velocity, 600 runs with random initial poses (position + orientation).
Evaluates robustness: convergence rate, tracking errors, solver timing, control effort.

>>> Change shared params in config/experiment_config.py — they propagate here. <<<
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Shared parameters from experiment_config (single source of truth)
# ═════════════════════════════════════════════════════════════════════════════

from config.experiment_config import (
    S_MAX_MANUAL,        # arc-length limit [m]
    FREC,                # control frequency [Hz]
    T_PREDICTION,        # MPC prediction horizon [s]
    N_WAYPOINTS,         # CasADi interpolation waypoints
    VTHETA_MAX,          # nominal progress velocity upper bound [m/s]
    P0, Q0,              # nominal initial conditions
)

# Re-export S_MAX with the name consumers expect.
# `None` means: let the runner use the full trajectory length computed at runtime.
S_MAX = S_MAX_MANUAL


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment 3 parameters
# ═════════════════════════════════════════════════════════════════════════════

# ── Velocity ─────────────────────────────────────────────────────────────
#    Fixed moderately-fast velocity [m/s].  Change this freely.
VELOCITY = 12

# ── Monte Carlo ──────────────────────────────────────────────────────────
N_RUNS = 10             # total number of random-pose runs per controller

# ── Initial-pose perturbations ───────────────────────────────────────────
#    Position:    p = p0 + δ,     δ ~ N(0, σ_p²·I₃)            [m]
#    Orientation: yaw-only perturbation around hover / nominal attitude.
#    Roll and pitch are kept at the nominal hover attitude; only heading varies.
SIGMA_P = 0.60          # position std per axis [m] for moderate Monte Carlo spread
SIGMA_Q = 0.18          # max yaw perturbation [rad] ≈ 10.3°
MAX_POS_OFFSET_NORM = 1.20  # [m] hard cap on ||p0_pert - P0|| to avoid extreme outliers

# ── Random seed ──────────────────────────────────────────────────────────
SEED = 2026

# ── Timing override ─────────────────────────────────────────────────────
#    With v=10 m/s and s_max=80 m the drone needs ~8 s.
#    Budget extra time for convergence from far-away initial poses.
T_FINAL = 40            # [s]

# ── Completion / divergence thresholds ───────────────────────────────────
COMPLETION_RATIO         = 0.95     # θ must reach ≥ 95% of s_max
POS_DIVERGENCE_THRESHOLD = 50.0     # [m] abort if pos error exceeds this
