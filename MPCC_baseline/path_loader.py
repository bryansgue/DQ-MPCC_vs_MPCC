"""Compatibility wrapper around the shared cached path reference builder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiment_config import (
    TRAJECTORY_T_FINAL,
    T_FINAL,
    FREC,
    N_WAYPOINTS,
    S_MAX_MANUAL,
    VTHETA_MAX,
    T_PREDICTION,
    ATTITUDE_REF_SPEED,
    ATTITUDE_REF_MAX_TILT_DEG,
    TRAJ_VALUE,
    trayectoria,
)
from config.path_reference import build_cached_path_reference
from utils.numpy_utils import (
    build_arc_length_parameterisation,
    build_waypoints,
    build_terminally_extended_path,
    euler_to_quaternion,
)

_CACHE_FILE = _THIS_DIR / "path_cache.npz"

def load_or_build_path_cache(verbose: bool = True):
    return build_cached_path_reference(
        cache_file=_CACHE_FILE,
        trajectory_t_final=TRAJECTORY_T_FINAL,
        t_final=T_FINAL,
        frec=FREC,
        n_waypoints=N_WAYPOINTS,
        s_max_manual=S_MAX_MANUAL,
        vtheta_max=VTHETA_MAX,
        t_prediction=T_PREDICTION,
        attitude_ref_speed=ATTITUDE_REF_SPEED,
        attitude_ref_max_tilt_deg=ATTITUDE_REF_MAX_TILT_DEG,
        traj_value=TRAJ_VALUE,
        trayectoria_fn=trayectoria,
        build_arc_length_parameterisation=build_arc_length_parameterisation,
        build_terminally_extended_path=build_terminally_extended_path,
        build_waypoints=build_waypoints,
        euler_to_quaternion=euler_to_quaternion,
        verbose=verbose,
    )
