#!/usr/bin/env python3
"""
precompile_paths.py — Shared path precompiler.

Builds the arc-length path, waypoints, and saves path_cache.npz for both
MPCC_baseline and DQ-MPCC_baseline.

Usage:
    python3 precompile_paths.py

After running this, each simulation script will pick up the fresh waypoints.
If the trajectory shape changed, set FORCE_REBUILD_OCP = True in each
simulation script so acados recompiles with the new bsplines.
"""

import os
import sys
import time
import numpy as np

# ── Project root setup ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from experiment_config import (
    T_FINAL, TRAJECTORY_T_FINAL, FREC, N_WAYPOINTS, S_MAX_MANUAL,
    VTHETA_MAX, T_PREDICTION,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    TRAJ_VALUE, trayectoria,
)
from config.path_reference import build_cached_path_reference


def build_path(module_dir: str, verbose: bool = True):
    """Build arc-length path and save cache for a given module directory."""
    mpcc_utils_dir = os.path.join(PROJECT_ROOT, "MPCC_baseline")
    if mpcc_utils_dir not in sys.path:
        sys.path.insert(0, mpcc_utils_dir)

    from utils.numpy_utils import (
        build_arc_length_parameterisation,
        build_terminally_extended_path,
        build_waypoints,
        euler_to_quaternion,
    )

    cache_file = os.path.join(module_dir, "path_cache.npz")

    # Force rebuild by deleting existing cache
    if os.path.isfile(cache_file):
        os.remove(cache_file)

    return build_cached_path_reference(
        cache_file=cache_file,
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


def main():
    print("=" * 70)
    print("  PRECOMPILE PATHS")
    print("=" * 70)
    print(f"  TRAJECTORY_T_FINAL = {TRAJECTORY_T_FINAL} s")
    print(f"  T_FINAL            = {T_FINAL} s")
    print(f"  FREC               = {FREC} Hz")
    print(f"  N_WAYPOINTS        = {N_WAYPOINTS}")
    print(f"  S_MAX_MANUAL       = {S_MAX_MANUAL}")
    print(f"  VTHETA_MAX         = {VTHETA_MAX} m/s")
    print(f"  T_PREDICTION       = {T_PREDICTION} s")
    print("=" * 70)

    tic = time.time()

    # ── Build path for MPCC_baseline ─────────────────────────────────────
    mpcc_dir = os.path.join(PROJECT_ROOT, "MPCC_baseline")
    print("\n[1/2] Building path for MPCC_baseline...")
    (
        arc_lengths, pos_ref,
        position_by_arc_length, tangent_by_arc_length,
        s_wp, pos_wp, tang_wp, quat_wp,
        s_max_full, s_max, S_MAX_SOLVER, delta_s_terminal,
    ) = build_path(mpcc_dir)

    # ── Build path for DQ-MPCC_baseline ──────────────────────────────────
    dq_dir = os.path.join(PROJECT_ROOT, "DQ-MPCC_baseline")
    print("[2/2] Building path for DQ-MPCC_baseline...")
    build_path(dq_dir)

    elapsed = time.time() - tic

    print(f"\n{'=' * 70}")
    print(f"  DONE in {elapsed:.1f} s")
    print(f"{'=' * 70}")
    print(f"  s_max_full (geometric)  = {s_max_full:.3f} m")
    print(f"  s_max (active limit)    = {s_max:.3f} m")
    print(f"  S_MAX_SOLVER (extended) = {S_MAX_SOLVER:.3f} m")
    print(f"  delta_terminal          = {delta_s_terminal:.3f} m")
    print(f"{'=' * 70}")
    print()
    print("  Next steps:")
    print("    If trajectory SHAPE changed, set FORCE_REBUILD_OCP = True")
    print("    in each simulation script so acados recompiles the bsplines.")
    print()
    print("    cd MPCC_baseline    && python3 MPCC_baseline.py")
    print("    cd DQ-MPCC_baseline && python3 DQ_MPCC_baseline.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
