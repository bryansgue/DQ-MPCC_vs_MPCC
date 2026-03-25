"""precompute_path.py — Precompute arc-length parameterisation and save to disk.

Run this ONCE whenever the trajectory or S_MAX_MANUAL changes.
Both mpcc_mujoco_node.py and MPCC_baseline_rates.py will load the cached
waypoints instead of recomputing them on every run.

Usage
-----
    cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline
    python3 MPCC_baseline_rates/precompute_path.py
"""

import os
import sys
import numpy as np

_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT   = os.path.dirname(_WORKSPACE_ROOT)
_SHARED_MPCC_ROOT = os.path.join(_PROJECT_ROOT, "MPCC_baseline")
for _path in (_PROJECT_ROOT, _SHARED_MPCC_ROOT, _WORKSPACE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.numpy_utils import (
    build_arc_length_parameterisation,
    build_waypoints,
    euler_to_quaternion,
)
from MPCC_baseline_rates.config.experiment_config import (
    T_TRAJ_BUILD, N_WAYPOINTS, S_MAX_MANUAL, trayectoria,
)

# ── Output path (next to this script) ────────────────────────────────────────
_CACHE_FILE = os.path.join(_WORKSPACE_ROOT, "path_cache.npz")


def build_and_save():
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()

    # Dense time vector for accurate arc-length integration
    N_pts = int(T_TRAJ_BUILD * 200)          # 200 pts/s → very smooth
    t_build = np.linspace(0.0, T_TRAJ_BUILD, N_pts)

    print(f"[PRECOMPUTE]  Building arc-length parameterisation "
          f"over T_TRAJ_BUILD={T_TRAJ_BUILD} s  ({N_pts} points)...")

    arc_lengths, pos_ref, position_by_arc, tangent_by_arc, s_max_full = \
        build_arc_length_parameterisation(xd, yd, zd, xd_p, yd_p, zd_p, t_build)

    print(f"[PRECOMPUTE]  Total arc length = {s_max_full:.3f} m")
    if S_MAX_MANUAL is not None and s_max_full < S_MAX_MANUAL:
        print(f"[PRECOMPUTE]  WARNING: s_max_full={s_max_full:.1f} < S_MAX_MANUAL={S_MAX_MANUAL}. "
              f"Increase T_TRAJ_BUILD in config.")

    # Waypoints up to S_MAX_MANUAL * 1.2 (solver needs a bit beyond the finish line)
    s_max_target = S_MAX_MANUAL if S_MAX_MANUAL is not None else s_max_full
    s_max_solver = min(s_max_target * 1.2, s_max_full)

    s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
        s_max_solver, N_WAYPOINTS, position_by_arc, tangent_by_arc,
        euler_to_quat_fn=euler_to_quaternion,
    )

    np.savez(
        _CACHE_FILE,
        s_wp=s_wp,
        pos_wp=pos_wp,
        tang_wp=tang_wp,
        quat_wp=quat_wp,
        s_max_full=np.array([s_max_full]),
        s_max_solver=np.array([s_max_solver]),
        arc_lengths=arc_lengths,
        pos_ref=pos_ref,
    )
    print(f"[PRECOMPUTE]  Saved {N_WAYPOINTS} waypoints → {_CACHE_FILE}")
    print(f"[PRECOMPUTE]  s_max_solver = {s_max_solver:.3f} m  |  S_MAX_MANUAL = {s_max_target:.1f} m")


if __name__ == "__main__":
    build_and_save()
