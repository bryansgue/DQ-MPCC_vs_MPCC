"""path_loader.py — Load precomputed waypoints or regenerate on-the-fly.

Called by both MPCC_baseline_rates.py and mpcc_mujoco_node.py.

If path_cache.npz exists: loads immediately (fast).
If not: builds from scratch (slow first run), but does NOT save — run
precompute_path.py explicitly to cache the result.
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

_CACHE_FILE = os.path.join(_WORKSPACE_ROOT, "path_cache.npz")


def load_path(verbose: bool = True):
    """Return arc-length parameterisation and waypoints.

    Returns
    -------
    s_wp         : (N_wp,)       arc-length values at waypoints
    pos_wp       : (3, N_wp)     positions at waypoints
    tang_wp      : (3, N_wp)     unit tangents at waypoints
    quat_wp      : (4, N_wp)     desired quaternions at waypoints
    s_max_full   : float         total arc length available
    s_max_solver : float         arc length given to solver (s_max * 1.2)
    arc_lengths  : (N_pts,)      dense arc-length array (for pos_ref)
    pos_ref      : (3, N_pts)    dense position reference

    position_by_arc_length : callable  θ → pos (3,)
    tangent_by_arc_length  : callable  θ → tangent (3,)
    """
    from utils.numpy_utils import (
        build_arc_length_parameterisation,
        build_waypoints,
        euler_to_quaternion,
    )
    from MPCC_baseline_rates.config.experiment_config import (
        T_TRAJ_BUILD, N_WAYPOINTS, S_MAX_MANUAL, trayectoria,
    )

    if os.path.isfile(_CACHE_FILE):
        if verbose:
            print(f"[PATH]  Loading cached waypoints from {_CACHE_FILE}")
        data = np.load(_CACHE_FILE)
        s_wp         = data["s_wp"]
        pos_wp       = data["pos_wp"]
        tang_wp      = data["tang_wp"]
        quat_wp      = data["quat_wp"]
        s_max_full   = float(data["s_max_full"][0])
        s_max_solver = float(data["s_max_solver"][0])
        arc_lengths  = data["arc_lengths"]
        pos_ref      = data["pos_ref"]

        # Rebuild callable interpolators from the dense reference
        from scipy.interpolate import interp1d
        position_by_arc_length = interp1d(
            arc_lengths, pos_ref, axis=1,
            kind="linear", fill_value="extrapolate"
        )
        tangent_by_arc_length = _build_tangent_interp(arc_lengths, pos_ref)

        if verbose:
            print(f"[PATH]  s_max_full={s_max_full:.3f} m  |  "
                  f"s_max_solver={s_max_solver:.3f} m  |  "
                  f"S_MAX_MANUAL={S_MAX_MANUAL} m")
    else:
        if verbose:
            print(f"[PATH]  Cache not found — building arc-length parameterisation "
                  f"(run precompute_path.py to cache)...")

        xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
        N_pts = int(T_TRAJ_BUILD * 200)
        t_build = np.linspace(0.0, T_TRAJ_BUILD, N_pts)

        arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, s_max_full = \
            build_arc_length_parameterisation(xd, yd, zd, xd_p, yd_p, zd_p, t_build)

        if S_MAX_MANUAL is not None and s_max_full < S_MAX_MANUAL:
            print(f"[PATH]  WARNING: s_max_full={s_max_full:.1f} < S_MAX_MANUAL={S_MAX_MANUAL}. "
                  f"Increase T_TRAJ_BUILD in config.")

        s_max_target = S_MAX_MANUAL if S_MAX_MANUAL is not None else s_max_full
        s_max_solver = min(s_max_target * 1.2, s_max_full)

        s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
            s_max_solver, N_WAYPOINTS, position_by_arc_length, tangent_by_arc_length,
            euler_to_quat_fn=euler_to_quaternion,
        )
        if verbose:
            print(f"[PATH]  s_max_full={s_max_full:.3f} m  |  "
                  f"s_max_solver={s_max_solver:.3f} m  |  "
                  f"S_MAX_MANUAL={S_MAX_MANUAL} m")

    s_max = S_MAX_MANUAL if (S_MAX_MANUAL is not None and S_MAX_MANUAL < s_max_full) \
            else s_max_full

    return (
        s_wp, pos_wp, tang_wp, quat_wp,
        s_max_full, s_max_solver, s_max,
        arc_lengths, pos_ref,
        position_by_arc_length, tangent_by_arc_length,
    )


def _build_tangent_interp(arc_lengths, pos_ref):
    """Build a tangent interpolator from dense pos_ref."""
    from scipy.interpolate import interp1d
    tangents = np.gradient(pos_ref, arc_lengths, axis=1)
    norms = np.linalg.norm(tangents, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    tangents /= norms
    return interp1d(arc_lengths, tangents, axis=1, kind="linear", fill_value="extrapolate")
