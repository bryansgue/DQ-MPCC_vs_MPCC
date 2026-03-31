"""Shared cached path/gamma construction for arc-length-based controllers.

This module centralises the geometric path preparation used by the one-run
baselines and can later be reused by experiment runners. The idea is to build
the arc-length reference once, sample it with enough waypoints, cache it, and
reload it when the configuration has not changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


def _build_tangent_interp(arc_lengths: np.ndarray, pos_ref: np.ndarray):
    tangents = np.gradient(pos_ref, arc_lengths, axis=1)
    norms = np.linalg.norm(tangents, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    tangents /= norms
    return interp1d(arc_lengths, tangents, axis=1, kind="linear", fill_value="extrapolate")


def build_cached_path_reference(
    *,
    cache_file: str | Path,
    trajectory_t_final: float,
    t_final: float,
    frec: float,
    n_waypoints: int,
    s_max_manual: float | None,
    vtheta_max: float,
    t_prediction: float,
    attitude_ref_speed: float,
    attitude_ref_max_tilt_deg: float,
    traj_value: float,
    trayectoria_fn: Callable,
    build_arc_length_parameterisation: Callable,
    build_terminally_extended_path: Callable,
    build_waypoints: Callable,
    euler_to_quaternion: Callable,
    verbose: bool = True,
):
    cache_file = Path(cache_file)

    def meta_dict() -> dict:
        return {
            "trajectory_t_final": float(trajectory_t_final),
            "t_final": float(t_final),
            "frec": float(frec),
            "n_waypoints": int(n_waypoints),
            "s_max_manual": None if s_max_manual is None else float(s_max_manual),
            "vtheta_max": float(vtheta_max),
            "t_prediction": float(t_prediction),
            "att_ref_speed": float(attitude_ref_speed),
            "att_ref_max_tilt_deg": float(attitude_ref_max_tilt_deg),
            "traj_value": float(traj_value),
        }

    def meta_matches(meta_json: str | np.ndarray) -> bool:
        if isinstance(meta_json, np.ndarray):
            meta_json = str(meta_json.item())
        try:
            cached = json.loads(meta_json)
        except Exception:
            return False
        return cached == meta_dict()

    def build_now():
        t_s = 1.0 / frec
        t = np.arange(0, t_final + t_s, t_s)
        t_path = np.linspace(0.0, trajectory_t_final, len(t))
        xd, yd, zd, xd_p, yd_p, zd_p = trayectoria_fn(t_path)
        t_finer = np.linspace(0.0, trajectory_t_final, len(t))

        arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, s_max_full = \
            build_arc_length_parameterisation(xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

        if s_max_manual is not None and s_max_manual < s_max_full:
            s_max = float(s_max_manual)
        else:
            s_max = float(s_max_full)

        delta_s_terminal = 1.10 * vtheta_max * t_prediction
        s_max_solver = float(s_max + delta_s_terminal)

        pos_by_arc_solver, tang_by_arc_solver = build_terminally_extended_path(
            position_by_arc_length,
            tangent_by_arc_length,
            s_max,
            s_max_solver,
            s_original_end=s_max_full,
        )

        s_wp, pos_wp, tang_wp, quat_wp = build_waypoints(
            s_max_solver,
            n_waypoints,
            pos_by_arc_solver,
            tang_by_arc_solver,
            euler_to_quat_fn=euler_to_quaternion,
            reference_speed=attitude_ref_speed,
            max_tilt_deg=attitude_ref_max_tilt_deg,
        )

        np.savez(
            cache_file,
            meta_json=json.dumps(meta_dict(), sort_keys=True),
            arc_lengths=arc_lengths,
            pos_ref=pos_ref,
            s_wp=s_wp,
            pos_wp=pos_wp,
            tang_wp=tang_wp,
            quat_wp=quat_wp,
            s_max=np.array([s_max]),
            s_max_full=np.array([s_max_full]),
            s_max_solver=np.array([s_max_solver]),
        )
        if verbose:
            print(f"[PATH] Built and cached path reference -> {cache_file}")

        return (
            arc_lengths,
            pos_ref,
            position_by_arc_length,
            tangent_by_arc_length,
            s_wp,
            pos_wp,
            tang_wp,
            quat_wp,
            s_max_full,
            s_max,
            s_max_solver,
            delta_s_terminal,
        )

    if cache_file.is_file():
        data = np.load(cache_file, allow_pickle=True)
        if meta_matches(data["meta_json"]):
            arc_lengths = data["arc_lengths"]
            pos_ref = data["pos_ref"]
            position_by_arc_length = interp1d(
                arc_lengths, pos_ref, axis=1, kind="linear", fill_value="extrapolate"
            )
            tangent_by_arc_length = _build_tangent_interp(arc_lengths, pos_ref)
            s_max_full = float(data["s_max_full"][0])
            s_max = float(data["s_max"][0])
            s_max_solver = float(data["s_max_solver"][0])
            delta_s_terminal = s_max_solver - s_max
            if verbose:
                print(f"[PATH] Loaded cached path reference from {cache_file}")
            return (
                arc_lengths,
                pos_ref,
                position_by_arc_length,
                tangent_by_arc_length,
                data["s_wp"],
                data["pos_wp"],
                data["tang_wp"],
                data["quat_wp"],
                s_max_full,
                s_max,
                s_max_solver,
                delta_s_terminal,
            )
        if verbose:
            print("[PATH] Cache metadata changed - rebuilding path reference")
    return build_now()
