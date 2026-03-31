#!/usr/bin/env python3
"""Export a consolidated .mat summary of manual/refined tuning results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.tuning_registry import get_known_weight_sets, get_active_weight_summary  # noqa: E402


OUT_PATH = ROOT / "results" / "tuning" / "data" / "tuning_summary.mat"
DQ_LOCAL_JSON = ROOT / "DQ-MPCC_baseline" / "tuning" / "best_weights_local.json"
MPCC_LOCAL_JSON = ROOT / "MPCC_baseline" / "tuning" / "best_weights_local.json"


def _evaluate_subprocess(controller: str, weights: dict) -> dict[str, float]:
    module_dir = ROOT / ("DQ-MPCC_baseline" if controller == "dq" else "MPCC_baseline")
    module_name = "DQ_MPCC_simulation_tuner" if controller == "dq" else "MPCC_simulation_tuner"
    snippet = f"""
import json, sys
sys.path.insert(0, {str(module_dir)!r})
sys.path.insert(0, {str(ROOT)!r})
from {module_name} import run_simulation
weights = json.loads({json.dumps(json.dumps(weights))})
result = run_simulation(weights=weights, verbose=False)
payload = {{
    "rmse_contorno": float(result["rmse_contorno"]),
    "rmse_lag": float(result["rmse_lag"]),
    "path_completed": float(result["path_completed"]),
    "mean_vtheta": float(result["mean_vtheta"]),
    "mean_vpath_ratio": float(result["mean_vpath_ratio"]),
    "effective_speed": float(result["mean_vtheta"]) * float(result["mean_vpath_ratio"]),
    "solver_fail_ratio": float(result["solver_fail_ratio"]),
    "mean_solver_ms": float(result["mean_solver_ms"]),
}}
if "rmse_attitude" in result:
    payload["rmse_attitude"] = float(result["rmse_attitude"])
print(json.dumps(payload))
"""
    proc = subprocess.run(
        ["python3", "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": "/tmp/mpl"},
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    json_lines = [line for line in lines if line.startswith("{") and line.endswith("}")]
    return json.loads(json_lines[-1])


def _pack_weights(prefix: str, rotation_key: str, weights: dict, payload: dict):
    payload[f"{prefix}_q_ec"] = np.asarray(weights["Q_ec"], dtype=float)
    payload[f"{prefix}_q_el"] = np.asarray(weights["Q_el"], dtype=float)
    payload[f"{prefix}_{rotation_key.lower()}"] = np.asarray(weights[rotation_key], dtype=float)
    payload[f"{prefix}_u_mat"] = np.asarray(weights["U_mat"], dtype=float)
    payload[f"{prefix}_q_omega"] = np.asarray(weights["Q_omega"], dtype=float)
    payload[f"{prefix}_q_s"] = float(weights["Q_s"])


def _pack_metrics(prefix: str, metrics: dict, payload: dict):
    for key, value in metrics.items():
        payload[f"{prefix}_{key}"] = float(value)


def _pack_local_meta(prefix: str, json_path: Path, payload: dict):
    if not json_path.is_file():
        return
    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)

    if "best_J_multi" in data:
        payload[f"{prefix}_best_j_multi"] = float(data["best_J_multi"])
    if "best_trial" in data:
        payload[f"{prefix}_best_trial"] = int(data["best_trial"])

    objective_info = data.get("objective_info", {})
    if "strategy" in objective_info:
        payload[f"{prefix}_strategy"] = objective_info["strategy"]
    if "type" in objective_info:
        payload[f"{prefix}_objective_type"] = objective_info["type"]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    known = get_known_weight_sets()
    active_dq = get_active_weight_summary("dq")
    active_mpcc = get_active_weight_summary("mpcc")

    mat_dict: dict[str, object] = {
        "dq_active_label": active_dq["label"],
        "dq_active_source": active_dq["source"],
        "mpcc_active_label": active_mpcc["label"],
        "mpcc_active_source": active_mpcc["source"],
    }

    dq_rotation = known["dq"]["rotation_key"]
    mpcc_rotation = known["mpcc"]["rotation_key"]

    _pack_weights("dq_manual", dq_rotation, known["dq"]["manual"], mat_dict)
    _pack_metrics("dq_manual", _evaluate_subprocess("dq", known["dq"]["manual"]), mat_dict)
    if known["dq"].get("refined"):
        _pack_weights("dq_refined", dq_rotation, known["dq"]["refined"], mat_dict)
        _pack_metrics("dq_refined", _evaluate_subprocess("dq", known["dq"]["refined"]), mat_dict)
        _pack_local_meta("dq_refined", DQ_LOCAL_JSON, mat_dict)

    _pack_weights("mpcc_manual", mpcc_rotation, known["mpcc"]["manual"], mat_dict)
    _pack_metrics("mpcc_manual", _evaluate_subprocess("mpcc", known["mpcc"]["manual"]), mat_dict)
    if known["mpcc"].get("refined"):
        _pack_weights("mpcc_refined", mpcc_rotation, known["mpcc"]["refined"], mat_dict)
        _pack_metrics("mpcc_refined", _evaluate_subprocess("mpcc", known["mpcc"]["refined"]), mat_dict)
        _pack_local_meta("mpcc_refined", MPCC_LOCAL_JSON, mat_dict)

    savemat(OUT_PATH, mat_dict, do_compression=True)
    print(f"[OK] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
