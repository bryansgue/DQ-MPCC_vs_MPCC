"""Helpers for tracking manual/refined controller weights and exporting them.

This module centralises the notion of:
  - known weight variants per controller (manual / refined)
  - active weight labelling (manual / refined / custom)
  - compact metadata payloads for .mat files and result scripts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json_weights(path: Path, key: str = "weights") -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload.get(key)


def _normalise_weights(weights: dict[str, Any], rotation_key: str) -> dict[str, Any]:
    return {
        "Q_ec": [float(v) for v in weights["Q_ec"]],
        "Q_el": [float(v) for v in weights["Q_el"]],
        rotation_key: [float(v) for v in weights[rotation_key]],
        "U_mat": [float(v) for v in weights["U_mat"]],
        "Q_omega": [float(v) for v in weights["Q_omega"]],
        "Q_s": float(weights["Q_s"]),
    }


def _weights_close(a: dict[str, Any], b: dict[str, Any], rotation_key: str) -> bool:
    keys = ["Q_ec", "Q_el", rotation_key, "U_mat", "Q_omega"]
    for key in keys:
        if not np.allclose(np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float),
                           rtol=1e-9, atol=1e-9):
            return False
    return abs(float(a["Q_s"]) - float(b["Q_s"])) <= 1e-9


def get_known_weight_sets() -> dict[str, dict[str, Any]]:
    from config.experiment_config import (
        DQ_MANUAL_WEIGHTS,
        MPCC_MANUAL_WEIGHTS,
        MPCC_MANUAL_WEIGHTS_PREV,
    )

    dq_refined = (
        _load_json_weights(PROJECT_ROOT / "DQ-MPCC_baseline" / "tuning" / "best_weights_local.json")
        or _load_json_weights(PROJECT_ROOT / "DQ-MPCC_baseline" / "tuning" / "final_refined_weights.json")
    )
    mpcc_refined = (
        _load_json_weights(PROJECT_ROOT / "MPCC_baseline" / "tuning" / "best_weights_local.json")
        or _load_json_weights(PROJECT_ROOT / "MPCC_baseline" / "tuning" / "final_refined_weights.json")
    )

    return {
        "dq": {
            "rotation_key": "Q_phi",
            "manual": _normalise_weights(DQ_MANUAL_WEIGHTS, "Q_phi"),
            "refined": _normalise_weights(dq_refined, "Q_phi") if dq_refined else None,
        },
        "mpcc": {
            "rotation_key": "Q_q",
            "manual": _normalise_weights(MPCC_MANUAL_WEIGHTS_PREV, "Q_q"),
            "active_manual": _normalise_weights(MPCC_MANUAL_WEIGHTS, "Q_q"),
            "refined": _normalise_weights(mpcc_refined, "Q_q") if mpcc_refined else None,
        },
    }


def get_active_weight_summary(controller: str) -> dict[str, Any]:
    from config.experiment_config import (
        USE_TUNED_WEIGHTS_DQ,
        DQ_WEIGHTS,
        DQ_WEIGHTS_PATH,
        USE_TUNED_WEIGHTS_MPCC,
        MPCC_WEIGHTS,
        MPCC_WEIGHTS_PATH,
    )

    known = get_known_weight_sets()[controller]
    rotation_key = known["rotation_key"]

    active_path = None
    if controller == "dq":
        active = _normalise_weights(DQ_WEIGHTS, rotation_key)
        tuned = bool(USE_TUNED_WEIGHTS_DQ and DQ_WEIGHTS_PATH.is_file())
        active_path = DQ_WEIGHTS_PATH if tuned else None
    else:
        active = _normalise_weights(MPCC_WEIGHTS, rotation_key)
        tuned = bool(USE_TUNED_WEIGHTS_MPCC and MPCC_WEIGHTS_PATH.is_file())
        active_path = MPCC_WEIGHTS_PATH if tuned else None

    label = "custom"
    if known.get("refined") and _weights_close(active, known["refined"], rotation_key):
        label = "refined"
    elif _weights_close(active, known["manual"], rotation_key):
        label = "manual"
    elif controller == "mpcc" and "active_manual" in known and _weights_close(
        active, known["active_manual"], rotation_key
    ):
        label = "manual_active"
    elif tuned and active_path is not None:
        name = active_path.name
        if name == "best_weights.json":
            label = "best_global"
        elif name == "best_weights_local.json":
            label = "best_local"
        elif name == "final_refined_weights.json":
            label = "final_refined"
        elif name == "final_refined_oriented_weights.json":
            label = "final_refined_oriented"
        elif name == "final_refined_relaxed_weights.json":
            label = "final_refined_relaxed"

    return {
        "controller": controller,
        "rotation_key": rotation_key,
        "weights": active,
        "label": label,
        "source": "tuned_json" if tuned else "experiment_config",
        "path": str(active_path) if active_path is not None else "",
    }


def flatten_weight_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    rotation_key = summary["rotation_key"]
    weights = summary["weights"]
    return {
        f"{prefix}_gain_label": summary["label"],
        f"{prefix}_gain_source": summary["source"],
        f"{prefix}_gain_path": summary.get("path", ""),
        f"{prefix}_q_ec": np.asarray(weights["Q_ec"], dtype=float),
        f"{prefix}_q_el": np.asarray(weights["Q_el"], dtype=float),
        f"{prefix}_{rotation_key.lower()}": np.asarray(weights[rotation_key], dtype=float),
        f"{prefix}_u_mat": np.asarray(weights["U_mat"], dtype=float),
        f"{prefix}_q_omega": np.asarray(weights["Q_omega"], dtype=float),
        f"{prefix}_q_s": float(weights["Q_s"]),
    }
