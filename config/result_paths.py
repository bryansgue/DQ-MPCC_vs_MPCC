"""Canonical output directories for generated experiment artifacts.

This keeps raw data, figures, tables, reports and temporary compile files
separated, instead of mixing everything in a single folder per experiment.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_dirs(name: str) -> dict[str, Path]:
    base = _ensure(RESULTS_ROOT / name)
    return {
        "base": base,
        "data": _ensure(base / "data"),
        "figures": _ensure(base / "figures"),
        "tables": _ensure(base / "tables"),
        "reports": _ensure(base / "reports"),
        "compile": _ensure(base / "compile"),
    }


def latex_out_dirs(name: str) -> dict[str, Path]:
    base = _ensure(PROJECT_ROOT / "latex" / "out" / name)
    return {
        "base": base,
        "compile": _ensure(base / "compile"),
    }
