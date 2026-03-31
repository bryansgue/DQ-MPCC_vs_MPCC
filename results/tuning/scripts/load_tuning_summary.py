#!/usr/bin/env python3
"""Load and print the consolidated tuning_summary.mat payload."""

from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[3]
MAT_PATH = ROOT / "results" / "tuning" / "data" / "tuning_summary.mat"


def _scalar(d, key, default=None):
    val = d.get(key, default)
    if hasattr(val, "flat"):
        return val.flat[0]
    return val


def main():
    if not MAT_PATH.is_file():
        raise SystemExit(f"Missing {MAT_PATH}. Run results/tuning/scripts/export_tuning_summary_mat.py first.")

    d = loadmat(MAT_PATH, squeeze_me=True)
    print(f"Loaded {MAT_PATH}")
    print(f"DQ active   : {_scalar(d, 'dq_active_label')} ({_scalar(d, 'dq_active_source')})")
    print(f"MPCC active : {_scalar(d, 'mpcc_active_label')} ({_scalar(d, 'mpcc_active_source')})")
    for prefix in ("dq_manual", "dq_refined", "mpcc_manual", "mpcc_refined"):
        if f"{prefix}_rmse_contorno" not in d:
            continue
        print(
            f"{prefix}: "
            f"rmse_c={float(_scalar(d, f'{prefix}_rmse_contorno')):.4f}, "
            f"rmse_l={float(_scalar(d, f'{prefix}_rmse_lag')):.4f}, "
            f"path={float(_scalar(d, f'{prefix}_path_completed')):.4f}, "
            f"veff={float(_scalar(d, f'{prefix}_effective_speed')):.4f}"
        )


if __name__ == "__main__":
    main()
