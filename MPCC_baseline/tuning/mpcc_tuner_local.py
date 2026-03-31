"""
mpcc_tuner_local.py  –  Local MPCC bilevel refinement around the current manual gains.

This tuner is intentionally local, but no longer freezes effort blocks:
  • it starts from MPCC_MANUAL_WEIGHTS
  • it refines Q_ec, Q_el, Q_q, U_mat, Q_omega, Q_s
  • the search stays in a local neighbourhood of the current manual solution

The upper-level objective is paper-oriented but intentionally simple:
  • manual tuning defines the operating basin first
  • bilevel performs only local refinement around that basin
  • metrics are normalised with respect to the current manual reference
  • effective path-following speed matters more than pure theta progress
  • only contouring, lag, effective speed, completion, and solver failure
    appear in the outer objective
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (PROJECT_ROOT, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from MPCC_simulation_tuner import run_simulation
from tuning_config import (
    W_INCOMPLETE,
    W_FAIL,
    DEFAULT_N_TRIALS,
    N_STARTUP_TRIALS,
    OPTUNA_SEED,
    TUNING_VELOCITIES,
)
from experiment_config import MPCC_MANUAL_WEIGHTS

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


W_CONTOUR_LOCAL = 1.5
W_LAG_LOCAL = 0.35
W_EFF_SPEED = 0.80

_REFERENCE_BY_VELOCITY = None


def _bounded_scale(value: float, low_scale: float, high_scale: float,
                   min_value: float = 1e-4) -> tuple[float, float]:
    """Return a positive local search interval around one scalar weight."""
    low = max(min_value, value * low_scale)
    high = max(low * 1.0001, value * high_scale)
    return low, high


def trial_to_weights(
    trial,
    gain_low_scale: float,
    gain_high_scale: float,
    qomega_low_scale: float,
    qomega_high_scale: float,
    umat_low_scale: float,
    umat_high_scale: float,
    qs_low_scale: float,
    qs_high_scale: float,
) -> dict:
    """Suggest local perturbations around the current manual gains."""
    base = MPCC_MANUAL_WEIGHTS

    q_ec = []
    for i, val in enumerate(base["Q_ec"]):
        low, high = _bounded_scale(float(val), gain_low_scale, gain_high_scale)
        q_ec.append(trial.suggest_float(f"Q_ec{i}", low, high, log=True))

    q_el = []
    for i, val in enumerate(base["Q_el"]):
        low, high = _bounded_scale(float(val), gain_low_scale, gain_high_scale)
        q_el.append(trial.suggest_float(f"Q_el{i}", low, high, log=True))

    q_q = []
    for i, val in enumerate(base["Q_q"]):
        low, high = _bounded_scale(float(val), gain_low_scale, gain_high_scale)
        q_q.append(trial.suggest_float(f"Q_q{i}", low, high, log=True))

    u_mat = []
    for i, val in enumerate(base["U_mat"]):
        low, high = _bounded_scale(float(val), umat_low_scale, umat_high_scale)
        u_mat.append(trial.suggest_float(f"U_mat{i}", low, high, log=True))

    q_omega = []
    for i, val in enumerate(base["Q_omega"]):
        low, high = _bounded_scale(float(val), qomega_low_scale, qomega_high_scale)
        q_omega.append(trial.suggest_float(f"Q_omega{i}", low, high, log=True))

    q_s_low, q_s_high = _bounded_scale(float(base["Q_s"]), qs_low_scale, qs_high_scale)
    q_s = trial.suggest_float("Q_s", q_s_low, q_s_high, log=True)

    return {
        "Q_ec": q_ec,
        "Q_el": q_el,
        "Q_q": q_q,
        "U_mat": u_mat,
        "Q_omega": q_omega,
        "Q_s": q_s,
    }


def _evaluate_single_velocity(weights: dict, v_max: float) -> tuple[float, dict]:
    """Run one simulation and compute the local paper-oriented objective."""
    try:
        result = run_simulation(weights=weights, verbose=False, vtheta_max=v_max)
    except Exception as e:
        return 1e6, {"error": str(e)}

    ref = _REFERENCE_BY_VELOCITY[v_max]
    compl = float(result["path_completed"])
    rmse_c = float(result.get("rmse_contorno", np.inf))
    rmse_l = float(result.get("rmse_lag", np.inf))
    mean_vt = float(result.get("mean_vtheta", 0.0))
    mean_ratio = float(result.get("mean_vpath_ratio", 0.0))
    success = bool(result.get("success", True))
    fail_ratio = float(result.get("solver_fail_ratio", 0.0))

    if not np.isfinite(rmse_c) or not np.isfinite(rmse_l):
        return 1e6, result

    eff_speed = mean_vt * mean_ratio
    ref_eff_speed = ref["effective_speed"]

    J = (
        W_CONTOUR_LOCAL * (rmse_c / ref["rmse_contorno"])
        + W_LAG_LOCAL * (rmse_l / ref["rmse_lag"])
        - W_EFF_SPEED * (eff_speed / ref_eff_speed)
    )

    if compl < 0.99:
        J += W_INCOMPLETE * (1.0 - compl)
    if not success:
        J += W_FAIL
    J += W_FAIL * fail_ratio

    result["_effective_speed"] = eff_speed
    result["_J"] = J
    return J, result


def _build_reference_metrics():
    """Evaluate the current manual set once and use it as the local reference."""
    reference = {}
    print("[REF] Evaluating current MPCC_MANUAL_WEIGHTS across tuning velocities ...")
    for v_max in TUNING_VELOCITIES:
        result = run_simulation(
            weights=MPCC_MANUAL_WEIGHTS,
            verbose=False,
            vtheta_max=v_max,
        )
        reference[v_max] = {
            "rmse_contorno": max(float(result.get("rmse_contorno", np.inf)), 1e-6),
            "rmse_lag": max(float(result.get("rmse_lag", np.inf)), 1e-6),
            "mean_vtheta": float(result.get("mean_vtheta", 0.0)),
            "mean_vpath_ratio": max(float(result.get("mean_vpath_ratio", 0.0)), 1e-6),
            "effective_speed": max(
                float(result.get("mean_vtheta", 0.0))
                * float(result.get("mean_vpath_ratio", 0.0)),
                1e-6,
            ),
        }
        print(
            f"  [REF v={v_max}] rmse_c={reference[v_max]['rmse_contorno']:.3f}  "
            f"rmse_l={reference[v_max]['rmse_lag']:.3f}  "
            f"v̄θ={reference[v_max]['mean_vtheta']:.2f}  "
            f"ratio={reference[v_max]['mean_vpath_ratio']:.3f}  "
            f"v_eff={reference[v_max]['effective_speed']:.2f}",
            flush=True,
        )
    return reference


def _improvement_percent(old_value: float, new_value: float, lower_is_better: bool) -> float:
    """Return percentage improvement from old to new."""
    if abs(old_value) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (old_value - new_value) / old_value
    return 100.0 * (new_value - old_value) / old_value


def _evaluate_summary(weights: dict) -> dict:
    """Run one production-aligned headless evaluation and extract compact metrics."""
    result = run_simulation(weights=weights, verbose=False)
    return {
        "rmse_contorno": float(result["rmse_contorno"]),
        "rmse_lag": float(result["rmse_lag"]),
        "rmse_attitude": float(result["rmse_attitude"]),
        "mean_vtheta": float(result["mean_vtheta"]),
        "mean_vpath_ratio": float(result["mean_vpath_ratio"]),
        "effective_speed": float(result["mean_vtheta"]) * float(result["mean_vpath_ratio"]),
        "path_completed": float(result["path_completed"]),
        "solver_fail_ratio": float(result["solver_fail_ratio"]),
    }


def _build_comparison(best_weights: dict) -> dict:
    """Compare refined weights against the current manual reference."""
    manual = _evaluate_summary(MPCC_MANUAL_WEIGHTS)
    refined = _evaluate_summary(best_weights)
    return {
        "manual": manual,
        "refined": refined,
        "improvement_percent": {
            "rmse_contorno": _improvement_percent(
                manual["rmse_contorno"], refined["rmse_contorno"], lower_is_better=True
            ),
            "rmse_lag": _improvement_percent(
                manual["rmse_lag"], refined["rmse_lag"], lower_is_better=True
            ),
            "rmse_attitude": _improvement_percent(
                manual["rmse_attitude"], refined["rmse_attitude"], lower_is_better=True
            ),
            "effective_speed": _improvement_percent(
                manual["effective_speed"], refined["effective_speed"], lower_is_better=False
            ),
            "path_completed": _improvement_percent(
                manual["path_completed"], refined["path_completed"], lower_is_better=False
            ),
        },
    }


def make_objective(
    gain_low_scale: float,
    gain_high_scale: float,
    qomega_low_scale: float,
    qomega_high_scale: float,
    umat_low_scale: float,
    umat_high_scale: float,
    qs_low_scale: float,
    qs_high_scale: float,
):
    """Return the Optuna objective function with fixed local-search settings."""
    def objective(trial) -> float:
        weights = trial_to_weights(
            trial,
            gain_low_scale,
            gain_high_scale,
            qomega_low_scale,
            qomega_high_scale,
            umat_low_scale,
            umat_high_scale,
            qs_low_scale,
            qs_high_scale,
        )

        t0_total = time.time()
        J_per_vel = {}
        info_per_vel = {}

        for v_max in TUNING_VELOCITIES:
            J_v, info_v = _evaluate_single_velocity(weights, v_max)
            J_per_vel[v_max] = J_v
            info_per_vel[v_max] = info_v

        J_multi = float(np.mean(list(J_per_vel.values())))
        wall_total = time.time() - t0_total

        parts = []
        for v in TUNING_VELOCITIES:
            info = info_per_vel[v]
            if isinstance(info, dict) and "path_completed" in info:
                parts.append(
                    f"v={v}: J={J_per_vel[v]:.3f} "
                    f"path={info['path_completed']*100:.0f}% "
                    f"v̄θ={float(info.get('mean_vtheta', 0)):.1f} "
                    f"ratio={float(info.get('mean_vpath_ratio', 0)):.3f} "
                    f"v_eff={float(info.get('_effective_speed', 0)):.2f} "
                    f"rmse_c={float(info.get('rmse_contorno', 0)):.3f}"
                )
            else:
                parts.append(f"v={v}: FAIL")

        print(
            f"  [Local {trial.number:3d}]  J_multi={J_multi:8.3f}  "
            f"{'  |  '.join(parts)}  ({wall_total:.1f}s)",
            flush=True,
        )

        trial.set_user_attr("J_multi", J_multi)
        trial.set_user_attr("wall_time", wall_total)
        for v in TUNING_VELOCITIES:
            info = info_per_vel[v]
            if isinstance(info, dict) and "path_completed" in info:
                trial.set_user_attr(f"J_v{v}", float(J_per_vel[v]))
                trial.set_user_attr(f"path_v{v}", float(info["path_completed"]))
                trial.set_user_attr(f"vtheta_v{v}", float(info.get("mean_vtheta", 0)))
                trial.set_user_attr(f"ratio_v{v}", float(info.get("mean_vpath_ratio", 0)))
                trial.set_user_attr(f"veff_v{v}", float(info.get("_effective_speed", 0)))
                trial.set_user_attr(f"rmse_c_v{v}", float(info.get("rmse_contorno", 0)))

        return J_multi

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Local MPCC bilevel refinement around MPCC_MANUAL_WEIGHTS"
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "cmaes"])
    parser.add_argument("--study-name", type=str, default="mpcc_tuning_local")
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna storage URI, e.g. sqlite:///local.db")
    parser.add_argument("--gain-low-scale", type=float, default=0.75,
                        help="Lower multiplicative bound for Q_ec/Q_el/Q_q")
    parser.add_argument("--gain-high-scale", type=float, default=1.35,
                        help="Upper multiplicative bound for Q_ec/Q_el/Q_q")
    parser.add_argument("--qomega-low-scale", type=float, default=0.90,
                        help="Lower multiplicative bound for Q_omega")
    parser.add_argument("--qomega-high-scale", type=float, default=1.10,
                        help="Upper multiplicative bound for Q_omega")
    parser.add_argument("--umat-low-scale", type=float, default=0.92,
                        help="Lower multiplicative bound for U_mat")
    parser.add_argument("--umat-high-scale", type=float, default=1.08,
                        help="Upper multiplicative bound for U_mat")
    parser.add_argument("--qs-low-scale", type=float, default=0.75,
                        help="Lower multiplicative bound for Q_s")
    parser.add_argument("--qs-high-scale", type=float, default=1.35,
                        help="Upper multiplicative bound for Q_s")
    args = parser.parse_args()

    if not HAS_OPTUNA:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    print("=" * 70)
    print("  MPCC LOCAL BILEVEL REFINEMENT")
    print("=" * 70)
    print(f"\n  Sampler      : {args.sampler.upper()}")
    print(f"  Trials       : {args.n_trials}")
    print(f"  Velocities   : {TUNING_VELOCITIES}")
    print(f"  Study        : {args.study_name}")
    print(f"  Storage      : {args.db or 'in-memory'}")
    print(f"  Base weights : {MPCC_MANUAL_WEIGHTS}")
    print(
        f"  Local windows: gains in [{args.gain_low_scale}, {args.gain_high_scale}] × base, "
        f"Q_omega in [{args.qomega_low_scale}, {args.qomega_high_scale}] × base, "
        f"U_mat in [{args.umat_low_scale}, {args.umat_high_scale}] × base, "
        f"Q_s in [{args.qs_low_scale}, {args.qs_high_scale}] × base"
    )
    print(
        "  Objective    : normalised contour/lag + "
        "effective-speed reward + completion/failure penalties\n"
    )

    print("[1/3] Building infrastructure (trajectory + solver) ...")
    t0 = time.time()
    _ = run_simulation(weights=None, verbose=False)
    print(f"[1/3] Done. Baseline simulation took {time.time()-t0:.1f}s\n")

    global _REFERENCE_BY_VELOCITY
    _REFERENCE_BY_VELOCITY = _build_reference_metrics()
    print()

    if args.sampler == "cmaes":
        sampler = CmaEsSampler(seed=OPTUNA_SEED)
    else:
        sampler = TPESampler(seed=OPTUNA_SEED, n_startup_trials=N_STARTUP_TRIALS)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction="minimize",
        storage=args.db,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        study.enqueue_trial({
            "Q_ec0": MPCC_MANUAL_WEIGHTS["Q_ec"][0],
            "Q_ec1": MPCC_MANUAL_WEIGHTS["Q_ec"][1],
            "Q_ec2": MPCC_MANUAL_WEIGHTS["Q_ec"][2],
            "Q_el0": MPCC_MANUAL_WEIGHTS["Q_el"][0],
            "Q_el1": MPCC_MANUAL_WEIGHTS["Q_el"][1],
            "Q_el2": MPCC_MANUAL_WEIGHTS["Q_el"][2],
            "Q_q0": MPCC_MANUAL_WEIGHTS["Q_q"][0],
            "Q_q1": MPCC_MANUAL_WEIGHTS["Q_q"][1],
            "Q_q2": MPCC_MANUAL_WEIGHTS["Q_q"][2],
            "U_mat0": MPCC_MANUAL_WEIGHTS["U_mat"][0],
            "U_mat1": MPCC_MANUAL_WEIGHTS["U_mat"][1],
            "U_mat2": MPCC_MANUAL_WEIGHTS["U_mat"][2],
            "U_mat3": MPCC_MANUAL_WEIGHTS["U_mat"][3],
            "Q_omega0": MPCC_MANUAL_WEIGHTS["Q_omega"][0],
            "Q_omega1": MPCC_MANUAL_WEIGHTS["Q_omega"][1],
            "Q_omega2": MPCC_MANUAL_WEIGHTS["Q_omega"][2],
            "Q_s": MPCC_MANUAL_WEIGHTS["Q_s"],
        })

    objective = make_objective(
        args.gain_low_scale,
        args.gain_high_scale,
        args.qomega_low_scale,
        args.qomega_high_scale,
        args.umat_low_scale,
        args.umat_high_scale,
        args.qs_low_scale,
        args.qs_high_scale
    )

    print(f"[2/3] Running {args.n_trials} local trials ...\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_weights = trial_to_weights(
        best,
        args.gain_low_scale,
        args.gain_high_scale,
        args.qomega_low_scale,
        args.qomega_high_scale,
        args.umat_low_scale,
        args.umat_high_scale,
        args.qs_low_scale,
        args.qs_high_scale
    )

    print("\n" + "=" * 70)
    print("  LOCAL REFINEMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Best trial  : #{best.number}")
    print(f"  Best J_multi: {best.value:.4f}")
    print(f"  Best weights:")
    for key, val in best_weights.items():
        print(f"    {key:10s} = {val}")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    history_records = []
    best_so_far = float("inf")
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            J_val = trial.value
            best_so_far = min(best_so_far, J_val)
            record = {
                "trial": trial.number,
                "J_multi": J_val,
                "best_J_so_far": best_so_far,
                "wall_time": trial.user_attrs.get("wall_time"),
            }
            for v in TUNING_VELOCITIES:
                record[f"J_v{v}"] = trial.user_attrs.get(f"J_v{v}")
                record[f"path_v{v}"] = trial.user_attrs.get(f"path_v{v}")
                record[f"vtheta_v{v}"] = trial.user_attrs.get(f"vtheta_v{v}")
                record[f"ratio_v{v}"] = trial.user_attrs.get(f"ratio_v{v}")
                record[f"veff_v{v}"] = trial.user_attrs.get(f"veff_v{v}")
                record[f"rmse_c_v{v}"] = trial.user_attrs.get(f"rmse_c_v{v}")
            history_records.append(record)

    hist_path = os.path.join(out_dir, "tuning_history_local.json")
    with open(hist_path, "w", encoding="utf-8") as fp:
        json.dump({
            "controller": "MPCC",
            "strategy": "local-refinement",
            "base_weights": MPCC_MANUAL_WEIGHTS,
            "tuning_velocities": TUNING_VELOCITIES,
            "n_trials": len(history_records),
            "sampler": args.sampler,
            "objective_info": {
                "type": "MEAN over velocities of baseline-normalised contour/lag/effective-speed objective",
                "W_INCOMPLETE": W_INCOMPLETE,
                "W_FAIL": W_FAIL,
                "W_CONTOUR_LOCAL": W_CONTOUR_LOCAL,
                "W_LAG_LOCAL": W_LAG_LOCAL,
                "W_EFF_SPEED": W_EFF_SPEED,
                "reference_by_velocity": _REFERENCE_BY_VELOCITY,
            },
            "trials": history_records,
        }, fp, indent=2)

    comparison = _build_comparison(best_weights)

    out_path = os.path.join(out_dir, "best_weights_local.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({
            "best_J_multi": best.value,
            "best_trial": best.number,
            "weights": {
                key: (val if isinstance(val, (int, float)) else list(val))
                for key, val in best_weights.items()
            },
            "flat_params": {key: float(val) for key, val in best.params.items()},
            "objective_info": {
                "type": "MEAN over velocities of baseline-normalised contour/lag/effective-speed objective",
                "strategy": "local-refinement",
                "base_weights": MPCC_MANUAL_WEIGHTS,
                "W_CONTOUR_LOCAL": W_CONTOUR_LOCAL,
                "W_LAG_LOCAL": W_LAG_LOCAL,
                "W_EFF_SPEED": W_EFF_SPEED,
                "reference_by_velocity": _REFERENCE_BY_VELOCITY,
            },
            "comparison_vs_manual": comparison,
        }, fp, indent=2)

    print(f"\n[3/3] ✓ Local best weights saved to {out_path}")
    print(f"      ✓ Local tuning history saved to {hist_path}")
    print(
        "      ✓ Comparison vs manual: "
        f"rmse_c {comparison['manual']['rmse_contorno']:.3f}->{comparison['refined']['rmse_contorno']:.3f}, "
        f"rmse_l {comparison['manual']['rmse_lag']:.3f}->{comparison['refined']['rmse_lag']:.3f}, "
        f"v_eff {comparison['manual']['effective_speed']:.3f}->{comparison['refined']['effective_speed']:.3f}"
    )


if __name__ == "__main__":
    main()
