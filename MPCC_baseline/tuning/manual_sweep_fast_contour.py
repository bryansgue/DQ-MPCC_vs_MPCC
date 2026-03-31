"""
Deterministic manual sweep around the current MPCC manual weights.

Goal:
  - keep contouring small and stable
  - maximise effective path speed, not only virtual progress speed

Selection rule:
  1. evaluate the current manual set as the baseline
  2. sweep Q_ec, Q_el, Q_s around that baseline
  3. refine Q_q around the best stage-1 candidate
  4. among candidates that keep contour close to the baseline, pick the one
     with the highest effective speed = mean_vtheta * mean_vpath_ratio
  5. fall back to the lowest combined score if no candidate dominates
"""

import itertools
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (PROJECT_ROOT, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from MPCC_simulation_tuner import run_simulation
from experiment_config import MPCC_MANUAL_WEIGHTS


def clone_weights(base):
    return {
        "Q_ec": list(base["Q_ec"]),
        "Q_el": list(base["Q_el"]),
        "Q_q": list(base["Q_q"]),
        "U_mat": list(base["U_mat"]),
        "Q_omega": list(base["Q_omega"]),
        "Q_s": float(base["Q_s"]),
    }


def scale_vec(vec, scale):
    return [float(scale * v) for v in vec]


def evaluate(name, weights):
    res = run_simulation(weights=weights, verbose=False, vtheta_max=10)
    eff_speed = float(res["mean_vtheta"]) * float(res["mean_vpath_ratio"])
    score = (
        1.00 * float(res["rmse_contorno"])
        + 0.25 * float(res["rmse_lag"])
        + 0.20 * max(0.0, 0.70 - float(res["mean_vpath_ratio"]))
        - 0.03 * eff_speed
    )
    out = {
        "name": name,
        "weights": weights,
        "rmse_c": float(res["rmse_contorno"]),
        "rmse_l": float(res["rmse_lag"]),
        "rmse_att": float(res["rmse_attitude"]),
        "mean_vtheta": float(res["mean_vtheta"]),
        "mean_vpath_ratio": float(res["mean_vpath_ratio"]),
        "effective_speed": eff_speed,
        "path_completed": float(res["path_completed"]),
        "score": float(score),
    }
    print(
        f"{name:20s} | ec={out['rmse_c']:.3f} "
        f"| el={out['rmse_l']:.3f} "
        f"| vth={out['mean_vtheta']:.3f} "
        f"| ratio={out['mean_vpath_ratio']:.3f} "
        f"| veff={out['effective_speed']:.3f} "
        f"| score={out['score']:.3f}",
        flush=True,
    )
    return out


def pick_best(results, baseline):
    feasible = []
    for r in results:
        if r["path_completed"] < 0.99:
            continue
        if r["rmse_c"] <= 1.10 * baseline["rmse_c"] and r["mean_vpath_ratio"] >= baseline["mean_vpath_ratio"] - 0.03:
            feasible.append(r)

    if feasible:
        feasible.sort(key=lambda r: (-r["effective_speed"], r["rmse_c"], r["score"]))
        return feasible[0], feasible

    results = [r for r in results if r["path_completed"] >= 0.99]
    results.sort(key=lambda r: r["score"])
    return results[0], results


def main():
    base = clone_weights(MPCC_MANUAL_WEIGHTS)
    print("=" * 72)
    print("MANUAL SWEEP: FAST CONTOURING MPCC")
    print("=" * 72)
    baseline = evaluate("baseline", base)

    stage1 = [baseline]
    qec_scales = [1.00, 1.10, 1.20]
    qel_scales = [1.00, 1.05]
    qs_scales = [1.00, 1.05, 1.10, 1.15]

    for qec_s, qel_s, qs_s in itertools.product(qec_scales, qel_scales, qs_scales):
        if qec_s == 1.0 and qel_s == 1.0 and qs_s == 1.0:
            continue
        w = clone_weights(base)
        w["Q_ec"] = scale_vec(base["Q_ec"], qec_s)
        w["Q_el"] = scale_vec(base["Q_el"], qel_s)
        w["Q_s"] = float(base["Q_s"] * qs_s)
        name = f"s1_ec{qec_s:.2f}_el{qel_s:.2f}_qs{qs_s:.2f}"
        stage1.append(evaluate(name, w))

    stage1_best, stage1_ranked = pick_best(stage1, baseline)
    print("\nBest after stage 1:")
    print(json.dumps(stage1_best, indent=2))

    stage2 = [stage1_best]
    qq_rollpitch_scales = [0.90, 1.00, 1.10]
    qq_yaw_scales = [0.90, 1.00, 1.10]
    base2 = clone_weights(stage1_best["weights"])

    for rp_s, yaw_s in itertools.product(qq_rollpitch_scales, qq_yaw_scales):
        if rp_s == 1.0 and yaw_s == 1.0:
            continue
        w = clone_weights(base2)
        w["Q_q"] = [
            float(base2["Q_q"][0] * rp_s),
            float(base2["Q_q"][1] * rp_s),
            float(base2["Q_q"][2] * yaw_s),
        ]
        name = f"s2_qrp{rp_s:.2f}_qy{yaw_s:.2f}"
        stage2.append(evaluate(name, w))

    final_best, final_ranked = pick_best(stage2 + stage1, baseline)

    out_path = os.path.join(PROJECT_ROOT, "tuning", "manual_sweep_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_best, f, indent=2)

    print("\nFinal best:")
    print(json.dumps(final_best, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
