#!/usr/bin/env python3
"""Generate a reviewer-oriented standalone PDF for the combined local tuning report."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import textwrap
from pathlib import Path
import sys

from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "results" / "tuning" / "reports"
MAT_PATH = ROOT / "results" / "tuning" / "data" / "tuning_summary.mat"
EXPORT_SCRIPT = ROOT / "results" / "tuning" / "scripts" / "export_tuning_summary_mat.py"
SECTION_TEX = OUT_DIR / "tuning_combined_section.tex"
SUMMARY_TEX = OUT_DIR / "tuning_combined_summary.tex"
STANDALONE_PDF = OUT_DIR / "tuning_combined_section.pdf"
COMPACT_TEX = OUT_DIR / "tuning_compact_section.tex"
COMPACT_SUMMARY_TEX = OUT_DIR / "tuning_compact_summary.tex"
COMPACT_PDF = OUT_DIR / "tuning_compact_section.pdf"

DQ_LOCAL_BEST = ROOT / "DQ-MPCC_baseline" / "tuning" / "best_weights_local.json"
DQ_LOCAL_HIST = ROOT / "DQ-MPCC_baseline" / "tuning" / "tuning_history_local.json"
MPCC_LOCAL_BEST = ROOT / "MPCC_baseline" / "tuning" / "best_weights_local.json"
MPCC_LOCAL_HIST = ROOT / "MPCC_baseline" / "tuning" / "tuning_history_local.json"

try:
    from config.experiment_config import (
        N_STARTUP_TRIALS,
        OPTUNA_SEED,
        TUNING_T_FINAL,
        TUNING_FREC,
        TUNING_T_PREDICTION,
        TUNING_N_WAYPOINTS,
        TRAJECTORY_T_FINAL,
    )
except Exception:
    N_STARTUP_TRIALS = 10
    OPTUNA_SEED = 42
    TUNING_T_FINAL = 30
    TUNING_FREC = 100
    TUNING_T_PREDICTION = 0.3
    TUNING_N_WAYPOINTS = 30
    TRAJECTORY_T_FINAL = 30


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _scalar(payload, key: str, default=None):
    value = payload.get(key, default)
    if hasattr(value, "flat"):
        return value.flat[0]
    return value


def _vector(payload, key: str) -> list[float]:
    value = payload[key]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flat: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(float(v) for v in item)
            else:
                flat.append(float(item))
        return flat
    return [float(value)]


def _pct(old: float, new: float, lower_is_better: bool) -> float:
    if abs(old) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (old - new) / old
    return 100.0 * (new - old) / old


def _fmt_scalar(value: float) -> str:
    if abs(value) < 0.01 and value != 0.0:
        exp = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10 ** exp)
        return f"{mantissa:.1f}\\!\\times\\!10^{{{exp}}}"
    if abs(value) < 0.1:
        return f"{value:.3f}"
    if abs(value) < 1.0:
        return f"{value:.2f}"
    if abs(value) < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _fmt_vec(values: list[float]) -> str:
    return ", ".join(_fmt_scalar(v) for v in values)


def _window(base_vals: list[float], low_scale: float, high_scale: float) -> str:
    lows = [v * low_scale for v in base_vals]
    highs = [v * high_scale for v in base_vals]
    return f"$[{_fmt_vec(lows)}] \\rightarrow [{_fmt_vec(highs)}]$"


def _window_scalar(base_val: float, low_scale: float, high_scale: float) -> str:
    return f"$[{_fmt_scalar(base_val * low_scale)}, {_fmt_scalar(base_val * high_scale)}]$"


def _best_wall_time(history: dict) -> float:
    best_trial = min(history["trials"], key=lambda item: item["best_J_so_far"])
    return float(best_trial.get("wall_time") or 0.0)


def _mean_wall_time(history: dict) -> float:
    vals = [float(t["wall_time"]) for t in history["trials"] if t.get("wall_time") is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _controller_summary(payload, prefix: str, controller_name: str, attitude: bool) -> dict:
    manual_c = float(_scalar(payload, f"{prefix}_manual_rmse_contorno"))
    refined_c = float(_scalar(payload, f"{prefix}_refined_rmse_contorno"))
    manual_l = float(_scalar(payload, f"{prefix}_manual_rmse_lag"))
    refined_l = float(_scalar(payload, f"{prefix}_refined_rmse_lag"))
    manual_v = float(_scalar(payload, f"{prefix}_manual_effective_speed"))
    refined_v = float(_scalar(payload, f"{prefix}_refined_effective_speed"))
    manual_p = float(_scalar(payload, f"{prefix}_manual_path_completed"))
    refined_p = float(_scalar(payload, f"{prefix}_refined_path_completed"))
    manual_fail = float(_scalar(payload, f"{prefix}_manual_solver_fail_ratio"))
    refined_fail = float(_scalar(payload, f"{prefix}_refined_solver_fail_ratio"))

    summary = {
        "name": controller_name,
        "manual_c": manual_c,
        "refined_c": refined_c,
        "manual_l": manual_l,
        "refined_l": refined_l,
        "manual_v": manual_v,
        "refined_v": refined_v,
        "manual_p": manual_p,
        "refined_p": refined_p,
        "manual_fail": manual_fail,
        "refined_fail": refined_fail,
        "imp_c": _pct(manual_c, refined_c, True),
        "imp_l": _pct(manual_l, refined_l, True),
        "imp_v": _pct(manual_v, refined_v, False),
        "best_j_multi": float(_scalar(payload, f"{prefix}_refined_best_j_multi", 0.0)),
        "best_trial": int(_scalar(payload, f"{prefix}_refined_best_trial", 0)),
        "q_ec": _vector(payload, f"{prefix}_refined_q_ec"),
        "q_el": _vector(payload, f"{prefix}_refined_q_el"),
        "q_rot": _vector(payload, f"{prefix}_refined_q_phi" if prefix == "dq" else f"{prefix}_refined_q_q"),
        "u_mat": _vector(payload, f"{prefix}_refined_u_mat"),
        "q_omega": _vector(payload, f"{prefix}_refined_q_omega"),
        "q_s": float(_scalar(payload, f"{prefix}_refined_q_s")),
    }

    if attitude:
        manual_a = float(_scalar(payload, f"{prefix}_manual_rmse_attitude"))
        refined_a = float(_scalar(payload, f"{prefix}_refined_rmse_attitude"))
        summary["manual_a"] = manual_a
        summary["refined_a"] = refined_a
        summary["imp_a"] = _pct(manual_a, refined_a, True)

    return summary


def _build_context(dq: dict, mpcc: dict, dq_best: dict, dq_hist: dict, mpcc_best: dict, mpcc_hist: dict) -> dict:
    return {
        "dq": {
            "manual": dq_hist["base_weights"],
            "refined": dq_best["weights"],
            "tuning_velocities": dq_hist["tuning_velocities"],
            "n_trials": int(dq_hist["n_trials"]),
            "sampler": dq_hist["sampler"].upper(),
            "low_scale": 0.75,
            "high_scale": 1.35,
            "qomega_low_scale": 0.90,
            "qomega_high_scale": 1.10,
            "umat_low_scale": 0.92,
            "umat_high_scale": 1.08,
            "qs_low_scale": 0.75,
            "qs_high_scale": 1.35,
            "frozen_blocks": r"none",
            "tuned_blocks": r"$\bm{q}_{\phi}$, $\bm{q}_{c}$, $\bm{q}_{\ell}$, $\bm{w}_u$, $\bm{q}_{\omega}$, $Q_s$",
            "objective": (
                r"$J_{\mathrm{multi}}=\mathrm{mean}_v[$contour $+$ lag $-$ effective-speed "
                r"$+$ failure/completion penalties$]$"
            ),
            "best_wall_s": _best_wall_time(dq_hist),
            "mean_wall_s": _mean_wall_time(dq_hist),
        },
        "mpcc": {
            "manual": mpcc_hist["base_weights"],
            "refined": mpcc_best["weights"],
            "tuning_velocities": mpcc_hist["tuning_velocities"],
            "n_trials": int(mpcc_hist["n_trials"]),
            "sampler": mpcc_hist["sampler"].upper(),
            "low_scale": 0.75,
            "high_scale": 1.35,
            "qomega_low_scale": 0.90,
            "qomega_high_scale": 1.10,
            "umat_low_scale": 0.92,
            "umat_high_scale": 1.08,
            "qs_low_scale": 0.75,
            "qs_high_scale": 1.35,
            "frozen_blocks": r"none",
            "tuned_blocks": r"$\bm{q}_{q}$, $\bm{q}_{c}$, $\bm{q}_{\ell}$, $\bm{w}_u$, $\bm{q}_{\omega}$, $Q_s$",
            "objective": (
                r"$J_{\mathrm{multi}}=\mathrm{mean}_v[$contour $+$ lag $-$ effective-speed "
                r"$+$ failure/completion penalties$]$"
            ),
            "best_wall_s": _best_wall_time(mpcc_hist),
            "mean_wall_s": _mean_wall_time(mpcc_hist),
        },
    }


def _build_intro(dq_ctx: dict, mpcc_ctx: dict) -> str:
    t_s = 1.0 / float(TUNING_FREC)
    n_prediction = int(round(float(TUNING_T_PREDICTION) / t_s))
    n_rollout_steps = int(round(float(TUNING_T_FINAL) / t_s)) - n_prediction + 1
    return textwrap.dedent(
        f"""
        This standalone report documents the \emph{{local}} bilevel refinement stage used to compare the
        manually stabilised operating points against the final refined gains for both controllers.
        It is intentionally self-contained for review: it states which gains were allowed to move,
        which gains were frozen by design, the multiplicative search windows, the final selected weights,
        and the corresponding closed-loop changes observed in the headless baseline simulations.

        The key point is that this is not a global retuning from scratch. The refinement starts from the
        manual gains and searches only a bounded neighbourhood around them. In the current local-search
        setup, the effort and angular-rate penalty blocks are also allowed to move, so the reported gains
        correspond to a true local search over all cost-weight blocks listed in the tables below.

        Each rollout in the tuning loop uses the same headless MiL configuration: maximum simulated duration
        $T_{{\mathrm{{final}}}}={TUNING_T_FINAL}$~s, trajectory-geometry window $T_{{\mathrm{{traj}}}}={TRAJECTORY_T_FINAL}$~s,
        control frequency $f_s={TUNING_FREC}$~Hz ($T_s={t_s:.2f}$~s), prediction horizon
        $T_N={TUNING_T_PREDICTION}$~s discretised into $N={n_prediction}$ shooting nodes, and
        $N_w={TUNING_N_WAYPOINTS}$ arc-length interpolation waypoints. Under this setup, one rollout evaluates
        approximately {n_rollout_steps} closed-loop integration steps, and each Optuna trial evaluates one full rollout
        for every operating point in $\\mathcal{{V}}$.

        The DQ local study used {dq_ctx['n_trials']} {dq_ctx['sampler']} trials over tuning velocities
        $\\{{{', '.join(str(v) for v in dq_ctx['tuning_velocities'])}\\}}$~m/s, with mean trial time
        {dq_ctx['mean_wall_s']:.1f}~s. The MPCC local study used {mpcc_ctx['n_trials']} {mpcc_ctx['sampler']}
        trials over the same velocity set, with mean trial time {mpcc_ctx['mean_wall_s']:.1f}~s.
        """
    ).strip()


def _build_setup_table(dq_ctx: dict, mpcc_ctx: dict) -> str:
    return textwrap.dedent(
        rf"""
        \begin{{table}}[!t]
        \centering
        \caption{{Local bilevel-refinement protocol used to generate the final refined gains.}}
        \label{{tab:tuning_local_setup}}
        \footnotesize
        \renewcommand{{\arraystretch}}{{1.15}}
        \setlength{{\tabcolsep}}{{4pt}}
        \begin{{tabular}}{{p{{2.2cm}} p{{2.8cm}} p{{2.6cm}} p{{2.1cm}} p{{2.2cm}}}}
        \toprule
        \textbf{{Controller}} & \textbf{{Tuned blocks}} & \textbf{{Frozen blocks}} & \textbf{{Local window}} & \textbf{{Objective summary}} \\
        \midrule
        DQ-MPCC & {dq_ctx['tuned_blocks']} & {dq_ctx['frozen_blocks']} & 
        $[\times {dq_ctx['low_scale']:.2f}, \times {dq_ctx['high_scale']:.2f}]$ for $\bm{{q}}_{{\phi}},\bm{{q}}_c,\bm{{q}}_\ell$;
        $[\times {dq_ctx['umat_low_scale']:.2f}, \times {dq_ctx['umat_high_scale']:.2f}]$ for $\bm{{w}}_u$;
        $[\times {dq_ctx['qomega_low_scale']:.2f}, \times {dq_ctx['qomega_high_scale']:.2f}]$ for $\bm{{q}}_{{\omega}}$;
        $[\times {dq_ctx['qs_low_scale']:.2f}, \times {dq_ctx['qs_high_scale']:.2f}]$ for $Q_s$ &
        contour, lag, effective speed, progress, path ratio, effort, completion, solver failures \\
        MPCC & {mpcc_ctx['tuned_blocks']} & {mpcc_ctx['frozen_blocks']} &
        $[\times {mpcc_ctx['low_scale']:.2f}, \times {mpcc_ctx['high_scale']:.2f}]$ for $\bm{{q}}_q,\bm{{q}}_c,\bm{{q}}_\ell$;
        $[\times {mpcc_ctx['umat_low_scale']:.2f}, \times {mpcc_ctx['umat_high_scale']:.2f}]$ for $\bm{{w}}_u$;
        $[\times {mpcc_ctx['qomega_low_scale']:.2f}, \times {mpcc_ctx['qomega_high_scale']:.2f}]$ for $\bm{{q}}_{{\omega}}$;
        $[\times {mpcc_ctx['qs_low_scale']:.2f}, \times {mpcc_ctx['qs_high_scale']:.2f}]$ for $Q_s$ &
        contour, lag, effective speed, completion, solver failures \\
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
    ).strip()


def _build_objective_coeff_table(dq_hist: dict, mpcc_hist: dict) -> str:
    dq_obj = dq_hist["objective_info"]
    mpcc_obj = mpcc_hist["objective_info"]
    rows = [
        ("$W_{\\mathrm{contour}}$", dq_obj.get("W_CONTOUR_LOCAL", "--"), mpcc_obj.get("W_CONTOUR_LOCAL", "--")),
        ("$W_{\\mathrm{lag}}$", dq_obj.get("W_LAG_LOCAL", "--"), mpcc_obj.get("W_LAG_LOCAL", "--")),
        ("$W_{\\mathrm{eff\\ speed}}$", dq_obj.get("W_EFF_SPEED", "--"), mpcc_obj.get("W_EFF_SPEED", "--")),
        ("$W_{\\mathrm{incomplete}}$", dq_obj.get("W_INCOMPLETE", "--"), mpcc_obj.get("W_INCOMPLETE", "--")),
        ("$W_{\\mathrm{fail}}$", dq_obj.get("W_FAIL", "--"), mpcc_obj.get("W_FAIL", "--")),
        ("$W_{\\mathrm{progress}}$", dq_obj.get("W_PROGRESS_LOCAL", "--"), "--"),
        ("$W_{\\mathrm{progress,hard}}$", dq_obj.get("W_PROGRESS_HARD", "--"), "--"),
        ("Min progress ratio", dq_obj.get("MIN_PROGRESS_RATIO", "--"), "--"),
        ("$W_{\\mathrm{path\ ratio}}$", dq_obj.get("W_VPATH", "--"), "--"),
        ("$W_{\\mathrm{path\ ratio,hard}}$", dq_obj.get("W_VPATH_HARD", "--"), "--"),
        ("Min path ratio", dq_obj.get("VPATH_RATIO_MIN", "--"), "--"),
        ("$W_{\\mathrm{att}}$", dq_obj.get("W_ATT_LOCAL", "--"), "--"),
        ("$W_{\\mathrm{effort}}$", dq_obj.get("W_EFFORT_LOCAL", "--"), "--"),
    ]
    body = "\n".join(
        f"{name} & {val_dq if isinstance(val_dq, str) else _fmt_scalar(float(val_dq))} & "
        f"{val_mpcc if isinstance(val_mpcc, str) else _fmt_scalar(float(val_mpcc))} \\\\"
        for name, val_dq, val_mpcc in rows
    )
    return textwrap.dedent(
        r"""
        \begin{table}[!t]
        \centering
        \caption{Exact upper-level objective coefficients used in the local refinement. Blank entries indicate terms not used by that controller.}
        \label{tab:tuning_objective_coeffs}
        \footnotesize
        \renewcommand{\arraystretch}{1.10}
        \setlength{\tabcolsep}{4pt}
        \begin{tabular}{l c c}
        \toprule
        \textbf{Coefficient} & \textbf{DQ-MPCC} & \textbf{Baseline MPCC} \\
        \midrule
        """
    ).strip() + "\n" + body + "\n" + textwrap.dedent(
        r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
    ).strip()


def _build_reference_metrics_table(controller_name: str, label: str, history: dict, include_effort: bool) -> str:
    ref = history["objective_info"]["reference_by_velocity"]
    include_effort = include_effort and all(
        "mean_effort" in ref[str(v)] for v in history["tuning_velocities"]
    )
    rows = []
    for v in history["tuning_velocities"]:
        metrics = ref[str(v)]
        effort = f" & {float(metrics['mean_effort']):.1f}" if include_effort else ""
        rows.append(
            f"{v} & {float(metrics['rmse_contorno']):.3f} & {float(metrics['rmse_lag']):.3f} & "
            f"{float(metrics['mean_vtheta']):.3f} & {float(metrics['mean_vpath_ratio']):.3f} & "
            f"{float(metrics['effective_speed']):.3f}{effort} \\\\"
        )
    effort_col = " & \\textbf{Effort}" if include_effort else ""
    effort_fmt = " c" if include_effort else ""
    return textwrap.dedent(
        rf"""
        \begin{{table}}[!t]
        \centering
        \caption{{Manual-reference metrics used to normalise the {controller_name} local objective at each tuning velocity.}}
        \label{{{label}}}
        \footnotesize
        \renewcommand{{\arraystretch}}{{1.10}}
        \setlength{{\tabcolsep}}{{4pt}}
        \begin{{tabular}}{{c c c c c c{effort_fmt}}}
        \toprule
        \textbf{{$v_{{\theta,\max}}$}} & $\mathbf{{RMSE}}_c$ & $\mathbf{{RMSE}}_\ell$ & $\bar{{v}}_\theta$ & Ratio & $v_{{\mathrm{{eff}}}}${effort_col} \\
        \midrule
        """
    ).strip() + "\n" + "\n".join(rows) + "\n" + textwrap.dedent(
        r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
    ).strip()


def _weight_row(name: str, manual: str, search: str, refined: str, status: str) -> str:
    return f"{name} & {manual} & {search} & {refined} & {status} \\\\"


def _build_controller_weight_table(title: str, label: str, manual: dict, refined: dict, rotation_key: str,
                                   low_scale: float, high_scale: float,
                                   qs_low_scale: float, qs_high_scale: float,
                                   best_j: float, best_trial: int) -> str:
    rot_label = r"$\bm{q}_{\phi}$" if rotation_key == "Q_phi" else r"$\bm{q}_{q}$"
    body = [
        _weight_row(rot_label, f"$[{_fmt_vec(manual[rotation_key])}]$", _window(manual[rotation_key], low_scale, high_scale),
                    f"$[{_fmt_vec(refined[rotation_key])}]$", "optimised"),
        _weight_row(r"$\bm{q}_{c}$", f"$[{_fmt_vec(manual['Q_ec'])}]$", _window(manual["Q_ec"], low_scale, high_scale),
                    f"$[{_fmt_vec(refined['Q_ec'])}]$", "optimised"),
        _weight_row(r"$\bm{q}_{\ell}$", f"$[{_fmt_vec(manual['Q_el'])}]$", _window(manual["Q_el"], low_scale, high_scale),
                    f"$[{_fmt_vec(refined['Q_el'])}]$", "optimised"),
        _weight_row(r"$U^f$", f"${_fmt_scalar(manual['U_mat'][0])}$", _window_scalar(float(manual["U_mat"][0]), low_scale, high_scale),
                    f"${_fmt_scalar(refined['U_mat'][0])}$", "optimised"),
        _weight_row(r"$[U^{\tau_x},U^{\tau_y},U^{\tau_z}]$", f"$[{_fmt_vec(manual['U_mat'][1:])}]$", _window(manual["U_mat"][1:], low_scale, high_scale),
                    f"$[{_fmt_vec(refined['U_mat'][1:])}]$", "optimised"),
        _weight_row(r"$\bm{q}_{\omega}$", f"$[{_fmt_vec(manual['Q_omega'])}]$", _window(manual["Q_omega"], low_scale, high_scale),
                    f"$[{_fmt_vec(refined['Q_omega'])}]$", "optimised"),
        _weight_row(r"$Q_s$", f"${_fmt_scalar(manual['Q_s'])}$", _window_scalar(float(manual["Q_s"]), qs_low_scale, qs_high_scale),
                    f"${_fmt_scalar(refined['Q_s'])}$", "optimised"),
        r"\midrule",
        f"$J_{{\\mathrm{{multi}}}}^{{\\star}}$ & \\multicolumn{{2}}{{c}}{{--}} & ${best_j:.3f}$ & best local objective \\\\",
        f"Best trial & \\multicolumn{{2}}{{c}}{{--}} & \\#{best_trial} & Optuna index \\\\",
    ]
    return textwrap.dedent(
        rf"""
        \begin{{table}}[!t]
        \centering
        \caption{{{title}}}
        \label{{{label}}}
        \footnotesize
        \renewcommand{{\arraystretch}}{{1.12}}
        \setlength{{\tabcolsep}}{{4pt}}
        \begin{{tabular}}{{l c c c c}}
        \toprule
        \textbf{{Weight}} & \textbf{{Manual}} & \textbf{{Search interval}} & \textbf{{Refined}} & \textbf{{Status}} \\
        \midrule
        """
    ).strip() + "\n" + "\n".join(body) + "\n" + textwrap.dedent(
        r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
    ).strip()


def _build_performance_table(dq: dict, mpcc: dict) -> str:
    return textwrap.dedent(
        rf"""
        \begin{{table}}[H]
        \centering
        \caption{{Closed-loop comparison between manual and refined gains used in the final baseline runs.}}
        \label{{tab:tuning_manual_vs_refined}}
        \footnotesize
        \renewcommand{{\arraystretch}}{{1.12}}
        \setlength{{\tabcolsep}}{{3.5pt}}
        \begin{{tabular}}{{lcccccc}}
        \toprule
        \textbf{{Controller}} & \textbf{{Variant}} & $\mathbf{{RMSE}}_c$ & $\mathbf{{RMSE}}_\ell$ & $\mathbf{{v}}_{{\mathrm{{eff}}}}$ & \textbf{{Path}} & \textbf{{Fail}} \\
        \midrule
        DQ-MPCC & Manual  & {dq['manual_c']:.3f} & {dq['manual_l']:.3f} & {dq['manual_v']:.3f} & {100.0*dq['manual_p']:.2f}\% & {100.0*dq['manual_fail']:.2f}\% \\
        DQ-MPCC & Refined & {dq['refined_c']:.3f} & {dq['refined_l']:.3f} & {dq['refined_v']:.3f} & {100.0*dq['refined_p']:.2f}\% & {100.0*dq['refined_fail']:.2f}\% \\
        \cmidrule(lr){{2-7}}
        DQ-MPCC & $\Delta$ (\%) & {dq['imp_c']:.1f} & {dq['imp_l']:.1f} & {dq['imp_v']:.1f} & -- & -- \\
        \midrule
        MPCC    & Manual  & {mpcc['manual_c']:.3f} & {mpcc['manual_l']:.3f} & {mpcc['manual_v']:.3f} & {100.0*mpcc['manual_p']:.2f}\% & {100.0*mpcc['manual_fail']:.2f}\% \\
        MPCC    & Refined & {mpcc['refined_c']:.3f} & {mpcc['refined_l']:.3f} & {mpcc['refined_v']:.3f} & {100.0*mpcc['refined_p']:.2f}\% & {100.0*mpcc['refined_fail']:.2f}\% \\
        \cmidrule(lr){{2-7}}
        MPCC    & $\Delta$ (\%) & {mpcc['imp_c']:.1f} & {mpcc['imp_l']:.1f} & {mpcc['imp_v']:.1f} & -- & -- \\
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
    ).strip()


def _build_compact_controller_table(controller: str, label: str, manual: dict, refined: dict,
                                    rotation_key: str, best_j: float, best_trial: int) -> str:
    rot_label = r"$\bm{q}_{\phi}$" if rotation_key == "Q_phi" else r"$\bm{q}_{q}$"
    rows = [
        (rot_label, f"$[{_fmt_vec(manual[rotation_key])}]$", f"$[{_fmt_vec(refined[rotation_key])}]$"),
        (r"$\bm{q}_{c}$", f"$[{_fmt_vec(manual['Q_ec'])}]$", f"$[{_fmt_vec(refined['Q_ec'])}]$"),
        (r"$\bm{q}_{\ell}$", f"$[{_fmt_vec(manual['Q_el'])}]$", f"$[{_fmt_vec(refined['Q_el'])}]$"),
        (r"$\bm{w}_{u}$", f"$[{_fmt_vec(manual['U_mat'])}]$", f"$[{_fmt_vec(refined['U_mat'])}]$"),
        (r"$\bm{q}_{\omega}$", f"$[{_fmt_vec(manual['Q_omega'])}]$", f"$[{_fmt_vec(refined['Q_omega'])}]$"),
        (r"$Q_s$", f"${_fmt_scalar(float(manual['Q_s']))}$", f"${_fmt_scalar(float(refined['Q_s']))}$"),
    ]
    body = "\n".join(f"{name} & {manual_val} & {refined_val} \\\\" for name, manual_val, refined_val in rows)
    return textwrap.dedent(
        rf"""
        \begin{{table}}[!t]
        \centering
        \caption{{{controller} compact gain summary. Best local objective: ${best_j:.3f}$ at trial \#{best_trial}.}}
        \label{{{label}}}
        \footnotesize
        \renewcommand{{\arraystretch}}{{1.10}}
        \setlength{{\tabcolsep}}{{4pt}}
        \begin{{tabular}}{{l c c}}
        \toprule
        \textbf{{Weight}} & \textbf{{Manual}} & \textbf{{Refined}} \\
        \midrule
        """
    ).strip() + "\n" + body + "\n" + textwrap.dedent(
        r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
    ).strip()


def _compact_window_text(ctx: dict) -> str:
    return (
        f"tracking/orientation in $[\\times {ctx['low_scale']:.2f},\\times {ctx['high_scale']:.2f}]$, "
        f"$\\bm{{w}}_u$ in $[\\times {ctx['umat_low_scale']:.2f},\\times {ctx['umat_high_scale']:.2f}]$, "
        f"$\\bm{{q}}_\\omega$ in $[\\times {ctx['qomega_low_scale']:.2f},\\times {ctx['qomega_high_scale']:.2f}]$, "
        f"$Q_s$ in $[\\times {ctx['qs_low_scale']:.2f},\\times {ctx['qs_high_scale']:.2f}]$"
    )


def _build_compact_combined_table(dq_ctx: dict, mpcc_ctx: dict, dq: dict, mpcc: dict) -> str:
    return textwrap.dedent(
        rf"""
        \begin{{table}}[H]
        \centering
        \caption{{Local tuning summary for both controllers.}}
        \label{{tab:tuning_compact_combined}}
        \scriptsize
        \renewcommand{{\arraystretch}}{{1.05}}
        \setlength{{\tabcolsep}}{{2pt}}
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin{{tabular}}{{lccc}}
        \toprule
        \textbf{{Weight}} & \textbf{{Manual}} & \textbf{{DQ-MPCC}} & \textbf{{MPCC}} \\
        \midrule
        $\bm{{q}}_{{\phi}}/\bm{{q}}_{{q}}$
        & $[{_fmt_vec(dq_ctx['manual']['Q_phi'])}]$
        & $[{_fmt_vec(dq_ctx['refined']['Q_phi'])}]$
        & $[{_fmt_vec(mpcc_ctx['refined']['Q_q'])}]$ \\
        $\bm{{q}}_{{c}}$
        & $[{_fmt_vec(dq_ctx['manual']['Q_ec'])}]$
        & $[{_fmt_vec(dq_ctx['refined']['Q_ec'])}]$
        & $[{_fmt_vec(mpcc_ctx['refined']['Q_ec'])}]$ \\
        $\bm{{q}}_{{\ell}}$
        & $[{_fmt_vec(dq_ctx['manual']['Q_el'])}]$
        & $[{_fmt_vec(dq_ctx['refined']['Q_el'])}]$
        & $[{_fmt_vec(mpcc_ctx['refined']['Q_el'])}]$ \\
        $\bm{{w}}_{{u}}$
        & $[{_fmt_vec(dq_ctx['manual']['U_mat'])}]$
        & $[{_fmt_vec(dq_ctx['refined']['U_mat'])}]$
        & $[{_fmt_vec(mpcc_ctx['refined']['U_mat'])}]$ \\
        $\bm{{q}}_{{\omega}}$
        & $[{_fmt_vec(dq_ctx['manual']['Q_omega'])}]$
        & $[{_fmt_vec(dq_ctx['refined']['Q_omega'])}]$
        & $[{_fmt_vec(mpcc_ctx['refined']['Q_omega'])}]$ \\
        $Q_s$
        & ${_fmt_scalar(float(dq_ctx['manual']['Q_s']))}$
        & ${_fmt_scalar(float(dq_ctx['refined']['Q_s']))}$
        & ${_fmt_scalar(float(mpcc_ctx['refined']['Q_s']))}$ \\
        \midrule
        $J^\star$ & -- & {dq['best_j_multi']:.3f} & {mpcc['best_j_multi']:.3f} \\
        Trial & -- & \#{dq['best_trial']} & \#{mpcc['best_trial']} \\
        \bottomrule
        \end{{tabular}}%
        }}
        \end{{table}}
        """
    ).strip()


def _build_reviewer_notes(dq: dict, mpcc: dict, dq_ctx: dict, mpcc_ctx: dict) -> str:
    return textwrap.dedent(
        f"""
        \\paragraph{{Reviewer-facing interpretation.}}
        The two local studies should be read as refinements carried out under the same outer-level objective and
        the same velocity set, so the reported $J_{{\\mathrm{{multi}}}}^\\star$ values are directly comparable as
        optimisation outcomes. The remaining differences arise from the controller formulations themselves and from
        the different closed-loop responses they induce under the same tuning protocol.

        For DQ-MPCC, the refined gains reduce contouring RMSE by {dq['imp_c']:.1f}\\% and lag RMSE by
        {dq['imp_l']:.1f}\\%, while increasing effective speed by {dq['imp_v']:.1f}\\% and keeping both
        path completion and solver-failure ratio effectively unchanged. For the baseline MPCC, the refined
        gains reduce contouring RMSE by {mpcc['imp_c']:.1f}\\%, lag RMSE by {mpcc['imp_l']:.1f}\\%, and
        attitude RMSE by {mpcc['imp_a']:.1f}\\%, but reduce effective speed by {abs(mpcc['imp_v']):.1f}\\%.

        In the refreshed symmetric setup, the local search also includes the effort and angular-rate penalties.
        That makes the refined vectors more defensible for review, because the reported final gains are no longer
        conditioned on a partially frozen manual regularisation layer.
        """
    ).strip()


def _build_compact_section_tex(payload, dq_best: dict, dq_hist: dict, mpcc_best: dict, mpcc_hist: dict) -> str:
    dq = _controller_summary(payload, "dq", "DQ-MPCC", attitude=False)
    mpcc = _controller_summary(payload, "mpcc", "MPCC", attitude=True)
    ctx = _build_context(dq, mpcc, dq_best, dq_hist, mpcc_best, mpcc_hist)
    dq_ctx = ctx["dq"]
    mpcc_ctx = ctx["mpcc"]
    t_s = 1.0 / float(TUNING_FREC)
    n_prediction = int(round(float(TUNING_T_PREDICTION) / t_s))
    n_rollout_steps = int(round(float(TUNING_T_FINAL) / t_s)) - n_prediction + 1
    intro = textwrap.dedent(
        f"""
        \\subsubsection{{Local Tuning Report}}

        This subsection reports the outcome of the local bilevel refinement under the common setup defined earlier:
        $\\mathcal{{V}}=\\{{{', '.join(str(v) for v in dq_ctx['tuning_velocities'])}\\}}$ m/s, {dq_ctx['n_trials']} TPE trials per controller,
        shared upper-level objective, and the hierarchical local search space defined in the experimental setup.
        Table~\\ref{{tab:tuning_compact_combined}} reports the manual and refined gains selected by the local search,
        while Table~\\ref{{tab:tuning_manual_vs_refined}} summarises the corresponding closed-loop effect in the final MiL baseline runs.
        """
    ).strip()
    combined_table = _build_compact_combined_table(dq_ctx, mpcc_ctx, dq, mpcc)
    perf = _build_performance_table(dq, mpcc)
    note = textwrap.dedent(
        f"""
        Overall, both controllers benefited from the local refinement, but the gain shifts and closed-loop trade-offs are
        controller dependent. DQ-MPCC improves both tracking and effective speed after refinement, whereas the baseline MPCC
        prioritises tracking accuracy at the expense of effective speed. This distinction is acceptable in the present context:
        the purpose of the local bilevel stage is not to enforce identical behavioural changes, but to ensure that both
        controllers are compared after the same mathematically defined refinement procedure, over the same representative
        velocity set, and with the same explicit distinction between optimised and fixed weights.
        """
    ).strip()
    return textwrap.dedent(
        f"""
        % Auto-generated by latex/generate_local_refine_combined.py
        % Compact reviewer-oriented version.
        % Do not edit manually.

        {intro}

        {combined_table}

        {perf}

        {note}
        """
    ).strip() + "\n"


def _build_section_tex(payload, dq_best: dict, dq_hist: dict, mpcc_best: dict, mpcc_hist: dict) -> str:
    dq = _controller_summary(payload, "dq", "DQ-MPCC", attitude=False)
    mpcc = _controller_summary(payload, "mpcc", "MPCC", attitude=True)
    ctx = _build_context(dq, mpcc, dq_best, dq_hist, mpcc_best, mpcc_hist)
    dq_ctx = ctx["dq"]
    mpcc_ctx = ctx["mpcc"]

    intro = _build_intro(dq_ctx, mpcc_ctx)
    setup = _build_setup_table(dq_ctx, mpcc_ctx)
    obj_coeffs = _build_objective_coeff_table(dq_hist, mpcc_hist)
    dq_refs = _build_reference_metrics_table("DQ-MPCC", "tab:dq_reference_metrics", dq_hist, include_effort=True)
    mpcc_refs = _build_reference_metrics_table("Baseline MPCC", "tab:mpcc_reference_metrics", mpcc_hist, include_effort=False)
    dq_weights = _build_controller_weight_table(
        "DQ-MPCC local refinement: manual gains, multiplicative search window, and selected refined gains.",
        "tab:dq_local_weights",
        dq_ctx["manual"],
        dq_ctx["refined"],
        "Q_phi",
        dq_ctx["low_scale"],
        dq_ctx["high_scale"],
        dq_ctx["qs_low_scale"],
        dq_ctx["qs_high_scale"],
        dq["best_j_multi"],
        dq["best_trial"],
    )
    mpcc_weights = _build_controller_weight_table(
        "Baseline MPCC local refinement: manual gains, multiplicative search window, and selected refined gains.",
        "tab:mpcc_local_weights",
        mpcc_ctx["manual"],
        mpcc_ctx["refined"],
        "Q_q",
        mpcc_ctx["low_scale"],
        mpcc_ctx["high_scale"],
        mpcc_ctx["qs_low_scale"],
        mpcc_ctx["qs_high_scale"],
        mpcc["best_j_multi"],
        mpcc["best_trial"],
    )
    perf = _build_performance_table(dq, mpcc)
    notes = _build_reviewer_notes(dq, mpcc, dq_ctx, mpcc_ctx)
    return textwrap.dedent(
        f"""
        % Auto-generated by latex/generate_local_refine_combined.py
        % Source data:
        %   - results/tuning/tuning_summary.mat
        %   - DQ-MPCC_baseline/tuning/best_weights_local.json
        %   - DQ-MPCC_baseline/tuning/tuning_history_local.json
        %   - MPCC_baseline/tuning/best_weights_local.json
        %   - MPCC_baseline/tuning/tuning_history_local.json
        % Do not edit manually.

        \\section*{{Combined Local Tuning Report}}

        {intro}

        {setup}

        {obj_coeffs}

        {dq_refs}

        {mpcc_refs}

        {dq_weights}

        {mpcc_weights}

        {perf}

        {notes}
        """
    ).strip() + "\n"


def _build_summary_text(dq_best: dict, dq_hist: dict, mpcc_best: dict, mpcc_hist: dict) -> str:
    return textwrap.dedent(
        f"""
        DQ local refinement: {dq_hist['n_trials']} {dq_hist['sampler'].upper()} trials over velocities {dq_hist['tuning_velocities']},
        tuning Q_phi/Q_ec/Q_el/U_mat/Q_omega/Q_s around manual gains with multiplicative window [0.75, 1.35].
        Best local objective = {dq_best['best_J_multi']:.3f} at trial #{dq_best['best_trial']}.

        MPCC local refinement: {mpcc_hist['n_trials']} {mpcc_hist['sampler'].upper()} trials over velocities {mpcc_hist['tuning_velocities']},
        tuning Q_q/Q_ec/Q_el/U_mat/Q_omega/Q_s around manual gains with multiplicative window [0.75, 1.35].
        Best local objective = {mpcc_best['best_J_multi']:.3f} at trial #{mpcc_best['best_trial']}.
        """
    ).strip() + "\n"


def _build_standalone_wrapper(section_name: str) -> str:
    return textwrap.dedent(
        rf"""
        \documentclass[11pt]{{article}}
        \usepackage[margin=0.85in]{{geometry}}
        \usepackage{{amsmath,amssymb,booktabs,bm,array,longtable,graphicx,float}}
        \usepackage[T1]{{fontenc}}
        \usepackage[utf8]{{inputenc}}
        \begin{{document}}
        \input{{{section_name}}}
        \end{{document}}
        """
    ).lstrip()


def _compile_standalone(tex_path: Path, pdf_path: Path) -> bool:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("[WARN] pdflatex not found in PATH; skipping PDF compilation.")
        return False

    wrapper = OUT_DIR / f"_compile_{tex_path.stem}.tex"
    wrapper.write_text(_build_standalone_wrapper(tex_path.name), encoding="utf-8")

    result = None
    for _ in range(2):
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", wrapper.name],
            cwd=OUT_DIR,
            capture_output=True,
            text=True,
        )

    generated_pdf = wrapper.with_suffix(".pdf")
    if generated_pdf.is_file():
        generated_pdf.replace(pdf_path)
        if result is not None and result.returncode != 0:
            print(f"[WARN] pdflatex returned {result.returncode}, but standalone PDF was generated.")
        print(f"[OK] Wrote {pdf_path}")
        return True

    if result is not None:
        print(f"[WARN] standalone PDF compilation failed with code {result.returncode}.")
        if result.stdout:
            print(result.stdout[-500:])
    return False


def _refresh_tuning_summary() -> None:
    subprocess.run(["python3", str(EXPORT_SCRIPT)], cwd=ROOT, check=True)


def main() -> None:
    _refresh_tuning_summary()
    payload = loadmat(MAT_PATH, squeeze_me=True)
    dq_best = _load_json(DQ_LOCAL_BEST)
    dq_hist = _load_json(DQ_LOCAL_HIST)
    mpcc_best = _load_json(MPCC_LOCAL_BEST)
    mpcc_hist = _load_json(MPCC_LOCAL_HIST)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    section_tex = _build_section_tex(payload, dq_best, dq_hist, mpcc_best, mpcc_hist)
    compact_tex = _build_compact_section_tex(payload, dq_best, dq_hist, mpcc_best, mpcc_hist)
    summary_text = _build_summary_text(dq_best, dq_hist, mpcc_best, mpcc_hist)

    SECTION_TEX.write_text(section_tex, encoding="utf-8")
    SUMMARY_TEX.write_text(summary_text, encoding="utf-8")
    COMPACT_TEX.write_text(compact_tex, encoding="utf-8")
    COMPACT_SUMMARY_TEX.write_text(summary_text, encoding="utf-8")

    print(f"[OK] Wrote {SECTION_TEX}")
    print(f"[OK] Wrote {SUMMARY_TEX}")
    print(f"[OK] Wrote {COMPACT_TEX}")
    print(f"[OK] Wrote {COMPACT_SUMMARY_TEX}")

    _compile_standalone(SECTION_TEX, STANDALONE_PDF)
    _compile_standalone(COMPACT_TEX, COMPACT_PDF)


if __name__ == "__main__":
    main()
