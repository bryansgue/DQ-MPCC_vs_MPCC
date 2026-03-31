#!/usr/bin/env python3
"""
generate_montecarlo.py — Auto-generate LaTeX section for Experiment 3.

Reads `results/experiment3/data/montecarlo_data.mat` and produces the
base/raw LaTeX file for Experiment 3 with:

  • Table I   — Monte Carlo summary (convergence, RMSE, solver time, effort)
  • Interpretive paragraphs with ALL numbers auto-filled

This script is intentionally the base/raw generation stage only.
Use `results/experiment3/reports/generate_montecarlo_verified.py` for the verification and LLM layers.

Usage:
    python results/experiment3/reports/generate_montecarlo.py               # → raw .tex + raw PDF + terminal summary
    python results/experiment3/reports/generate_montecarlo.py --no-compile # only generates raw .tex
"""

import os, sys, argparse, textwrap, shutil, subprocess
import numpy as np
from scipy.io import loadmat

# ═════════════════════════════════════════════════════════════════════════════
#  Paths & config
# ═════════════════════════════════════════════════════════════════════════════
_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT)))
_LATEX_SHARED = os.path.join(_ROOT, 'latex')

sys.path.insert(0, _ROOT)
sys.path.insert(0, _LATEX_SHARED)
from llm_verify_utils import (
    build_revision_trace_latex,
    generate_llm_verified_tex,
    generate_text_diff,
    llm_is_configured,
)
from config.result_paths import experiment_dirs

_EXP3_DIR = experiment_dirs('experiment3')
_MAT     = os.path.join(str(_EXP3_DIR['data']), 'montecarlo_data.mat')
_FIGURES_DIR = str(_EXP3_DIR['figures'])
_REPORTS_DIR = str(_EXP3_DIR['reports'])
_TEX_OUT = os.path.join(str(_EXP3_DIR['reports']), 'experiment3_analysis.tex')
_VERIFIED_TEX_OUT = os.path.join(str(_EXP3_DIR['reports']), 'experiment3_analysis_verified.tex')
_LLM_TEX_OUT = os.path.join(str(_EXP3_DIR['reports']), 'experiment3_analysis_llm.tex')
_LLM_DIFF_OUT = os.path.join(str(_EXP3_DIR['reports']), 'experiment3_analysis_llm.diff')
_COMPILE_DIR = str(_EXP3_DIR['compile'])

try:
    from config.montecarlo_config import (
        VELOCITY, N_RUNS, SIGMA_P, SIGMA_Q, S_MAX, T_FINAL, SEED,
        COMPLETION_RATIO,
    )
except ImportError:
    VELOCITY = 10; N_RUNS = 600; SIGMA_P = 2.0; SIGMA_Q = 0.5
    S_MAX = 80; T_FINAL = 40; SEED = 2026; COMPLETION_RATIO = 0.95


def _scalar(d, key, fallback):
    val = d.get(key, fallback)
    return float(np.atleast_1d(val).ravel()[0])


def _int_scalar(d, key, fallback):
    val = d.get(key, fallback)
    return int(np.atleast_1d(val).ravel()[0])


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def _get(d, ctrl, key):
    arr = d.get(f'{ctrl}_{key}', np.array([]))
    return np.atleast_1d(arr).astype(float)


def load_data():
    if not os.path.isfile(_MAT):
        print(f"[ERROR] {_MAT} not found. Run results/experiment3/scripts/run_experiment3.py first.")
        sys.exit(1)
    d = loadmat(_MAT, squeeze_me=True)
    return d


# ═════════════════════════════════════════════════════════════════════════════
#  Compute statistics
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(d):
    """Return a dict of summary statistics for each controller."""
    N = int(d['N_runs'])
    vel = float(d['velocity'])
    stats = {}

    for ctrl, tag in [('dq', 'DQ-MPCC'), ('base', 'Baseline MPCC')]:
        succ = _get(d, ctrl, 'success').astype(bool)
        n_ok = int(np.sum(succ))
        n_fail = N - n_ok
        conv_rate = n_ok / N * 100

        s = dict(tag=tag, n_ok=n_ok, n_fail=n_fail, conv_rate=conv_rate)

        for key in ['rmse_ec', 'rmse_el', 'rmse_pos', 'rmse_ori',
                     't_lap', 'mean_vtheta', 'ctrl_effort',
                     'solve_mean', 'solve_max', 'solve_std']:
            arr = _get(d, ctrl, key)
            if len(arr) > 0:
                s[f'{key}_mean'] = np.mean(arr)
                s[f'{key}_std']  = np.std(arr)
                s[f'{key}_med']  = np.median(arr)
                s[f'{key}_p5']   = np.percentile(arr, 5)
                s[f'{key}_p95']  = np.percentile(arr, 95)
            else:
                for suffix in ['_mean', '_std', '_med', '_p5', '_p95']:
                    s[f'{key}{suffix}'] = float('nan')

        stats[ctrl] = s

    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  LaTeX generation
# ═════════════════════════════════════════════════════════════════════════════

def _f(val, fmt='.4f'):
    """Format float, handle NaN."""
    if np.isnan(val):
        return '--'
    return f'{val:{fmt}}'


def _pct(dq_val, base_val):
    """Relative change string: e.g. '-12.3\\%'."""
    if np.isnan(dq_val) or np.isnan(base_val) or base_val == 0:
        return '--'
    delta = (dq_val - base_val) / base_val * 100
    sign = '+' if delta > 0 else ''
    return f'{sign}{delta:.1f}\\%'


def _latex_escape(text):
    return (text.replace('\\', r'\textbackslash{}')
                .replace('&', r'\&')
                .replace('%', r'\%')
                .replace('_', r'\_')
                .replace('#', r'\#'))


def _claim_title(claim):
    claim = claim.strip().rstrip('.')
    if len(claim) <= 70:
        return claim
    cut = claim[:67].rsplit(' ', 1)[0]
    return cut + '...'


def _build_verification_section(title, items):
    lines = [rf'\subsubsection{{{title}}}']
    for status, claim, detail in items:
        lines.append(
            rf'\subsubsection{{{_latex_escape(_claim_title(claim))}}}'
        )
        lines.append(r'\textbf{Status:} ' + _latex_escape(status) + r'\\')
        lines.append(_latex_escape(detail))
        lines.append('')
    return '\n'.join(lines)


def build_verification_items(d, stats):
    dq = stats['dq']
    ba = stats['base']
    dq_s = _get(d, 'dq', 'success').astype(bool)
    ba_s = _get(d, 'base', 'success').astype(bool)
    n_c = min(int(np.sum(dq_s)), int(np.sum(ba_s)))
    dq_ori_arr = _get(d, 'dq', 'rmse_ori')[:n_c]
    ba_ori_arr = _get(d, 'base', 'rmse_ori')[:n_c]
    ori_win = int((dq_ori_arr < ba_ori_arr).sum()) if n_c > 0 else 0

    items = []
    if dq['conv_rate'] >= 95.0 and ba['conv_rate'] >= 95.0:
        items.append((
            'Supported',
            'Both controllers achieve near-perfect convergence under the sampled perturbations.',
            f'DQ = {dq["conv_rate"]:.1f}% and Baseline = {ba["conv_rate"]:.1f}%.'
        ))
    else:
        items.append((
            'Qualified',
            'Robust convergence is present but should be described with the measured rates rather than near-perfect language.',
            f'DQ = {dq["conv_rate"]:.1f}% and Baseline = {ba["conv_rate"]:.1f}%.'
        ))

    if (dq['rmse_ori_mean'] < ba['rmse_ori_mean'] and
            dq['rmse_ori_med'] < ba['rmse_ori_med'] and
            ori_win > n_c / 2):
        items.append((
            'Supported',
            'DQ-MPCC preserves an orientation-tracking advantage in the Monte Carlo runs.',
            f'Orientation wins = {ori_win}/{n_c}, mean delta = {_pct(dq["rmse_ori_mean"], ba["rmse_ori_mean"])}.'
        ))
    else:
        items.append((
            'Qualified',
            'The orientation advantage is weaker than a blanket dominance claim and should be stated more cautiously.',
            f'Orientation wins = {ori_win}/{n_c}, mean delta = {_pct(dq["rmse_ori_mean"], ba["rmse_ori_mean"])}.'
        ))

    if dq['rmse_ec_mean'] > ba['rmse_ec_mean']:
        items.append((
            'Supported',
            'Large perturbations inflate contouring error for DQ-MPCC relative to the baseline in this dataset.',
            f'Mean contouring delta = {_pct(dq["rmse_ec_mean"], ba["rmse_ec_mean"])}.'
        ))
    else:
        items.append((
            'Contradicted',
            'The statement that the transient phase inflates DQ contouring error relative to baseline is not supported here.',
            f'Mean contouring delta = {_pct(dq["rmse_ec_mean"], ba["rmse_ec_mean"])}.'
        ))

    if dq['ctrl_effort_mean'] < ba['ctrl_effort_mean']:
        items.append((
            'Supported',
            'DQ-MPCC uses smoother actuation according to the accumulated control effort metric.',
            f'Control-effort delta = {_pct(dq["ctrl_effort_mean"], ba["ctrl_effort_mean"])}.'
        ))
    else:
        items.append((
            'Qualified',
            'Actuation smoothness is not better for DQ-MPCC in this run, so the text should avoid a stronger claim.',
            f'Control-effort delta = {_pct(dq["ctrl_effort_mean"], ba["ctrl_effort_mean"])}.'
        ))

    items.append((
        'External context',
        'The explicit cross-reference to Experiment 2 remains contextual rather than verified from this Monte Carlo dataset alone.',
        'The verified version should treat that comparison as supporting context, not direct evidence from Experiment 3.'
    ))
    return items


def build_evidence_summary(d, stats):
    dq = stats['dq']
    ba = stats['base']
    return {
        'velocity': float(d['velocity']),
        'n_runs': int(d['N_runs']),
        'dq': {
            'conv_rate': float(dq['conv_rate']),
            'rmse_ec_mean': float(dq['rmse_ec_mean']),
            'rmse_el_mean': float(dq['rmse_el_mean']),
            'rmse_pos_mean': float(dq['rmse_pos_mean']),
            'rmse_ori_mean': float(dq['rmse_ori_mean']),
            'ctrl_effort_mean': float(dq['ctrl_effort_mean']),
            'solve_mean_ms': float(dq['solve_mean_mean'] * 1e3),
        },
        'baseline': {
            'conv_rate': float(ba['conv_rate']),
            'rmse_ec_mean': float(ba['rmse_ec_mean']),
            'rmse_el_mean': float(ba['rmse_el_mean']),
            'rmse_pos_mean': float(ba['rmse_pos_mean']),
            'rmse_ori_mean': float(ba['rmse_ori_mean']),
            'ctrl_effort_mean': float(ba['ctrl_effort_mean']),
            'solve_mean_ms': float(ba['solve_mean_mean'] * 1e3),
        },
    }


def apply_verified_rewrites(tex, d, stats):
    dq = stats['dq']
    ba = stats['base']
    dq_s = _get(d, 'dq', 'success').astype(bool)
    ba_s = _get(d, 'base', 'success').astype(bool)
    n_c = min(int(np.sum(dq_s)), int(np.sum(ba_s)))
    dq_ori_arr = _get(d, 'dq', 'rmse_ori')[:n_c]
    ba_ori_arr = _get(d, 'base', 'rmse_ori')[:n_c]
    ori_win = int((dq_ori_arr < ba_ori_arr).sum()) if n_c > 0 else 0
    ori_supported = (dq['rmse_ori_mean'] < ba['rmse_ori_mean'] and
                     dq['rmse_ori_med'] < ba['rmse_ori_med'] and
                     ori_win > n_c / 2)

    if dq['conv_rate'] < 95.0 or ba['conv_rate'] < 95.0:
        tex = tex.replace(
            'Both controllers achieve near-perfect convergence despite large\n'
            'initial-pose perturbations ($\\sigma_p = {SIGMA_P}$~m,\n'
            '$\\sigma_q = {SIGMA_Q}$~rad), confirming reliable operation\n'
            'under uncertainty.'.format(SIGMA_P=SIGMA_P, SIGMA_Q=SIGMA_Q),
            'The measured convergence rates remain high despite large\n'
            'initial-pose perturbations ($\\sigma_p = {SIGMA_P}$~m,\n'
            '$\\sigma_q = {SIGMA_Q}$~rad), supporting robust operation\n'
            'under uncertainty without implying perfect convergence.'.format(
                SIGMA_P=SIGMA_P, SIGMA_Q=SIGMA_Q)
        )

    if not ori_supported:
        tex = tex.replace(
            'The orientation subplot confirms that DQ-MPCC consistently reduces\n'
            '$e_{\\mathrm{ori}}$: its median',
            'The orientation subplot indicates an orientation-tracking advantage\n'
            'for DQ-MPCC in this Monte Carlo sample: its median'
        )
        tex = tex.replace(
            'yet the orientation advantage is preserved',
            'while an orientation advantage is still observed'
        )

    tex = tex.replace(
        f'This trade-off mirrors the Pareto front in Experiment~2\n'
        f'(Fig.~\\ref{{fig:pareto}}): at $v = {float(d["velocity"]):.0f}$~m/s with nominal\n'
        f'initial conditions, DQ-MPCC achieves $e_c = {0.4340}$\n'
        f'($-4.0\\%$) and $e_{{\\mathrm{{ori}}}} = {0.7520}$\n'
        f'($-16.0\\%$) relative to the baseline.',
        f'As contextual support from Experiment~2\n'
        f'(Fig.~\\ref{{fig:pareto}}), the nominal-condition case at $v = {float(d["velocity"]):.0f}$~m/s\n'
        f'also reported lower orientation error for DQ-MPCC, although that\n'
        f'cross-experiment comparison is not itself verified from the present Monte Carlo dataset.'
    )
    tex = tex.replace(
        'Notably, the lag error $e_l$ shows a consistent trend across\n'
        'both experiments (Exp.~2: $+34.6\\%$, Exp.~3: {el_delta}),\n'
        'indicating that DQ-MPCC systematically trades\n'
        'progress speed for orientation alignment---a desirable\n'
        'property for applications requiring accurate attitude tracking.'.format(
            el_delta=_pct(dq['rmse_el_mean'], ba['rmse_el_mean'])
        ),
        'In this Monte Carlo dataset, the lag error $e_l$ remains a trade-off variable\n'
        '(Exp.~3: {el_delta}), which is consistent with the interpretation that\n'
        'DQ-MPCC may exchange progress speed for orientation alignment under\n'
        'large perturbations.'.format(
            el_delta=_pct(dq['rmse_el_mean'], ba['rmse_el_mean'])
        )
    )
    if dq['ctrl_effort_mean'] >= ba['ctrl_effort_mean']:
        tex = tex.replace(
            'indicating comparable\n'
            'actuation for the dual-quaternion formulation.',
            'indicating that the dual-quaternion formulation does not reduce\n'
            'actuation effort in this dataset.'
        )
    return tex


def generate_tex(d, stats):
    vel = _scalar(d, 'velocity', VELOCITY)
    N   = _int_scalar(d, 'N_runs', N_RUNS)
    sigma_p = _scalar(d, 'sigma_p', SIGMA_P)
    sigma_q = _scalar(d, 'sigma_q', SIGMA_Q)
    s_max = _scalar(d, 's_max', S_MAX if S_MAX is not None else np.nan)
    t_final = _scalar(d, 't_final', T_FINAL)
    completion_ratio = _scalar(d, 'completion_ratio', COMPLETION_RATIO)
    dq  = stats['dq']
    ba  = stats['base']

    lines = []
    L = lines.append

    L(r'% ═══════════════════════════════════════════════════════════════════')
    L(r'%  AUTO-GENERATED — do not edit manually')
    L(f'%  Source: latex/generate_montecarlo.py')
    L(r'% ═══════════════════════════════════════════════════════════════════')
    L('')
    L(r'\subsection{Experiment~3: Monte Carlo Robustness Evaluation}')
    L(r'\label{sec:experiment3}')
    L('')

    # ── Intro paragraph ──────────────────────────────────────────────────
    L(textwrap.dedent(f"""\
    To evaluate robustness under uncertain initial conditions, we perform
    a Monte Carlo study with $N = {N}$ trials at a fixed progress-velocity
    limit $v_{{\\theta}}^{{\\max}} = {vel:.0f}$~m/s.  Each trial samples
    a random initial pose: position is perturbed as
    $\\mathbf{{p}}_0 \\sim \\mathcal{{N}}(\\bar{{\\mathbf{{p}}}}_0,\\,
    \\sigma_p^2 \\mathbf{{I}}_3)$ with $\\sigma_p = {sigma_p}$~m,
    and orientation as a random rotation with
    $\\|\\mathrm{{Log}}(\\delta q)\\| \\sim U(0,\\, {sigma_q})$~rad
    ($\\approx {np.degrees(sigma_q):.0f}^\\circ$).
    The path length is $s_{{\\max}} = {s_max:.2f}$~m with a time budget
    of $T = {t_final:.0f}$~s.
    A trial is deemed \\emph{{successful}} if the progress variable reaches
    $\\theta \\geq {completion_ratio*100:.0f}\\%$ of $s_{{\\max}}$."""))
    L('')

    # ── Table ────────────────────────────────────────────────────────────
    L(r'\begin{table}[!t]')
    L(r'  \centering')
    L(r'  \caption{Monte Carlo robustness comparison '
      f'($v = {vel:.0f}$~m/s, $N = {N}$).}}')
    L(r'  \label{tab:montecarlo}')
    L(r'  \setlength{\tabcolsep}{4pt}')
    L(r'  \begin{tabular}{l c c c}')
    L(r'    \toprule')
    L(r'    \textbf{Metric} & \textbf{DQ-MPCC} & \textbf{Baseline} & $\boldsymbol{\Delta}$ \\')
    L(r'    \midrule')

    # Convergence
    L(f'    Conv. rate [\\%]'
      f' & {dq["conv_rate"]:.1f}'
      f' & {ba["conv_rate"]:.1f}'
      f' & {_pct(dq["conv_rate"], ba["conv_rate"])}'
      r' \\')

    # RMSE metrics (mean ± std)
    for key, label in [
        ('rmse_ec',  r'RMSE $e_c$'),
        ('rmse_el',  r'RMSE $e_l$'),
        ('rmse_pos', r'RMSE $e_{\mathrm{pos}}$'),
        ('rmse_ori', r'RMSE $e_{\mathrm{ori}}$'),
    ]:
        dq_m, dq_s = dq[f'{key}_mean'], dq[f'{key}_std']
        ba_m, ba_s = ba[f'{key}_mean'], ba[f'{key}_std']
        L(f'    {label}'
          f' & ${_f(dq_m)} \\pm {_f(dq_s)}$'
          f' & ${_f(ba_m)} \\pm {_f(ba_s)}$'
          f' & {_pct(dq_m, ba_m)}'
          r' \\')

    L(r'    \midrule')

    # Solver time
    dq_solve = dq['solve_mean_mean'] * 1e3
    ba_solve = ba['solve_mean_mean'] * 1e3
    dq_solve_s = dq['solve_mean_std'] * 1e3
    ba_solve_s = ba['solve_mean_std'] * 1e3
    L(f'    Solver time [ms]'
      f' & ${_f(dq_solve, ".2f")} \\pm {_f(dq_solve_s, ".2f")}$'
      f' & ${_f(ba_solve, ".2f")} \\pm {_f(ba_solve_s, ".2f")}$'
      f' & {_pct(dq_solve, ba_solve)}'
      r' \\')

    # Control effort
    dq_ce = dq['ctrl_effort_mean']
    ba_ce = ba['ctrl_effort_mean']
    dq_ce_s = dq['ctrl_effort_std']
    ba_ce_s = ba['ctrl_effort_std']
    L(f'    Ctrl. effort'
      f' & ${_f(dq_ce, ".1f")} \\pm {_f(dq_ce_s, ".1f")}$'
      f' & ${_f(ba_ce, ".1f")} \\pm {_f(ba_ce_s, ".1f")}$'
      f' & {_pct(dq_ce, ba_ce)}'
      r' \\')

    # Lap time
    dq_tl = dq['t_lap_mean']
    ba_tl = ba['t_lap_mean']
    L(f'    Lap time [s]'
      f' & ${_f(dq_tl, ".2f")} \\pm {_f(dq["t_lap_std"], ".2f")}$'
      f' & ${_f(ba_tl, ".2f")} \\pm {_f(ba["t_lap_std"], ".2f")}$'
      f' & {_pct(dq_tl, ba_tl)}'
      r' \\')

    L(r'    \bottomrule')
    L(r'  \end{tabular}')
    L(r'\end{table}')
    L('')

    # ── Interpretation paragraphs ────────────────────────────────────────
    # Convergence
    if dq['conv_rate'] >= ba['conv_rate']:
        conv_winner = 'DQ-MPCC'
        conv_note = (f'DQ-MPCC achieves a convergence rate of '
                     f'{dq["conv_rate"]:.1f}\\%, compared to '
                     f'{ba["conv_rate"]:.1f}\\% for the baseline.')
    else:
        conv_winner = 'Baseline'
        conv_note = (f'The baseline achieves a convergence rate of '
                     f'{ba["conv_rate"]:.1f}\\%, compared to '
                     f'{dq["conv_rate"]:.1f}\\% for DQ-MPCC.')

    L(textwrap.dedent(f"""\
    \\paragraph{{Convergence}}
    {conv_note}
    Both controllers achieve near-perfect convergence despite large
    initial-pose perturbations ($\\sigma_p = {sigma_p}$~m,
    $\\sigma_q = {sigma_q}$~rad), confirming reliable operation
    under uncertainty."""))
    L('')

    # ── Robustness figure ────────────────────────────────────────────────
    L(textwrap.dedent("""\
    \\begin{figure}[!t]
      \\centering
      \\includegraphics[width=\\columnwidth]{fig_mc_robustness.pdf}
      \\caption{Distribution of per-run RMSE across the Monte Carlo trials
               with random initial poses ($\\sigma_p = """ + f"""{sigma_p}""" + r"""$~m,
               $\\sigma_q = """ + f"""{sigma_q}""" + r"""$~rad) at $v_{\theta}^{\max} = """ + f"""{vel:.0f}""" + r"""$~m/s.
               Violin contours show probability density; individual points
               overlay each trial.  Median values are annotated.}
      \\label{fig:mc_robustness}
    \\end{figure}"""))
    L('')

    # Tracking accuracy + Pareto cross-reference
    ec_delta = _pct(dq['rmse_ec_mean'], ba['rmse_ec_mean'])
    el_delta = _pct(dq['rmse_el_mean'], ba['rmse_el_mean'])
    ori_delta = _pct(dq['rmse_ori_mean'], ba['rmse_ori_mean'])
    pos_delta = _pct(dq['rmse_pos_mean'], ba['rmse_pos_mean'])

    # Compute per-run win counts (on common subset of successful trials)
    dq_s  = _get(d, 'dq',  'success').astype(bool)
    ba_s  = _get(d, 'base', 'success').astype(bool)
    n_c   = min(int(np.sum(dq_s)), int(np.sum(ba_s)))
    dq_ec_arr  = _get(d, 'dq',  'rmse_ec')[:n_c]
    ba_ec_arr  = _get(d, 'base', 'rmse_ec')[:n_c]
    dq_ori_arr = _get(d, 'dq',  'rmse_ori')[:n_c]
    ba_ori_arr = _get(d, 'base', 'rmse_ori')[:n_c]
    ec_win  = int((dq_ec_arr  < ba_ec_arr).sum())
    ori_win = int((dq_ori_arr < ba_ori_arr).sum())

    L(textwrap.dedent(f"""\
    \\paragraph{{Tracking accuracy and Pareto consistency}}
    Fig.~\\ref{{fig:mc_robustness}} displays the error distributions
    for all four RMSE metrics.
    The orientation subplot shows that DQ-MPCC reduces
    $e_{{\\mathrm{{ori}}}}$: its median
    (${_f(dq['rmse_ori_med'])})$ lies well below the baseline median
    (${_f(ba['rmse_ori_med'])})$, with DQ-MPCC winning in
    {ori_win}\\,/\\,{n_c} individual trials
    ($\\Delta\\bar{{e}}_{{\\mathrm{{ori}}}} = {ori_delta}$).
    Crucially, the baseline's $e_{{\\mathrm{{ori}}}}$ distribution
    is tightly concentrated ($\\sigma = {_f(ba['rmse_ori_std'])}$),
    while DQ-MPCC's wider spread
    ($\\sigma = {_f(dq['rmse_ori_std'])}$) reflects a
    pose-dependent adaptation enabled by the Lie-algebraic coupling
    between translation and rotation.
    Under large pose perturbations ($\\sigma_p = {sigma_p}$~m),
    the transient convergence phase inflates the contouring error
    to $\\bar{{e}}_c = {_f(dq['rmse_ec_mean'])}$ ({ec_delta}),
    while the orientation advantage is preserved ({ori_delta}).
    The lag error $e_l$ remains a trade-off variable in this Monte Carlo
    dataset ({el_delta}), which should be interpreted together with the
    improved orientation tracking rather than as an isolated metric."""))
    L('')

    # Computational cost
    solve_delta = _pct(dq_solve, ba_solve)
    L(textwrap.dedent(f"""\
    \\paragraph{{Computational cost}}
    The mean solver time per MPC step is
    ${_f(dq_solve, ".2f")}$~ms for DQ-MPCC and
    ${_f(ba_solve, ".2f")}$~ms for the baseline ({solve_delta}).
    Both remain well within the ${1000/100:.0f}$~ms real-time budget
    ($f = {100}$~Hz)."""))
    L('')

    # Control effort
    ce_delta = _pct(dq_ce, ba_ce)
    L(textwrap.dedent(f"""\
    \\paragraph{{Control effort}}
    The accumulated control-rate effort
    $\\sum\\|\\Delta \\mathbf{{u}}\\|^2$ averages
    ${_f(dq_ce, ".1f")}$ for DQ-MPCC and
    ${_f(ba_ce, ".1f")}$ for the baseline ({ce_delta}),
    indicating {'smoother' if dq_ce < ba_ce else 'comparable'}
    actuation for the dual-quaternion formulation."""))
    L('')

    # ── 3D trajectory figure ─────────────────────────────────────────────
    L(textwrap.dedent(f"""\
    \\begin{{figure}}[!t]
      \\centering
      \\includegraphics[width=\\columnwidth]{{fig_mc_3d.pdf}}
      \\caption{{Isometric view of all {N} Monte Carlo trajectories
               (colored) overlaid on the reference path (black) at
               $v_{{\\theta}}^{{\\max}} = {vel:.0f}$~m/s.
               Left: Baseline MPCC.  Right: DQ-MPCC.
               Start positions (dots) reflect the random perturbation
               $\\sigma_p = {sigma_p}$~m.  Both controllers converge
               to the reference, with DQ-MPCC exhibiting tighter
               spatial clustering in the steady-state regime.}}
      \\label{{fig:mc_3d}}
    \\end{{figure}}"""))
    L('')

    return '\n'.join(lines)


def generate_verified_tex(d, stats):
    base_tex = apply_verified_rewrites(generate_tex(d, stats), d, stats)
    verification = _build_verification_section(
        'Verified Claim Audit',
        build_verification_items(d, stats),
    )
    return base_tex + '\n' + verification


def compile_pdf(tex_name, pdf_name):
    pdflatex = shutil.which('pdflatex')
    if pdflatex is None:
        print('[WARN] pdflatex not found in PATH; skipping PDF compilation.')
        return False

    tex_path = tex_name if os.path.isabs(tex_name) else os.path.join(_REPORTS_DIR, tex_name)
    tex_rel = os.path.relpath(tex_path, _COMPILE_DIR).replace(os.sep, '/')
    figures_rel = os.path.relpath(_FIGURES_DIR, _COMPILE_DIR).replace(os.sep, '/')
    wrapper_name = f'_compile_{pdf_name}.tex'
    wrapper = os.path.join(_COMPILE_DIR, wrapper_name)
    with open(wrapper, 'w') as f:
        f.write(textwrap.dedent(r"""
            \documentclass[11pt]{article}
            \usepackage[margin=1in]{geometry}
            \usepackage{amsmath,amssymb,booktabs,graphicx}
            \begin{document}
            \graphicspath{{""" + figures_rel + r"""/}}
            \input{""" + tex_rel + r"""}
            \end{document}
        """).lstrip())

    print(f'\n[INFO] Compiling PDF with pdflatex -> {pdf_name}.pdf ...')
    result = None
    for _ in range(2):
        result = subprocess.run(
            [pdflatex, '-interaction=nonstopmode', wrapper_name],
            cwd=_COMPILE_DIR,
            capture_output=True,
            text=True,
        )

    pdf = os.path.join(_REPORTS_DIR, f'{pdf_name}.pdf')
    generated_pdf = os.path.join(_COMPILE_DIR, wrapper_name.replace('.tex', '.pdf'))
    if os.path.isfile(generated_pdf) and generated_pdf != pdf:
        os.replace(generated_pdf, pdf)
    if os.path.isfile(pdf):
        if result is not None and result.returncode != 0:
            print(f'[WARN] pdflatex returned {result.returncode}, but PDF was generated.')
        print(f'✓  {pdf}')
        return True

    code = result.returncode if result is not None else 'unknown'
    print(f'[WARN] pdflatex failed with exit code {code}.')
    if result is not None and result.stdout:
        print(result.stdout[-500:])
    if result is not None and result.stderr:
        print(result.stderr[-500:])
    return False


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-compile', action='store_true',
                        help='Skip PDF compilation after generating .tex')
    args = parser.parse_args()

    print(f'[INFO] Loading Monte Carlo data ...')
    d = load_data()
    stats = compute_stats(d)

    # ── Terminal summary ─────────────────────────────────────────────────
    N = int(d['N_runs'])
    vel = float(d['velocity'])
    print(f'\n{"="*70}')
    print(f'  Monte Carlo:  v = {vel} m/s,  N = {N}')
    print(f'{"="*70}')
    for ctrl, tag in [('dq', 'DQ-MPCC'), ('base', 'Baseline')]:
        s = stats[ctrl]
        print(f'\n  [{tag}]  conv={s["conv_rate"]:.1f}%  '
              f'ec={_f(s["rmse_ec_mean"])}  '
              f'el={_f(s["rmse_el_mean"])}  '
              f'pos={_f(s["rmse_pos_mean"])}  '
              f'ori={_f(s["rmse_ori_mean"])}  '
              f'solve={s["solve_mean_mean"]*1e3:.2f}ms  '
              f'effort={s["ctrl_effort_mean"]:.1f}')
    print()

    # ── Generate LaTeX ───────────────────────────────────────────────────
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    tex = generate_tex(d, stats)
    with open(_TEX_OUT, 'w') as f:
        f.write(tex)
    print(f'[INFO] Generating {_TEX_OUT} ...')
    print(f'✓  {_TEX_OUT}')
    print(f'   ({len(tex)} chars, {tex.count(chr(10))+1} lines)')

    if not args.no_compile:
        compile_pdf('experiment3_analysis.tex', 'experiment3_analysis')

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
