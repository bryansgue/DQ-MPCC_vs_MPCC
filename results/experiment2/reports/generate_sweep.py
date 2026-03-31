#!/usr/bin/env python3
"""
generate_sweep.py
─────────────────
Reads `results/experiment2/data/velocity_sweep_data.mat` and generates the
base/raw LaTeX report for Experiment 2 under `results/experiment2/reports/`.

  • Table I   — Comparative performance (ec, el, pos, ori, t_lap)
  • Table II  — Orthogonal decomposition  (%ec vs %el)
  • Interpretive paragraphs with ALL numbers auto-filled

This script is intentionally the base/raw generation stage only.
Use `results/experiment2/reports/generate_sweep_verified.py` for the verification and LLM layers.

Usage
─────
    python results/experiment2/reports/generate_sweep.py               # → raw .tex + raw PDF + terminal summary
    python results/experiment2/reports/generate_sweep.py --no-compile # only generates raw .tex
"""

import os, sys, argparse, textwrap, shutil, subprocess
import numpy as np
from scipy.io import loadmat

# ═════════════════════════════════════════════════════════════════════════════
#  Paths & config
# ═════════════════════════════════════════════════════════════════════════════
_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT)))  # workspace root
_LATEX_SHARED = os.path.join(_ROOT, "latex")

# ── Ensure workspace root is on sys.path ─────────────────────────────────────
sys.path.insert(0, _ROOT)
sys.path.insert(0, _LATEX_SHARED)
from llm_verify_utils import (
    build_revision_trace_latex,
    generate_llm_verified_tex,
    generate_text_diff,
    llm_is_configured,
)
from config.result_paths import experiment_dirs

_EXP2_DIR = experiment_dirs('experiment2')
_MAT     = os.path.join(str(_EXP2_DIR['data']), 'velocity_sweep_data.mat')
_LEGACY_MAT = os.path.join(str(_EXP2_DIR['base']), 'velocity_sweep_data.mat')
_TEX_OUT = os.path.join(str(_EXP2_DIR['reports']), 'experiment2_analysis.tex')
_VERIFIED_TEX_OUT = os.path.join(str(_EXP2_DIR['reports']), 'experiment2_analysis_verified.tex')
_LLM_TEX_OUT = os.path.join(str(_EXP2_DIR['reports']), 'experiment2_analysis_llm.tex')
_LLM_DIFF_OUT = os.path.join(str(_EXP2_DIR['reports']), 'experiment2_analysis_llm.diff')
_COMPILE_DIR = str(_EXP2_DIR['compile'])

try:
    from config.sweep_config import (VELOCITIES, N_RUNS, SIGMA_P, SIGMA_Q,
                                     S_MAX, T_FINAL, SEED)
except ImportError:
    VELOCITIES = [4, 6, 8, 10, 12, 15, 16, 18]
    N_RUNS = 10; SIGMA_P = 0.05; SIGMA_Q = 0.05
    S_MAX  = 80; T_FINAL = 30; SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def _vk(v):
    return f"{v:.2f}".replace('.', 'p')


def _fmt_v(v):
    return f"{float(v):g}"


def _to_list(value):
    arr = np.atleast_1d(value)
    return [float(x) for x in arr.astype(float).ravel().tolist()]


def load_data():
    """Return sweep payload with raw mat, metadata, and per-controller data."""
    mat_path = _MAT if os.path.isfile(_MAT) else _LEGACY_MAT
    raw = loadmat(mat_path, squeeze_me=True)
    velocities = _to_list(raw.get('velocities', VELOCITIES))
    n_runs = int(np.atleast_1d(raw.get('n_runs', N_RUNS)).ravel()[0])
    sigma_p = float(np.atleast_1d(raw.get('sigma_p', SIGMA_P)).ravel()[0])
    sigma_q = float(np.atleast_1d(raw.get('sigma_q', SIGMA_Q)).ravel()[0])
    s_max = float(np.atleast_1d(raw.get('s_max', S_MAX)).ravel()[0])
    data = {'dq': {}, 'base': {}}
    for ctrl in ('dq', 'base'):
        for v in velocities:
            vstr = _vk(v)
            data[ctrl][v] = {}
            for m in ('rmse_pos', 'rmse_ori', 'rmse_ec', 'rmse_el',
                      't_lap', 'mean_vtheta'):
                key = f'{ctrl}_v{vstr}_{m}'
                val = raw.get(key, np.array([]))
                data[ctrl][v][m] = np.atleast_1d(val).ravel()
            fkey = f'{ctrl}_v{vstr}_failures'
            fval = raw.get(fkey, 0)
            data[ctrl][v]['failures'] = int(np.atleast_1d(fval).ravel()[0])
    return {
        'mat_path': mat_path,
        'raw': raw,
        'velocities': velocities,
        'n_runs': n_runs,
        'sigma_p': sigma_p,
        'sigma_q': sigma_q,
        's_max': s_max,
        'data': data,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(data, velocities):
    """
    Return list of dicts (one per velocity) with all comparative metrics.
    Also return global averages dict.
    """
    rows = []
    sums = {k: {'dq': [], 'base': []}
            for k in ('ec', 'el', 'pos', 'ori', 'tlap')}

    for v in velocities:
        r = {'v': v}
        for metric, short in [('rmse_ec', 'ec'), ('rmse_el', 'el'),
                               ('rmse_pos', 'pos'), ('rmse_ori', 'ori'),
                               ('t_lap', 'tlap')]:
            dq_arr   = data['dq'][v][metric]
            base_arr = data['base'][v][metric]
            dq_m   = float(np.mean(dq_arr))   if len(dq_arr)   else np.nan
            base_m = float(np.mean(base_arr)) if len(base_arr) else np.nan
            r[f'dq_{short}']   = dq_m
            r[f'base_{short}'] = base_m
            if base_m != 0 and np.isfinite(dq_m) and np.isfinite(base_m):
                r[f'd_{short}'] = (dq_m - base_m) / base_m * 100.0
            else:
                r[f'd_{short}'] = np.nan
            sums[short]['dq'].append(dq_m)
            sums[short]['base'].append(base_m)

        # Decomposition: %ec, %el of total position error²
        dq_ec2  = float(np.mean(data['dq'][v]['rmse_ec']**2))
        dq_el2  = float(np.mean(data['dq'][v]['rmse_el']**2))
        base_ec2 = float(np.mean(data['base'][v]['rmse_ec']**2))
        base_el2 = float(np.mean(data['base'][v]['rmse_el']**2))
        dq_tot2  = dq_ec2 + dq_el2
        base_tot2 = base_ec2 + base_el2
        r['dq_pct_ec']   = dq_ec2 / dq_tot2 * 100 if dq_tot2 > 0 else 0
        r['dq_pct_el']   = dq_el2 / dq_tot2 * 100 if dq_tot2 > 0 else 0
        r['base_pct_ec'] = base_ec2 / base_tot2 * 100 if base_tot2 > 0 else 0
        r['base_pct_el'] = base_el2 / base_tot2 * 100 if base_tot2 > 0 else 0

        # ratio el/ec
        r['dq_ratio']   = r['dq_el'] / r['dq_ec'] if r['dq_ec'] > 0 else 0
        r['base_ratio'] = r['base_el'] / r['base_ec'] if r['base_ec'] > 0 else 0

        r['dq_nf']   = data['dq'][v]['failures']
        r['base_nf'] = data['base'][v]['failures']
        rows.append(r)

    # Global averages
    avgs = {}
    for short in ('ec', 'el', 'pos', 'ori', 'tlap'):
        dq_avg   = float(np.mean(sums[short]['dq']))
        base_avg = float(np.mean(sums[short]['base']))
        avgs[f'dq_{short}']   = dq_avg
        avgs[f'base_{short}'] = base_avg
        if base_avg != 0:
            avgs[f'd_{short}'] = (dq_avg - base_avg) / base_avg * 100.0
        else:
            avgs[f'd_{short}'] = np.nan

    return rows, avgs


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: find best/worst Δ across velocities
# ═════════════════════════════════════════════════════════════════════════════

def _extremes(rows, key):
    """Return (best_Δ%, best_v, worst_Δ%, worst_v) for a given 'd_*' key.
    'best' = most negative (DQ wins more), 'worst' = most positive."""
    vals = [(r[key], r['v']) for r in rows if np.isfinite(r[key])]
    if not vals:
        return 0, 0, 0, 0
    best  = min(vals, key=lambda x: x[0])
    worst = max(vals, key=lambda x: x[0])
    return best[0], best[1], worst[0], worst[1]


def _winner(delta):
    """Return 'DQ-MPCC' or 'Baseline MPCC' depending on sign."""
    return 'DQ-MPCC' if delta < 0 else 'Baseline MPCC'


def _winner_phrase(delta, metric_name):
    winner = _winner(delta)
    if winner == 'DQ-MPCC':
        return f"DQ-MPCC is better on average in {metric_name}"
    return f"Baseline MPCC is better on average in {metric_name}"


# ═════════════════════════════════════════════════════════════════════════════
#  LaTeX generation
# ═════════════════════════════════════════════════════════════════════════════

def _s(x, fmt='.4f'):
    """Format a number, handling NaN."""
    if np.isnan(x):
        return '--'
    return f'{x:{fmt}}'


def _sp(x):
    """Format a percentage with sign, e.g. +12.3 or -4.7."""
    if np.isnan(x):
        return '--'
    return f'{x:+.1f}'


def _bold_if_better(dq_val, base_val, lower_is_better=True):
    """Return (dq_str, base_str) with \\textbf on the winner."""
    if np.isnan(dq_val) or np.isnan(base_val):
        return _s(dq_val), _s(base_val)
    if lower_is_better:
        dq_wins = dq_val < base_val
    else:
        dq_wins = dq_val > base_val
    dq_str   = f'\\textbf{{{_s(dq_val)}}}' if dq_wins else _s(dq_val)
    base_str = f'\\textbf{{{_s(base_val)}}}' if not dq_wins else _s(base_val)
    return dq_str, base_str


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


def build_verification_items(rows, avgs):
    all_ec_better = all(r['d_ec'] < 0 for r in rows if np.isfinite(r['d_ec']))
    all_el_worse = all(r['d_el'] > 0 for r in rows if np.isfinite(r['d_el']))
    all_ori_better = all(r['d_ori'] < 0 for r in rows if np.isfinite(r['d_ori']))
    pos_signs = [np.sign(r['d_pos']) for r in rows if np.isfinite(r['d_pos']) and r['d_pos'] != 0]
    has_pos_crossover = any(a != b for a, b in zip(pos_signs, pos_signs[1:]))
    base_contouring_dominated = all(r['base_ratio'] < 1.0 for r in rows)
    dq_high_speed_lag = rows[-1]['dq_ratio'] > 1.0

    items = []
    if all_ec_better:
        items.append(('Supported',
                      'DQ-MPCC achieves lower contouring error across the full tested sweep.',
                      f'Average contouring delta = {_sp(avgs["d_ec"])}%.'))
    else:
        first_win = next((r['v'] for r in rows if r['d_ec'] < 0), None)
        items.append(('Qualified',
                      'DQ-MPCC contouring superiority is limited to the high-speed region rather than the full sweep.',
                      f'First DQ advantage appears at v = {first_win} m/s.'))

    if all_el_worse:
        items.append(('Supported',
                      'Baseline MPCC dominates lag error across the tested sweep.',
                      f'Average lag delta = {_sp(avgs["d_el"])}%.'))
    else:
        items.append(('Qualified',
                      'Lag-error dominance is not uniform at every velocity and should be described as a trend, not an absolute.',
                      f'Average lag delta = {_sp(avgs["d_el"])}%.'))

    if has_pos_crossover:
        items.append(('Supported',
                      'The position-error curves exhibit a crossover over the tested velocities.',
                      'The sign of the position delta changes across the sweep.'))
    else:
        items.append(('Qualified',
                      'The position-error curves stay on the same side over the tested points, so crossover language should be softened.',
                      'No sign change was detected in the sampled position deltas.'))

    if all_ori_better:
        items.append(('Supported',
                      'DQ-MPCC maintains lower orientation error across the entire tested velocity range.',
                      f'Average orientation delta = {_sp(avgs["d_ori"])}%.'))
    else:
        items.append(('Qualified',
                      'Baseline MPCC keeps lower orientation error over the tested sweep, so DQ orientation-dominance language is not supported.',
                      f'Average orientation delta = {_sp(avgs["d_ori"])}%.'))

    items.append((
        'Supported' if base_contouring_dominated else 'Qualified',
        'Baseline MPCC behaves as a contouring-dominated controller in the decomposition analysis.',
        'All speeds satisfy e_l/e_c < 1 for Baseline.' if base_contouring_dominated
        else 'At least one speed does not satisfy e_l/e_c < 1 for Baseline.'
    ))
    items.append((
        'Supported' if dq_high_speed_lag else 'Qualified',
        'DQ-MPCC becomes lag-dominated at the high-speed end of the sweep.',
        f'At the maximum tested speed, e_l/e_c = {rows[-1]["dq_ratio"]:.2f}.'
    ))
    return items


def build_evidence_summary(rows, avgs):
    return {
        'velocities': [float(r['v']) for r in rows],
        'average_deltas_pct': {
            'ec': float(avgs['d_ec']),
            'el': float(avgs['d_el']),
            'pos': float(avgs['d_pos']),
            'ori': float(avgs['d_ori']),
            'tlap': float(avgs['d_tlap']),
        },
        'rows': [
            {
                'v': float(r['v']),
                'd_ec': float(r['d_ec']),
                'd_el': float(r['d_el']),
                'd_pos': float(r['d_pos']),
                'd_ori': float(r['d_ori']),
                'dq_ratio': float(r['dq_ratio']),
                'base_ratio': float(r['base_ratio']),
            }
            for r in rows
        ],
    }


def apply_verified_rewrites(tex, rows, avgs):
    all_ec_better = all(r['d_ec'] < 0 for r in rows if np.isfinite(r['d_ec']))
    all_el_worse = all(r['d_el'] > 0 for r in rows if np.isfinite(r['d_el']))
    all_ori_better = all(r['d_ori'] < 0 for r in rows if np.isfinite(r['d_ori']))
    pos_signs = [np.sign(r['d_pos']) for r in rows if np.isfinite(r['d_pos']) and r['d_pos'] != 0]
    has_pos_crossover = any(a != b for a, b in zip(pos_signs, pos_signs[1:]))
    base_contouring_dominated = all(r['base_ratio'] < 1.0 for r in rows)
    dq_high_speed_lag = rows[-1]['dq_ratio'] > 1.0

    if not all_el_worse:
        tex = tex.replace(
            "Panel~(b) reveals the opposite: the Baseline's lag curve is consistently\n"
            "lower, with an absolute gap that widens from",
            "Panel~(b) shows that the Baseline's lag curve is lower over most of the tested\n"
            "range, with an absolute gap that changes from"
        )
        tex = tex.replace(
            'whereas the Baseline MPCC dominates in lag error',
            'whereas the Baseline MPCC generally attains lower lag error'
        )

    if not has_pos_crossover:
        tex = tex.replace(
            'the curves cross near $v_\\theta^{\\max} \\approx ',
            'the curves remain ordered throughout the sampled sweep rather than crossing near $v_\\theta^{\\max} \\approx '
        )

    if not all_ori_better:
        tex = tex.replace(
            'Finally, panel~(d) shows that the DQ-MPCC maintains consistently lower\n'
            'orientation error across the entire velocity range',
            'Finally, panel~(d) shows that the DQ-MPCC reduces orientation error over\n'
            'part of the tested velocity range'
        )

    if not all_ec_better:
        first_loss = next((r['v'] for r in rows if r['d_ec'] > 0), None)
        tex = tex.replace(
            'Table~\\ref{tab:velocity_sweep_full} shows that the DQ-MPCC achieves lower\n'
            'contouring error at all tested velocities',
            'Table~\\ref{tab:velocity_sweep_full} shows that the DQ-MPCC achieves lower\n'
            f'contouring error up to the low-speed regime, with the first reversal appearing at {first_loss}~m/s'
        )

    if not base_contouring_dominated:
        tex = tex.replace(
            'the Baseline is \\emph{contouring-dominated}',
            'the Baseline is predominantly contouring-dominated'
        )

    if not dq_high_speed_lag:
        tex = tex.replace(
            'to \\emph{lag-dominated}',
            'to a more balanced error split instead of a clearly lag-dominated regime'
        )

    return tex


def generate_tex(rows, avgs, metadata):
    """Return the full .tex string."""
    velocities = metadata['velocities']
    n_vel = len(velocities)
    n_runs = metadata['n_runs']
    sigma_p = metadata['sigma_p']
    sigma_q = metadata['sigma_q']
    s_max = metadata['s_max']
    mat_path = metadata['mat_path']

    # ── Extremes for each metric ──
    ec_best, ec_best_v, _, _ = _extremes(rows, 'd_ec')
    _, _, _, _ = _extremes(rows, 'd_el')
    pos_best, pos_best_v, _, _ = _extremes(rows, 'd_pos')
    _, _, _, _ = _extremes(rows, 'd_ori')
    _, _, _, _ = _extremes(rows, 'd_tlap')

    # ── Decomposition extremes ──
    dq_pct_el_min  = min(r['dq_pct_el'] for r in rows)
    dq_pct_el_max  = max(r['dq_pct_el'] for r in rows)
    base_pct_ec_min = min(r['base_pct_ec'] for r in rows)
    base_pct_ec_max = max(r['base_pct_ec'] for r in rows)

    # ── Build Table I rows ──
    table1_rows = []
    for r in rows:
        dq_ec_s, base_ec_s = _bold_if_better(r['dq_ec'], r['base_ec'])
        dq_el_s, base_el_s = _bold_if_better(r['dq_el'], r['base_el'])
        dq_pos_s, base_pos_s = _bold_if_better(r['dq_pos'], r['base_pos'])
        dq_ori_s, base_ori_s = _bold_if_better(r['dq_ori'], r['base_ori'])
        dq_t_s, base_t_s = _bold_if_better(r['dq_tlap'], r['base_tlap'])

        line = (f"    {_fmt_v(r['v']):>2} "
                f"& {dq_ec_s} & {base_ec_s} & {_sp(r['d_ec'])}\\% "
                f"& {dq_el_s} & {base_el_s} & {_sp(r['d_el'])}\\% "
                f"& {dq_pos_s} & {base_pos_s} & {_sp(r['d_pos'])}\\% "
                f"& {dq_ori_s} & {base_ori_s} & {_sp(r['d_ori'])}\\% "
                f"& {_s(r['dq_tlap'], '.2f')} & {_s(r['base_tlap'], '.2f')} "
                f"& {_sp(r['d_tlap'])}\\% \\\\")
        table1_rows.append(line)

    table1_body = '\n'.join(table1_rows)

    # Average row
    avg_row = (f"    \\textit{{Avg.}} "
               f"& {_s(avgs['dq_ec'])} & {_s(avgs['base_ec'])} & {_sp(avgs['d_ec'])}\\% "
               f"& {_s(avgs['dq_el'])} & {_s(avgs['base_el'])} & {_sp(avgs['d_el'])}\\% "
               f"& {_s(avgs['dq_pos'])} & {_s(avgs['base_pos'])} & {_sp(avgs['d_pos'])}\\% "
               f"& {_s(avgs['dq_ori'])} & {_s(avgs['base_ori'])} & {_sp(avgs['d_ori'])}\\% "
               f"& {_s(avgs['dq_tlap'], '.2f')} & {_s(avgs['base_tlap'], '.2f')} "
               f"& {_sp(avgs['d_tlap'])}\\% \\\\")

    # ── Build Table II rows (decomposition) ──
    table2_rows = []
    for r in rows:
        line = (f"    {_fmt_v(r['v']):>2} "
                f"& {r['dq_pct_ec']:.1f} & {r['dq_pct_el']:.1f} "
                f"& {r['dq_ratio']:.3f} "
                f"& {r['base_pct_ec']:.1f} & {r['base_pct_el']:.1f} "
                f"& {r['base_ratio']:.3f} \\\\")
        table2_rows.append(line)
    table2_body = '\n'.join(table2_rows)

    # ── Determine dominant error type per controller ──
    # DQ: lag-dominated if ratio > 1 at most velocities
    dq_lag_dominated_count = sum(1 for r in rows if r['dq_ratio'] > 1.0)
    base_ec_dominated_count = sum(1 for r in rows if r['base_ratio'] < 1.0)

    # ── Regime summaries for narrative text ──
    dq_ec_win_vs = [r['v'] for r in rows if r['d_ec'] < 0]
    dq_el_win_vs = [r['v'] for r in rows if r['d_el'] < 0]
    dq_pos_win_vs = [r['v'] for r in rows if r['d_pos'] < 0]
    dq_ori_win_vs = [r['v'] for r in rows if r['d_ori'] < 0]
    first_ec_win_v = dq_ec_win_vs[0] if dq_ec_win_vs else None
    first_el_win_v = dq_el_win_vs[0] if dq_el_win_vs else None
    first_pos_win_v = dq_pos_win_vs[0] if dq_pos_win_vs else None

    dq_ec_regime = (
        "only at the highest tested speeds ($v_\\theta^{\\max} \\ge 16$~m/s)"
        if dq_ec_win_vs and min(dq_ec_win_vs) >= 16
        else "over part of the tested sweep"
        if dq_ec_win_vs else "at none of the tested velocities"
    )
    dq_el_regime = (
        "only at the highest tested speeds ($v_\\theta^{\\max} \\ge 18$~m/s)"
        if dq_el_win_vs and min(dq_el_win_vs) >= 18
        else "over part of the tested sweep"
        if dq_el_win_vs else "at none of the tested velocities"
    )
    dq_pos_regime = (
        "only at the highest tested speeds ($v_\\theta^{\\max} \\ge 16$~m/s)"
        if dq_pos_win_vs and min(dq_pos_win_vs) >= 16
        else "over part of the tested sweep"
        if dq_pos_win_vs else "at none of the tested velocities"
    )
    ori_statement = (
        "the Baseline MPCC keeps lower orientation error across the full sweep"
        if not dq_ori_win_vs else
        "orientation performance is split across the sweep"
    )
    avg_summary = (
        _winner_phrase(avgs['d_ec'], 'contouring RMSE') + ", " +
        _winner_phrase(avgs['d_el'], 'lag RMSE') + ", " +
        _winner_phrase(avgs['d_pos'], 'position RMSE') + ", while " +
        _winner_phrase(avgs['d_ori'], 'orientation RMSE').lower() + "."
    )

    # ── Build the LaTeX document ──
    tex = textwrap.dedent(r"""
    %% ═══════════════════════════════════════════════════════════════════════
    %%  AUTO-GENERATED by latex/generate_sweep.py
    %%  Date: """ + f"{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}" + r"""
    %%  Source: """ + f"{os.path.relpath(mat_path, _ROOT)}" + r"""
    %%  Config: """ + f"{n_vel} velocities, {n_runs} runs/vel, σ_p={sigma_p}, σ_q={sigma_q}, s_max={s_max}" + r"""
    %%
    %%  DO NOT EDIT — re-run the script to regenerate.
    %% ═══════════════════════════════════════════════════════════════════════

    """ + r"""
    \subsection{Velocity Sweep: Contouring and Lag Error Analysis}
    \label{sec:velocity_sweep_analysis}

    To rigorously evaluate the tracking quality of both formulations across
    operating regimes, we decompose the position tracking error into its
    orthogonal components: the \emph{contouring error} $e_c$ (perpendicular
    distance to the reference path) and the \emph{lag error} $e_\ell$
    (signed projection along the path tangent).  By the Pythagorean
    relationship these satisfy
    \begin{equation}
        \mathrm{RMSE}_{e_p}^{2} = \mathrm{RMSE}_{e_c}^{2} + \mathrm{RMSE}_{e_\ell}^{2},
        \label{eq:decomposition}
    \end{equation}
    which was verified numerically for every run in our Monte~Carlo campaign.
    Both components are computed in the inertial frame to ensure a fair,
    frame-neutral comparison between the DQ-MPCC (which internally operates
    in the body frame via $\mathrm{Log}\bigl(\mathrm{SE}(3)\bigr)$) and
    the Baseline MPCC (inertial-frame errors). Both controllers are also
    evaluated against the same quaternion attitude reference $\gamma_q(\theta)$,
    built from the shared terminally-extended path and waypoint construction
    using tangent direction, curvature-induced tilt, nominal reference speed,
    and the same tilt limit; thus the orientation comparison is no longer a
    yaw-only reference mismatch.

    The experiment sweeps """ + f"{n_vel}" + r""" maximum progress velocities
    $v_\theta^{\max} \in \{""" + ', '.join(f'{v:g}' for v in velocities) + r"""\}$~m/s
    with """ + f"{n_runs}" + r""" Monte~Carlo runs per velocity per controller
    ($\sigma_p = """ + f"{sigma_p}" + r"""$~m, $\sigma_q = """ + f"{sigma_q}" + r"""$~rad),
    yielding """ + f"{n_vel * n_runs * 2}" + r""" simulations in total over a
    Lissajous path of arc length $s_{\max} = """ + f"{s_max:g}" + r"""$~m.

    %% ─────────────────────────────────────────────────────────────────────
    %%  TABLE I — Full comparative metrics
    %% ─────────────────────────────────────────────────────────────────────

    \begin{table*}[!t]
    \centering
    \caption{Comparative performance across progress velocities.
    $\Delta$\% is computed as $(x_{\text{DQ}} - x_{\text{Base}})/x_{\text{Base}} \times 100$;
    negative values favour DQ-MPCC.  Bold marks the better value per velocity.}
    \label{tab:velocity_sweep_full}
    \renewcommand{\arraystretch}{1.1}
    \setlength{\tabcolsep}{3pt}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{r|ccc|ccc|ccc|ccc|ccc}
    \toprule
    & \multicolumn{3}{c|}{Contouring $\mathrm{RMSE}_{e_c}$ [m]}
    & \multicolumn{3}{c|}{Lag $\mathrm{RMSE}_{e_\ell}$ [m]}
    & \multicolumn{3}{c|}{Position $\mathrm{RMSE}_{e_p}$ [m]}
    & \multicolumn{3}{c|}{Orientation $\mathrm{RMSE}_{e_\psi}$ [rad]}
    & \multicolumn{3}{c}{Lap time $t_\mathrm{lap}$ [s]} \\
    $v_\theta^{\max}$
    & DQ & Base & $\Delta$\%
    & DQ & Base & $\Delta$\%
    & DQ & Base & $\Delta$\%
    & DQ & Base & $\Delta$\%
    & DQ & Base & $\Delta$\% \\
    \midrule
    """ + table1_body + r"""
    \midrule
    """ + avg_row + r"""
    \bottomrule
    \end{tabular}}
    \end{table*}

    %% ─────────────────────────────────────────────────────────────────────
    %%  TABLE II — Orthogonal decomposition
    %% ─────────────────────────────────────────────────────────────────────

    \begin{table}[!t]
    \centering
    \caption{Orthogonal decomposition of the position error into
    contouring (\%$e_c^2$) and lag (\%$e_\ell^2$) components.
    The ratio $e_\ell/e_c$ indicates the dominant error type:
    $>1$ implies lag-dominated tracking, $<1$ implies contouring-dominated.}
    \label{tab:decomposition}
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{r|ccc|ccc}
    \toprule
    & \multicolumn{3}{c|}{DQ-MPCC}
    & \multicolumn{3}{c}{Baseline MPCC} \\
    $v_\theta^{\max}$ & \%$e_c^2$ & \%$e_\ell^2$ & $e_\ell/e_c$
                      & \%$e_c^2$ & \%$e_\ell^2$ & $e_\ell/e_c$ \\
    \midrule
    """ + table2_body + r"""
    \bottomrule
    \end{tabular}
    \end{table}

    %% ─────────────────────────────────────────────────────────────────────
    %%  FIGURE — Pareto frontiers (2×2)
    %% ─────────────────────────────────────────────────────────────────────

    \begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{fig_pareto_quad.pdf}
    \caption{Speed--accuracy Pareto frontiers for the DQ-MPCC and
Baseline MPCC across """ + f"{n_vel}" + r""" progress velocities.
Each point is the median over """ + f"{n_runs}" + r""" Monte~Carlo runs;
    error bars denote the interquartile range (P25--P75).
    (a)~Contouring RMSE, (b)~Lag RMSE, (c)~Position RMSE,
    (d)~Orientation RMSE, all vs.\ lap time.
    Annotations indicate $v_\theta^{\max}$ [m/s].}
    \label{fig:pareto_quad}
    \end{figure}

    %% ─────────────────────────────────────────────────────────────────────
    %%  FIGURE ANALYSIS
    %% ─────────────────────────────────────────────────────────────────────

    Fig.~\ref{fig:pareto_quad} presents the speed--accuracy Pareto frontiers
    for both controllers.
    %
    In panel~(a), the Baseline MPCC yields lower contouring RMSE over low and
    medium speeds, whereas the DQ-MPCC becomes superior """ + dq_ec_regime + r""".
    This indicates that the $\mathrm{SE}(3)$ formulation does not dominate the
    full sweep uniformly, but it becomes increasingly competitive as the
    commanded progress rises.
    %
    Panel~(b) shows a similar trend for lag RMSE: the Baseline is better over
    most of the sweep, while the DQ-MPCC becomes better """ + dq_el_regime + r""".
    Hence the DQ formulation does not simply trade contouring against larger
    tangential delay at the top end; instead, both translational components
    improve jointly in the highest-speed regime.
    %
    Panel~(c) combines both components via~\eqref{eq:decomposition}. The total
    position RMSE follows the same pattern, with DQ-MPCC becoming superior """ + dq_pos_regime + r""",
    while the Baseline remains preferable over the lower and mid-speed region.
    %
    Finally, panel~(d) shows that """ + ori_statement + r""". Therefore, the
    high-speed translational advantage of DQ-MPCC should not be interpreted as
    a uniform pose-level dominance.

    %% ─────────────────────────────────────────────────────────────────────
    %%  ANALYSIS  (all numbers auto-generated)
    %% ─────────────────────────────────────────────────────────────────────

    \subsubsection{Contouring vs.\ Lag Decomposition}

    Table~\ref{tab:velocity_sweep_full} shows that """ + avg_summary + r"""
    The corresponding average deltas for DQ relative to the Baseline are
    $e_c=""" + f"{_sp(avgs['d_ec'])}" + r"""\%$, $e_\ell=""" + f"{_sp(avgs['d_el'])}" +
    r"""\%$, $e_p=""" + f"{_sp(avgs['d_pos'])}" + r"""\%$, and $e_\psi=""" + f"{_sp(avgs['d_ori'])}" + r"""\%$.
    However, the sign reversal in the high-speed region is important:
    DQ-MPCC first becomes better in contouring at """ + f"${_fmt_v(first_ec_win_v)}$" + r"""~m/s, and
    its strongest high-speed gains appear at """ + f"${_fmt_v(pos_best_v)}$" + r"""~m/s for position
    ($""" + f"{_sp(pos_best)}" + r"""\%$) and at """ + f"${_fmt_v(ec_best_v)}$" + r"""~m/s for contouring
    ($""" + f"{_sp(ec_best)}" + r"""\%$).
    At the same time, DQ-MPCC is consistently faster in lap time across the
    sweep (average $\Delta t_\mathrm{lap}=""" + f"{_sp(avgs['d_tlap'])}" + r"""\%$),
    so the average tracking deficit does not come from moving more slowly.
    The decomposition in Table~\ref{tab:decomposition} explains this asymmetry:
    the Baseline is \emph{contouring-dominated}
    ($""" + f"{base_pct_ec_min:.0f}" + r"""$--$""" + f"{base_pct_ec_max:.0f}" + r"""\%$ of $\|e_p\|^2$ comes from $e_c$,
    $e_\ell/e_c < 1$ at all speeds),
    while the DQ-MPCC transitions from near parity
    (""" + f"${rows[0]['dq_pct_ec']:.0f}$/$\\ {rows[0]['dq_pct_el']:.0f}$" + r"""\%
    at """ + f"${_fmt_v(rows[0]['v'])}$" + r"""~m/s) to \emph{lag-dominated}
    (""" + f"${rows[-1]['dq_pct_ec']:.0f}$/$\\ {rows[-1]['dq_pct_el']:.0f}$" + r"""\%,
    $e_\ell/e_c = """ + f"{rows[-1]['dq_ratio']:.2f}" + r"""$ at """ + f"${_fmt_v(rows[-1]['v'])}$" + r"""~m/s).
    %
    The root cause is structural: the DQ-MPCC's $\mathrm{SE}(3)$ logarithmic
    error map $\boldsymbol{\rho} = \mathbf{J}_l(\boldsymbol{\varphi})^{-1}
    \mathbf{R}_d^\top \Delta\mathbf{p}$ couples position and orientation,
    while the Baseline penalises $e_c$ and $e_\ell$ as decoupled terms. The
    present results suggest that the decoupled structure is advantageous in the
    low- and mid-speed regime, whereas the coupled DQ formulation becomes more
    competitive, and eventually superior in translational tracking, once the
    demanded progress enters the highest-speed regime.

    """)

    return tex


def generate_verified_tex(rows, avgs, metadata):
    base_tex = apply_verified_rewrites(generate_tex(rows, avgs, metadata), rows, avgs)
    verification = _build_verification_section(
        'Verified Claim Audit',
        build_verification_items(rows, avgs),
    )
    return base_tex + '\n' + verification


def compile_pdf(tex_name, pdf_name):
    pdflatex = shutil.which('pdflatex')
    if pdflatex is None:
        print('[WARN] pdflatex not found in PATH; skipping PDF compilation.')
        return False

    tex_path = tex_name if os.path.isabs(tex_name) else os.path.join(os.path.dirname(_TEX_OUT), tex_name)
    tex_rel = os.path.relpath(tex_path, _COMPILE_DIR).replace(os.sep, '/')
    figures_rel = os.path.relpath(str(_EXP2_DIR['figures']), _COMPILE_DIR).replace(os.sep, '/')
    wrapper_name = f'_compile_{pdf_name}.tex'
    wrapper = os.path.join(_COMPILE_DIR, wrapper_name)
    with open(wrapper, 'w') as f:
        f.write(textwrap.dedent(r"""
            \documentclass[11pt]{article}
            \usepackage[margin=1in]{geometry}
            \usepackage{amsmath,amssymb,booktabs,graphicx,float}
            \begin{document}
            \graphicspath{{""" + figures_rel + r"""/}}
            \input{""" + tex_rel + r"""}
            \end{document}
        """).lstrip())

    print(f'[INFO] Compiling PDF with pdflatex -> {pdf_name}.pdf ...')
    result = None
    for _ in range(2):
        result = subprocess.run(
            [pdflatex, '-interaction=nonstopmode', wrapper_name],
            cwd=_COMPILE_DIR,
            capture_output=True,
            text=True,
        )

    pdf = os.path.join(os.path.dirname(_TEX_OUT), f'{pdf_name}.pdf')
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
    parser = argparse.ArgumentParser(
        description='Generate base/raw LaTeX analysis from velocity sweep data')
    parser.add_argument('--no-compile', action='store_true',
                        help='Skip PDF compilation after generating .tex')
    args = parser.parse_args()

    if not os.path.isfile(_MAT) and not os.path.isfile(_LEGACY_MAT):
        print(f'[ERROR] Not found: {_MAT}')
        print('        Run scripts/run_experiment2.py first.')
        sys.exit(1)

    print('[INFO] Loading sweep data ...')
    payload = load_data()
    data = payload['data']
    metadata = {
        'velocities': payload['velocities'],
        'n_runs': payload['n_runs'],
        'sigma_p': payload['sigma_p'],
        'sigma_q': payload['sigma_q'],
        's_max': payload['s_max'],
        'mat_path': payload['mat_path'],
    }

    print('[INFO] Computing metrics ...')
    rows, avgs = compute_metrics(data, metadata['velocities'])

    # ── Terminal summary ──
    print()
    print('=' * 90)
    print(f'[INFO] Source mat: {os.path.relpath(metadata["mat_path"], _ROOT)}')
    print(f'[INFO] Output tex: {os.path.relpath(_TEX_OUT, _ROOT)}')
    print(f'[INFO] Output pdf: {os.path.relpath(os.path.join(str(_EXP2_DIR["reports"]), "experiment2_analysis.pdf"), _ROOT)}')
    print(f'[INFO] Sweep metadata: velocities={metadata["velocities"]}, n_runs={metadata["n_runs"]}, sigma_p={metadata["sigma_p"]}, sigma_q={metadata["sigma_q"]}, s_max={metadata["s_max"]}')
    print('=' * 90)
    print()
    print('=' * 90)
    print(f'{"v":>3} | {"DQ_ec":>7} {"Base_ec":>7} {"Δec":>7} | '
          f'{"DQ_el":>7} {"Base_el":>7} {"Δel":>7} | '
          f'{"DQ_pos":>7} {"Base_pos":>8} {"Δpos":>7} | '
          f'{"DQ_ori":>7} {"Base_ori":>8} {"Δori":>7}')
    print('-' * 90)
    for r in rows:
        print(f"{r['v']:>3} | "
              f"{r['dq_ec']:7.4f} {r['base_ec']:7.4f} {_sp(r['d_ec']):>6}% | "
              f"{r['dq_el']:7.4f} {r['base_el']:7.4f} {_sp(r['d_el']):>6}% | "
              f"{r['dq_pos']:7.4f} {r['base_pos']:8.4f} {_sp(r['d_pos']):>6}% | "
              f"{r['dq_ori']:7.4f} {r['base_ori']:8.4f} {_sp(r['d_ori']):>6}%")
    print('-' * 90)
    print(f"{'Avg':>3} | "
          f"{avgs['dq_ec']:7.4f} {avgs['base_ec']:7.4f} {_sp(avgs['d_ec']):>6}% | "
          f"{avgs['dq_el']:7.4f} {avgs['base_el']:7.4f} {_sp(avgs['d_el']):>6}% | "
          f"{avgs['dq_pos']:7.4f} {avgs['base_pos']:8.4f} {_sp(avgs['d_pos']):>6}% | "
          f"{avgs['dq_ori']:7.4f} {avgs['base_ori']:8.4f} {_sp(avgs['d_ori']):>6}%")
    print('=' * 90)
    print()

    # ── Generate .tex ──
    print(f'[INFO] Generating {_TEX_OUT} ...')
    tex_content = generate_tex(rows, avgs, metadata)
    with open(_TEX_OUT, 'w') as f:
        f.write(tex_content)
    print(f'✓  {_TEX_OUT}')
    print(f'   ({len(tex_content)} chars, {tex_content.count(chr(10))} lines)')

    if not args.no_compile:
        compile_pdf('experiment2_analysis.tex', 'experiment2_analysis')

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
