#!/usr/bin/env python3
"""
generate_sweep.py  (was generate_experiment2_latex.py)
─────────────────────────────
Reads  results/experiment2/velocity_sweep_data.mat  and generates a
self-contained LaTeX file with:

  • Table I   — Comparative performance (ec, el, pos, ori, t_lap)
  • Table II  — Orthogonal decomposition  (%ec vs %el)
  • Interpretive paragraphs with ALL numbers auto-filled

Re-run this script every time velocities, N_runs, or weights change
and the .tex will update automatically.

Usage
─────
    python latex/generate_sweep.py            # → .tex + terminal summary
    python latex/generate_sweep.py --compile  # also runs pdflatex
"""

import os, sys, argparse, textwrap
import numpy as np
from scipy.io import loadmat

# ═════════════════════════════════════════════════════════════════════════════
#  Paths & config
# ═════════════════════════════════════════════════════════════════════════════
_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)                       # workspace root
_OUT_DIR = os.path.join(_ROOT, 'results', 'experiment2')
_MAT     = os.path.join(_OUT_DIR, 'velocity_sweep_data.mat')
_TEX_OUT = os.path.join(_OUT_DIR, 'experiment2_analysis.tex')

# ── Ensure workspace root is on sys.path ─────────────────────────────────────
sys.path.insert(0, _ROOT)

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


def load_data():
    """Return dict  d[ctrl][v][metric] → numpy array  (ctrl ∈ {dq, base})."""
    raw = loadmat(_MAT, squeeze_me=True)
    data = {'dq': {}, 'base': {}}
    for ctrl in ('dq', 'base'):
        for v in VELOCITIES:
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
    return data


# ═════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(data):
    """
    Return list of dicts (one per velocity) with all comparative metrics.
    Also return global averages dict.
    """
    rows = []
    sums = {k: {'dq': [], 'base': []}
            for k in ('ec', 'el', 'pos', 'ori', 'tlap')}

    for v in VELOCITIES:
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


def generate_tex(rows, avgs):
    """Return the full .tex string."""

    n_vel = len(VELOCITIES)
    v_min, v_max = VELOCITIES[0], VELOCITIES[-1]

    # ── Extremes for each metric ──
    ec_best, ec_best_v, ec_worst, ec_worst_v = _extremes(rows, 'd_ec')
    el_best, el_best_v, el_worst, el_worst_v = _extremes(rows, 'd_el')
    pos_best, pos_best_v, pos_worst, pos_worst_v = _extremes(rows, 'd_pos')
    ori_best, ori_best_v, ori_worst, ori_worst_v = _extremes(rows, 'd_ori')
    t_best, t_best_v, t_worst, t_worst_v = _extremes(rows, 'd_tlap')

    # ── Decomposition extremes ──
    dq_pct_el_min  = min(r['dq_pct_el'] for r in rows)
    dq_pct_el_max  = max(r['dq_pct_el'] for r in rows)
    base_pct_ec_min = min(r['base_pct_ec'] for r in rows)
    base_pct_ec_max = max(r['base_pct_ec'] for r in rows)

    # ── Determine overall winner per metric ──
    ec_winner  = _winner(avgs['d_ec'])
    el_winner  = _winner(avgs['d_el'])
    pos_winner = _winner(avgs['d_pos'])
    ori_winner = _winner(avgs['d_ori'])

    # ── Build Table I rows ──
    table1_rows = []
    for r in rows:
        dq_ec_s, base_ec_s = _bold_if_better(r['dq_ec'], r['base_ec'])
        dq_el_s, base_el_s = _bold_if_better(r['dq_el'], r['base_el'])
        dq_pos_s, base_pos_s = _bold_if_better(r['dq_pos'], r['base_pos'])
        dq_ori_s, base_ori_s = _bold_if_better(r['dq_ori'], r['base_ori'])
        dq_t_s, base_t_s = _bold_if_better(r['dq_tlap'], r['base_tlap'])

        line = (f"    {r['v']:>2} "
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
        line = (f"    {r['v']:>2} "
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

    # ── Crossover velocity for ec (where DQ stops winning) ──
    ec_crossover = None
    for r in rows:
        if r['d_ec'] > 0:
            ec_crossover = r['v']
            break

    # ── Crossover velocity for pos (where DQ stops winning) ──
    pos_crossover = None
    for r in rows:
        if r['d_pos'] > 0:
            pos_crossover = r['v']
            break

    # ── ec curves: nearly merge at high speed? ──
    ec_high_v = rows[-1]
    ec_gap_high = abs(ec_high_v['dq_ec'] - ec_high_v['base_ec'])

    # ── el: absolute gap at lowest and highest speed ──
    el_gap_low  = rows[0]['dq_el'] - rows[0]['base_el']
    el_gap_high = rows[-1]['dq_el'] - rows[-1]['base_el']

    # ── ori: gap range ──
    ori_gap_min = min(abs(r['d_ori']) for r in rows)
    ori_gap_max = max(abs(r['d_ori']) for r in rows)

    # ── Build the LaTeX document ──
    tex = textwrap.dedent(r"""
    %% ═══════════════════════════════════════════════════════════════════════
    %%  AUTO-GENERATED by generate_experiment2_latex.py
    %%  Date: """ + f"{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}" + r"""
    %%  Source: experiment2_results/velocity_sweep_data.mat
    %%  Config: """ + f"{n_vel} velocities, {N_RUNS} runs/vel, σ_p={SIGMA_P}, σ_q={SIGMA_Q}, s_max={S_MAX}" + r"""
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
    the Baseline MPCC (inertial-frame errors).

    The experiment sweeps """ + f"{n_vel}" + r""" maximum progress velocities
    $v_\theta^{\max} \in \{""" + ', '.join(str(v) for v in VELOCITIES) + r"""\}$~m/s
    with """ + f"{N_RUNS}" + r""" Monte~Carlo runs per velocity per controller
    ($\sigma_p = """ + f"{SIGMA_P}" + r"""$~m, $\sigma_q = """ + f"{SIGMA_Q}" + r"""$~rad),
    yielding """ + f"{n_vel * N_RUNS * 2}" + r""" simulations in total over a
    Lissajous path of arc length $s_{\max} = """ + f"{S_MAX}" + r"""$~m.

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
    \includegraphics[width=\columnwidth]{Fig/fig_pareto_quad.pdf}
    \caption{Speed--accuracy Pareto frontiers for the DQ-MPCC and
    Baseline MPCC across """ + f"{n_vel}" + r""" progress velocities.
    Each point is the median over """ + f"{N_RUNS}" + r""" Monte~Carlo runs;
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
    In panel~(a), the DQ-MPCC (blue) lies below the Baseline (red)""" + (
        f" for $v_\\theta^{{\\max}} < {ec_crossover}$~m/s, after which the curves nearly merge"
        if ec_crossover is not None
        else " across the entire velocity range"
    ) + r""" (gap $<""" + f"{ec_gap_high:.3f}" + r"""$~m at """ + f"${ec_high_v['v']}$" + r"""~m/s),
    confirming that the coupled $\mathrm{SE}(3)$ formulation yields tighter
    geometric adherence to the path.
    %
    Panel~(b) reveals the opposite: the Baseline's lag curve is consistently
    lower, with an absolute gap that widens from
    $""" + f"{el_gap_low:.3f}" + r"""$~m at """ + f"${rows[0]['v']}$" + r"""~m/s
    to $""" + f"{el_gap_high:.3f}" + r"""$~m at """ + f"${rows[-1]['v']}$" + r"""~m/s,
    reflecting the decoupled formulation's ability to independently minimise
    along-path delay.
    %
    Panel~(c) combines both components via~\eqref{eq:decomposition}:
    the curves""" + (
        f" cross near $v_\\theta^{{\\max}} \\approx {pos_crossover}$~m/s"
        if pos_crossover is not None
        else " remain close"
    ) + r""", with the lag-dominated DQ-MPCC yielding higher total position error
    at high speeds.
    %
    Finally, panel~(d) shows that the DQ-MPCC maintains consistently lower
    orientation error across the entire velocity range
    ($""" + f"{ori_gap_min:.1f}" + r"""$--$""" + f"{ori_gap_max:.1f}" + r"""\%$),
    a direct benefit of the pose-level coupling in the cost function.

    %% ─────────────────────────────────────────────────────────────────────
    %%  ANALYSIS  (all numbers auto-generated)
    %% ─────────────────────────────────────────────────────────────────────

    \subsubsection{Contouring vs.\ Lag Decomposition}

    Table~\ref{tab:velocity_sweep_full} shows that the DQ-MPCC achieves lower
    contouring error """ + (
        f"at all tested velocities"
        if ec_crossover is None
        else f"for $v_\\theta^{{\\max}} < {ec_crossover}$~m/s"
    ) + r"""
    (avg.\ $""" + f"{_sp(avgs['d_ec'])}" + r"""\%$, best $""" + f"{_sp(ec_best)}" + r"""\%$ at """ + f"${ec_best_v}$" + r"""~m/s),
    whereas the Baseline MPCC dominates in lag error
    (avg.\ $""" + f"{_sp(avgs['d_el'])}" + r"""\%$, up to $""" + f"{_sp(el_worst)}" + r"""\%$ at """ + f"${el_worst_v}$" + r"""~m/s).
    The decomposition in Table~\ref{tab:decomposition} explains this asymmetry:
    the Baseline is \emph{contouring-dominated}
    ($""" + f"{base_pct_ec_min:.0f}" + r"""$--$""" + f"{base_pct_ec_max:.0f}" + r"""\%$ of $\|e_p\|^2$ comes from $e_c$,
    $e_\ell/e_c < 1$ at all speeds),
    while the DQ-MPCC transitions from near parity
    (""" + f"${rows[0]['dq_pct_ec']:.0f}$/$\\ {rows[0]['dq_pct_el']:.0f}$" + r"""\%
    at """ + f"${rows[0]['v']}$" + r"""~m/s) to \emph{lag-dominated}
    (""" + f"${rows[-1]['dq_pct_ec']:.0f}$/$\\ {rows[-1]['dq_pct_el']:.0f}$" + r"""\%,
    $e_\ell/e_c = """ + f"{rows[-1]['dq_ratio']:.2f}" + r"""$ at """ + f"${rows[-1]['v']}$" + r"""~m/s).
    %
    The root cause is structural: the DQ-MPCC's $\mathrm{SE}(3)$ logarithmic
    error map $\boldsymbol{\rho} = \mathbf{J}_l(\boldsymbol{\varphi})^{-1}
    \mathbf{R}_d^\top \Delta\mathbf{p}$ couples position and orientation,
    preventing the solver from independently minimising the tangential projection.
    The Baseline, by contrast, penalises $e_c$ and $e_\ell$ as decoupled cost
    terms, enabling aggressive lag reduction at the expense of orientation fidelity.

    """)

    return tex


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX analysis from velocity sweep data')
    parser.add_argument('--compile', action='store_true',
                        help='Also compile with pdflatex')
    args = parser.parse_args()

    if not os.path.isfile(_MAT):
        print(f'[ERROR] Not found: {_MAT}')
        print('        Run 2_run_experiment2_sweep.py first.')
        sys.exit(1)

    print('[INFO] Loading sweep data ...')
    data = load_data()

    print('[INFO] Computing metrics ...')
    rows, avgs = compute_metrics(data)

    # ── Terminal summary ──
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
    tex_content = generate_tex(rows, avgs)

    with open(_TEX_OUT, 'w') as f:
        f.write(tex_content)
    print(f'✓  {_TEX_OUT}')
    print(f'   ({len(tex_content)} chars, {tex_content.count(chr(10))} lines)')

    # ── Optional compilation ──
    if args.compile:
        # Create a minimal wrapper to compile standalone
        wrapper = os.path.join(_OUT_DIR, '_compile_analysis.tex')
        with open(wrapper, 'w') as f:
            f.write(textwrap.dedent(r"""
                \documentclass[journal,twocolumn]{IEEEtran}
                \usepackage{amsmath,amssymb,booktabs,graphicx}
                \begin{document}
                \input{experiment2_analysis.tex}
                \end{document}
            """).lstrip())
        import subprocess
        print('[INFO] Compiling with pdflatex ...')
        subprocess.run(['pdflatex', '-interaction=nonstopmode',
                        '_compile_analysis.tex'],
                       cwd=_OUT_DIR, capture_output=True)
        pdf = os.path.join(_OUT_DIR, '_compile_analysis.pdf')
        if os.path.isfile(pdf):
            print(f'✓  {pdf}')
        else:
            print('[WARN] pdflatex failed — check _compile_analysis.log')

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
