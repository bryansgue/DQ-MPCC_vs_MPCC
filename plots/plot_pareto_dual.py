#!/usr/bin/env python3
"""
plot_experiment2_pareto_dual.py — Dual Pareto frontier:
  Left  panel: Position    RMSE vs t_lap   (MPCC dominates)
  Right panel: Orientation RMSE vs t_lap   (DQ-MPCC dominates)

Reads  experiment2_results/velocity_sweep_data.mat.

Usage
-----
    python plot_experiment2_pareto_dual.py
    python plot_experiment2_pareto_dual.py --mock
    python plot_experiment2_pareto_dual.py --no-show
"""

import os, sys, argparse
import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt

# ── Style ────────────────────────────────────────────────────────────────────
_USE_TEX = False
try:
    matplotlib.rcParams['text.usetex'] = True
    _fig_test = plt.figure(); plt.close(_fig_test)
    _USE_TEX = True
except Exception:
    pass

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'figure.dpi':        150,
    'text.usetex':       _USE_TEX,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

C_DQ   = '#1f77b4'   # blue
C_BASE = '#d62728'   # red

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)

sys.path.insert(0, _ROOT)
from config.result_paths import experiment_dirs
try:
    from config.sweep_config import VELOCITIES
except ImportError:
    VELOCITIES = [8, 12, 15]

_EXP2_DIRS = experiment_dirs("experiment2")
_DATA_DIR = str(_EXP2_DIRS["data"])
_FIG_DIR = str(_EXP2_DIRS["figures"])
_LEGACY_DATA = os.path.join(_ROOT, 'results', 'experiment2', 'velocity_sweep_data.mat')


# ═════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_sweep_data():
    mat_path = os.path.join(_DATA_DIR, 'velocity_sweep_data.mat')
    if not os.path.isfile(mat_path) and os.path.isfile(_LEGACY_DATA):
        mat_path = _LEGACY_DATA
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Not found: {mat_path}")
    d = loadmat(mat_path)

    results  = {'dq': {}, 'base': {}}
    failures = {'dq': {}, 'base': {}}
    for ctrl in ['dq', 'base']:
        for v in VELOCITIES:
            vstr = f"{v:.2f}".replace('.', 'p')
            rp = d.get(f'{ctrl}_v{vstr}_rmse_pos',    np.array([])).ravel()
            ro = d.get(f'{ctrl}_v{vstr}_rmse_ori',    np.array([])).ravel()
            tl = d.get(f'{ctrl}_v{vstr}_t_lap',       np.array([])).ravel()
            mv = d.get(f'{ctrl}_v{vstr}_mean_vtheta', np.array([])).ravel()
            nf = int(d.get(f'{ctrl}_v{vstr}_failures', np.array([0])).ravel()[0])
            results[ctrl][v] = {'rmse_pos': rp, 'rmse_ori': ro,
                                't_lap': tl, 'mean_vtheta': mv}
            failures[ctrl][v] = nf
    return results, failures


def generate_mock_data():
    """Synthetic data for layout prototyping (N=20 runs per point)."""
    np.random.seed(42)
    N = 20
    results  = {'dq': {}, 'base': {}}
    failures = {'dq': {}, 'base': {}}

    # Position: MPCC wins; Orientation: DQ-MPCC wins
    dq_pos_nom   = {8: 0.42, 12: 0.56, 15: 0.72}
    base_pos_nom = {8: 0.35, 12: 0.48, 15: 0.60}
    dq_ori_nom   = {8: 0.38, 12: 0.55, 15: 0.68}
    base_ori_nom = {8: 0.50, 12: 0.72, 15: 0.95}
    dq_tlap_nom  = {8: 11.5, 12: 8.5,  15: 7.0}
    base_tlap_nom= {8: 12.0, 12: 9.0,  15: 7.5}

    for v in VELOCITIES:
        for ctrl, pos_nom, ori_nom, tlap_nom in [
            ('dq',   dq_pos_nom.get(v, 0.5), dq_ori_nom.get(v, 0.5), dq_tlap_nom.get(v, 8.0)),
            ('base', base_pos_nom.get(v, 0.4), base_ori_nom.get(v, 0.7), base_tlap_nom.get(v, 8.5)),
        ]:
            rp = np.abs(np.random.normal(pos_nom, pos_nom * 0.10, N))
            ro = np.abs(np.random.normal(ori_nom, ori_nom * 0.10, N))
            tl = np.abs(np.random.normal(tlap_nom, tlap_nom * 0.04, N))
            mv = 80.0 / tl
            results[ctrl][v]  = {'rmse_pos': rp, 'rmse_ori': ro,
                                 't_lap': tl, 'mean_vtheta': mv}
            failures[ctrl][v] = 0
    return results, failures


def _stats(arr):
    """Return (median, p25, p75) or (nan, nan, nan) if empty."""
    a = arr[np.isfinite(arr)] if len(arr) > 0 else np.array([])
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


# ═════════════════════════════════════════════════════════════════════════════
#  Core panel plotter
# ═════════════════════════════════════════════════════════════════════════════

def _plot_one_panel(ax, results, failures, metric_key, ylabel, panel_letter):
    """
    Plot one Pareto panel (either 'rmse_pos' or 'rmse_ori') on the given ax.
    """
    controllers = [
        ('dq',   'DQ-MPCC',       C_DQ,   'o', '-'),
        ('base', 'Baseline MPCC', C_BASE, 's', '--'),
    ]

    for ctrl, label, color, marker, ls in controllers:
        x_med, x_lo, x_hi = [], [], []
        y_med, y_lo, y_hi = [], [], []
        v_labels = []

        for v in VELOCITIES:
            metric = results[ctrl][v][metric_key]
            tl     = results[ctrl][v]['t_lap']

            if len(metric) == 0 or len(tl) == 0:
                continue

            mask = np.isfinite(metric) & np.isfinite(tl)
            m_ok = metric[mask]
            tl_ok = tl[mask]
            if len(m_ok) == 0:
                continue

            rm, rlo, rhi = _stats(m_ok)
            tm, tlo, thi = _stats(tl_ok)

            y_med.append(rm);  y_lo.append(rlo);  y_hi.append(rhi)
            x_med.append(tm);  x_lo.append(tlo);  x_hi.append(thi)
            v_labels.append(v)

        if len(x_med) == 0:
            continue

        x_med = np.array(x_med)
        y_med = np.array(y_med)
        x_err = np.array([x_med - np.array(x_lo),
                          np.array(x_hi) - x_med])
        y_err = np.array([y_med - np.array(y_lo),
                          np.array(y_hi) - y_med])

        # Sort by t_lap
        order    = np.argsort(x_med)
        x_med    = x_med[order]
        y_med    = y_med[order]
        x_err    = x_err[:, order]
        y_err    = y_err[:, order]
        v_sorted = [v_labels[i] for i in order]

        # Connecting line
        ax.plot(x_med, y_med,
                color=color, linestyle=ls, linewidth=1.4,
                zorder=2, alpha=0.7)

        # Error bars (IQR)
        ax.errorbar(x_med, y_med,
                    xerr=x_err, yerr=y_err,
                    fmt='none',
                    ecolor=color, elinewidth=1.0,
                    capsize=3, capthick=1.0,
                    zorder=3, alpha=0.85)

        # Markers
        ax.scatter(x_med, y_med,
                   color=color, marker=marker,
                   s=50, zorder=5,
                   edgecolors='white', linewidths=0.5,
                   label=label)

        # Velocity annotations — alternate upper-right / lower-left
        for idx, (xi, yi, vv) in enumerate(zip(x_med, y_med, v_sorted)):
            nf  = failures[ctrl][vv]
            nok = len(results[ctrl][vv][metric_key])
            if _USE_TEX:
                ann = rf'$v_{{\theta}}^{{\max}}\!=\!{vv}$'
            else:
                ann = f'{vv} m/s'
            if nf > 0:
                ann += f'\n({nok}/{nok+nf} ok)'

            if idx % 2 == 0:
                xyoff = (5, 4)
                ha, va = 'left', 'bottom'
            else:
                xyoff = (-5, -4)
                ha, va = 'right', 'top'

            ax.annotate(
                ann, xy=(xi, yi), xytext=xyoff,
                textcoords='offset points',
                fontsize=6.5, color=color,
                ha=ha, va=va,
            )

    # ── Formatting ──
    ax.set_xlabel(r'Lap time $t_{\mathrm{lap}}$ [s]')
    ax.set_ylabel(ylabel)

    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Arrow: "faster" → LEFT
    ax.annotate('', xy=(0.75, 0.06), xytext=(0.87, 0.06),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.1))
    ax.text(0.74, 0.06, 'faster', transform=ax.transAxes,
            va='center', ha='right', fontsize=7, color='gray')

    # Arrow: "more accurate" → DOWN
    ax.annotate('', xy=(0.05, 0.08), xytext=(0.05, 0.18),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.1))
    ax.text(0.05, 0.20, 'more\naccurate', transform=ax.transAxes,
            va='bottom', ha='center', fontsize=7, color='gray')

    # Panel letter (a), (b)
    ax.text(0.02, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)


# ═════════════════════════════════════════════════════════════════════════════
#  Main figure
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto_dual(results, failures, save=True, show=True):
    """Dual Pareto: Position RMSE (left) + Orientation RMSE (right)."""

    fig, (ax_pos, ax_ori) = plt.subplots(1, 2, figsize=(13, 5.2),
                                          sharey=False, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.12,
                        wspace=0.26)

    # ── Left panel: Position RMSE ──
    if _USE_TEX:
        ylabel_pos = r'Position RMSE $\mathrm{RMSE}_{\mathbf{e}_p}$ [m]'
    else:
        ylabel_pos = r'Position RMSE [m]'
    _plot_one_panel(ax_pos, results, failures,
                    metric_key='rmse_pos',
                    ylabel=ylabel_pos,
                    panel_letter='(a)')

    # ── Right panel: Orientation RMSE ──
    if _USE_TEX:
        ylabel_ori = r'Orientation RMSE $\mathrm{RMSE}_{\mathbf{e}_\psi}$ [rad]'
    else:
        ylabel_ori = r'Orientation RMSE [rad]'
    _plot_one_panel(ax_ori, results, failures,
                    metric_key='rmse_ori',
                    ylabel=ylabel_ori,
                    panel_letter='(b)')

    # ── Suptitle ──
    if _USE_TEX:
        suptitle = (r'\textbf{Speed--Accuracy Pareto Frontier:}'
                    r'  Position (a) vs.\ Orientation (b)')
        subtitle = (r'Error bars = IQR (P25--P75) across Monte Carlo runs'
                    r'  $\bullet$  same $v_{\theta}^{\max}$ per pair of points')
    else:
        suptitle = 'Speed–Accuracy Pareto Frontier: Position (a) vs. Orientation (b)'
        subtitle = ('Error bars = IQR (P25–P75) across Monte Carlo runs'
                    '  •  same v_θ^max per pair of points')

    fig.suptitle(f'{suptitle}\n{subtitle}', fontsize=10, y=0.98)

    if save:
        for ext in ('pdf', 'png'):
            out = os.path.join(_FIG_DIR, f'fig_pareto_dual.{ext}')
            fig.savefig(out, dpi=300, bbox_inches='tight')
        print('✓ fig_pareto_dual saved (pdf + png)')

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2 — Dual Pareto: Position + Orientation RMSE vs t_lap')
    parser.add_argument('--mock',    action='store_true',
                        help='Use synthetic data for prototyping')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not call plt.show()')
    args = parser.parse_args()

    if args.mock:
        print('[INFO] Using synthetic mock data')
        results, failures = generate_mock_data()
    else:
        try:
            print('[INFO] Loading real sweep data...')
            results, failures = load_sweep_data()
            v0   = VELOCITIES[0]
            has_tlap = len(results['dq'][v0]['t_lap']) > 0 or \
                       len(results['base'][v0]['t_lap']) > 0
            if not has_tlap:
                print('[WARN] .mat has no t_lap data — falling back to mock.')
                results, failures = generate_mock_data()
            else:
                print('[INFO] Data loaded OK')
        except FileNotFoundError as e:
            print(f'[WARN] {e}')
            print('[INFO] Falling back to mock data')
            results, failures = generate_mock_data()

    plot_pareto_dual(results, failures, save=True, show=not args.no_show)

    if not args.no_show:
        plt.show()

    print('\n✓ Dual Pareto plot done.')


if __name__ == '__main__':
    main()
