#!/usr/bin/env python3
"""
plot_experiment2_pareto_quad.py — 2×2 Pareto frontier:
  (a) Contouring  RMSE vs t_lap
  (b) Lag         RMSE vs t_lap
  (c) Position    RMSE vs t_lap
  (d) Orientation RMSE vs t_lap

All errors computed in the **inertial frame** for fair comparison.
Reads  experiment2_results/velocity_sweep_data.mat.

Usage
-----
    python plot_experiment2_pareto_quad.py
    python plot_experiment2_pareto_quad.py --no-show
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
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'figure.dpi':        150,
    'text.usetex':       _USE_TEX,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

C_DQ   = '#1f77b4'   # blue
C_BASE = '#d62728'   # red

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)                       # workspace root

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
            rp  = d.get(f'{ctrl}_v{vstr}_rmse_pos',    np.array([])).ravel()
            ro  = d.get(f'{ctrl}_v{vstr}_rmse_ori',    np.array([])).ravel()
            rec = d.get(f'{ctrl}_v{vstr}_rmse_ec',     np.array([])).ravel()
            rel = d.get(f'{ctrl}_v{vstr}_rmse_el',     np.array([])).ravel()
            tl  = d.get(f'{ctrl}_v{vstr}_t_lap',       np.array([])).ravel()
            mv  = d.get(f'{ctrl}_v{vstr}_mean_vtheta', np.array([])).ravel()
            nf  = int(d.get(f'{ctrl}_v{vstr}_failures', np.array([0])).ravel()[0])
            results[ctrl][v] = {
                'rmse_pos': rp, 'rmse_ori': ro,
                'rmse_ec': rec, 'rmse_el': rel,
                't_lap': tl, 'mean_vtheta': mv,
            }
            failures[ctrl][v] = nf
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

def _plot_one_panel(ax, results, failures, metric_key, ylabel, panel_letter,
                    show_legend=True, show_xlabel=True, show_annotations=True):
    """
    Plot one Pareto panel on the given axes.
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
            metric = results[ctrl][v].get(metric_key, np.array([]))
            tl     = results[ctrl][v]['t_lap']

            if len(metric) == 0 or len(tl) == 0:
                continue

            # Pair-wise filter
            n = min(len(metric), len(tl))
            metric = metric[:n]
            tl = tl[:n]
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
                color=color, linestyle=ls, linewidth=1.3,
                zorder=2, alpha=0.7)

        # Error bars (IQR)
        ax.errorbar(x_med, y_med,
                    xerr=x_err, yerr=y_err,
                    fmt='none',
                    ecolor=color, elinewidth=0.9,
                    capsize=2.5, capthick=0.9,
                    zorder=3, alpha=0.8)

        # Markers
        ax.scatter(x_med, y_med,
                   color=color, marker=marker,
                   s=40, zorder=5,
                   edgecolors='white', linewidths=0.5,
                   label=label)

        # Velocity annotations
        if show_annotations:
            for idx, (xi, yi, vv) in enumerate(zip(x_med, y_med, v_sorted)):
                nf  = failures[ctrl][vv]
                nok = len(results[ctrl][vv].get(metric_key, []))
                if _USE_TEX:
                    ann = rf'${vv}$'
                else:
                    ann = f'{vv}'
                if nf > 0:
                    ann += f'\n({nok}/{nok+nf})'

                if idx % 2 == 0:
                    xyoff = (4, 3)
                    ha, va = 'left', 'bottom'
                else:
                    xyoff = (-4, -3)
                    ha, va = 'right', 'top'

                ax.annotate(
                    ann, xy=(xi, yi), xytext=xyoff,
                    textcoords='offset points',
                    fontsize=6, color=color,
                    ha=ha, va=va,
                )

    # ── Formatting ──
    if show_xlabel:
        ax.set_xlabel(r'Lap time $t_{\mathrm{lap}}$ [s]')
    ax.set_ylabel(ylabel)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Arrow: "faster" → LEFT
    ax.annotate('', xy=(0.73, 0.07), xytext=(0.87, 0.07),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(0.72, 0.07, 'faster', transform=ax.transAxes,
            va='center', ha='right', fontsize=6.5, color='gray')

    # Arrow: "better" → DOWN
    ax.annotate('', xy=(0.06, 0.09), xytext=(0.06, 0.19),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(0.06, 0.21, 'better', transform=ax.transAxes,
            va='bottom', ha='center', fontsize=6.5, color='gray')

    # Panel letter
    ax.text(0.02, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')

    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=7)


# ═════════════════════════════════════════════════════════════════════════════
#  Main figure  (2×2)
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto_quad(results, failures, save=True, show=True):
    """2×2 Pareto: Contouring, Lag, Position, Orientation  vs  t_lap."""

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07,
                        wspace=0.24, hspace=0.22)

    # ── Check if ec/el data exists ──
    v0 = VELOCITIES[0]
    has_ec = len(results['dq'][v0].get('rmse_ec', [])) > 0
    has_el = len(results['dq'][v0].get('rmse_el', [])) > 0

    # ── (a) Contouring RMSE ──
    if _USE_TEX:
        ylabel_ec = r'Contouring RMSE $\mathrm{RMSE}_{e_c}$ [m]'
    else:
        ylabel_ec = 'Contouring RMSE [m]'
    if has_ec:
        _plot_one_panel(axes[0, 0], results, failures,
                        metric_key='rmse_ec', ylabel=ylabel_ec,
                        panel_letter='(a)', show_legend=True,
                        show_xlabel=False)
    else:
        axes[0, 0].text(0.5, 0.5, 'No contouring data\n(re-run sweep)',
                        transform=axes[0, 0].transAxes, ha='center', va='center',
                        fontsize=11, color='gray')
        axes[0, 0].set_ylabel(ylabel_ec)
        axes[0, 0].text(0.02, 0.97, '(a)', transform=axes[0, 0].transAxes,
                        fontsize=11, fontweight='bold', va='top', ha='left')

    # ── (b) Lag RMSE ──
    if _USE_TEX:
        ylabel_el = r'Lag RMSE $\mathrm{RMSE}_{e_\ell}$ [m]'
    else:
        ylabel_el = 'Lag RMSE [m]'
    if has_el:
        _plot_one_panel(axes[0, 1], results, failures,
                        metric_key='rmse_el', ylabel=ylabel_el,
                        panel_letter='(b)', show_legend=False,
                        show_xlabel=False)
    else:
        axes[0, 1].text(0.5, 0.5, 'No lag data\n(re-run sweep)',
                        transform=axes[0, 1].transAxes, ha='center', va='center',
                        fontsize=11, color='gray')
        axes[0, 1].set_ylabel(ylabel_el)
        axes[0, 1].text(0.02, 0.97, '(b)', transform=axes[0, 1].transAxes,
                        fontsize=11, fontweight='bold', va='top', ha='left')

    # ── (c) Position RMSE ──
    if _USE_TEX:
        ylabel_pos = r'Position RMSE $\mathrm{RMSE}_{\mathbf{e}_p}$ [m]'
    else:
        ylabel_pos = 'Position RMSE [m]'
    _plot_one_panel(axes[1, 0], results, failures,
                    metric_key='rmse_pos', ylabel=ylabel_pos,
                    panel_letter='(c)', show_legend=False,
                    show_xlabel=True)

    # ── (d) Orientation RMSE ──
    if _USE_TEX:
        ylabel_ori = r'Orientation RMSE $\mathrm{RMSE}_{\mathbf{e}_\psi}$ [rad]'
    else:
        ylabel_ori = 'Orientation RMSE [rad]'
    _plot_one_panel(axes[1, 1], results, failures,
                    metric_key='rmse_ori', ylabel=ylabel_ori,
                    panel_letter='(d)', show_legend=False,
                    show_xlabel=True)

    # ── Suptitle ──
    if _USE_TEX:
        suptitle = (r'\textbf{Speed--Accuracy Pareto Frontiers}'
                    r'  (inertial-frame metrics)')
        subtitle = (r'Error bars = IQR (P25--P75) across Monte Carlo runs'
                    r'  $\bullet$  annotations = $v_{\theta}^{\max}$ [m/s]')
    else:
        suptitle = 'Speed–Accuracy Pareto Frontiers (inertial-frame metrics)'
        subtitle = ('Error bars = IQR (P25–P75) across Monte Carlo runs'
                    '  •  annotations = v_θ^max [m/s]')

    fig.suptitle(f'{suptitle}\n{subtitle}', fontsize=10, y=0.98)

    if save:
        for ext in ('pdf', 'png'):
            out = os.path.join(_FIG_DIR, f'fig_pareto_quad.{ext}')
            fig.savefig(out, dpi=300, bbox_inches='tight')
        print('✓ fig_pareto_quad saved (pdf + png)')

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2 — 2×2 Pareto: ec, el, pos, ori  vs  t_lap')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not call plt.show()')
    args = parser.parse_args()

    try:
        print('[INFO] Loading real sweep data...')
        results, failures = load_sweep_data()
        v0 = VELOCITIES[0]
        has_tlap = (len(results['dq'][v0]['t_lap']) > 0 or
                    len(results['base'][v0]['t_lap']) > 0)
        if not has_tlap:
            print('[WARN] .mat has no t_lap data.')
            return
        print('[INFO] Data loaded OK')

        # Report availability of ec/el
        has_ec = len(results['dq'][v0].get('rmse_ec', [])) > 0
        has_el = len(results['dq'][v0].get('rmse_el', [])) > 0
        if has_ec and has_el:
            print('[INFO] Contouring and Lag data found ✓')
        else:
            print('[WARN] No ec/el data — panels (a)(b) will show placeholder.')
            print('       Re-run 2_run_experiment2_sweep.py to generate them.')

    except FileNotFoundError as e:
        print(f'[WARN] {e}')
        return

    plot_pareto_quad(results, failures, save=True, show=not args.no_show)

    if not args.no_show:
        plt.show()

    print('\n✓ Quad Pareto plot done.')


if __name__ == '__main__':
    main()
