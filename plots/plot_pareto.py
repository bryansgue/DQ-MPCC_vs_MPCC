#!/usr/bin/env python3
"""
plot_experiment2_pareto.py — Pareto frontier: RMSE_pos vs t_lap.

Reads  experiment2_results/velocity_sweep_data.mat  and produces a single
figure where:

  • X-axis : median lap time  t_lap  [s]     (smaller = faster)
  • Y-axis : median position RMSE  [m]       (smaller = more accurate)
  • Each point  = one (controller, velocity) pair
  • Error bars  = IQR across the N Monte Carlo runs   (P25–P75)
  • Annotation  = v_theta_max value of that point
  • Connecting line  = Pareto curve of each controller

The controller whose curve sits lower-left dominates in BOTH speed AND accuracy.

Usage
-----
    python plot_experiment2_pareto.py            # real data
    python plot_experiment2_pareto.py --mock     # synthetic data for prototyping
    python plot_experiment2_pareto.py --no-show
"""

import os, sys, argparse
import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
            rec = d.get(f'{ctrl}_v{vstr}_rmse_ec',    np.array([])).ravel()
            rel = d.get(f'{ctrl}_v{vstr}_rmse_el',    np.array([])).ravel()
            tl = d.get(f'{ctrl}_v{vstr}_t_lap',       np.array([])).ravel()
            mv = d.get(f'{ctrl}_v{vstr}_mean_vtheta', np.array([])).ravel()
            nf = int(d.get(f'{ctrl}_v{vstr}_failures', np.array([0])).ravel()[0])
            results[ctrl][v] = {'rmse_pos': rp, 'rmse_ori': ro,
                                'rmse_ec': rec, 'rmse_el': rel,
                                't_lap': tl, 'mean_vtheta': mv}
            failures[ctrl][v] = nf
    return results, failures


def generate_mock_data():
    """Synthetic data for layout prototyping (N=20 runs per point)."""
    np.random.seed(42)
    N = 20
    results  = {'dq': {}, 'base': {}}
    failures = {'dq': {}, 'base': {}}

    # As v_max increases: MPCC gets less accurate fast, DQ-MPCC degrades slowly
    # t_lap decreases for both (faster speed → less time to finish)
    dq_rmse_nom   = {8: 0.18, 12: 0.24, 15: 0.32}
    base_rmse_nom = {8: 0.35, 12: 0.52, 15: 0.80}
    dq_tlap_nom   = {8: 13.0, 12: 9.5,  15: 7.8}
    base_tlap_nom = {8: 14.5, 12: 11.0, 15: 9.2}

    for v in VELOCITIES:
        for ctrl, rmse_nom, tlap_nom in [
            ('dq',   dq_rmse_nom[v],   dq_tlap_nom[v]),
            ('base', base_rmse_nom[v], base_tlap_nom[v]),
        ]:
            rp = np.abs(np.random.normal(rmse_nom, rmse_nom * 0.12, N))
            ro = np.abs(np.random.normal(rmse_nom * 1.3, rmse_nom * 0.15, N))
            tl = np.abs(np.random.normal(tlap_nom, tlap_nom * 0.04, N))
            mv = 100.0 / tl   # s_max / t_lap
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
#  Main plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto(results, failures, save=True, show=True):
    """Single Pareto figure: RMSE_pos vs t_lap, with IQR error bars."""

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)

    controllers = [
        ('dq',   'DQ-MPCC',      C_DQ,   'o', '-'),
        ('base', 'Baseline MPCC', C_BASE, 's', '--'),
    ]

    for ctrl, label, color, marker, ls in controllers:
        x_med, x_lo, x_hi = [], [], []   # t_lap statistics
        y_med, y_lo, y_hi = [], [], []   # rmse_pos statistics
        v_labels = []
        valid_velocities = []

        for v in VELOCITIES:
            rp = results[ctrl][v]['rmse_pos']
            tl = results[ctrl][v]['t_lap']

            # Need both metrics to plot a point
            if len(rp) == 0 or len(tl) == 0:
                continue

            # Filter paired valid samples (both finite)
            mask = np.isfinite(rp) & np.isfinite(tl)
            rp_ok = rp[mask]
            tl_ok = tl[mask]
            if len(rp_ok) == 0:
                continue

            rm, rlo, rhi = _stats(rp_ok)
            tm, tlo, thi = _stats(tl_ok)

            y_med.append(rm);  y_lo.append(rlo);  y_hi.append(rhi)
            x_med.append(tm);  x_lo.append(tlo);  x_hi.append(thi)
            v_labels.append(v)
            valid_velocities.append(v)

        if len(x_med) == 0:
            continue

        x_med = np.array(x_med)
        y_med = np.array(y_med)
        x_err = np.array([np.array(x_med) - np.array(x_lo),
                          np.array(x_hi)  - np.array(x_med)])
        y_err = np.array([np.array(y_med) - np.array(y_lo),
                          np.array(y_hi)  - np.array(y_med)])

        # Sort by t_lap so the connecting line makes sense
        order   = np.argsort(x_med)
        x_med   = x_med[order]
        y_med   = y_med[order]
        x_err   = x_err[:, order]
        y_err   = y_err[:, order]
        v_sorted = [v_labels[i] for i in order]

        # Connecting line (Pareto frontier)
        ax.plot(x_med, y_med,
                color=color, linestyle=ls, linewidth=1.4,
                zorder=2, alpha=0.7)

        # Error bars (IQR)
        ax.errorbar(x_med, y_med,
                    xerr=x_err, yerr=y_err,
                    fmt='none',
                    ecolor=color, elinewidth=1.1,
                    capsize=4, capthick=1.1,
                    zorder=3, alpha=0.85)

        # Markers (one per velocity)
        ax.scatter(x_med, y_med,
                   color=color, marker=marker,
                   s=60, zorder=5,
                   edgecolors='white', linewidths=0.6,
                   label=label)

        # Velocity annotations next to each point
        # Alternate offset direction to reduce overlap with many velocities
        for idx, (xi, yi, vv) in enumerate(zip(x_med, y_med, v_sorted)):
            nf  = failures[ctrl][vv]
            nok = len(results[ctrl][vv]['rmse_pos'])
            if _USE_TEX:
                ann = rf'$v_{{\theta}}^{{\max}}\!=\!{vv}$'
            else:
                ann = f'{vv} m/s'
            if nf > 0:
                ann += f'\n({nok}/{nok+nf} ok)'
            # Alternate: even points go upper-right, odd go lower-left
            if idx % 2 == 0:
                xyoff = (6, 5)
                ha, va = 'left', 'bottom'
            else:
                xyoff = (-6, -5)
                ha, va = 'right', 'top'
            ax.annotate(
                ann,
                xy=(xi, yi),
                xytext=xyoff,
                textcoords='offset points',
                fontsize=7.5,
                color=color,
                ha=ha, va=va,
            )

    # ── Axes labels and formatting ────────────────────────────────────────
    ax.set_xlabel(r'Lap time $t_{\mathrm{lap}}$ [s]  (median over $N$ runs)')
    ax.set_ylabel(r'Position RMSE $\mathrm{RMSE}_{\mathbf{e}_p}$ [m]  (median over $N$ runs)')

    title = (r'\textbf{Speed--Accuracy Pareto Frontier}' if _USE_TEX
             else 'Speed–Accuracy Pareto Frontier')
    subtitle = (r'Error bars = IQR (P25--P75) across Monte Carlo runs  '
                r'$\bullet$  same $v_{\theta}^{\max}$ per pair of points'
                if _USE_TEX
                else 'Error bars = IQR (P25–P75) across Monte Carlo runs'
                     '  •  same v_θ^max per pair of points')
    ax.set_title(f'{title}\n{subtitle}', fontsize=9, loc='left')

    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Arrow annotations explaining axes direction
    # Smaller t_lap = faster → arrow points LEFT
    ax.annotate('', xy=(0.78, 0.06), xytext=(0.88, 0.06),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(0.77, 0.06, 'faster', transform=ax.transAxes,
            va='center', ha='right', fontsize=7.5, color='gray')

    # Smaller RMSE = more accurate → arrow points DOWN
    ax.annotate('', xy=(0.04, 0.08), xytext=(0.04, 0.18),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(0.04, 0.20, 'more\naccurate', transform=ax.transAxes,
            va='bottom', ha='center', fontsize=7.5, color='gray')

    # Legend
    ax.legend(loc='upper right', framealpha=0.9)

    if save:
        for ext in ('pdf', 'png'):
            out = os.path.join(_FIG_DIR, f'fig_pareto_rmse_vs_tlap.{ext}')
            fig.savefig(out, dpi=300, bbox_inches='tight')
        print('✓ fig_pareto_rmse_vs_tlap saved (pdf + png)')

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2 — Pareto: RMSE_pos vs t_lap')
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
            # Quick sanity check: does the .mat have t_lap?
            v0   = VELOCITIES[0]
            vstr = f"{v0:.2f}".replace('.', 'p')
            has_tlap = len(results['dq'][v0]['t_lap']) > 0 or \
                       len(results['base'][v0]['t_lap']) > 0
            if not has_tlap:
                print('[WARN] .mat has no t_lap data — did you run the '
                      'updated 2_run_experiment2_sweep.py?')
                print('[INFO] Falling back to mock data for layout preview.')
                results, failures = generate_mock_data()
            else:
                print('[INFO] Data loaded OK')
        except FileNotFoundError as e:
            print(f'[WARN] {e}')
            print('[INFO] Falling back to mock data')
            results, failures = generate_mock_data()

    plot_pareto(results, failures, save=True, show=not args.no_show)

    if not args.no_show:
        plt.show()

    print('\n✓ Pareto plot done.')


if __name__ == '__main__':
    main()
