#!/usr/bin/env python3
"""
plot_experiment2_boxplot.py — Generate velocity-sweep boxplot figure.

Reads  experiment2_results/velocity_sweep_data.mat  and produces
the double boxplot (RMSE_pos + RMSE_ori vs v_theta_max).

Also generates a LaTeX summary table.

Usage:
    python plot_experiment2_boxplot.py          # from real data
    python plot_experiment2_boxplot.py --mock   # synthetic data for prototyping
    python plot_experiment2_boxplot.py --no-show
"""

import os, sys, argparse
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ────────────────────────────────────────────────────────────────────
_USE_TEX = True
try:
    matplotlib.rcParams.update({'text.usetex': True})
    fig_test = plt.figure(); plt.close(fig_test)
except Exception:
    _USE_TEX = False

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.size':          9,
    'axes.labelsize':     9,
    'legend.fontsize':    8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'figure.dpi':         150,
    'text.usetex':        _USE_TEX,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

C_DQ   = '#1f77b4'
C_BASE = '#d62728'

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)
_OUT_DIR = os.path.join(_ROOT, "results", "experiment2")
os.makedirs(_OUT_DIR, exist_ok=True)

# Import VELOCITIES from config so there's one source of truth
sys.path.insert(0, _ROOT)
try:
    from config.sweep_config import VELOCITIES
except ImportError:
    VELOCITIES = [4.74, 5.32, 6.1]


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_sweep_data():
    """Load from .mat file."""
    mat_path = os.path.join(_OUT_DIR, 'velocity_sweep_data.mat')
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Not found: {mat_path}")
    d = loadmat(mat_path)

    results = {'dq': {}, 'base': {}}
    failures = {'dq': {}, 'base': {}}
    for ctrl in ['dq', 'base']:
        for v in VELOCITIES:
            vstr = f"{v:.2f}".replace('.', 'p')
            rp = d.get(f'{ctrl}_v{vstr}_rmse_pos', np.array([])).ravel()
            ro = d.get(f'{ctrl}_v{vstr}_rmse_ori', np.array([])).ravel()
            nf = int(d.get(f'{ctrl}_v{vstr}_failures', 0))
            results[ctrl][v] = {'rmse_pos': rp, 'rmse_ori': ro}
            failures[ctrl][v] = nf
    return results, failures


def generate_mock_data():
    """Realistic synthetic data for layout prototyping."""
    np.random.seed(42)
    N = 50
    results = {'dq': {}, 'base': {}}
    failures = {'dq': {}, 'base': {}}

    speed_scale = {v: 1.0 + (v - VELOCITIES[0]) / (VELOCITIES[-1] - VELOCITIES[0]) * 1.3
                   for v in VELOCITIES} if len(VELOCITIES) > 1 else {VELOCITIES[0]: 1.0}

    for vel in VELOCITIES:
        sc = speed_scale[vel]
        results['dq'][vel] = {
            'rmse_pos': np.abs(np.random.normal(0.03*sc, 0.008*sc, N)),
            'rmse_ori': np.abs(np.random.normal(0.04*sc, 0.010*sc, N)),
        }
        base_pos = np.abs(np.random.normal(0.06*sc, 0.020*sc, N))
        base_ori = np.abs(np.random.normal(0.08*sc, 0.025*sc, N))
        if sc > 1.5:
            n_out = max(1, int((sc - 1.0) * 3))
            idx = np.random.choice(N, n_out, replace=False)
            base_pos[idx] *= 3.5
            base_ori[idx] *= 3.5
        results['base'][vel] = {'rmse_pos': base_pos, 'rmse_ori': base_ori}
        failures['dq'][vel]   = 0
        failures['base'][vel] = max(0, int((sc - 1.0) * 2))

    return results, failures


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _style_boxplot(bp, color):
    """Apply consistent style to a boxplot."""
    for patch in bp['boxes']:
        patch.set_facecolor(matplotlib.colors.to_rgba(color, 0.4))
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for whisker in bp['whiskers']:
        whisker.set_color(color)
        whisker.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color(color)
        cap.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set(marker='+', markersize=4,
                  markeredgecolor=color, markeredgewidth=0.8)


def _draw_data(ax, data_list, positions, color, width, N_BOX_MIN=5):
    """Smart renderer: boxplot when N≥N_BOX_MIN, individual points + mean bar otherwise.

    Parameters
    ----------
    data_list : list of ndarray   – one array per velocity (may include empty arrays).
    positions : list of float     – x position for each velocity.
    color     : str               – matplotlib color.
    width     : float             – box / error-bar width.
    N_BOX_MIN : int               – minimum N to use boxplot (default 5).
    """
    # Separate indices by rendering mode
    box_pos, box_data = [], []
    dot_pos, dot_data = [], []
    for pos, d in zip(positions, data_list):
        if len(d) == 0:
            continue
        if len(d) >= N_BOX_MIN:
            box_pos.append(pos)
            box_data.append(d)
        else:
            dot_pos.append(pos)
            dot_data.append(d)

    # ── Boxplot for large N ───────────────────────────────────────────────
    if box_data:
        bp = ax.boxplot(box_data, positions=box_pos, widths=width,
                        patch_artist=True, notch=False,
                        showfliers=True, whis=1.5)
        _style_boxplot(bp, color)

    # ── Individual points + mean bar for small N ──────────────────────────
    rng = np.random.default_rng(0)
    for pos, d in zip(dot_pos, dot_data):
        # Jitter points horizontally so they don't overlap
        jitter = rng.uniform(-width * 0.25, width * 0.25, size=len(d))
        ax.scatter(pos + jitter, d,
                   color=color, s=22, zorder=4, alpha=0.85,
                   edgecolors='none')
        # Mean line
        ax.hlines(np.mean(d), pos - width * 0.40, pos + width * 0.40,
                  colors=color, linewidths=1.8, zorder=5)
        # Min–max whisker
        ax.vlines(pos, np.min(d), np.max(d),
                  colors=color, linewidths=1.0, linestyles='--', zorder=3,
                  alpha=0.6)


def plot_velocity_sweep_boxplot(results, failures, save=True):
    """Velocity sweep figure: RMSE_pos (top) + RMSE_ori (bottom) vs v_theta_max.

    Rendering is adaptive:
      • N ≥ 5  → full boxplot  (median, IQR box, Tukey whiskers, outlier +)
      • N < 5  → individual dots + mean bar + min/max whisker
      • N = 0  → empty slot with failure annotation

    Y-limits are computed automatically from the data.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.subplots_adjust(hspace=0.08, left=0.10, right=0.97,
                        top=0.94, bottom=0.12)

    width = 0.35
    positions_dq   = [i * 1.0 - 0.20 for i in range(len(VELOCITIES))]
    positions_base = [i * 1.0 + 0.20 for i in range(len(VELOCITIES))]

    for ax, metric, ylabel in [
        (ax1, 'rmse_pos', r'Position RMSE [m]'),
        (ax2, 'rmse_ori', r'Orientation RMSE [rad]'),
    ]:
        data_dq   = [results['dq'][v][metric]   for v in VELOCITIES]
        data_base = [results['base'][v][metric] for v in VELOCITIES]

        _draw_data(ax, data_dq,   positions_dq,   C_DQ,   width)
        _draw_data(ax, data_base, positions_base, C_BASE, width)

        # Auto y-limits: gather all valid data
        all_vals = np.concatenate(
            [d for d in data_dq + data_base if len(d) > 0]
        ) if any(len(d) > 0 for d in data_dq + data_base) else np.array([0.0])
        y_max = float(np.max(all_vals)) * 1.20
        y_max = max(y_max, 0.1)
        ax.set_ylim(0, y_max)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # N label above each group (shows how many successful runs)
    for i, v in enumerate(VELOCITIES):
        y_top = ax1.get_ylim()[1]
        for ctrl, positions, color in [
            ('dq',   positions_dq,   C_DQ),
            ('base', positions_base, C_BASE),
        ]:
            n_ok = len(results[ctrl][v]['rmse_pos'])
            nf   = failures[ctrl][v]
            total = n_ok + nf
            if n_ok == 0:
                # All runs failed — annotate clearly
                lbl = (r'$\times$all' if _USE_TEX else 'x all')
                ax1.text(positions[i], y_top * 0.96, lbl,
                         ha='center', va='top', fontsize=7,
                         color=color, fontweight='bold')
            elif nf > 0:
                # Some failures — show n_ok / total
                lbl = rf'$n$={n_ok}/{total}' if _USE_TEX else f'n={n_ok}/{total}'
                ax1.text(positions[i], y_top * 0.96, lbl,
                         ha='center', va='top', fontsize=6.5,
                         color=color, fontweight='bold')
            else:
                # All succeeded — just show N quietly
                lbl = rf'$n$={n_ok}' if _USE_TEX else f'n={n_ok}'
                ax1.text(positions[i], y_top * 0.97, lbl,
                         ha='center', va='top', fontsize=6,
                         color=color, alpha=0.75)

    # Legend on top subplot
    patch_dq   = mpatches.Patch(facecolor=matplotlib.colors.to_rgba(C_DQ, 0.4),
                                edgecolor=C_DQ, linewidth=1.2, label='DQ-MPCC')
    patch_base = mpatches.Patch(facecolor=matplotlib.colors.to_rgba(C_BASE, 0.4),
                                edgecolor=C_BASE, linewidth=1.2, label='Baseline MPCC')
    ax1.legend(handles=[patch_dq, patch_base], loc='upper left', framealpha=0.9)

    # X-axis labels on bottom subplot only
    tick_pos = [i * 1.0 for i in range(len(VELOCITIES))]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([f'{v} m/s' for v in VELOCITIES])
    ax2.set_xlabel(r'Maximum progress speed $v_{\theta,\max}$')
    ax2.set_xlim(-0.6, len(VELOCITIES) - 0.4)

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUT_DIR,
                        f'fig_velocity_sweep_boxplot.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_velocity_sweep_boxplot saved")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  LaTeX table
# ═════════════════════════════════════════════════════════════════════════════

def save_sweep_latex_table(results, failures, filename=None):
    """Generate LaTeX table: median RMSE ± IQR for each velocity."""
    if filename is None:
        filename = os.path.join(_OUT_DIR, 'velocity_sweep_table.tex')

    lines = []
    lines.append(r'% Requires: \usepackage{booktabs,siunitx}')
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Velocity sweep: median RMSE (IQR) over 50 Monte Carlo runs.}')
    lines.append(r'  \label{tab:velocity_sweep}')
    lines.append(r'  \setlength{\tabcolsep}{3pt}')
    lines.append(r'  \begin{tabular}{l cc cc}')
    lines.append(r'    \toprule')
    lines.append(r'    & \multicolumn{2}{c}{\textbf{DQ-MPCC}}')
    lines.append(r'    & \multicolumn{2}{c}{\textbf{Baseline MPCC}} \\')
    lines.append(r'    \cmidrule(lr){2-3} \cmidrule(lr){4-5}')
    lines.append(r'    $v_{\theta,\max}$ [m/s]')
    lines.append(r'    & $\|\mathbf{e}_p\|$ [m] & $\|\mathrm{Log}(q_e)\|$ [rad]')
    lines.append(r'    & $\|\mathbf{e}_p\|$ [m] & $\|\mathrm{Log}(q_e)\|$ [rad] \\')
    lines.append(r'    \midrule')

    for v in VELOCITIES:
        vals = []
        for ctrl in ['dq', 'base']:
            for metric in ['rmse_pos', 'rmse_ori']:
                d = results[ctrl][v][metric]
                if len(d) > 0:
                    med = np.median(d)
                    q1, q3 = np.percentile(d, [25, 75])
                    iqr = q3 - q1
                    vals.append(f'{med:.3f} ({iqr:.3f})')
                else:
                    vals.append(r'--')
        nf_base = failures['base'][v]
        fail_note = f' ($\\times${nf_base})' if nf_base > 0 else ''
        lines.append(f'    {v}{fail_note} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'✓ LaTeX table saved to {filename}')


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: boxplot")
    parser.add_argument('--mock', action='store_true',
                        help='Use synthetic data for prototyping')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not call plt.show()')
    args = parser.parse_args()

    if args.mock:
        print("[INFO] Using synthetic mock data")
        results, failures = generate_mock_data()
    else:
        try:
            print("[INFO] Loading real sweep data...")
            results, failures = load_sweep_data()
            print("[INFO] Data loaded OK")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print("[INFO] Falling back to mock data")
            results, failures = generate_mock_data()

    plot_velocity_sweep_boxplot(results, failures)
    save_sweep_latex_table(results, failures)

    if not args.no_show:
        plt.show()

    print("\n✓ Experiment 2 plotting done.")


if __name__ == '__main__':
    main()
