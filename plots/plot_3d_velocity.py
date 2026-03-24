#!/usr/bin/env python3
"""
plot_3d_velocity.py — 3-D isometric trajectory comparison (velocity sweep).

Reads  results/experiment2/trajectory_3d_data.mat  and generates a
two-panel figure:

  Left  : MPCC Baseline     Right : DQ-MPCC

Design:
  • All simulation velocities are drawn in **light grey** (background family).
  • ONE highlighted velocity (V_HIGHLIGHT, future real-flight candidate)
    is drawn in **red** with thicker linewidth, as if it were "the real run".
  • If experimental data keys (exp_base_x, exp_dq_x, …) exist in the .mat,
    they are overlaid with a distinct marker style.

Usage
-----
    python plots/plot_3d_velocity.py
    python plots/plot_3d_velocity.py --no-show
    python plots/plot_3d_velocity.py --vreal 10
"""

import os, sys, argparse
import numpy as np
from scipy.io import loadmat

# ── Axes3D workaround (system vs pip mpl_toolkits conflict) ──────────────
import mpl_toolkits
_local_mpl = os.path.expanduser(
    '~/.local/lib/python3.10/site-packages/mpl_toolkits')
if os.path.isdir(_local_mpl):
    mpl_toolkits.__path__ = [_local_mpl]

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import projection_registry
if '3d' not in projection_registry._all_projection_types:
    projection_registry.register(Axes3D)

import matplotlib.lines as mlines

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
    'font.size':         9,
    'axes.labelsize':    9,
    'legend.fontsize':   7.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'figure.dpi':        150,
    'text.usetex':       _USE_TEX,
})

COL_W = 3.5          # IEEE single-column width in inches

# ── Velocity to highlight (simulated "real flight") ──────────────────────
V_HIGHLIGHT = 10     # m/s  — change when you pick the real-flight speed

# ── Colours ──────────────────────────────────────────────────────────────
C_GREY  = '#b0b0b0'  # all other sims
C_REAL  = '#d62728'   # highlighted / future real
C_REF   = 'k'        # reference path

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)
_DATA    = os.path.join(_ROOT, 'results', 'experiment2',
                        'trajectory_3d_data.mat')
_OUT_DIR = os.path.join(_ROOT, 'results', 'experiment2')
os.makedirs(_OUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Plot
# ═════════════════════════════════════════════════════════════════════════════

def fig_3d_velocity(v_highlight=V_HIGHLIGHT, save=True, show=True):
    """Two-panel 3-D view: Baseline (left) vs DQ-MPCC (right).

    All velocities in grey; *v_highlight* in red (future real-flight).
    If experimental keys exist in the .mat they are overlaid automatically.
    """

    d    = loadmat(_DATA, squeeze_me=True)
    vels = np.atleast_1d(d['velocities']).astype(int)

    ref_x = np.asarray(d['ref_x']).ravel()
    ref_y = np.asarray(d['ref_y']).ravel()
    ref_z = np.asarray(d['ref_z']).ravel()

    # Check if experimental data exists
    has_exp = 'exp_base_x' in d

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(COL_W * 2, 3.8))

    panels = [
        ('base', r'MPCC Baseline'),
        ('dq',   r'DQ-MPCC'),
    ]

    axes = []
    for i, (prefix, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        axes.append(ax)

        # Reference path
        ax.plot(ref_x, ref_y, ref_z,
                color=C_REF, ls='--', lw=1.4, alpha=0.50,
                label='Reference', zorder=0)

        # ── Grey background: all OTHER velocities ────────────────────
        first_grey = True
        for v in vels:
            if v == v_highlight:
                continue
            x = np.asarray(d[f'{prefix}_v{v}_x']).ravel()
            y = np.asarray(d[f'{prefix}_v{v}_y']).ravel()
            z = np.asarray(d[f'{prefix}_v{v}_z']).ravel()
            lbl = r'Other $v_{\theta}^{\max}$' if first_grey else None
            ax.plot(x, y, z,
                    color=C_GREY, lw=1.2, alpha=0.3,
                    label=lbl, zorder=1)
            first_grey = False

        # ── Red highlight: the "real-flight" velocity ────────────────
        if v_highlight in vels:
            x = np.asarray(d[f'{prefix}_v{v_highlight}_x']).ravel()
            y = np.asarray(d[f'{prefix}_v{v_highlight}_y']).ravel()
            z = np.asarray(d[f'{prefix}_v{v_highlight}_z']).ravel()
            lbl_hl = (rf'Sim $v_{{\theta}}^{{\max}}={v_highlight}$\,m/s')
            ax.plot(x, y, z,
                    color=C_REAL, lw=2, alpha=0.92,
                    label=lbl_hl, zorder=50)

        # ── Experimental overlay (if available) ──────────────────────
        if has_exp:
            ex = np.asarray(d[f'exp_{prefix}_x']).ravel()
            ey = np.asarray(d[f'exp_{prefix}_y']).ravel()
            ez = np.asarray(d[f'exp_{prefix}_z']).ravel()
            ax.plot(ex, ey, ez,
                    color=C_REAL, lw=1.0, ls=':', alpha=0.90,
                    marker='o', markersize=1.5, markevery=20,
                    label='Experimental', zorder=100)

        ax.set_xlabel(r'$x$ [m]', labelpad=2)
        ax.set_ylabel(r'$y$ [m]', labelpad=2)
        ax.set_zlabel(r'$z$ [m]', labelpad=2)
        ax.set_title(title, pad=4, fontsize=10)
        ax.view_init(elev=28, azim=-52)
        ax.tick_params(pad=1)

        # ── Equal scale (1 m = 1 m) on all 3 axes, tight limits ────
        all_x = np.concatenate([l.get_data_3d()[0] for l in ax.get_lines()])
        all_y = np.concatenate([l.get_data_3d()[1] for l in ax.get_lines()])
        all_z = np.concatenate([l.get_data_3d()[2] for l in ax.get_lines()])

        margin = 0.5  # metres of padding around data
        x_min, x_max = all_x.min() - margin, all_x.max() + margin
        y_min, y_max = all_y.min() - margin, all_y.max() + margin
        z_min, z_max = all_z.min() - margin, all_z.max() + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # box_aspect proportional to data range → 1 m looks the same on every axis
        rx, ry, rz = (x_max - x_min), (y_max - y_min), (z_max - z_min)
        ax.set_box_aspect([rx, ry, rz])

        # Clean panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('grey')
        ax.yaxis.pane.set_edgecolor('grey')
        ax.zaxis.pane.set_edgecolor('grey')

    # ── Shared legend below ──────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               fontsize=7.5)

    fig.subplots_adjust(wspace=0.22, top=0.92, bottom=0.10)

    if save:
        for ext in ('pdf', 'png'):
            out = os.path.join(_OUT_DIR, f'fig_3d_velocity.{ext}')
            fig.savefig(out, dpi=300, bbox_inches='tight')
        print('✓ fig_3d_velocity saved (pdf + png)')

    if show:
        plt.show()

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='3-D velocity trajectory comparison')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--vreal', type=int, default=V_HIGHLIGHT,
                        help='Velocity to highlight in red (default: %(default)s)')
    args = parser.parse_args()

    fig_3d_velocity(v_highlight=args.vreal, save=True, show=not args.no_show)
    print('\n✓ 3-D velocity plot done.')


if __name__ == '__main__':
    main()
