#!/usr/bin/env python3
"""
plot_montecarlo.py — Visualisation for Experiment 3: Monte Carlo Random Poses.

Generates publication-quality figures:
  1. Boxplot comparison of RMSE metrics  (ec, el, pos, ori)
  2. Histogram of solver times
  3. Control effort comparison
  4. Convergence rate bar chart
  5. Scatter: initial distance vs RMSE_ec  (robustness)
  6. Combined 2×3 summary figure

Usage:
    python plots/plot_montecarlo.py
"""

import os, sys
import numpy as np
from scipy.io import loadmat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_SCRIPT)
sys.path.insert(0, _ROOT)
from config.result_paths import experiment_dirs

_EXP3_DIR = experiment_dirs('experiment3')
_DATA   = os.path.join(str(_EXP3_DIR['data']), 'montecarlo_data.mat')
_OUT    = str(_EXP3_DIR['figures'])
_LEGACY_DATA = os.path.join(_ROOT, 'results', 'experiment3', 'montecarlo_data.mat')
os.makedirs(_OUT, exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt

# ── Style — IEEEtran ─────────────────────────────────────────────────────────
_USE_TEX = True
try:
    matplotlib.rcParams.update({'text.usetex': True})
    _fig_test = plt.figure(); plt.close(_fig_test)
except Exception:
    _USE_TEX = False

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
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

COL_W = 3.5   # IEEEtran single column [in]


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def _load():
    if not os.path.isfile(_DATA):
        if not os.path.isfile(_LEGACY_DATA):
            print(f"[ERROR] Data file not found: {_DATA}")
            print(f"        Run 'python results/experiment3/scripts/run_experiment3.py' first.")
            sys.exit(1)
        d = loadmat(_LEGACY_DATA, squeeze_me=True)
        return d
    d = loadmat(_DATA, squeeze_me=True)
    return d


def _get(d, ctrl, key):
    """Get array from mat dict."""
    arr = d.get(f'{ctrl}_{key}', np.array([]))
    return np.atleast_1d(arr).astype(float)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 1 — RMSE Boxplots (ec, el, pos, ori)
# ═════════════════════════════════════════════════════════════════════════════

def fig_boxplots(d):
    metrics = [('rmse_ec', '$e_c$'), ('rmse_el', '$e_l$'),
               ('rmse_pos', '$e_{pos}$'), ('rmse_ori', '$e_{ori}$')]

    fig, axes = plt.subplots(1, 4, figsize=(COL_W * 2, 2.2))

    for ax, (key, label) in zip(axes, metrics):
        dq_vals   = _get(d, 'dq', key)
        base_vals = _get(d, 'base', key)

        bp = ax.boxplot([dq_vals, base_vals],
                        tick_labels=['DQ', 'Base'],
                        patch_artist=True,
                        widths=0.5,
                        medianprops=dict(color='black', linewidth=1.2),
                        flierprops=dict(marker='.', markersize=2, alpha=0.4))
        bp['boxes'][0].set_facecolor(C_DQ)
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(C_BASE)
        bp['boxes'][1].set_alpha(0.5)
        ax.set_title(label)
        ax.set_ylabel('RMSE' if ax == axes[0] else '')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_boxplots.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_boxplots")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Solver timing histograms
# ═════════════════════════════════════════════════════════════════════════════

def fig_solver_time(d):
    dq_t   = _get(d, 'dq', 'solve_mean') * 1e3     # ms
    base_t = _get(d, 'base', 'solve_mean') * 1e3

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    bins = np.linspace(0, max(np.max(dq_t), np.max(base_t)) * 1.1, 40)
    ax.hist(dq_t, bins=bins, alpha=0.6, color=C_DQ, label='DQ-MPCC')
    ax.hist(base_t, bins=bins, alpha=0.6, color=C_BASE, label='Baseline')
    ax.set_xlabel('Mean solver time per step [ms]')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_solver_time.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_solver_time")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Control effort comparison
# ═════════════════════════════════════════════════════════════════════════════

def fig_control_effort(d):
    dq_e   = _get(d, 'dq', 'ctrl_effort')
    base_e = _get(d, 'base', 'ctrl_effort')

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    bp = ax.boxplot([dq_e, base_e],
                    tick_labels=['DQ-MPCC', 'Baseline'],
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1.2))
    bp['boxes'][0].set_facecolor(C_DQ)
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_BASE)
    bp['boxes'][1].set_alpha(0.5)
    ax.set_ylabel(r'Control effort $\sum\|\Delta u\|^2$')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_ctrl_effort.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_ctrl_effort")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Convergence rate bar chart
# ═════════════════════════════════════════════════════════════════════════════

def fig_convergence(d):
    dq_succ   = _get(d, 'dq', 'success')
    base_succ = _get(d, 'base', 'success')
    N = int(d['N_runs'])

    dq_rate   = np.sum(dq_succ) / N * 100
    base_rate = np.sum(base_succ) / N * 100

    fig, ax = plt.subplots(figsize=(COL_W * 0.7, 2.2))
    bars = ax.bar(['DQ-MPCC', 'Baseline'], [dq_rate, base_rate],
                  color=[C_DQ, C_BASE], alpha=0.7, width=0.5)
    ax.set_ylabel('Convergence rate [\\%]')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, [dq_rate, base_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}\\%', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_convergence.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_convergence")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 5 — 2×2 violin plots: ec, el, ori, pos
# ═════════════════════════════════════════════════════════════════════════════

def fig_robustness_scatter(d):
    dq_ec    = _get(d, 'dq',   'rmse_ec')
    base_ec  = _get(d, 'base', 'rmse_ec')
    dq_el    = _get(d, 'dq',   'rmse_el')
    base_el  = _get(d, 'base', 'rmse_el')
    dq_ori   = _get(d, 'dq',   'rmse_ori')
    base_ori = _get(d, 'base', 'rmse_ori')
    dq_pos   = _get(d, 'dq',   'rmse_pos')
    base_pos = _get(d, 'base', 'rmse_pos')

    panels = [
        (dq_ec,  base_ec,  'RMSE $e_c$ [m]',         'Contouring error'),
        (dq_el,  base_el,  'RMSE $e_l$ [m]',         'Lag error'),
        (dq_ori, base_ori, 'RMSE $e_{ori}$ [rad]',   'Orientation error'),
        (dq_pos, base_pos, 'RMSE $e_{pos}$ [m]',     'Position error'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(COL_W * 2, 4.2))

    for ax, (dq_v, base_v, ylabel, title) in zip(axes.flat, panels):
        parts = ax.violinplot([dq_v, base_v],
                              positions=[1, 2],
                              showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], [C_DQ, C_BASE]):
            pc.set_facecolor(col)
            pc.set_alpha(0.55)
        for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
            parts[part].set_color('black')
            parts[part].set_linewidth(0.8)

        # Overlay individual points (strip)
        np.random.seed(0)
        jitter = np.random.uniform(-0.08, 0.08, len(dq_v))
        ax.scatter(1 + jitter[:len(dq_v)],   dq_v,
                   s=4, color=C_DQ,   alpha=0.35, zorder=3)
        jitter2 = np.random.uniform(-0.08, 0.08, len(base_v))
        ax.scatter(2 + jitter2[:len(base_v)], base_v,
                   s=4, color=C_BASE, alpha=0.35, zorder=3)

        # Annotate medians
        med_dq   = np.median(dq_v)
        med_base = np.median(base_v)
        y_range  = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1
        offset   = (max(dq_v.max(), base_v.max()) - min(dq_v.min(), base_v.min())) * 0.03
        ax.text(1, med_dq   + offset, f'{med_dq:.3f}',
                ha='center', va='bottom', fontsize=6.5, color=C_DQ,   fontweight='bold')
        ax.text(2, med_base + offset, f'{med_base:.3f}',
                ha='center', va='bottom', fontsize=6.5, color=C_BASE, fontweight='bold')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['DQ-MPCC', 'Baseline'], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    # Shared legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_DQ,   alpha=0.6, label='DQ-MPCC'),
                        Patch(color=C_BASE, alpha=0.6, label='Baseline')],
               loc='upper center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_robustness.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_robustness")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 6 — Combined 2×3 summary
# ═════════════════════════════════════════════════════════════════════════════

def fig_summary(d):
    init_dists  = np.atleast_1d(d['init_dists']).astype(float)
    dq_succ     = _get(d, 'dq', 'success').astype(bool)
    base_succ   = _get(d, 'base', 'success').astype(bool)
    N           = int(d['N_runs'])
    vel         = float(d['velocity'])

    dq_ok_idx   = np.where(dq_succ)[0]
    base_ok_idx = np.where(base_succ)[0]

    fig, axes = plt.subplots(2, 3, figsize=(COL_W * 2, 4.0))

    # (0,0) Convergence
    ax = axes[0, 0]
    dq_rate   = np.sum(dq_succ) / N * 100
    base_rate = np.sum(base_succ) / N * 100
    bars = ax.bar(['DQ', 'Base'], [dq_rate, base_rate],
                  color=[C_DQ, C_BASE], alpha=0.7, width=0.5)
    ax.set_ylabel('Conv. rate [\\%]')
    ax.set_ylim(0, 105)
    for b, v in zip(bars, [dq_rate, base_rate]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    # (0,1) RMSE ec boxplot
    ax = axes[0, 1]
    dq_ec   = _get(d, 'dq', 'rmse_ec')
    base_ec = _get(d, 'base', 'rmse_ec')
    bp = ax.boxplot([dq_ec, base_ec], tick_labels=['DQ', 'Base'],
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1))
    bp['boxes'][0].set_facecolor(C_DQ);  bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_BASE); bp['boxes'][1].set_alpha(0.5)
    ax.set_ylabel('RMSE $e_c$')

    # (0,2) RMSE ori boxplot
    ax = axes[0, 2]
    dq_ori   = _get(d, 'dq', 'rmse_ori')
    base_ori = _get(d, 'base', 'rmse_ori')
    bp = ax.boxplot([dq_ori, base_ori], tick_labels=['DQ', 'Base'],
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1))
    bp['boxes'][0].set_facecolor(C_DQ);  bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_BASE); bp['boxes'][1].set_alpha(0.5)
    ax.set_ylabel('RMSE $e_{ori}$')

    # (1,0) Solver time
    ax = axes[1, 0]
    dq_t   = _get(d, 'dq', 'solve_mean') * 1e3
    base_t = _get(d, 'base', 'solve_mean') * 1e3
    if len(dq_t) > 0 and len(base_t) > 0:
        bins = np.linspace(0, max(np.max(dq_t), np.max(base_t)) * 1.1, 30)
        ax.hist(dq_t, bins=bins, alpha=0.6, color=C_DQ, label='DQ')
        ax.hist(base_t, bins=bins, alpha=0.6, color=C_BASE, label='Base')
        ax.legend(fontsize=6)
    ax.set_xlabel('Solve time [ms]')
    ax.set_ylabel('Freq.')

    # (1,1) Control effort
    ax = axes[1, 1]
    dq_e   = _get(d, 'dq', 'ctrl_effort')
    base_e = _get(d, 'base', 'ctrl_effort')
    bp = ax.boxplot([dq_e, base_e], tick_labels=['DQ', 'Base'],
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1))
    bp['boxes'][0].set_facecolor(C_DQ);  bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_BASE); bp['boxes'][1].set_alpha(0.5)
    ax.set_ylabel(r'$\sum\|\Delta u\|^2$')

    # (1,2) Mini violin: e_ori (key DQ advantage)
    ax = axes[1, 2]
    dq_ori_s  = _get(d, 'dq',  'rmse_ori')
    base_ori_s= _get(d, 'base','rmse_ori')
    parts = ax.violinplot([dq_ori_s, base_ori_s],
                          positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts['bodies'], [C_DQ, C_BASE]):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[part].set_color('black'); parts[part].set_linewidth(0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['DQ', 'Base'], fontsize=7)
    ax.set_ylabel('RMSE $e_{ori}$')

    fig.suptitle(f'Experiment 3: Monte Carlo ($v = {vel:.0f}$ m/s, $N = {N}$)',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_summary.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_summary")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 7 — Isometric 3D view of all trajectories (side-by-side)
# ═════════════════════════════════════════════════════════════════════════════

def fig_3d_trajectories(d):
    """Two-panel isometric 3D: all Monte Carlo trajectories for each controller,
    with the reference path in black.  One column per controller."""
    # Fix mpl_toolkits namespace conflict (system vs pip matplotlib)
    import mpl_toolkits
    _local_mpl = os.path.expanduser(
        '~/.local/lib/python3.10/site-packages/mpl_toolkits')
    if os.path.isdir(_local_mpl):
        mpl_toolkits.__path__ = [_local_mpl]
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    # Register 3d projection if not yet available
    from matplotlib.projections import projection_registry
    if '3d' not in projection_registry._all_projection_types:
        projection_registry.register(Axes3D)

    # Reference path
    ref_x = np.atleast_1d(d.get('ref_x', np.array([]))).astype(float)
    ref_y = np.atleast_1d(d.get('ref_y', np.array([]))).astype(float)
    ref_z = np.atleast_1d(d.get('ref_z', np.array([]))).astype(float)
    if ref_x.size == 0:
        print("  ⚠ fig_mc_3d: no reference path in data — skipping")
        return

    # Load trajectories (object arrays from .mat)
    dq_traj_raw   = d.get('dq_trajectories', np.array([]))
    base_traj_raw = d.get('base_trajectories', np.array([]))

    def _unpack(raw):
        """Return list of (3, K) arrays."""
        out = []
        raw = np.atleast_1d(raw)
        for item in raw:
            arr = np.atleast_2d(item).astype(float)
            if arr.shape[0] == 3:
                out.append(arr)
            elif arr.shape[1] == 3:
                out.append(arr.T)
        return out

    def _trim_large_jumps(traj, max_jump=2.5):
        if traj.shape[1] < 2:
            return traj
        jumps = np.linalg.norm(np.diff(traj, axis=1), axis=0)
        bad = np.where(jumps > max_jump)[0]
        if bad.size == 0:
            return traj
        return traj[:, :bad[0] + 1]

    dq_trajs   = _unpack(dq_traj_raw)
    base_trajs = _unpack(base_traj_raw)

    if len(dq_trajs) == 0 and len(base_trajs) == 0:
        print("  ⚠ fig_mc_3d: no trajectories in data — skipping "
              "(re-run experiment3 to save them)")
        return

    fig = plt.figure(figsize=(COL_W * 2, 3.8))

    elev, azim = 30, -55

    for col, (trajs, label, c_traj) in enumerate([
        (base_trajs, 'Baseline MPCC', C_BASE),
        (dq_trajs,   'DQ-MPCC',      C_DQ),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')

        # Reference path (thick black)
        ax.plot(ref_x, ref_y, ref_z, 'k-', lw=1.8, alpha=0.85,
                label='Reference', zorder=10)

        # All trajectories
        for traj in trajs:
            traj = _trim_large_jumps(traj)
            ax.plot(traj[0], traj[1], traj[2],
                    color=c_traj, lw=0.35, alpha=0.35)
            # Mark start point
            ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0],
                       color=c_traj, s=6, alpha=0.5, zorder=5)

        ax.set_xlabel('$x$ [m]', fontsize=7, labelpad=2)
        ax.set_ylabel('$y$ [m]', fontsize=7, labelpad=2)
        ax.set_zlabel('$z$ [m]', fontsize=7, labelpad=2)
        ax.set_title(label, fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)

        # Equal aspect (approx)
        all_pts = np.column_stack([ref_x, ref_y, ref_z])
        for tr in trajs:
            all_pts = np.vstack([all_pts, tr.T])
        mx = all_pts.max(axis=0)
        mn = all_pts.min(axis=0)
        rng = (mx - mn).max() / 2 * 1.05
        mid = (mx + mn) / 2
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_3d.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_3d")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 8 — XY / XZ projections with initial points
# ═════════════════════════════════════════════════════════════════════════════

def fig_planar_projections(d):
    ref_x = np.atleast_1d(d.get('ref_x', np.array([]))).astype(float)
    ref_y = np.atleast_1d(d.get('ref_y', np.array([]))).astype(float)
    ref_z = np.atleast_1d(d.get('ref_z', np.array([]))).astype(float)
    if ref_x.size == 0:
        print("  ⚠ fig_mc_planar: no reference path in data — skipping")
        return

    dq_traj_raw   = d.get('dq_trajectories', np.array([]))
    base_traj_raw = d.get('base_trajectories', np.array([]))

    def _unpack(raw):
        out = []
        raw = np.atleast_1d(raw)
        for item in raw:
            arr = np.atleast_2d(item).astype(float)
            if arr.shape[0] == 3:
                out.append(arr)
            elif arr.shape[1] == 3:
                out.append(arr.T)
        return out

    def _trim_large_jumps(traj, max_jump=2.5):
        if traj.shape[1] < 2:
            return traj
        jumps = np.linalg.norm(np.diff(traj, axis=1), axis=0)
        bad = np.where(jumps > max_jump)[0]
        if bad.size == 0:
            return traj
        return traj[:, :bad[0] + 1]

    dq_trajs   = _unpack(dq_traj_raw)
    base_trajs = _unpack(base_traj_raw)
    if len(dq_trajs) == 0 and len(base_trajs) == 0:
        print("  ⚠ fig_mc_planar: no trajectories in data — skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(COL_W * 2, 4.2))
    panels = [
        (axes[0, 0], base_trajs, 'Baseline MPCC', 'xy'),
        (axes[0, 1], dq_trajs,   'DQ-MPCC',       'xy'),
        (axes[1, 0], base_trajs, 'Baseline MPCC', 'xz'),
        (axes[1, 1], dq_trajs,   'DQ-MPCC',       'xz'),
    ]

    for ax, trajs, title, plane in panels:
        if plane == 'xy':
            ax.plot(ref_x, ref_y, 'k-', lw=1.6, alpha=0.85, label='Reference')
            for traj in trajs:
                traj = _trim_large_jumps(traj)
                ax.plot(traj[0], traj[1], color=C_BASE if 'Base' in title else C_DQ,
                        lw=0.45, alpha=0.35)
                ax.scatter(traj[0, 0], traj[1, 0], color=C_BASE if 'Base' in title else C_DQ,
                           s=10, alpha=0.65, marker='o')
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$y$ [m]')
            ax.set_title(f'{title} — XY')
        else:
            ax.plot(ref_x, ref_z, 'k-', lw=1.6, alpha=0.85, label='Reference')
            for traj in trajs:
                traj = _trim_large_jumps(traj)
                ax.plot(traj[0], traj[2], color=C_BASE if 'Base' in title else C_DQ,
                        lw=0.45, alpha=0.35)
                ax.scatter(traj[0, 0], traj[2, 0], color=C_BASE if 'Base' in title else C_DQ,
                           s=10, alpha=0.65, marker='o')
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$z$ [m]')
            ax.set_title(f'{title} — XZ')
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(_OUT, f'fig_mc_planar.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  ✓ fig_mc_planar")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("[INFO] Loading Monte Carlo data ...")
    d = _load()
    vel = float(d['velocity'])
    N   = int(d['N_runs'])
    print(f"[INFO] v = {vel} m/s,  N = {N}")

    fig_boxplots(d)
    fig_solver_time(d)
    fig_control_effort(d)
    fig_convergence(d)
    fig_robustness_scatter(d)
    fig_summary(d)
    fig_3d_trajectories(d)
    fig_planar_projections(d)

    print(f"\n✓ All figures saved to {_OUT}")


if __name__ == '__main__':
    main()
