#!/usr/bin/env python3
"""
plot_tuning_convergence.py — Visualise Optuna tuning convergence.

Reads  tuning_history.json  produced by mpcc_tuner / dq_mpcc_tuner
and generates publication-quality convergence plots.

Usage
─────
    # Plot both controllers (default)
    python plot_tuning_convergence.py

    # Plot only MPCC
    python plot_tuning_convergence.py --mpcc-only

    # Plot only DQ-MPCC
    python plot_tuning_convergence.py --dq-only
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MPCC_HIST    = os.path.join(WORKSPACE, 'MPCC_baseline',    'tuning', 'tuning_history.json')
DQ_MPCC_HIST = os.path.join(WORKSPACE, 'DQ-MPCC_baseline', 'tuning', 'tuning_history.json')
OUT_DIR      = WORKSPACE   # save figures at workspace root


def load_history(path: str) -> dict | None:
    """Load a tuning_history.json, return None if missing."""
    if not os.path.isfile(path):
        print(f"  ⚠  Not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  ✓ Loaded {path}  ({data['n_trials']} trials)")
    return data


def extract_arrays(data: dict) -> dict:
    """Pull numpy arrays from the trial list."""
    trials = data['trials']
    return {
        'trial':          np.array([t['trial']          for t in trials]),
        'J':              np.array([t['J']              for t in trials]),
        'best_J':         np.array([t['best_J_so_far']  for t in trials]),
        'mpcc_cost':      np.array([t['mpcc_cost']      for t in trials]),
        'rmse_contorno':  np.array([t['rmse_contorno']  for t in trials]),
        'rmse_lag':       np.array([t['rmse_lag']       for t in trials]),
        'mean_vtheta':    np.array([t['mean_vtheta']    for t in trials if t['mean_vtheta'] is not None]),
        't_lap':          np.array([t['t_lap']          for t in trials if t['t_lap'] is not None]),
        'path_completed': np.array([t['path_completed'] for t in trials]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_single(data: dict, label: str, color: str, ax_J, ax_best):
    """Scatter J per trial + best-so-far line on two axes."""
    a = extract_arrays(data)
    ax_J.scatter(a['trial'], a['J'], s=12, alpha=0.45, color=color, label=label)
    ax_best.plot(a['trial'], a['best_J'], linewidth=2, color=color, label=label)


def plot_sub_metrics_single(data: dict, label: str, color: str,
                            ax_mpcc, ax_rmse, ax_vel, ax_tlap):
    """Plot sub-metric convergence curves."""
    a = extract_arrays(data)
    t = a['trial']
    ax_mpcc.scatter(t, a['mpcc_cost'],     s=10, alpha=0.4, color=color, label=label)
    ax_rmse.scatter(t, a['rmse_contorno'], s=10, alpha=0.4, color=color, label=f'{label} contorno')
    ax_rmse.scatter(t, a['rmse_lag'],      s=10, alpha=0.25, color=color, marker='x', label=f'{label} lag')
    if len(a['mean_vtheta']) == len(t):
        ax_vel.scatter(t, a['mean_vtheta'], s=10, alpha=0.4, color=color, label=label)
    if len(a['t_lap']) == len(t):
        ax_tlap.scatter(t, a['t_lap'],      s=10, alpha=0.4, color=color, label=label)


def make_plots(mpcc_data, dq_data, out_dir: str):
    """Generate all convergence figures."""

    # ── Figure 1: J convergence (scatter + best-so-far) ──────────────────
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Tuning Convergence — Objective $J$', fontsize=14, fontweight='bold')

    ax1a.set_title('All Trials (scatter)')
    ax1a.set_xlabel('Trial')
    ax1a.set_ylabel('$J$')
    ax1a.set_yscale('log')
    ax1a.grid(True, alpha=0.3)

    ax1b.set_title('Best $J$ so far')
    ax1b.set_xlabel('Trial')
    ax1b.set_ylabel('Best $J$')
    ax1b.grid(True, alpha=0.3)

    if mpcc_data:
        plot_convergence_single(mpcc_data, 'MPCC', '#2196F3', ax1a, ax1b)
    if dq_data:
        plot_convergence_single(dq_data, 'DQ-MPCC', '#E91E63', ax1a, ax1b)

    ax1a.legend()
    ax1b.legend()
    fig1.tight_layout()
    p1 = os.path.join(out_dir, 'tuning_convergence_J.png')
    fig1.savefig(p1, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved {p1}")

    # ── Figure 2: Sub-metrics ─────────────────────────────────────────────
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Tuning Sub-metrics per Trial', fontsize=14, fontweight='bold')

    ax2a.set_title('Mean MPCC Stage Cost')
    ax2a.set_xlabel('Trial'); ax2a.set_ylabel('$\\bar{J}_{\\mathrm{mpcc}}$')
    ax2a.grid(True, alpha=0.3)

    ax2b.set_title('RMSE (Contorno & Lag)')
    ax2b.set_xlabel('Trial'); ax2b.set_ylabel('RMSE [m]')
    ax2b.grid(True, alpha=0.3)

    ax2c.set_title('Mean Progress Velocity $\\bar{v}_\\theta$')
    ax2c.set_xlabel('Trial'); ax2c.set_ylabel('$\\bar{v}_\\theta$ [m/s]')
    ax2c.grid(True, alpha=0.3)

    ax2d.set_title('Lap Time $t_{\\mathrm{lap}}$')
    ax2d.set_xlabel('Trial'); ax2d.set_ylabel('$t_{\\mathrm{lap}}$ [s]')
    ax2d.grid(True, alpha=0.3)

    if mpcc_data:
        plot_sub_metrics_single(mpcc_data, 'MPCC', '#2196F3', ax2a, ax2b, ax2c, ax2d)
    if dq_data:
        plot_sub_metrics_single(dq_data, 'DQ-MPCC', '#E91E63', ax2a, ax2b, ax2c, ax2d)

    for ax in (ax2a, ax2b, ax2c, ax2d):
        ax.legend(fontsize=8)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, 'tuning_convergence_metrics.png')
    fig2.savefig(p2, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved {p2}")

    # ── Figure 3: Best-so-far overlay (compact, paper-friendly) ──────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.set_title('Tuning Convergence — Best $J$ vs Trial', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Trial', fontsize=12)
    ax3.set_ylabel('Best $J$', fontsize=12)
    ax3.grid(True, alpha=0.3)

    if mpcc_data:
        a = extract_arrays(mpcc_data)
        ax3.plot(a['trial'], a['best_J'], linewidth=2.2, color='#2196F3',
                 label=f"MPCC  (final J={a['best_J'][-1]:.3f})")
    if dq_data:
        a = extract_arrays(dq_data)
        ax3.plot(a['trial'], a['best_J'], linewidth=2.2, color='#E91E63',
                 label=f"DQ-MPCC  (final J={a['best_J'][-1]:.3f})")

    ax3.legend(fontsize=11)
    fig3.tight_layout()
    p3 = os.path.join(out_dir, 'tuning_convergence_best.png')
    fig3.savefig(p3, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved {p3}")

    plt.close('all')

    # ── Save convergence data to .mat ─────────────────────────────────────
    mat_dict = {}
    for tag, d in [('mpcc', mpcc_data), ('dq_mpcc', dq_data)]:
        if d is None:
            continue
        a = extract_arrays(d)
        mat_dict[f'{tag}_trial']          = a['trial']
        mat_dict[f'{tag}_J']              = a['J']
        mat_dict[f'{tag}_best_J']         = a['best_J']
        mat_dict[f'{tag}_mpcc_cost']      = a['mpcc_cost']
        mat_dict[f'{tag}_rmse_contorno']  = a['rmse_contorno']
        mat_dict[f'{tag}_rmse_lag']       = a['rmse_lag']
        mat_dict[f'{tag}_mean_vtheta']    = a['mean_vtheta']
        mat_dict[f'{tag}_t_lap']          = a['t_lap']
        mat_dict[f'{tag}_path_completed'] = a['path_completed']

    mat_path = os.path.join(out_dir, 'tuning_convergence_data.mat')
    sio.savemat(mat_path, mat_dict, do_compression=True)
    print(f"  ✓ Saved {mat_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CONVERGENCE SUMMARY")
    print("="*60)
    for name, d in [('MPCC', mpcc_data), ('DQ-MPCC', dq_data)]:
        if d is None:
            continue
        a = extract_arrays(d)
        best_idx = int(np.argmin(a['J']))
        print(f"\n  {name}:")
        print(f"    Trials completed : {len(a['J'])}")
        print(f"    Best J           : {a['best_J'][-1]:.4f}  (trial #{a['trial'][best_idx]})")
        print(f"    Best MPCC cost   : {a['mpcc_cost'][best_idx]:.4f}")
        print(f"    Best RMSE_c      : {a['rmse_contorno'][best_idx]:.4f} m")
        print(f"    Best RMSE_l      : {a['rmse_lag'][best_idx]:.4f} m")
        if len(a['mean_vtheta']) > best_idx:
            print(f"    Best mean v_θ    : {a['mean_vtheta'][best_idx]:.2f} m/s")
        if len(a['t_lap']) > best_idx:
            print(f"    Best t_lap       : {a['t_lap'][best_idx]:.1f} s")
        # Convergence rate: at which trial did we reach 95% of final improvement?
        J_init = a['best_J'][0]
        J_final = a['best_J'][-1]
        threshold = J_final + 0.05 * (J_init - J_final)
        converged_at = np.argmax(a['best_J'] <= threshold)
        print(f"    95% converged at : trial #{a['trial'][converged_at]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot Optuna tuning convergence curves')
    parser.add_argument('--mpcc-only', action='store_true',
                        help='Plot only MPCC results')
    parser.add_argument('--dq-only', action='store_true',
                        help='Plot only DQ-MPCC results')
    parser.add_argument('--mpcc-hist', type=str, default=MPCC_HIST,
                        help='Path to MPCC tuning_history.json')
    parser.add_argument('--dq-hist', type=str, default=DQ_MPCC_HIST,
                        help='Path to DQ-MPCC tuning_history.json')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='Directory to save figures')
    args = parser.parse_args()

    print("="*60)
    print("  TUNING CONVERGENCE PLOTTER")
    print("="*60)

    mpcc_data = None
    dq_data   = None

    if not args.dq_only:
        mpcc_data = load_history(args.mpcc_hist)
    if not args.mpcc_only:
        dq_data = load_history(args.dq_hist)

    if mpcc_data is None and dq_data is None:
        print("\n  ✗ No history files found. Run the tuners first.")
        sys.exit(1)

    make_plots(mpcc_data, dq_data, args.out_dir)
    print("  Done.\n")


if __name__ == '__main__':
    main()
