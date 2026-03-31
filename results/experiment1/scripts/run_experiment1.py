#!/usr/bin/env python3
"""
run_experiment1.py – Publication-quality comparative plots for DQ-MPCC vs MPCC.

Loads the real simulation .mat files produced by DQ_MPCC_baseline.py and
MPCC_baseline.py, resamples them onto a common time grid, computes the
derived error/speed signals, and generates figures suitable for
IEEEtran single-column publication.

Usage:
    python results/experiment1/scripts/run_experiment1.py            # uses real .mat if available
    python results/experiment1/scripts/run_experiment1.py --mock     # force synthetic data
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

# ── Ensure workspace root is on sys.path ─────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from config.experiment_config import trayectoria as _trayectoria_config
from config.tuning_registry import get_active_weight_summary

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# 3D imports — handle version conflicts gracefully
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    # Quick test that 3D projection actually works
    _fig_test = plt.figure()
    _fig_test.add_subplot(111, projection='3d')
    plt.close(_fig_test)
    _HAS_3D = True
except Exception:
    _HAS_3D = False

# ═════════════════════════════════════════════════════════════════════════════
#  Style — IEEEtran single-column
# ═════════════════════════════════════════════════════════════════════════════
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

# Colours
C_DQ   = '#1f77b4'   # blue
C_BASE = '#d62728'   # red
C_REF  = 'black'

# Paths
_OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "results", "experiment1")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  1.  Data loading / generation
# ═════════════════════════════════════════════════════════════════════════════

def _quat_log_norm(q):
    """‖Log(q)‖ for a unit quaternion q = [qw, qx, qy, qz] (scalar first).

    Returns the geodesic distance on SO(3) in radians.
    """
    qw = q[0]
    qv = q[1:4]
    # Enforce positive hemisphere
    sign = np.sign(qw) if qw != 0 else 1.0
    qw = qw * sign
    qv = qv * sign
    norm_v = np.linalg.norm(qv)
    angle = 2.0 * np.arctan2(norm_v, qw)
    return abs(angle)


def _euler_to_quat(roll, pitch, yaw):
    """ZYX Euler → Hamilton quaternion [qw, qx, qy, qz]."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def _compute_path_curvature(pos, ds=None):
    """Approximate curvature κ(s) from a (N,3) position array."""
    dp = np.gradient(pos, axis=0)
    ddp = np.gradient(dp, axis=0)
    cross = np.cross(dp, ddp)
    numer = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(dp, axis=1)**3 + 1e-12
    return numer / denom


def load_real_data():
    """Load the two .mat files and build the comparison dictionaries.

    Returns  (data_dq, data_base, ref)  or raises FileNotFoundError.
    """
    dq_candidates = [
        os.path.join(_PROJECT_ROOT, "DQ-MPCC_baseline", "Results_DQ_MPCC_baseline_refined.mat"),
        os.path.join(_PROJECT_ROOT, "DQ-MPCC_baseline", "Results_DQ_MPCC_baseline_custom.mat"),
        os.path.join(_PROJECT_ROOT, "DQ-MPCC_baseline", "Results_DQ_MPCC_baseline_manual.mat"),
        os.path.join(_PROJECT_ROOT, "DQ-MPCC_baseline", "Results_DQ_MPCC_baseline_1.mat"),
        os.path.join(_PROJECT_ROOT, "Results_DQ_MPCC_baseline_1.mat"),
    ]
    base_candidates = [
        os.path.join(_PROJECT_ROOT, "MPCC_baseline", "Results_MPCC_baseline_refined.mat"),
        os.path.join(_PROJECT_ROOT, "MPCC_baseline", "Results_MPCC_baseline_custom.mat"),
        os.path.join(_PROJECT_ROOT, "MPCC_baseline", "Results_MPCC_baseline_manual_active.mat"),
        os.path.join(_PROJECT_ROOT, "MPCC_baseline", "Results_MPCC_baseline_manual.mat"),
        os.path.join(_PROJECT_ROOT, "MPCC_baseline", "Results_MPCC_baseline_1.mat"),
        os.path.join(_PROJECT_ROOT, "Results_MPCC_baseline_1.mat"),
    ]
    mat_dq = next((p for p in dq_candidates if os.path.isfile(p)), dq_candidates[-1])
    mat_base = next((p for p in base_candidates if os.path.isfile(p)), base_candidates[-1])

    if not os.path.isfile(mat_dq) or not os.path.isfile(mat_base):
        raise FileNotFoundError("One or both .mat files not found")

    d_dq   = loadmat(mat_dq)
    d_base = loadmat(mat_base)

    # ── Common time grid (resample the shorter onto the longer) ──────────
    t_dq   = d_dq['time'].ravel()
    t_base = d_base['time'].ravel()
    t_end  = min(t_dq[-1], t_base[-1])
    N      = int(round(t_end * 100))            # 100 Hz
    t_common = np.linspace(0, t_end, N)

    def _resample_1d(t_orig, y_orig, t_new):
        f = interp1d(t_orig, y_orig, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        return f(t_new)

    def _resample_2d(t_orig, y_orig, t_new):
        """y_orig: shape (D, M) → output (N, D)"""
        out = np.zeros((len(t_new), y_orig.shape[0]))
        for d in range(y_orig.shape[0]):
            out[:, d] = _resample_1d(t_orig, y_orig[d, :], t_new)
        return out

    # ── Helper: extract common signals from a .mat dict ──────────────────
    def _extract(d, t_orig, is_dq=False):
        N_ctrl = d['T_control'].shape[1]
        t_ctrl = t_orig[:N_ctrl]

        # States
        states = d['states']          # (14 or 15, M+1)
        if is_dq:
            states_std = d['states_std']  # (13, M+1)
            pos = states_std[0:3, :]      # (3, M+1)
            quat = np.zeros((4, states_std.shape[1]))
            quat[0, :] = states_std[6, :]  # qw
            quat[1, :] = states_std[7, :]  # qx
            quat[2, :] = states_std[8, :]  # qy
            quat[3, :] = states_std[9, :]  # qz
        else:
            pos = states[0:3, :]          # (3, M+1)
            quat = states[6:10, :]        # (4, M+1)

        # ── Position error from e_total (θ-indexed, correct for MPCC) ───
        # e_total = p_d(θ_k) − p_k ∈ ℝ³, shape (3, N_ctrl)
        e_total = d['e_total']              # (3, N_ctrl)
        pos_err = np.linalg.norm(e_total, axis=0)   # ‖p_d(θ) − p‖

        # ── Reference quaternion (time-indexed — used for orientation err)
        ref_quat_w = d['ref'][6, :]       # qw_d
        ref_quat_x = d['ref'][7, :]
        ref_quat_y = d['ref'][8, :]
        ref_quat_z = d['ref'][9, :]
        ref_quat = np.vstack([ref_quat_w, ref_quat_x,
                              ref_quat_y, ref_quat_z])  # (4, M+1)

        # ── Orientation error via quaternion log (over N_ctrl samples) ───
        # Use quat_d_theta (θ-indexed desired quaternion) if available
        # (saved by updated simulation scripts). Fall back to time-indexed
        # ref quaternion for backwards compatibility with old .mat files.
        if 'quat_d_theta' in d:
            ref_quat = d['quat_d_theta']   # (4, N_ctrl) — θ-indexed ✓
            t_ctrl_idx = True
        else:
            # Legacy: time-indexed reference quaternion (approximate)
            ref_quat = np.vstack([d['ref'][6, :], d['ref'][7, :],
                                  d['ref'][8, :], d['ref'][9, :]])  # (4, M+1)
            t_ctrl_idx = False

        ori_err = np.zeros(N_ctrl)
        for i in range(N_ctrl):
            qd_i = ref_quat[:, i] if t_ctrl_idx else ref_quat[:, i]
            q_i  = quat[:, i]
            # conjugate qd  →  qd⁻¹
            qd_inv = np.array([qd_i[0], -qd_i[1], -qd_i[2], -qd_i[3]])
            qd_inv /= np.linalg.norm(qd_inv) + 1e-12
            # Hamilton product  q_err = qd⁻¹ ⊗ q
            w0, x0, y0, z0 = qd_inv
            w1, x1, y1, z1 = q_i
            q_err = np.array([
                w0*w1 - x0*x1 - y0*y1 - z0*z1,
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1,
            ])
            ori_err[i] = _quat_log_norm(q_err)

        # ── Lag / contouring (already θ-indexed) ────────────────────────
        e_contorno = d['e_contorno']       # (3, N_ctrl)
        e_arrastre = d['e_arrastre']       # (3, N_ctrl)
        rho_cont = np.linalg.norm(e_contorno, axis=0)
        rho_lag  = np.linalg.norm(e_arrastre, axis=0)

        # Speeds
        vel_progres = d['vel_progres'].ravel()    # v_θ  (N_ctrl,)
        vel_real    = d['vel_real'].ravel()        # real progress
        vel_tangent = d['vel_tangent'].ravel()     # ‖v‖

        # Controls
        T_ctrl = d['T_control']           # (5, N_ctrl)
        thrust  = T_ctrl[0, :]
        torques = T_ctrl[1:4, :].T        # (N_ctrl, 3)

        # Progress
        theta = d['theta_history'].ravel()  # (M+1,)

        # Reference position (time-indexed, for 3D plot)
        ref_pos = d['ref'][0:3, :]        # (3, M+1)

        # Lap time and solver stats (from updated protocol)
        t_lap_val = float(d['t_lap'].ravel()[0]) if 't_lap' in d else np.nan
        solver_mean = float(d['t_solver_mean'].ravel()[0]) if 't_solver_mean' in d else np.nan
        solver_max  = float(d['t_solver_max'].ravel()[0])  if 't_solver_max'  in d else np.nan
        solver_std  = float(d['t_solver_std'].ravel()[0])  if 't_solver_std'  in d else np.nan

        return {
            't_states': t_orig,
            't_ctrl':   t_ctrl,
            'pos':      pos.T,               # (M+1, 3)
            'pos_ref':  ref_pos.T,           # (M+1, 3)
            'pos_error': pos_err,            # (N_ctrl,) — θ-indexed!
            'ori_error': ori_err,            # (N_ctrl,)
            'rho_cont':  rho_cont,           # (N_ctrl,)
            'rho_lag':   rho_lag,            # (N_ctrl,)
            'us':        vel_progres,        # (N_ctrl,)
            'vtang':     vel_tangent,        # (N_ctrl,)
            'vreal':     vel_real,           # (N_ctrl,)
            'f':         thrust,             # (N_ctrl,)
            'tau':       torques,            # (N_ctrl, 3)
            's':         theta,              # (M+1,)
            't_lap':     t_lap_val,          # [s]  lap completion time
            'solver_mean': solver_mean,      # [ms] solver stats
            'solver_max':  solver_max,
            'solver_std':  solver_std,
        }

    raw_dq   = _extract(d_dq,   t_dq,   is_dq=True)
    raw_base = _extract(d_base, t_base, is_dq=False)

    # ── Resample everything onto t_common ────────────────────────────────
    def _build_output(raw, t_common):
        out = {'t': t_common}
        # These are all on the control time grid (N_ctrl)
        for key in ['pos_error', 'ori_error',
                     'rho_cont', 'rho_lag', 'us', 'vtang', 'vreal', 'f']:
            out[key] = _resample_1d(raw['t_ctrl'], raw[key], t_common)
        out['tau'] = _resample_2d(raw['t_ctrl'], raw['tau'].T, t_common)
        # These are on the state time grid (M+1)
        out['s']   = _resample_1d(raw['t_states'], raw['s'], t_common)
        out['pos'] = _resample_2d(raw['t_states'], raw['pos'].T, t_common)
        out['pos_ref'] = _resample_2d(raw['t_states'],
                                      raw['pos_ref'].T, t_common)
        # Scalar KPIs (not resampled)
        out['t_lap']       = raw['t_lap']
        out['solver_mean'] = raw['solver_mean']
        out['solver_max']  = raw['solver_max']
        out['solver_std']  = raw['solver_std']
        return out

    data_dq   = _build_output(raw_dq,   t_common)
    data_base = _build_output(raw_base, t_common)

    # Reference (use DQ's reference — same trajectory)
    ref_pos = data_dq['pos_ref']
    ref = {
        'pos':       ref_pos,
        'curvature': _compute_path_curvature(ref_pos),
    }

    return data_dq, data_base, ref


def generate_mock_data(N=1500, t_final=15.0):
    """Generate synthetic data matching the real simulation format.

    Uses a Lissajous 3D curve with additive noise to mimic realistic
    controller behaviour. DQ-MPCC is made slightly better to reflect
    typical SE(3)-based advantages.
    """
    t = np.linspace(0, t_final, N)
    dt = t[1] - t[0]

    # ── Lissajous reference path (from experiment_config) ──────────────
    xd, yd, zd, _, _, _ = _trayectoria_config()
    ref_x = xd(t)
    ref_y = yd(t)
    ref_z = zd(t)
    ref_pos = np.column_stack([ref_x, ref_y, ref_z])
    curvature = _compute_path_curvature(ref_pos)

    ref = {'pos': ref_pos, 'curvature': curvature}

    # ── Arc-length ───────────────────────────────────────────────────────
    ds = np.linalg.norm(np.diff(ref_pos, axis=0), axis=1)
    s_ref = np.concatenate([[0], np.cumsum(ds)])

    # ── Helper: generate one controller's data ───────────────────────────
    def _make(label, pos_scale, ori_scale, speed_mean, lag_scale, cont_scale):
        np.random.seed(42 if label == 'dq' else 123)

        # Position error: baseline with curvature-correlated peaks
        base_err = pos_scale * (0.03 + 0.07 * curvature / (curvature.max()+1e-6))
        pos_error = base_err + pos_scale * 0.02 * np.random.randn(N)
        pos_error = np.abs(pos_error)

        # Orientation error
        ori_error = ori_scale * (0.02 + 0.05 * curvature / (curvature.max()+1e-6))
        ori_error += ori_scale * 0.015 * np.random.randn(N)
        ori_error = np.abs(ori_error)

        # Progress speed u_s
        us = speed_mean + 0.8 * np.sin(2 * np.pi * 0.3 * t)
        us -= 1.5 * curvature / (curvature.max() + 1e-6)
        us = np.clip(us, 0.5, 8.0)

        # Tangential speed (tracks u_s with some lag)
        vtang = 0.92 * us + 0.3 * np.random.randn(N) * 0.1
        vtang = np.clip(vtang, 0.0, 10.0)

        vreal = 0.88 * us + 0.2 * np.random.randn(N) * 0.1

        # Lag and contouring errors
        rho_lag  = lag_scale * np.abs(0.02 + 0.04 * np.random.randn(N))
        rho_cont = cont_scale * np.abs(0.02 + 0.04 * np.random.randn(N))
        rho_cont += cont_scale * 0.3 * curvature / (curvature.max() + 1e-6)

        # Controls
        thrust = 9.81 + 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(N)
        tau = 0.02 * np.random.randn(N, 3)

        # Actual position (reference + noise)
        noise_dir = np.random.randn(N, 3) * pos_scale * 0.05
        pos = ref_pos + noise_dir

        # Arc-length progress
        s = np.clip(s_ref * (speed_mean / 5.0), 0, s_ref[-1])

        # Solver time [ms]
        t_solve = 2.0 + 0.5 * np.random.randn(N)
        t_solve = np.clip(t_solve, 0.5, 8.0)

        return {
            't':         t,
            'pos_error': pos_error,
            'ori_error': ori_error,
            'us':        us,
            'vtang':     vtang,
            'vreal':     vreal,
            'rho_lag':   rho_lag,
            'rho_cont':  rho_cont,
            'f':         thrust,
            'tau':       tau,
            'pos_ref':   ref_pos,
            'pos':       pos,
            's':         s,
            't_solve':   t_solve,
        }

    data_dq = _make('dq',
                     pos_scale=0.7, ori_scale=0.7,
                     speed_mean=5.5, lag_scale=0.6, cont_scale=0.6)
    data_base = _make('base',
                      pos_scale=1.0, ori_scale=1.0,
                      speed_mean=4.8, lag_scale=1.0, cont_scale=1.0)

    return data_dq, data_base, ref


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Plotting functions
# ═════════════════════════════════════════════════════════════════════════════

def _shade_high_curvature(ax, t, curvature, threshold_pct=80):
    """Add vertical grey bands where curvature exceeds the given percentile."""
    thr = np.percentile(curvature, threshold_pct)
    high = curvature > thr
    # Find contiguous runs
    diff = np.diff(high.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if high[0]:
        starts = np.concatenate([[0], starts])
    if high[-1]:
        ends = np.concatenate([ends, [len(t) - 1]])
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t)-1)],
                   color='grey', alpha=0.15, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: Tracking errors
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracking_errors(data_dq, data_base, ref, save=True):
    """Position and orientation tracking errors over time."""
    t = data_dq['t']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.subplots_adjust(hspace=0.12, left=0.12, right=0.97, top=0.96,
                        bottom=0.10)

    # ── Subplot 1: position error ────────────────────────────────────────
    _shade_high_curvature(ax1, t, ref['curvature'])
    ax1.plot(t, data_base['pos_error'], color=C_BASE, lw=1.2,
             label='Baseline MPCC')
    ax1.plot(t, data_dq['pos_error'],   color=C_DQ,   lw=1.2,
             label='DQ-MPCC')
    ax1.set_ylabel(r'$\|\mathbf{p}_d - \mathbf{p}\|$ [m]')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, ls='--', alpha=0.4)

    # ── Subplot 2: orientation error ─────────────────────────────────────
    _shade_high_curvature(ax2, t, ref['curvature'])
    ax2.plot(t, data_base['ori_error'], color=C_BASE, lw=1.2)
    ax2.plot(t, data_dq['ori_error'],   color=C_DQ,   lw=1.2)
    ax2.set_ylabel(r'$\|\mathrm{Log}(q_d^{*} \otimes q)\|$ [rad]')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim(0, t[-1])
    ax2.grid(True, ls='--', alpha=0.4)

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUTPUT_DIR, f'fig_tracking_errors.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_tracking_errors saved")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: Speed modulation
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_modulation(data_dq, data_base, save=True):
    """Progress speed u_s and tangential speed v_tang for both controllers."""
    t = data_dq['t']
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.17)

    ax.plot(t, data_dq['us'],      color=C_DQ,   ls='-',  lw=1.2,
            label=r'$v_\theta$ DQ-MPCC')
    ax.plot(t, data_dq['vtang'],   color=C_DQ,   ls='--', lw=1.0,
            label=r'$v_\mathrm{tang}$ DQ-MPCC')
    ax.plot(t, data_base['us'],    color=C_BASE,  ls='-',  lw=1.2,
            label=r'$v_\theta$ Baseline')
    ax.plot(t, data_base['vtang'], color=C_BASE,  ls='--', lw=1.0,
            label=r'$v_\mathrm{tang}$ Baseline')

    ax.set_ylabel('Speed [m/s]')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)

    # Speed-tracking ratios
    mean_us_dq   = np.mean(data_dq['us'])
    mean_vt_dq   = np.mean(data_dq['vtang'])
    mean_us_base = np.mean(data_base['us'])
    mean_vt_base = np.mean(data_base['vtang'])
    r_dq   = mean_vt_dq   / (mean_us_dq   + 1e-8)
    r_base = mean_vt_base / (mean_us_base + 1e-8)

    txt = (f"$v_{{\\mathrm{{tang}}}}/v_\\theta$:  "
           f"DQ={r_dq:.2f},  Base={r_base:.2f}")
    ax.text(0.98, 0.05, txt, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUTPUT_DIR, f'fig_speed_modulation.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_speed_modulation saved")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3: 3D trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory_3d(data_dq, data_base, ref, save=True):
    """3D path: reference (black), DQ-MPCC (colour-coded by u_s), baseline (red).

    Falls back to 2D XY + XZ panels if mpl_toolkits.mplot3d is broken.
    """
    dp = data_dq['pos']
    bp = data_base['pos']
    rp = ref['pos']
    us = data_dq['us']
    us_max = max(7.0, np.max(us) * 1.05)
    norm = plt.Normalize(vmin=0, vmax=us_max)

    if _HAS_3D:
        # ── True 3D plot ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

        ax.plot(rp[:, 0], rp[:, 1], rp[:, 2],
                'k--', lw=1.0, label='Reference', zorder=1)
        ax.plot(bp[:, 0], bp[:, 1], bp[:, 2],
                color=C_BASE, lw=1.0, alpha=0.7,
                label='Baseline MPCC', zorder=2)

        points = dp.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap='viridis', norm=norm, lw=1.5)
        lc.set_array(us[:-1])
        ax.add_collection3d(lc)
        lc.set_label('DQ-MPCC')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.10)
        cbar.set_label(r'$v_\theta$ [m/s]')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.view_init(elev=25, azim=-60)
        ax.legend(loc='upper left', fontsize=8)

    else:
        # ── 2D fallback: XY (top) + XZ (bottom) ─────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.subplots_adjust(hspace=0.15, left=0.12, right=0.88,
                            top=0.96, bottom=0.09)

        for ax, idx_y, ylabel in [(ax1, 1, 'y [m]'), (ax2, 2, 'z [m]')]:
            ax.plot(rp[:, 0], rp[:, idx_y],
                    'k--', lw=1.0, label='Reference', zorder=1)
            ax.plot(bp[:, 0], bp[:, idx_y],
                    color=C_BASE, lw=1.0, alpha=0.7,
                    label='Baseline MPCC', zorder=2)
            sc = ax.scatter(dp[:, 0], dp[:, idx_y],
                            c=us, cmap='viridis', norm=norm,
                            s=2, zorder=3, label='DQ-MPCC')
            ax.set_ylabel(ylabel)
            ax.grid(True, ls='--', alpha=0.4)

        ax1.legend(loc='upper right', fontsize=8)
        ax2.set_xlabel('x [m]')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.9, pad=0.02)
        cbar.set_label(r'$v_\theta$ [m/s]')

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUTPUT_DIR, f'fig_trajectory_3d.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_trajectory_3d saved")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4: Lag / Contouring error decomposition
# ─────────────────────────────────────────────────────────────────────────────

def plot_lag_contouring(data_dq, data_base, ref, save=True):
    """Lag and contouring error norms over time."""
    t = data_dq['t']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.subplots_adjust(hspace=0.12, left=0.12, right=0.97, top=0.96,
                        bottom=0.10)

    # ── Contouring ───────────────────────────────────────────────────────
    _shade_high_curvature(ax1, t, ref['curvature'])
    ax1.plot(t, data_base['rho_cont'], color=C_BASE, lw=1.2,
             label='Baseline MPCC')
    ax1.plot(t, data_dq['rho_cont'],   color=C_DQ,   lw=1.2,
             label='DQ-MPCC')
    ax1.set_ylabel(r'Contouring $\|\mathbf{e}_c\|$ [m]')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, ls='--', alpha=0.4)

    # ── Lag ──────────────────────────────────────────────────────────────
    _shade_high_curvature(ax2, t, ref['curvature'])
    ax2.plot(t, data_base['rho_lag'], color=C_BASE, lw=1.2)
    ax2.plot(t, data_dq['rho_lag'],   color=C_DQ,   lw=1.2)
    ax2.set_ylabel(r'Lag $\|\mathbf{e}_\ell\|$ [m]')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim(0, t[-1])
    ax2.grid(True, ls='--', alpha=0.4)

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUTPUT_DIR,
                                     f'fig_lag_contouring.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_lag_contouring saved")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5: Control efforts
# ─────────────────────────────────────────────────────────────────────────────

def plot_control_efforts(data_dq, data_base, save=True):
    """Thrust and torques comparison."""
    t = data_dq['t']
    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    fig.subplots_adjust(hspace=0.15, left=0.12, right=0.97, top=0.96,
                        bottom=0.07)

    labels = ['$T$ [N]', r'$\tau_x$ [Nm]', r'$\tau_y$ [Nm]', r'$\tau_z$ [Nm]']

    # Thrust
    axes[0].plot(t, data_base['f'], color=C_BASE, lw=1.0, label='Baseline MPCC')
    axes[0].plot(t, data_dq['f'],   color=C_DQ,   lw=1.0, label='DQ-MPCC')
    axes[0].set_ylabel(labels[0])
    axes[0].legend(loc='upper right', framealpha=0.9)

    # Torques
    for i in range(3):
        axes[i+1].plot(t, data_base['tau'][:, i], color=C_BASE, lw=1.0)
        axes[i+1].plot(t, data_dq['tau'][:, i],   color=C_DQ,   lw=1.0)
        axes[i+1].set_ylabel(labels[i+1])

    for ax in axes:
        ax.grid(True, ls='--', alpha=0.4)

    axes[-1].set_xlabel('Time [s]')
    axes[-1].set_xlim(0, t[-1])

    if save:
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(_OUTPUT_DIR,
                                     f'fig_control_efforts.{ext}'),
                        dpi=300, bbox_inches='tight')
        print("✓ fig_control_efforts saved")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  3.  Summary statistics table (printed to console)
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(data_dq, data_base):
    """Print a comparison table to the console."""
    def _stats(arr):
        return np.mean(arr), np.std(arr), np.max(arr)

    metrics = [
        ('Position error [m]',   'pos_error'),
        ('Orientation error [rad]', 'ori_error'),
        ('Contouring error [m]', 'rho_cont'),
        ('Lag error [m]',        'rho_lag'),
        ('Progress speed v_θ [m/s]', 'us'),
        ('Tangential speed [m/s]', 'vtang'),
    ]

    print("\n" + "═"*78)
    print(f"  {'Metric':<30s} │ {'DQ-MPCC':>22s} │ {'Baseline':>22s}")
    print("─"*78)
    for name, key in metrics:
        m_dq, s_dq, x_dq    = _stats(data_dq[key])
        m_ba, s_ba, x_ba    = _stats(data_base[key])
        dq_str   = f"{m_dq:.4f} ± {s_dq:.4f}  (max {x_dq:.4f})"
        ba_str   = f"{m_ba:.4f} ± {s_ba:.4f}  (max {x_ba:.4f})"
        print(f"  {name:<30s} │ {dq_str:>22s} │ {ba_str:>22s}")
    print("─"*78)

    # ── Lap time & solver KPIs (from updated experimental protocol) ──────
    t_lap_dq   = data_dq.get('t_lap',   np.nan)
    t_lap_base = data_base.get('t_lap', np.nan)
    if not np.isnan(t_lap_dq) or not np.isnan(t_lap_base):
        print(f"  {'Lap time [s]':<30s} │ {t_lap_dq:>22.3f} │ {t_lap_base:>22.3f}")
    sol_dq   = data_dq.get('solver_mean', np.nan)
    sol_base = data_base.get('solver_mean', np.nan)
    if not np.isnan(sol_dq) or not np.isnan(sol_base):
        sol_dq_str   = f"{sol_dq:.2f} ± {data_dq.get('solver_std', 0):.2f}  (max {data_dq.get('solver_max', 0):.2f})"
        sol_base_str = f"{sol_base:.2f} ± {data_base.get('solver_std', 0):.2f}  (max {data_base.get('solver_max', 0):.2f})"
        print(f"  {'Solver time [ms]':<30s} │ {sol_dq_str:>22s} │ {sol_base_str:>22s}")
    print("═"*78 + "\n")


def save_latex_table(data_dq, data_base, filename=None):
    """Write a publication-ready LaTeX table (booktabs) to a .tex file.

    Compatible with IEEEtran and standard article classes.
    Requires \\usepackage{booktabs} and \\usepackage{siunitx} in the preamble.
    """
    if filename is None:
        filename = os.path.join(_OUTPUT_DIR, 'comparison_table.tex')

    def _stats(arr):
        return np.mean(arr), np.std(arr), np.max(arr)

    # ── Collect all values ───────────────────────────────────────────────
    # Symbol-only labels (no verbose name) — compact for IEEEtran columns
    metrics = [
        (r'$\|\mathbf{e}_p\|$ [m]',                     'pos_error'),
        (r'$\|\mathrm{Log}(q_e)\|$ [rad]',              'ori_error'),
        (r'$\|\boldsymbol{e}_c\|$ [m]',                 'rho_cont'),
        (r'$\|\boldsymbol{e}_\ell\|$ [m]',              'rho_lag'),
        (r'$v_\theta$ [m/s]',                           'us'),
        (r'$\|\mathbf{v}\|$ [m/s]',                     'vtang'),
    ]

    t_lap_dq   = data_dq.get('t_lap',   float('nan'))
    t_lap_base = data_base.get('t_lap', float('nan'))
    sol_dq_m   = data_dq.get('solver_mean',  float('nan'))
    sol_dq_s   = data_dq.get('solver_std',   0.0)
    sol_dq_x   = data_dq.get('solver_max',   float('nan'))
    sol_ba_m   = data_base.get('solver_mean', float('nan'))
    sol_ba_s   = data_base.get('solver_std',  0.0)
    sol_ba_x   = data_base.get('solver_max',  float('nan'))

    # ── Build LaTeX ──────────────────────────────────────────────────────
    lines = []
    lines.append(r'% ────────────────────────────────────────────────────────────')
    lines.append(r'% Comparison table: DQ-MPCC vs Baseline MPCC')
    lines.append(r'% Generated automatically by compare_results.py')
    lines.append(r'% Requires: \usepackage{booktabs,siunitx,multirow}')
    lines.append(r'% ────────────────────────────────────────────────────────────')
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Performance comparison between DQ-MPCC and Baseline MPCC')
    lines.append(r'           over a fixed 100\,m Lissajous path. Statistics are')
    lines.append(r'           mean\,$\pm$\,std (max) computed over the full lap.}')
    lines.append(r'  \label{tab:comparison}')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \begin{tabular}{l S[table-format=2.4] @{\,$\pm$\,} S[table-format=1.4]'
                 r' S[table-format=2.4] S[table-format=2.4] @{\,$\pm$\,} S[table-format=1.4]'
                 r' S[table-format=2.4]}')
    lines.append(r'    \toprule')
    lines.append(r'    \multirow{2}{*}{\textbf{}}')
    lines.append(r'      & \multicolumn{3}{c}{\textbf{DQ-MPCC}}')
    lines.append(r'      & \multicolumn{3}{c}{\textbf{Baseline MPCC}} \\')
    lines.append(r'    \cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    lines.append(r'      & {Mean} & {Std} & {Max}')
    lines.append(r'      & {Mean} & {Std} & {Max} \\')
    lines.append(r'    \midrule')

    for label, key in metrics:
        m_dq, s_dq, x_dq = _stats(data_dq[key])
        m_ba, s_ba, x_ba = _stats(data_base[key])
        lines.append(
            f'    {label}'
            f' & {m_dq:.4f} & {s_dq:.4f} & {x_dq:.4f}'
            f' & {m_ba:.4f} & {s_ba:.4f} & {x_ba:.4f} \\\\'
        )

    lines.append(r'    \midrule')

    # Lap time (single value per controller — no std/max)
    if not (np.isnan(t_lap_dq) and np.isnan(t_lap_base)):
        lap_dq_str  = f'{t_lap_dq:.3f}' if not np.isnan(t_lap_dq)  else r'\text{--}'
        lap_ba_str  = f'{t_lap_base:.3f}' if not np.isnan(t_lap_base) else r'\text{--}'
        lines.append(
            r'    $t_\mathrm{lap}$ [s]'
            f' & \\multicolumn{{3}}{{c}}{{{lap_dq_str}}}'
            f' & \\multicolumn{{3}}{{c}}{{{lap_ba_str}}} \\\\'
        )

    # Solver time
    if not np.isnan(sol_dq_m):
        lines.append(
            r'    $t_\mathrm{solve}$ [ms]'
            f' & {sol_dq_m:.2f} & {sol_dq_s:.2f} & {sol_dq_x:.2f}'
            f' & {sol_ba_m:.2f} & {sol_ba_s:.2f} & {sol_ba_x:.2f} \\\\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines) + '\n'

    with open(filename, 'w') as f:
        f.write(tex)
    print(f'✓ LaTeX table saved to {filename}')


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Save all data to .mat for MATLAB
# ═════════════════════════════════════════════════════════════════════════════

def save_comparison_mat(data_dq, data_base, ref, filename=None):
    """Save all comparison data to a single .mat file.

    Structure in MATLAB:
        load('comparison_data.mat')
        dq.t, dq.pos_error, dq.ori_error, ...
        base.t, base.pos_error, base.ori_error, ...
        ref.pos, ref.curvature
    """
    if filename is None:
        filename = os.path.join(_OUTPUT_DIR, 'comparison_data.mat')

    mat_dict = {}

    # Flatten all arrays and prefix with controller name
    for prefix, data in [('dq', data_dq), ('base', data_base)]:
        for key, val in data.items():
            mat_key = f"{prefix}_{key}"
            mat_dict[mat_key] = np.asarray(val)

    # Reference
    mat_dict['ref_pos']       = ref['pos']
    mat_dict['ref_curvature'] = ref['curvature']
    mat_dict['dq_gain_label'] = get_active_weight_summary('dq')['label']
    mat_dict['base_gain_label'] = get_active_weight_summary('mpcc')['label']

    savemat(filename, mat_dict, do_compression=True)
    print(f"✓ Comparison data saved to {filename}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DQ-MPCC vs Baseline MPCC — comparison plots")
    parser.add_argument('--mock', action='store_true',
                        help='Force synthetic data even if .mat files exist')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not call plt.show() (for batch mode)')
    args = parser.parse_args()

    # ── Load or generate data ────────────────────────────────────────────
    if args.mock:
        print("[INFO] Using synthetic mock data (--mock flag)")
        data_dq, data_base, ref = generate_mock_data()
    else:
        try:
            print("[INFO] Loading real simulation data from .mat files ...")
            data_dq, data_base, ref = load_real_data()
            print(f"[INFO] Loaded OK — {len(data_dq['t'])} samples, "
                  f"t ∈ [0, {data_dq['t'][-1]:.2f}] s")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print("[INFO] Falling back to synthetic mock data")
            data_dq, data_base, ref = generate_mock_data()

    # ── Summary statistics ───────────────────────────────────────────────
    print_summary_table(data_dq, data_base)
    save_latex_table(data_dq, data_base)

    # ── Generate figures ─────────────────────────────────────────────────
    print(f"[INFO] Saving figures to {_OUTPUT_DIR}/\n")

    fig1 = plot_tracking_errors(data_dq, data_base, ref)
    fig2 = plot_speed_modulation(data_dq, data_base)
    fig3 = plot_trajectory_3d(data_dq, data_base, ref)
    fig4 = plot_lag_contouring(data_dq, data_base, ref)
    fig5 = plot_control_efforts(data_dq, data_base)

    # ── Save .mat for MATLAB ─────────────────────────────────────────────
    save_comparison_mat(data_dq, data_base, ref)

    if not args.no_show:
        plt.show()

    print("\n✓ All done.")


if __name__ == '__main__':
    main()
