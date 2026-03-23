#!/usr/bin/env python3
"""
extract_paper_data.py — Extract ALL metrics from .mat files for the paper.

Loads Experiment 1 and Experiment 2 results, computes every number used
in the LaTeX tables and analysis text, and prints:
  • A copy-pasteable summary for each table
  • All derived Δ% comparisons
  • Controller weights + experimental configuration
  • Auto-generated LaTeX snippets (tables ready to paste)

Re-run this script whenever the .mat files change to get updated numbers.

Usage:
    python paper_sections/extract_paper_data.py
"""

import os, sys, math, re, importlib
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)
_DQ_DIR     = os.path.join(_ROOT, "DQ-MPCC_baseline")
_MPCC_DIR   = os.path.join(_ROOT, "MPCC_baseline")
_EXP2_DIR   = os.path.join(_ROOT, "experiment2_results")

sys.path.insert(0, _ROOT)

try:
    from scipy.io import loadmat
except ImportError:
    sys.exit("ERROR: scipy not installed.  pip install scipy")


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _scalar(mat, key, default=float('nan')):
    """Safely extract a scalar from a .mat dict."""
    if key not in mat:
        return default
    v = mat[key].ravel()
    return float(v[0]) if len(v) > 0 else default


def _array(mat, key):
    """Safely extract a 1-D array from a .mat dict."""
    if key not in mat:
        return np.array([])
    return mat[key].ravel()


def _rmse(arr):
    """RMSE of a 1-D array."""
    if len(arr) == 0:
        return float('nan')
    return math.sqrt(np.mean(arr ** 2))


def _delta_pct(val_dq, val_base):
    """Improvement of DQ over baseline:  negative = DQ is better."""
    if val_base == 0 or np.isnan(val_base) or np.isnan(val_dq):
        return float('nan')
    return (val_dq - val_base) / val_base * 100.0


def _quat_log_norm(q):
    """||Log(q)|| for unit quaternion [qw,qx,qy,qz]."""
    qw = q[0]
    qv = q[1:4]
    s = np.sign(qw) if qw != 0 else 1.0
    qw, qv = qw * s, qv * s
    return abs(2.0 * math.atan2(np.linalg.norm(qv), qw))


def _quat_mult(a, b):
    w0,x0,y0,z0 = a
    w1,x1,y1,z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


SEP = "═" * 78
SEP2 = "─" * 78


# ═════════════════════════════════════════════════════════════════════════════
#  1. Load experiment configuration
# ═════════════════════════════════════════════════════════════════════════════

def load_experiment_config():
    """Load shared + Exp2 configuration."""
    cfg = {}

    # experiment_config.py
    try:
        import experiment_config as ec
        cfg['P0']           = ec.P0.tolist()
        cfg['Q0']           = ec.Q0.tolist()
        cfg['VALUE']        = ec.VALUE
        cfg['T_FINAL']      = ec.T_FINAL
        cfg['FREC']         = ec.FREC
        cfg['T_PREDICTION'] = ec.T_PREDICTION
        cfg['N_WAYPOINTS']  = ec.N_WAYPOINTS
        cfg['S_MAX']        = getattr(ec, 'S_MAX_MANUAL', 100.0)
    except ImportError:
        print("[WARN] experiment_config.py not found — using defaults")
        cfg['FREC'] = 100
        cfg['T_PREDICTION'] = 0.3
        cfg['S_MAX'] = 100.0

    # experiment2_config.py
    try:
        import experiment2_config as e2c
        cfg['VELOCITIES'] = list(e2c.VELOCITIES)
        cfg['N_RUNS']     = e2c.N_RUNS
        cfg['SIGMA_P']    = e2c.SIGMA_P
        cfg['SIGMA_Q']    = e2c.SIGMA_Q
        cfg['SEED']       = e2c.SEED
        cfg['S_MAX_EXP2'] = e2c.S_MAX
        cfg['T_FINAL_EXP2'] = e2c.T_FINAL
    except ImportError:
        print("[WARN] experiment2_config.py not found — using defaults")
        cfg['VELOCITIES'] = [8, 12, 15]
        cfg['N_RUNS']     = 5
        cfg['SIGMA_P']    = 0.05
        cfg['SIGMA_Q']    = 0.05

    return cfg


# ═════════════════════════════════════════════════════════════════════════════
#  2. Load controller weights from source files
# ═════════════════════════════════════════════════════════════════════════════

def _parse_defaults_from_file(filepath, prefix='DEFAULT_'):
    """Parse DEFAULT_xxx = ... lines from a Python source file."""
    weights = {}
    if not os.path.isfile(filepath):
        return weights
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith(prefix):
                try:
                    key, val_str = line.split('=', 1)
                    key = key.strip()
                    val_str = val_str.split('#')[0].strip()  # remove comments
                    val = eval(val_str, {"__builtins__": {}},
                               {"G": 9.81, "np": np})
                    weights[key] = val
                except Exception:
                    pass
    return weights


def load_controller_weights():
    """Return weight dicts for both controllers."""
    dq_file   = os.path.join(_DQ_DIR, "ocp", "dq_mpcc_controller.py")
    mpcc_file = os.path.join(_MPCC_DIR, "ocp", "mpcc_controller.py")
    return {
        'DQ-MPCC': _parse_defaults_from_file(dq_file),
        'MPCC':    _parse_defaults_from_file(mpcc_file),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  3. Experiment 1 — single-run extraction
# ═════════════════════════════════════════════════════════════════════════════

def load_experiment1():
    """Load both Experiment 1 .mat files, compute all metrics."""

    mat_dq_path   = os.path.join(_DQ_DIR,   "Results_DQ_MPCC_baseline_1.mat")
    mat_mpcc_path = os.path.join(_MPCC_DIR,  "Results_MPCC_baseline_1.mat")

    if not os.path.isfile(mat_dq_path):
        print(f"[WARN] DQ-MPCC Exp1 .mat not found: {mat_dq_path}")
        return None
    if not os.path.isfile(mat_mpcc_path):
        print(f"[WARN] MPCC Exp1 .mat not found: {mat_mpcc_path}")
        return None

    d = loadmat(mat_dq_path)
    m = loadmat(mat_mpcc_path)

    def _extract_exp1(mat, label, is_dq=False):
        """Extract all Experiment 1 metrics from a single .mat."""
        out = {'label': label}

        # Scalars stored directly
        out['t_lap']         = _scalar(mat, 't_lap')
        out['t_solver_mean'] = _scalar(mat, 't_solver_mean')
        out['t_solver_max']  = _scalar(mat, 't_solver_max')
        out['t_solver_std']  = _scalar(mat, 't_solver_std')
        out['s_max']         = _scalar(mat, 's_max')

        # Number of control steps
        T_ctrl = mat['T_control']  # (5, N_ctrl)
        N_ctrl = T_ctrl.shape[1]
        out['N_steps'] = N_ctrl

        # Position error: e_total = p_d(θ) - p  ∈ R³
        e_total = mat['e_total']  # (3, N_ctrl)
        pos_err = np.linalg.norm(e_total, axis=0)
        out['rmse_pos']    = _rmse(pos_err)
        out['mean_pos_err'] = float(np.mean(pos_err))
        out['max_pos_err']  = float(np.max(pos_err))

        # Contouring & lag
        e_cont = np.linalg.norm(mat['e_contorno'], axis=0)
        e_lag  = np.linalg.norm(mat['e_arrastre'], axis=0)
        out['rmse_cont'] = _rmse(e_cont)
        out['rmse_lag']  = _rmse(e_lag)
        out['mean_cont'] = float(np.mean(e_cont))
        out['mean_lag']  = float(np.mean(e_lag))

        # Orientation error (via quat_d_theta if available)
        states = mat['states']
        if is_dq:
            # DQ state: [dq(8), vb(3), wb(3), θ]  → quaternion = dq[0:4]
            quat_state = states[0:4, :N_ctrl]
        else:
            # MPCC state: [p(3), v(3), q(4), w(3), θ] → quaternion = states[6:10]
            quat_state = states[6:10, :N_ctrl]

        if 'quat_d_theta' in mat:
            quat_des = mat['quat_d_theta'][:, :N_ctrl]  # (4, N_ctrl)
        else:
            quat_des = mat['ref'][6:10, :N_ctrl]

        ori_errs = []
        for i in range(N_ctrl):
            qd_i = quat_des[:, i]
            qs_i = quat_state[:, i]
            q_err = _quat_mult(_quat_conj(qd_i), qs_i)
            ori_errs.append(_quat_log_norm(q_err))
        ori_errs = np.array(ori_errs)
        out['rmse_ori']    = _rmse(ori_errs)
        out['mean_ori_err'] = float(np.mean(ori_errs))
        out['max_ori_err']  = float(np.max(ori_errs))

        # Progress speed v_θ
        vp = _array(mat, 'vel_progres')
        out['mean_vtheta'] = float(np.mean(vp)) if len(vp) > 0 else float('nan')

        # Control effort  Σ ||u||² (excluding v_theta)
        thrust  = T_ctrl[0, :]
        torques = T_ctrl[1:4, :]
        effort  = float(np.sum(thrust**2) + np.sum(torques**2))
        out['effort'] = effort

        # Theta final
        theta = _array(mat, 'theta_history')
        out['theta_final'] = float(theta[-1]) if len(theta) > 0 else float('nan')

        return out

    exp1_dq   = _extract_exp1(d, 'DQ-MPCC', is_dq=True)
    exp1_mpcc = _extract_exp1(m, 'MPCC Baseline', is_dq=False)

    return exp1_dq, exp1_mpcc


# ═════════════════════════════════════════════════════════════════════════════
#  4. Experiment 2 — velocity sweep extraction
# ═════════════════════════════════════════════════════════════════════════════

def load_experiment2(velocities):
    """Load the velocity sweep .mat and compute statistics per (ctrl, v)."""

    mat_path = os.path.join(_EXP2_DIR, "velocity_sweep_data.mat")
    if not os.path.isfile(mat_path):
        print(f"[WARN] Exp2 .mat not found: {mat_path}")
        return None

    d = loadmat(mat_path)

    exp2 = {}
    for ctrl in ['dq', 'base']:
        exp2[ctrl] = {}
        for v in velocities:
            vstr = f"{v:.2f}".replace('.', 'p')
            rp  = d.get(f'{ctrl}_v{vstr}_rmse_pos',    np.array([])).ravel()
            ro  = d.get(f'{ctrl}_v{vstr}_rmse_ori',    np.array([])).ravel()
            tl  = d.get(f'{ctrl}_v{vstr}_t_lap',       np.array([])).ravel()
            mv  = d.get(f'{ctrl}_v{vstr}_mean_vtheta', np.array([])).ravel()
            nf  = int(_scalar(d, f'{ctrl}_v{vstr}_failures', 0))

            # Filter NaN
            rp_ok = rp[np.isfinite(rp)] if len(rp) > 0 else np.array([])
            ro_ok = ro[np.isfinite(ro)] if len(ro) > 0 else np.array([])
            tl_ok = tl[np.isfinite(tl)] if len(tl) > 0 else np.array([])
            mv_ok = mv[np.isfinite(mv)] if len(mv) > 0 else np.array([])

            def _stats5(arr):
                if len(arr) == 0:
                    return {'median': np.nan, 'mean': np.nan,
                            'std': np.nan, 'p25': np.nan, 'p75': np.nan,
                            'min': np.nan, 'max': np.nan, 'n': 0,
                            'raw': arr}
                return {
                    'median': float(np.median(arr)),
                    'mean':   float(np.mean(arr)),
                    'std':    float(np.std(arr)),
                    'p25':    float(np.percentile(arr, 25)),
                    'p75':    float(np.percentile(arr, 75)),
                    'min':    float(np.min(arr)),
                    'max':    float(np.max(arr)),
                    'n':      len(arr),
                    'raw':    arr,
                }

            exp2[ctrl][v] = {
                'rmse_pos':    _stats5(rp_ok),
                'rmse_ori':    _stats5(ro_ok),
                't_lap':       _stats5(tl_ok),
                'mean_vtheta': _stats5(mv_ok),
                'failures':    nf,
            }

    # General metadata
    exp2['_meta'] = {
        'N_runs':           int(_scalar(d, 'N_runs', 0)),
        'sigma_p':          _scalar(d, 'sigma_p'),
        'sigma_q':          _scalar(d, 'sigma_q'),
        'completion_ratio': _scalar(d, 'completion_ratio'),
        's_max':            _scalar(d, 's_max'),
        'velocities':       _array(d, 'velocities').tolist(),
    }
    return exp2


# ═════════════════════════════════════════════════════════════════════════════
#  5. Console report
# ═════════════════════════════════════════════════════════════════════════════

def print_config(cfg, weights):
    """Print experiment configuration and controller weights."""
    print(f"\n{SEP}")
    print("  EXPERIMENT CONFIGURATION")
    print(SEP)
    for k in sorted(cfg.keys()):
        print(f"    {k:20s} = {cfg[k]}")

    print(f"\n{SEP}")
    print("  CONTROLLER WEIGHTS")
    print(SEP)
    for ctrl_name, w in weights.items():
        print(f"\n  ── {ctrl_name} ──")
        for k in sorted(w.keys()):
            v = w[k]
            if isinstance(v, list):
                v_str = "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
            elif isinstance(v, float):
                v_str = f"{v:.6f}"
            else:
                v_str = str(v)
            print(f"    {k:25s} = {v_str}")


def print_experiment1(exp1_dq, exp1_mpcc):
    """Print Experiment 1 comparison table."""
    print(f"\n{SEP}")
    print("  EXPERIMENT 1 — SINGLE RUN COMPARISON")
    print(SEP)

    metrics = [
        ('t_lap',         't_lap [s]',            '.2f'),
        ('rmse_pos',      'RMSE_pos [m]',         '.4f'),
        ('rmse_cont',     'RMSE_cont [m]',        '.4f'),
        ('rmse_lag',      'RMSE_lag [m]',         '.4f'),
        ('mean_pos_err',  'Mean |e_p| [m]',       '.4f'),
        ('max_pos_err',   'Max |e_p| [m]',        '.4f'),
        ('rmse_ori',      'RMSE_ori [rad]',       '.4f'),
        ('mean_vtheta',   'Mean v_θ [m/s]',       '.2f'),
        ('effort',        'Effort Σ||u||²',       '.2f'),
        ('t_solver_mean', 't_solver_mean [ms]',   '.2f'),
        ('t_solver_max',  't_solver_max [ms]',    '.2f'),
        ('N_steps',       'N_steps',              'd'),
        ('theta_final',   'θ_final [m]',          '.2f'),
    ]

    print(f"\n  {'Metric':<25s} │ {'DQ-MPCC':>14s} │ {'MPCC Baseline':>14s} │ {'Δ%':>8s}")
    print(f"  {SEP2}")
    for key, name, fmt in metrics:
        vd = exp1_dq[key]
        vm = exp1_mpcc[key]
        delta = _delta_pct(vd, vm)
        sd = f"{vd:{fmt}}" if not np.isnan(vd) else "N/A"
        sm = f"{vm:{fmt}}" if not np.isnan(vm) else "N/A"
        sd_str = f"{delta:+.1f}%" if not np.isnan(delta) else "—"
        print(f"  {name:<25s} │ {sd:>14s} │ {sm:>14s} │ {sd_str:>8s}")
    print()


def print_experiment2(exp2, velocities):
    """Print Experiment 2 velocity sweep table."""
    print(f"\n{SEP}")
    print("  EXPERIMENT 2 — VELOCITY SWEEP (Monte Carlo)")
    print(SEP)

    meta = exp2['_meta']
    print(f"    N_runs = {meta['N_runs']},  σ_p = {meta['sigma_p']},  "
          f"σ_q = {meta['sigma_q']},  s_max = {meta['s_max']}")

    # ── Position RMSE ────────────────────────────────────────────────────
    print(f"\n  {'v_max':>6s} │ {'DQ-MPCC RMSE_pos':>24s} │ "
          f"{'MPCC RMSE_pos':>24s} │ {'Δ%':>8s}")
    print(f"  {'[m/s]':>6s} │ {'median (std)':>24s} │ "
          f"{'median (std)':>24s} │ {'':>8s}")
    print(f"  {SEP2}")
    for v in velocities:
        dq  = exp2['dq'][v]['rmse_pos']
        ba  = exp2['base'][v]['rmse_pos']
        delta = _delta_pct(dq['median'], ba['median'])
        print(f"  {v:6d} │ {dq['median']:8.4f} (±{dq['std']:.4f}) n={dq['n']}  │ "
              f"{ba['median']:8.4f} (±{ba['std']:.4f}) n={ba['n']}  │ {delta:+7.1f}%")

    # ── Orientation RMSE ─────────────────────────────────────────────────
    print(f"\n  {'v_max':>6s} │ {'DQ-MPCC RMSE_ori':>24s} │ "
          f"{'MPCC RMSE_ori':>24s} │ {'Δ%':>8s}")
    print(f"  {SEP2}")
    for v in velocities:
        dq  = exp2['dq'][v]['rmse_ori']
        ba  = exp2['base'][v]['rmse_ori']
        delta = _delta_pct(dq['median'], ba['median'])
        print(f"  {v:6d} │ {dq['median']:8.4f} (±{dq['std']:.4f}) n={dq['n']}  │ "
              f"{ba['median']:8.4f} (±{ba['std']:.4f}) n={ba['n']}  │ {delta:+7.1f}%")

    # ── Lap time (if available) ──────────────────────────────────────────
    has_tlap = any(exp2['dq'][v]['t_lap']['n'] > 0 for v in velocities)
    if has_tlap:
        print(f"\n  {'v_max':>6s} │ {'DQ t_lap':>18s} │ "
              f"{'MPCC t_lap':>18s} │ {'Δ%':>8s}")
        print(f"  {SEP2}")
        for v in velocities:
            dq  = exp2['dq'][v]['t_lap']
            ba  = exp2['base'][v]['t_lap']
            delta = _delta_pct(dq['median'], ba['median'])
            if dq['n'] > 0:
                print(f"  {v:6d} │ {dq['median']:8.2f}s (±{dq['std']:.2f}) │ "
                      f"{ba['median']:8.2f}s (±{ba['std']:.2f}) │ {delta:+7.1f}%")
            else:
                print(f"  {v:6d} │ {'(no data)':>18s} │ {'(no data)':>18s} │    —")
    else:
        print("\n  [INFO] t_lap data not available in .mat — re-run experiment "
              "with updated 2_run_experiment2_sweep.py")

    # ── Mean v_theta (if available) ──────────────────────────────────────
    has_mvt = any(exp2['dq'][v]['mean_vtheta']['n'] > 0 for v in velocities)
    if has_mvt:
        print(f"\n  {'v_max':>6s} │ {'DQ mean_vθ':>18s} │ "
              f"{'MPCC mean_vθ':>18s}")
        print(f"  {SEP2}")
        for v in velocities:
            dq  = exp2['dq'][v]['mean_vtheta']
            ba  = exp2['base'][v]['mean_vtheta']
            if dq['n'] > 0:
                print(f"  {v:6d} │ {dq['median']:8.2f} m/s (±{dq['std']:.2f}) │ "
                      f"{ba['median']:8.2f} m/s (±{ba['std']:.2f})")

    # ── Failures ─────────────────────────────────────────────────────────
    print(f"\n  {'v_max':>6s} │ {'DQ fails':>10s} │ {'MPCC fails':>10s}")
    print(f"  {SEP2}")
    for v in velocities:
        print(f"  {v:6d} │ {exp2['dq'][v]['failures']:>10d} │ "
              f"{exp2['base'][v]['failures']:>10d}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  6. Auto-generate LaTeX snippets
# ═════════════════════════════════════════════════════════════════════════════

def generate_latex_exp1(exp1_dq, exp1_mpcc):
    """Return a string with the LaTeX table for Experiment 1."""

    rows = [
        ('RMSE$_{\\text{pos}}$ [m]',       'rmse_pos',      '.3f'),
        ('RMSE$_{\\text{cont}}$ [m]',      'rmse_cont',     '.3f'),
        ('RMSE$_{\\text{lag}}$ [m]',       'rmse_lag',      '.3f'),
        ('Mean $|\\mathbf{e}_p|$ [m]',     'mean_pos_err',  '.3f'),
        ('Max $|\\mathbf{e}_p|$ [m]',      'max_pos_err',   '.3f'),
        ('$\\bar v_\\theta$ [m/s]',        'mean_vtheta',   '.2f'),
        ('$t_{\\text{lap}}$ [s]',          't_lap',         '.2f'),
        ('Effort $\\sum\\|\\mathbf{u}\\|^2$', 'effort',    '.1f'),
        ('$\\bar t_{\\text{solver}}$ [ms]', 't_solver_mean', '.2f'),
        ('$t_{\\text{solver}}^{\\max}$ [ms]', 't_solver_max', '.2f'),
        ('$N_{\\text{steps}}$',            'N_steps',       'd'),
    ]

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Experiment~I --- single-run comparison')
    lines.append(f'           ($S_{{\\max}}=\\SI{{{exp1_dq["s_max"]:.0f}}}{{m}}$, '
                 f'$v_{{\\theta,\\max}}=\\SI{{15}}{{m/s}}$).}}')
    lines.append(r'  \label{tab:exp1}')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \begin{tabular}{lSS}')
    lines.append(r'    \toprule')
    lines.append(r'    {Metric} & {DQ-MPCC} & {MPCC Baseline} \\')
    lines.append(r'    \midrule')

    for label, key, fmt in rows:
        vd = exp1_dq[key]
        vm = exp1_mpcc[key]
        sd = f"{vd:{fmt}}" if not np.isnan(vd) else "---"
        sm = f"{vm:{fmt}}" if not np.isnan(vm) else "---"
        lines.append(f'    {{{label}}} & {sd} & {sm} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_latex_exp2(exp2, velocities):
    """Return a string with the LaTeX table for Experiment 2."""
    meta = exp2['_meta']

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(f'  \\caption{{Experiment~II --- median RMSE (std) over '
                 f'$N={meta["N_runs"]}$ Monte Carlo')
    lines.append(r'           runs per speed.  Zero failures for both controllers.}')
    lines.append(r'  \label{tab:exp2}')
    lines.append(r'  \setlength{\tabcolsep}{3pt}')
    lines.append(r'  \footnotesize')
    lines.append(r'  \begin{tabular}{c cc cc c}')
    lines.append(r'    \toprule')
    lines.append(r'    & \multicolumn{2}{c}{\textbf{DQ-MPCC}}')
    lines.append(r'    & \multicolumn{2}{c}{\textbf{MPCC Baseline}}')
    lines.append(r'    & \\')
    lines.append(r'    \cmidrule(lr){2-3} \cmidrule(lr){4-5}')
    lines.append(r'    $v_{\theta,\max}$')
    lines.append(r'    & {RMSE$_{\text{pos}}$}')
    lines.append(r'    & {RMSE$_{\text{ori}}$}')
    lines.append(r'    & {RMSE$_{\text{pos}}$}')
    lines.append(r'    & {RMSE$_{\text{ori}}$}')
    lines.append(r'    & {$\Delta\%_{\text{pos}}$} \\')
    lines.append(r'    {[m/s]}')
    lines.append(r'    & {[m]}')
    lines.append(r'    & {[rad]}')
    lines.append(r'    & {[m]}')
    lines.append(r'    & {[rad]}')
    lines.append(r'    & \\')
    lines.append(r'    \midrule')

    for v in velocities:
        dq_rp = exp2['dq'][v]['rmse_pos']
        dq_ro = exp2['dq'][v]['rmse_ori']
        ba_rp = exp2['base'][v]['rmse_pos']
        ba_ro = exp2['base'][v]['rmse_ori']
        delta = _delta_pct(dq_rp['median'], ba_rp['median'])

        lines.append(
            f'    {v}\n'
            f'    & {dq_rp["median"]:.3f}\\,{{\\scriptsize($\\pm${dq_rp["std"]:.3f})}}\n'
            f'    & {dq_ro["median"]:.3f}\\,{{\\scriptsize($\\pm${dq_ro["std"]:.3f})}}\n'
            f'    & {ba_rp["median"]:.3f}\\,{{\\scriptsize($\\pm${ba_rp["std"]:.3f})}}\n'
            f'    & {ba_ro["median"]:.3f}\\,{{\\scriptsize($\\pm${ba_ro["std"]:.3f})}}\n'
            f'    & ${delta:+.1f}\\%$ \\\\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  7. Save everything to a single JSON for programmatic access
# ═════════════════════════════════════════════════════════════════════════════

def save_paper_data_json(cfg, weights, exp1, exp2, velocities, outpath):
    """Save all extracted data as a JSON file for external tools."""
    import json

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    data = {
        'config': cfg,
        'weights': weights,
    }

    if exp1 is not None:
        exp1_dq, exp1_mpcc = exp1
        data['experiment1'] = {
            'dq': {k: _to_serializable(v) for k, v in exp1_dq.items()},
            'mpcc': {k: _to_serializable(v) for k, v in exp1_mpcc.items()},
        }

    if exp2 is not None:
        exp2_json = {'meta': exp2['_meta']}
        for ctrl in ['dq', 'base']:
            exp2_json[ctrl] = {}
            for v in velocities:
                vdata = {}
                for metric in ['rmse_pos', 'rmse_ori', 't_lap', 'mean_vtheta']:
                    s = exp2[ctrl][v][metric].copy()
                    s.pop('raw', None)
                    vdata[metric] = {k: _to_serializable(v2) for k, v2 in s.items()}
                vdata['failures'] = exp2[ctrl][v]['failures']
                exp2_json[ctrl][str(v)] = vdata
        data['experiment2'] = exp2_json

    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2, default=_to_serializable)
    print(f"  ✓ JSON data saved to {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*78}")
    print("  extract_paper_data.py — Extracting all metrics from .mat files")
    print(f"{'='*78}")

    # 1. Configuration
    cfg = load_experiment_config()
    weights = load_controller_weights()
    velocities = cfg.get('VELOCITIES', [8, 12, 15])

    print_config(cfg, weights)

    # 2. Experiment 1
    exp1 = load_experiment1()
    if exp1 is not None:
        exp1_dq, exp1_mpcc = exp1
        print_experiment1(exp1_dq, exp1_mpcc)

        # Auto-generate LaTeX table
        tex1 = generate_latex_exp1(exp1_dq, exp1_mpcc)
        tex1_path = os.path.join(_SCRIPT_DIR, "auto_table_exp1.tex")
        with open(tex1_path, 'w') as f:
            f.write(f"% AUTO-GENERATED by extract_paper_data.py — do not edit\n")
            f.write(tex1 + '\n')
        print(f"  ✓ LaTeX table → {tex1_path}")

    # 3. Experiment 2
    exp2 = load_experiment2(velocities)
    if exp2 is not None:
        print_experiment2(exp2, velocities)

        # Auto-generate LaTeX table
        tex2 = generate_latex_exp2(exp2, velocities)
        tex2_path = os.path.join(_SCRIPT_DIR, "auto_table_exp2.tex")
        with open(tex2_path, 'w') as f:
            f.write(f"% AUTO-GENERATED by extract_paper_data.py — do not edit\n")
            f.write(tex2 + '\n')
        print(f"  ✓ LaTeX table → {tex2_path}")

    # 4. Save JSON with all data
    json_path = os.path.join(_SCRIPT_DIR, "paper_data.json")
    save_paper_data_json(cfg, weights, exp1, exp2, velocities, json_path)

    # 5. Summary for quick reference
    print(f"\n{SEP}")
    print("  KEY NUMBERS FOR PAPER TEXT (copy-paste ready)")
    print(SEP)
    if exp1 is not None:
        d, m = exp1
        print(f"  Exp1 DQ:   RMSE_pos={d['rmse_pos']:.3f}m, t_lap={d['t_lap']:.2f}s, "
              f"v̄θ={d['mean_vtheta']:.2f}m/s, solver={d['t_solver_mean']:.2f}ms")
        print(f"  Exp1 MPCC: RMSE_pos={m['rmse_pos']:.3f}m, t_lap={m['t_lap']:.2f}s, "
              f"v̄θ={m['mean_vtheta']:.2f}m/s, solver={m['t_solver_mean']:.2f}ms")
        print(f"  Exp1 Δ%:   RMSE_pos={_delta_pct(d['rmse_pos'], m['rmse_pos']):+.1f}%, "
              f"t_lap={_delta_pct(d['t_lap'], m['t_lap']):+.1f}%")
    if exp2 is not None:
        print()
        for v in velocities:
            dq_m = exp2['dq'][v]['rmse_pos']['median']
            ba_m = exp2['base'][v]['rmse_pos']['median']
            dq_s = exp2['dq'][v]['rmse_pos']['std']
            ba_s = exp2['base'][v]['rmse_pos']['std']
            delta = _delta_pct(dq_m, ba_m)
            print(f"  Exp2 v={v:2d}: DQ={dq_m:.4f}±{dq_s:.4f},  "
                  f"MPCC={ba_m:.4f}±{ba_s:.4f},  Δ%={delta:+.1f}%")

    print(f"\n{SEP}")
    print("  DONE — outputs:")
    print(f"    {os.path.join(_SCRIPT_DIR, 'auto_table_exp1.tex')}")
    print(f"    {os.path.join(_SCRIPT_DIR, 'auto_table_exp2.tex')}")
    print(f"    {os.path.join(_SCRIPT_DIR, 'paper_data.json')}")
    print(SEP)


if __name__ == '__main__':
    main()
