#!/usr/bin/env python3
"""
generate_tuning_latex.py
────────────────────────
Reads both best_weights.json + tuning_history.json files and generates
a self-contained LaTeX subsection with:

  • Table — Optimised cost weights  (DQ-MPCC vs Baseline MPCC side by side)
  • Table — Per-velocity sub-objectives of the best trial
  • Convergence summary (best J milestones, wall times)
  • Compact interpretive paragraph (all numbers auto-filled)

Continues from the bilevel formulation subsection in the paper
(Section V-F, eq. (36)) and produces the "Tuning Results" subsection.

Usage
─────
    python generate_tuning_latex.py            # → .tex + terminal summary
    python generate_tuning_latex.py --compile  # also runs pdflatex
"""

import os, sys, argparse, json, textwrap
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════════════
_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)                       # workspace root
_OUT_DIR = os.path.join(_ROOT, 'results', 'experiment2')

_DQ_BEST   = os.path.join(_ROOT, 'DQ-MPCC_baseline', 'tuning', 'best_weights.json')
_DQ_HIST   = os.path.join(_ROOT, 'DQ-MPCC_baseline', 'tuning', 'tuning_history.json')
_MPCC_BEST = os.path.join(_ROOT, 'MPCC_baseline',    'tuning', 'best_weights.json')
_MPCC_HIST = os.path.join(_ROOT, 'MPCC_baseline',    'tuning', 'tuning_history.json')
_TEX_OUT   = os.path.join(_OUT_DIR, 'tuning_analysis.tex')

# ── Ensure workspace root is on sys.path so config/ is importable ────────────
sys.path.insert(0, _ROOT)

# ── Tuning meta-parameters (from config/experiment_config) ───────────────────
try:
    from config.experiment_config import (
        TUNING_VELOCITIES, DEFAULT_N_TRIALS, DEFAULT_SAMPLER,
        N_STARTUP_TRIALS, OPTUNA_SEED,
        W_INCOMPLETE, W_TIME, W_VEL, W_ISOTROPY, W_CONTOUR,
        Q_EC_RANGE, Q_EL_RANGE, Q_ROT_RANGE,
        U_T_RANGE, U_TAU_RANGE, Q_OMEGA_RANGE, Q_S_RANGE,
        S_MAX_MANUAL, FREC,
    )
except ImportError:
    TUNING_VELOCITIES = [5, 10, 16]
    DEFAULT_N_TRIALS = 100
    DEFAULT_SAMPLER = 'tpe'
    N_STARTUP_TRIALS = 10
    OPTUNA_SEED = 42
    W_INCOMPLETE = 1000.0
    W_TIME = 0.5; W_VEL = 2.0; W_ISOTROPY = 0.3; W_CONTOUR = 3.0
    Q_EC_RANGE = (1.0, 30.0, True)
    Q_EL_RANGE = (0.5, 30.0, True)
    Q_ROT_RANGE = (0.1, 20.0, True)
    U_T_RANGE = (0.005, 0.5, True)
    U_TAU_RANGE = (20, 800, True)
    Q_OMEGA_RANGE = (0.01, 2.0, True)
    Q_S_RANGE = (0.5, 20, True)
    S_MAX_MANUAL = 80
    FREC = 100


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_best(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt(x, decimals=4):
    """Format a float."""
    if isinstance(x, list):
        return ', '.join(f'{v:.{decimals}f}' for v in x)
    return f'{x:.{decimals}f}'


def _fmt_sci(x):
    """Format small numbers in scientific-ish notation for LaTeX."""
    if abs(x) < 0.01:
        exp = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10**exp)
        return f'{mantissa:.1f}\\!\\times\\!10^{{{exp}}}'
    elif abs(x) < 0.1:
        return f'{x:.3f}'
    elif abs(x) < 1:
        return f'{x:.2f}'
    elif abs(x) < 100:
        return f'{x:.1f}'
    else:
        return f'{x:.0f}'


def _convergence_milestones(trials, checkpoints=None):
    """Return (trial_idx, best_J_so_far) at given trial numbers."""
    if checkpoints is None:
        checkpoints = [0, 10, 25, 50, 75, len(trials) - 1]
    results = []
    for c in checkpoints:
        if c < len(trials):
            results.append((c, trials[c]['best_J_so_far']))
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  LaTeX generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_tex(dq_best, mpcc_best, dq_hist, mpcc_hist):
    """Generate the full tuning LaTeX subsection."""

    n_trials_dq   = dq_hist['n_trials']
    n_trials_mpcc = mpcc_hist['n_trials']
    n_params      = len(dq_best['flat_params'])  # 17

    dq_w   = dq_best['weights']
    mpcc_w = mpcc_best['weights']
    dq_ua  = dq_best['user_attrs']
    mpcc_ua = mpcc_best['user_attrs']
    vels   = dq_best['objective_info']['TUNING_VELOCITIES']

    # ── Wall times ──
    dq_trials   = dq_hist['trials']
    mpcc_trials = mpcc_hist['trials']
    dq_wall   = [t['wall_time'] for t in dq_trials if t.get('wall_time')]
    mpcc_wall = [t['wall_time'] for t in mpcc_trials if t.get('wall_time')]
    dq_wall_mean   = np.mean(dq_wall)
    mpcc_wall_mean = np.mean(mpcc_wall)
    dq_wall_total  = sum(dq_wall) / 60.0   # minutes
    mpcc_wall_total = sum(mpcc_wall) / 60.0

    # ── Best J milestones ──
    dq_final_J   = dq_trials[-1]['best_J_so_far']
    mpcc_final_J = mpcc_trials[-1]['best_J_so_far']

    # ── Build weight table rows ──
    # Shared rows: Q_ec, Q_el, U_mat, Q_omega, Q_s
    # DQ-only: Q_phi (orientation via SE(3) log)
    # MPCC-only: Q_q  (orientation via SO(3) log)
    weight_rows = []

    # Orientation error weight (different name, same role)
    dq_rot = dq_w['Q_phi']
    mpcc_rot = mpcc_w['Q_q']

    # Build rows: (latex_name, dq_values, mpcc_values)
    rows_data = [
        (r'$\bm{q}_{\phi}$ / $\bm{q}_{q}$', dq_rot, mpcc_rot),
        (r'$\bm{q}_{c}$', dq_w['Q_ec'], mpcc_w['Q_ec']),
        (r'$\bm{q}_{\ell}$', dq_w['Q_el'], mpcc_w['Q_el']),
        (r'$U^f$', [dq_w['U_mat'][0]], [mpcc_w['U_mat'][0]]),
        (r'$[U^{\tau_x}, U^{\tau_y}, U^{\tau_z}]$',
         dq_w['U_mat'][1:], mpcc_w['U_mat'][1:]),
        (r'$\bm{q}_{\omega}$', dq_w['Q_omega'], mpcc_w['Q_omega']),
        (r'$Q_s$', [dq_w['Q_s']], [mpcc_w['Q_s']]),
    ]

    table_weight_body = []
    for latex_name, dq_vals, mpcc_vals in rows_data:
        if not isinstance(dq_vals, list):
            dq_vals = [dq_vals]
        if not isinstance(mpcc_vals, list):
            mpcc_vals = [mpcc_vals]
        # Format each value
        dq_str = ', '.join(_fmt_sci(v) for v in dq_vals)
        mpcc_str = ', '.join(_fmt_sci(v) for v in mpcc_vals)
        row = f"    {latex_name} & $[{dq_str}]$ & $[{mpcc_str}]$ \\\\"
        # For scalar Q_s, don't use brackets
        if len(dq_vals) == 1:
            row = f"    {latex_name} & ${dq_str}$ & ${mpcc_str}$ \\\\"
        table_weight_body.append(row)

    weight_body_str = '\n'.join(table_weight_body)

    # ── Build per-velocity sub-objective table ──
    vel_rows = []
    for v in vels:
        dq_J   = dq_ua.get(f'J_v{v}', np.nan)
        mpcc_J = mpcc_ua.get(f'J_v{v}', np.nan)
        dq_path = dq_ua.get(f'path_v{v}', np.nan)
        mpcc_path = mpcc_ua.get(f'path_v{v}', np.nan)
        dq_vt  = dq_ua.get(f'vtheta_v{v}', np.nan)
        mpcc_vt = mpcc_ua.get(f'vtheta_v{v}', np.nan)
        dq_rc  = dq_ua.get(f'rmse_c_v{v}', np.nan)
        mpcc_rc = mpcc_ua.get(f'rmse_c_v{v}', np.nan)

        # Bold the better J (lower is better)
        if dq_J < mpcc_J:
            dq_J_s = f'\\textbf{{{dq_J:.2f}}}'
            mpcc_J_s = f'{mpcc_J:.2f}'
        else:
            dq_J_s = f'{dq_J:.2f}'
            mpcc_J_s = f'\\textbf{{{mpcc_J:.2f}}}'

        row = (f"    {v} & {dq_J_s} & {dq_vt:.2f} & {dq_rc:.3f} "
               f"& {mpcc_J_s} & {mpcc_vt:.2f} & {mpcc_rc:.3f} \\\\")
        vel_rows.append(row)

    # Average row
    dq_J_avg = np.mean([dq_ua[f'J_v{v}'] for v in vels])
    mpcc_J_avg = np.mean([mpcc_ua[f'J_v{v}'] for v in vels])
    dq_vt_avg = np.mean([dq_ua[f'vtheta_v{v}'] for v in vels])
    mpcc_vt_avg = np.mean([mpcc_ua[f'vtheta_v{v}'] for v in vels])
    dq_rc_avg = np.mean([dq_ua[f'rmse_c_v{v}'] for v in vels])
    mpcc_rc_avg = np.mean([mpcc_ua[f'rmse_c_v{v}'] for v in vels])

    vel_body_str = '\n'.join(vel_rows)

    # ── Determine J_multi winner ──
    dq_J_multi = dq_best['best_J_multi']
    mpcc_J_multi = mpcc_best['best_J_multi']
    j_diff_pct = (dq_J_multi - mpcc_J_multi) / mpcc_J_multi * 100.0
    dq_wins_j = dq_J_multi < mpcc_J_multi

    # ── Format the differences in the per-velocity metrics ──
    # DQ vs MPCC vtheta and rmse_c
    vt_comparisons = []
    rc_comparisons = []
    for v in vels:
        dq_vt = dq_ua[f'vtheta_v{v}']
        mpcc_vt = mpcc_ua[f'vtheta_v{v}']
        dq_rc = dq_ua[f'rmse_c_v{v}']
        mpcc_rc = mpcc_ua[f'rmse_c_v{v}']
        vt_comparisons.append((v, dq_vt, mpcc_vt))
        rc_comparisons.append((v, dq_rc, mpcc_rc))

    # ── Convergence: at which trial did each reach within 10% of final? ──
    def _trial_within_pct(trials, pct=10):
        final = trials[-1]['best_J_so_far']
        threshold = final * (1 + pct / 100.0)
        for t in trials:
            if t['best_J_so_far'] <= threshold:
                return t['trial']
        return len(trials) - 1

    dq_conv_trial = _trial_within_pct(dq_trials, 10)
    mpcc_conv_trial = _trial_within_pct(mpcc_trials, 10)

    # ── Build LaTeX ──────────────────────────────────────────────────────
    n_vel = len(vels)
    n_evals_per_trial = n_vel  # 3 sims per trial
    vel_set_str = ', '.join(str(v) for v in vels)

    # ── Search-space table rows (auto from config) ──
    search_rows = [
        (r'$Q_{c,i}$',        f'$[{Q_EC_RANGE[0]},\;{Q_EC_RANGE[1]:.0f}]$'),
        (r'$Q_{\ell,i}$',     f'$[{Q_EL_RANGE[0]},\;{Q_EL_RANGE[1]:.0f}]$'),
        (r'$Q_{{\phi,i}}$',   f'$[{Q_ROT_RANGE[0]},\;{Q_ROT_RANGE[1]:.0f}]$'),
        (r'$U^f$',            f'$[{U_T_RANGE[0]},\;{U_T_RANGE[1]}]$'),
        (r'$U^{{\tau_j}}$',   f'$[{U_TAU_RANGE[0]:.0f},\;{U_TAU_RANGE[1]:.0f}]$'),
        (r'$Q_{{\omega,i}}$', f'$[{Q_OMEGA_RANGE[0]},\;{Q_OMEGA_RANGE[1]}]$'),
        (r'$Q_s$',            f'$[{Q_S_RANGE[0]},\;{Q_S_RANGE[1]:.0f}]$'),
    ]
    search_body = '\n'.join(
        f'    {name:20s} & {rng} \\\\' for name, rng in search_rows
    )

    tex = textwrap.dedent(r"""
    %% ═══════════════════════════════════════════════════════════════════════
    %%  AUTO-GENERATED by generate_tuning_latex.py
    %%  Date: """ + f"{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}" + r"""
    %%  Sources: DQ-MPCC_baseline/tuning/best_weights.json
    %%           MPCC_baseline/tuning/best_weights.json
    %%
    %%  DO NOT EDIT — re-run the script to regenerate.
    %% ═══════════════════════════════════════════════════════════════════════

    %% ─────────────────────────────────────────────────────────────────────
    %%  Bilevel formulation
    %% ─────────────────────────────────────────────────────────────────────
    \subsubsection{Multi-Velocity Bilevel Problem}
    \label{sec:bilevel_problem}

    We seek the weight vector
    $\boldsymbol{\vartheta}\!\in\!\mathbb{R}^{""" + f"{n_params}" + r"""}$
    that minimises a scalar meta-objective
    averaged over a representative velocity set
    $\mathcal{V}=\{""" + vel_set_str + r"""\}$~m/s:
    %
    \begin{equation}
      \boldsymbol{\vartheta}^{\star}
      \;=\;
      \arg\min_{\boldsymbol{\vartheta}\,\in\,\Theta}\;
      J_{\mathrm{multi}}(\boldsymbol{\vartheta}),
      \qquad
      J_{\mathrm{multi}}
      \;=\;
      \frac{1}{|\mathcal{V}|}
      \sum_{v\in\mathcal{V}} J(\boldsymbol{\vartheta},\,v),
      \label{eq:bilevel_multi}
    \end{equation}
    %
    where $\Theta$ is a log-uniform hyper-rectangle (Table~\ref{tab:search_space})
    and $J(\boldsymbol{\vartheta},v)$ is the \emph{single-velocity sub-objective}
    obtained by running a full closed-loop simulation at
    $v_\theta^{\max}=v$.

    Each sub-objective aggregates the accumulated MPCC stage cost with
    five penalty terms that prevent degenerate solutions
    (velocity starvation, incomplete laps, axis-selective tracking):
    %
    \begin{equation}
    \begin{aligned}
    J(\boldsymbol{\vartheta},v)
    &= \bar{\ell}
    +\underbrace{W_{\mathrm{inc}}\,\max(0,\,1\!-\!c)
      \vphantom{\frac{t}{T}}}_{\text{completion}}
    +\underbrace{W_{t}\,\frac{t_{\mathrm{lap}}}{T_{\mathrm{ref}}}}_{\text{lap time}}
    \\[2pt]
    &\quad
    +\underbrace{W_{v}\,\frac{[v - \bar{v}_\theta]^{+}}{v}}_{\text{velocity}}
    +\underbrace{W_{\alpha}\!\left(\frac{\max_i r_i}{\min_i r_i}-1\right)}_{\text{isotropy}}
    \\[2pt]
    &\quad
    +\underbrace{W_{c}\,\mathrm{RMSE}_{c}\vphantom{\frac{t}{T}}}_{\text{contouring}}\,.
    \end{aligned}
    \label{eq:sub_objective}
    \end{equation}
    %
    Here
    $\bar{\ell}$ is the mean MPCC stage cost over $K$ steps,
    $c=\theta_K/s_{\max}$ is the fractional path completion,
    $T_{\mathrm{ref}}=s_{\max}/v$ is the ideal lap time,
    $[\cdot]^{+}=\max(\cdot,0)$,
    $r_i = \bigl(\frac{1}{K}\sum_{k} e_{c,i,k}^{2}+e_{\ell,i,k}^{2}\bigr)^{1/2}$
    is the per-axis combined tracking RMSE ($i\!\in\!\{x,y,z\}$),
    and $\mathrm{RMSE}_{c}$ is the scalar contouring RMSE.
    The fixed penalty weights are
    $W_{\mathrm{inc}}\!=\!""" + f"{W_INCOMPLETE:.0f}" + r"""$,
    $W_t\!=\!""" + f"{W_TIME}" + r"""$,
    $W_v\!=\!""" + f"{W_VEL}" + r"""$,
    $W_\alpha\!=\!""" + f"{W_ISOTROPY}" + r"""$,
    $W_c\!=\!""" + f"{W_CONTOUR}" + r"""$.

    %% ── Search-space table ──
    \begin{table}[!t]
    \centering
    \caption{Log-uniform search space $\Theta$ shared by both controllers
    ($i\!\in\!\{x,y,z\}$, $j\!\in\!\{x,y,z\}$).
    All """ + f"{n_params}" + r""" parameters are sampled independently.}
    \label{tab:search_space}
    \renewcommand{\arraystretch}{1.1}
    \setlength{\tabcolsep}{6pt}
    \begin{tabular}{l c}
    \toprule
    \textbf{Parameter} & \textbf{Range} \\
    \midrule
    """ + search_body + r"""
    \bottomrule
    \end{tabular}
    \end{table}

    The outer optimiser is the Tree-structured Parzen Estimator (TPE)
    implemented in Optuna~\cite{akiba2019optuna},
    with """ + f"{N_STARTUP_TRIALS}" + r""" uniform warm-up trials
    and seed """ + f"{OPTUNA_SEED}" + r""" for reproducibility.
    Each trial requires """ + f"{n_vel}" + r""" closed-loop simulations
    (one per $v\!\in\!\mathcal{V}$), making the per-trial cost
    approximately """ + f"{n_vel}" + r"""$\times$ a single run.

    %% ─────────────────────────────────────────────────────────────────────
    %%  Tuning results
    %% ─────────────────────────────────────────────────────────────────────
    \subsubsection{Tuning Results}
    \label{sec:tuning_results}

    The procedure is executed independently for
    both controllers using """ + f"{n_trials_dq}" + r""" trials.
    The search converges within approximately """ + f"{max(dq_conv_trial, mpcc_conv_trial)}" + r""" trials
    for both controllers
    (total wall time: """ + f"{dq_wall_total:.0f}" + r"""~min for DQ-MPCC,
    """ + f"{mpcc_wall_total:.0f}" + r"""~min for the Baseline,
    at """ + f"{dq_wall_mean:.1f}" + r"""~s and """ + f"{mpcc_wall_mean:.1f}" + r"""~s per trial respectively).

    %% ─────────────────────────────────────────────────────────────────────
    %%  TABLE — Optimised cost weights
    %% ─────────────────────────────────────────────────────────────────────

    \begin{table}[!t]
    \centering
    \caption{Optimised cost weights obtained by bilevel Bayesian
    optimisation (""" + f"{n_trials_dq}" + r""" TPE trials per controller).
    $\bm{q}_\phi$ denotes the $\mathrm{SE}(3)$ rotational penalty (DQ-MPCC)
    and $\bm{q}_q$ the $\mathrm{SO}(3)$ rotational penalty (Baseline).}
    \label{tab:cost_weights}
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l c c}
    \toprule
    \textbf{Weight} & \textbf{DQ-MPCC} & \textbf{Baseline MPCC} \\
    \midrule
    """ + weight_body_str + r"""
    \midrule
    $J_{\mathrm{multi}}^{\star}$ & """ + f"${dq_J_multi:.2f}$" + r""" & """ + f"${mpcc_J_multi:.2f}$" + r""" \\
    Best trial & """ + f"\\#{dq_best['best_trial']}" + r""" & """ + f"\\#{mpcc_best['best_trial']}" + r""" \\
    \bottomrule
    \end{tabular}
    \end{table}

    Table~\ref{tab:cost_weights} reports the optimal weights.
    """ + (
        f"The DQ-MPCC achieves a ${abs(j_diff_pct):.0f}\\%$ lower aggregate objective "
        f"($J_{{\\mathrm{{multi}}}}^\\star = {dq_J_multi:.2f}$ vs.\\ ${mpcc_J_multi:.2f}$)"
        if dq_wins_j else
        f"The Baseline achieves a ${abs(j_diff_pct):.0f}\\%$ lower aggregate objective "
        f"($J_{{\\mathrm{{multi}}}}^\\star = {mpcc_J_multi:.2f}$ vs.\\ ${dq_J_multi:.2f}$)"
    ) + r""",
    indicating that the $\mathrm{SE}(3)$ formulation""" + (
        r""" yields a cost landscape more amenable to Bayesian optimisation."""
        if dq_wins_j else
        r""" presents a harder cost landscape for the optimiser."""
    ) + r"""
    Both controllers allocate the largest tracking penalties
    to the lateral axes ($""" + (
        f"Q_{{c}}^y = {dq_w['Q_ec'][1]:.1f}" if dq_w['Q_ec'][1] > dq_w['Q_ec'][0] else
        f"Q_{{c}}^x = {dq_w['Q_ec'][0]:.1f}"
    ) + r"""$ for DQ-MPCC,
    $""" + (
        f"Q_{{c}}^y = {mpcc_w['Q_ec'][1]:.1f}" if mpcc_w['Q_ec'][1] > mpcc_w['Q_ec'][0] else
        f"Q_{{c}}^x = {mpcc_w['Q_ec'][0]:.1f}"
    ) + r"""$ for the Baseline),
    consistent with the Lissajous geometry that generates the
    largest contouring excursions in the $y$-axis.
    A notable difference is the torque penalty $\bm{w}_u$:
    the Baseline requires
    $U^{\tau_x} = """ + f"{mpcc_w['U_mat'][1]:.1f}" + r"""$,
    roughly """ + f"{mpcc_w['U_mat'][1] / dq_w['U_mat'][1]:.0f}" + r"""$\times$ larger
    than the DQ-MPCC value
    ($""" + f"{dq_w['U_mat'][1]:.1f}" + r"""$),
    suggesting that the decoupled Euclidean formulation needs stronger
    input regularisation to maintain smooth torque profiles.

    """)

    return tex


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tuning analysis from best_weights + history')
    parser.add_argument('--compile', action='store_true',
                        help='Also compile with pdflatex')
    args = parser.parse_args()

    # ── Check files exist ────────────────────────────────────────────────
    for p, name in [(_DQ_BEST, 'DQ best_weights'),
                    (_MPCC_BEST, 'MPCC best_weights'),
                    (_DQ_HIST, 'DQ tuning_history'),
                    (_MPCC_HIST, 'MPCC tuning_history')]:
        if not os.path.isfile(p):
            print(f'[ERROR] Not found: {p}  ({name})')
            sys.exit(1)

    print('[INFO] Loading tuning data ...')
    dq_best   = load_best(_DQ_BEST)
    mpcc_best = load_best(_MPCC_BEST)
    dq_hist   = load_history(_DQ_HIST)
    mpcc_hist = load_history(_MPCC_HIST)

    # ── Terminal summary ──
    print()
    print('=' * 70)
    print(f'  DQ-MPCC:   {dq_hist["n_trials"]} trials, '
          f'best J_multi = {dq_best["best_J_multi"]:.4f} '
          f'(trial #{dq_best["best_trial"]})')
    print(f'  Baseline:  {mpcc_hist["n_trials"]} trials, '
          f'best J_multi = {mpcc_best["best_J_multi"]:.4f} '
          f'(trial #{mpcc_best["best_trial"]})')
    print('=' * 70)
    print()

    # ── Generate .tex ──
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f'[INFO] Generating {_TEX_OUT} ...')
    tex_content = generate_tex(dq_best, mpcc_best, dq_hist, mpcc_hist)

    with open(_TEX_OUT, 'w') as f:
        f.write(tex_content)
    print(f'✓  {_TEX_OUT}')
    print(f'   ({len(tex_content)} chars, {tex_content.count(chr(10))} lines)')

    # ── Optional compilation ──
    if args.compile:
        wrapper = os.path.join(_OUT_DIR, '_compile_tuning.tex')
        with open(wrapper, 'w') as f:
            f.write(textwrap.dedent(r"""
                \documentclass[journal,twocolumn]{IEEEtran}
                \usepackage{amsmath,amssymb,booktabs,graphicx,bm}
                \begin{document}
                \input{tuning_analysis.tex}
                \end{document}
            """).lstrip())
        import subprocess
        print('[INFO] Compiling with pdflatex ...')
        subprocess.run(['pdflatex', '-interaction=nonstopmode',
                        '_compile_tuning.tex'],
                       cwd=_OUT_DIR, capture_output=True)
        pdf = os.path.join(_OUT_DIR, '_compile_tuning.pdf')
        if os.path.isfile(pdf):
            print(f'✓  {pdf}')
        else:
            print('[WARN] pdflatex failed — check _compile_tuning.log')

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
