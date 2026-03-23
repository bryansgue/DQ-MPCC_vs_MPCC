"""
dq_mpcc_tuner.py  –  Bilevel optimisation of DQ-MPCC cost weights.

Architecture
────────────
  LEVEL 1 (outer):  Optuna (Bayesian / TPE) or CMA-ES optimises the
                    gain vector θ_g = [Q_phi, Q_ec, Q_el, U_mat, Q_omega, Q_s]
  LEVEL 2 (inner):  Full DQ-MPCC simulation with acados (runs ~30 s per eval).
                    The solver was compiled ONCE with symbolic parameters.

The meta-cost uses the SAME cost function as the DQ-MPCC solver:

    J_stage = φ'Q_φ·φ + ρ_cont'Q_ec·ρ_cont + ρ_lag'Q_el·ρ_lag
            + u'U·u + ω'Q_ω·ω + Q_s·(v_max − v_θ)²

accumulated over the entire trajectory.  An additional penalty for
incomplete path traversal is added to discourage premature stops.

Usage
─────
    cd /home/bryansgue/dev/ros2/DQ-MPCC_vs_MPCC_baseline/DQ-MPCC_baseline
    python -m tuning.dq_mpcc_tuner                     # run with defaults
    python -m tuning.dq_mpcc_tuner --n-trials 200      # more trials
    python -m tuning.dq_mpcc_tuner --sampler cmaes     # use CMA-ES

After optimisation, the best weights are printed and saved to
    tuning/best_weights.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── Add project root + workspace root to path ────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
for p in (PROJECT_ROOT, WORKSPACE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from DQ_MPCC_simulation_tuner import run_simulation

# ── Shared tuning configuration ──────────────────────────────────────────────
from tuning_config import (
    W_INCOMPLETE, W_TIME, W_VEL, W_ISOTROPY, W_CONTOUR,
    FREC, VTHETA_MAX,
    DEFAULT_N_TRIALS, DEFAULT_SAMPLER, N_STARTUP_TRIALS, OPTUNA_SEED,
    Q_EC_RANGE, Q_EL_RANGE, Q_ROT_RANGE,
    U_T_RANGE, U_TAU_RANGE,
    Q_OMEGA_RANGE, Q_S_RANGE,
    TUNING_VELOCITIES,
)

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ══════════════════════════════════════════════════════════════════════════════
#  Search space definition  (bounds come from tuning_config.py)
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (name, low, high, log_scale)
SEARCH_SPACE = [
    # Q_phi: orientation error (so(3)) [φ_x, φ_y, φ_z]  (rotation log-map)
    ('Q_phix', *Q_ROT_RANGE),
    ('Q_phiy', *Q_ROT_RANGE),
    ('Q_phiz', *Q_ROT_RANGE),
    # Q_ec: contouring error [ρ_cx, ρ_cy, ρ_cz]
    ('Q_ecx',  *Q_EC_RANGE),
    ('Q_ecy',  *Q_EC_RANGE),
    ('Q_ecz',  *Q_EC_RANGE),
    # Q_el: lag error [ρ_lx, ρ_ly, ρ_lz]
    ('Q_elx',  *Q_EL_RANGE),
    ('Q_ely',  *Q_EL_RANGE),
    ('Q_elz',  *Q_EL_RANGE),
    # U_mat: control effort [T, τx, τy, τz]
    ('U_T',    *U_T_RANGE),
    ('U_tx',   *U_TAU_RANGE),
    ('U_ty',   *U_TAU_RANGE),
    ('U_tz',   *U_TAU_RANGE),
    # Q_omega: angular velocity [ωx, ωy, ωz]
    ('Q_wx',   *Q_OMEGA_RANGE),
    ('Q_wy',   *Q_OMEGA_RANGE),
    ('Q_wz',   *Q_OMEGA_RANGE),
    # Q_s: progress speed
    ('Q_s',    *Q_S_RANGE),
]


def trial_to_weights(trial) -> dict:
    """Convert Optuna trial suggestions into a weights dict."""
    params = {}
    for name, low, high, log in SEARCH_SPACE:
        if log:
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = trial.suggest_float(name, low, high)

    return {
        'Q_phi':   [params['Q_phix'], params['Q_phiy'], params['Q_phiz']],
        'Q_ec':    [params['Q_ecx'],  params['Q_ecy'],  params['Q_ecz']],
        'Q_el':    [params['Q_elx'],  params['Q_ely'],  params['Q_elz']],
        'U_mat':   [params['U_T'],    params['U_tx'],   params['U_ty'], params['U_tz']],
        'Q_omega': [params['Q_wx'],   params['Q_wy'],   params['Q_wz']],
        'Q_s':     params['Q_s'],
    }


def dict_to_weights(d: dict) -> dict:
    """Re-pack a flat dict (from best_params) into the weights dict format."""
    return {
        'Q_phi':   [d['Q_phix'], d['Q_phiy'], d['Q_phiz']],
        'Q_ec':    [d['Q_ecx'],  d['Q_ecy'],  d['Q_ecz']],
        'Q_el':    [d['Q_elx'],  d['Q_ely'],  d['Q_elz']],
        'U_mat':   [d['U_T'],    d['U_tx'],   d['U_ty'], d['U_tz']],
        'Q_omega': [d['Q_wx'],   d['Q_wy'],   d['Q_wz']],
        'Q_s':     d['Q_s'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single-velocity sub-objective
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_single_velocity(weights: dict, v_max: float,
                              verbose_label: str = "") -> tuple[float, dict]:
    """Run a DQ-MPCC simulation at a specific v_theta_max and return (J, info).

    Returns
    -------
    J     : float   sub-objective for this velocity
    info  : dict    metrics dictionary from run_simulation
    """
    try:
        result = run_simulation(weights=weights, verbose=False, vtheta_max=v_max)
    except Exception as e:
        return 1e6, {'error': str(e)}

    mpcc_cost = result['mean_mpcc_cost']
    compl     = result['path_completed']
    rmse_c    = result['rmse_contorno']
    v_mean    = float(result['mean_vtheta'])
    N_steps   = result['N_steps']
    s_max     = result['s_max']

    # Guard against NaN / Inf
    if not np.isfinite(mpcc_cost) or not np.isfinite(rmse_c):
        return 1e6, result

    J = mpcc_cost

    # Completion penalty
    if compl < 0.99:
        J += W_INCOMPLETE * (1.0 - compl)

    # Term A: penalise long lap times
    t_s   = 1.0 / FREC
    t_lap = N_steps * t_s
    T_ref = s_max / v_max
    J += W_TIME * (t_lap / T_ref)

    # Term B: penalise low mean progress velocity
    vel_ratio = max(0.0, (v_max - v_mean)) / v_max
    J += W_VEL * vel_ratio

    # Term C: axis-anisotropy
    e_cont_3 = result['e_contorno']
    e_lag_3  = result['e_lag']
    e_all_3  = np.sqrt(e_cont_3**2 + e_lag_3**2)
    rmse_xyz = np.sqrt(np.mean(e_all_3**2, axis=1))
    aniso    = rmse_xyz.max() / (rmse_xyz.min() + 1e-8) - 1.0
    J += W_ISOTROPY * aniso

    # Term D: contouring error
    J += W_CONTOUR * rmse_c

    result['_J'] = J
    result['_t_lap'] = N_steps * t_s
    result['_aniso'] = aniso
    return J, result


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-velocity objective function
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial) -> float:
    """Evaluate one set of gains across MULTIPLE representative velocities.

    The objective is the MEAN sub-objective over TUNING_VELOCITIES:
        J_multi = (1/|V|) * Σ_v  J(w, v)

    This produces a single robust weight set that generalises across
    the entire velocity sweep, avoiding over-fitting to one speed.
    """
    weights = trial_to_weights(trial)

    t0_total = time.time()
    J_per_vel = {}
    info_per_vel = {}

    for v_max in TUNING_VELOCITIES:
        J_v, info_v = _evaluate_single_velocity(weights, v_max)
        J_per_vel[v_max] = J_v
        info_per_vel[v_max] = info_v

    # Mean objective across velocities
    J_multi = np.mean(list(J_per_vel.values()))
    wall_total = time.time() - t0_total

    # ── Logging ──────────────────────────────────────────────────────────
    parts = []
    for v in TUNING_VELOCITIES:
        info = info_per_vel[v]
        if isinstance(info, dict) and 'path_completed' in info:
            parts.append(f"v={v}: J={J_per_vel[v]:.2f} "
                        f"path={info['path_completed']*100:.0f}% "
                        f"v̄θ={float(info.get('mean_vtheta', 0)):.1f}")
        else:
            parts.append(f"v={v}: FAIL")

    print(f"  [Trial {trial.number:3d}]  J_multi={J_multi:8.3f}  "
          f"{'  |  '.join(parts)}  ({wall_total:.1f}s)", flush=True)

    # ── Store sub-metrics ────────────────────────────────────────────────
    trial.set_user_attr('J_multi', float(J_multi))
    trial.set_user_attr('wall_time', wall_total)
    for v in TUNING_VELOCITIES:
        info = info_per_vel[v]
        if isinstance(info, dict) and 'path_completed' in info:
            trial.set_user_attr(f'J_v{v}', float(J_per_vel[v]))
            trial.set_user_attr(f'path_v{v}', float(info['path_completed']))
            trial.set_user_attr(f'vtheta_v{v}', float(info.get('mean_vtheta', 0)))
            trial.set_user_attr(f'rmse_c_v{v}', float(info.get('rmse_contorno', 0)))

    return J_multi


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Bilevel optimisation of DQ-MPCC cost weights (multi-velocity)')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    parser.add_argument('--sampler', type=str, default=DEFAULT_SAMPLER,
                        choices=['tpe', 'cmaes'],
                        help=f'Optuna sampler: tpe or cmaes (default: {DEFAULT_SAMPLER})')
    parser.add_argument('--study-name', type=str, default='dq_mpcc_tuning_multivel',
                        help='Optuna study name')
    parser.add_argument('--db', type=str, default=None,
                        help='Optuna storage (e.g. sqlite:///tuning.db). '
                             'If not set, in-memory.')
    args = parser.parse_args()

    if not HAS_OPTUNA:
        print("ERROR: optuna not installed.  Run:  pip install optuna")
        sys.exit(1)

    # ── Force first infrastructure build (compiles solver ONCE) ──────────
    print("="*70)
    print("  DQ-MPCC BILEVEL GAIN TUNER  (multi-velocity)")
    print("="*70)
    print(f"\n  Sampler    : {args.sampler.upper()}")
    print(f"  Trials     : {args.n_trials}")
    print(f"  Velocities : {TUNING_VELOCITIES}")
    print(f"  Study      : {args.study_name}")
    print(f"  Storage    : {args.db or 'in-memory'}\n")

    print("[1/3] Building infrastructure (trajectory + solver) ...")
    t0 = time.time()
    _ = run_simulation(weights=None, verbose=False)
    print(f"[1/3] Done. Baseline simulation took {time.time()-t0:.1f}s\n")

    # ── Create Optuna study ──────────────────────────────────────────────
    if args.sampler == 'cmaes':
        sampler = CmaEsSampler(seed=OPTUNA_SEED)
    else:
        sampler = TPESampler(seed=OPTUNA_SEED, n_startup_trials=N_STARTUP_TRIALS)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction='minimize',
        storage=args.db,
        load_if_exists=True,
    )

    # ── Run optimisation ─────────────────────────────────────────────────
    print(f"[2/3] Running {args.n_trials} trials "
          f"(×{len(TUNING_VELOCITIES)} velocities each) ...\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ──────────────────────────────────────────────────────────
    best = study.best_trial
    best_weights = dict_to_weights(best.params)

    print("\n" + "="*70)
    print("  OPTIMISATION COMPLETE  (multi-velocity)")
    print("="*70)
    print(f"\n  Best trial  : #{best.number}")
    print(f"  Best J_multi: {best.value:.4f}")
    print(f"  Velocities  : {TUNING_VELOCITIES}")
    for v in TUNING_VELOCITIES:
        J_v = best.user_attrs.get(f'J_v{v}', '?')
        p_v = best.user_attrs.get(f'path_v{v}', '?')
        vt  = best.user_attrs.get(f'vtheta_v{v}', '?')
        rc  = best.user_attrs.get(f'rmse_c_v{v}', '?')
        print(f"    v={v:2d}: J={J_v}  path={p_v}  v̄θ={vt}  RMSE_c={rc}")
    print(f"\n  Best weights:")
    for key, val in best_weights.items():
        print(f"    {key:10s} = {val}")

    # ── Save trial history (for convergence plots) ───────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))

    history_records = []
    best_so_far = float('inf')
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            J_val = t.value
            best_so_far = min(best_so_far, J_val)
            record = {
                'trial':          t.number,
                'J_multi':        J_val,
                'best_J_so_far':  best_so_far,
                'wall_time':      t.user_attrs.get('wall_time'),
            }
            # Per-velocity breakdown
            for v in TUNING_VELOCITIES:
                record[f'J_v{v}']      = t.user_attrs.get(f'J_v{v}')
                record[f'path_v{v}']   = t.user_attrs.get(f'path_v{v}')
                record[f'vtheta_v{v}'] = t.user_attrs.get(f'vtheta_v{v}')
                record[f'rmse_c_v{v}'] = t.user_attrs.get(f'rmse_c_v{v}')
            history_records.append(record)

    hist_path = os.path.join(out_dir, 'tuning_history.json')
    with open(hist_path, 'w') as fp:
        json.dump({
            'controller': 'DQ-MPCC',
            'strategy': 'multi-velocity',
            'tuning_velocities': TUNING_VELOCITIES,
            'n_trials': len(history_records),
            'sampler': args.sampler,
            'objective_info': {
                'type': 'MEAN over velocities of (dq_mpcc_cost + completion + time + vel + isotropy + contour)',
                'W_INCOMPLETE': W_INCOMPLETE,
                'W_TIME': W_TIME,
                'W_VEL': W_VEL,
                'W_ISOTROPY': W_ISOTROPY,
                'W_CONTOUR': W_CONTOUR,
                'TUNING_VELOCITIES': TUNING_VELOCITIES,
                'FREC': FREC,
            },
            'trials': history_records,
        }, fp, indent=2)
    print(f"\n  ✓ Trial history saved to {hist_path}")

    # ── Save to JSON ─────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, 'best_weights.json')

    save_data = {
        'best_J_multi': best.value,
        'best_trial': best.number,
        'weights': {k: (v if isinstance(v, (int, float)) else list(v))
                    for k, v in best_weights.items()},
        'flat_params': {k: float(v) for k, v in best.params.items()},
        'user_attrs': {k: float(v) for k, v in best.user_attrs.items()
                       if isinstance(v, (int, float))},
        'objective_info': {
            'type': 'MEAN over velocities of (dq_mpcc_cost + completion + time + vel + isotropy + contour)',
            'strategy': 'multi-velocity',
            'TUNING_VELOCITIES': TUNING_VELOCITIES,
            'W_INCOMPLETE': W_INCOMPLETE,
            'W_TIME': W_TIME,
            'W_VEL': W_VEL,
            'W_ISOTROPY': W_ISOTROPY,
            'W_CONTOUR': W_CONTOUR,
            'FREC': FREC,
        },
    }

    with open(out_path, 'w') as fp:
        json.dump(save_data, fp, indent=2)
    print(f"\n[3/3] ✓ Best weights saved to {out_path}")

    # ── Print copy-pasteable format ──────────────────────────────────────
    print("\n  ── Copy-paste into dq_mpcc_controller.py ──")
    print(f"  DEFAULT_Q_PHI   = {best_weights['Q_phi']}")
    print(f"  DEFAULT_Q_EC    = {best_weights['Q_ec']}")
    print(f"  DEFAULT_Q_EL    = {best_weights['Q_el']}")
    print(f"  DEFAULT_U_MAT   = {best_weights['U_mat']}")
    print(f"  DEFAULT_Q_OMEGA = {best_weights['Q_omega']}")
    print(f"  DEFAULT_Q_S     = {best_weights['Q_s']}")
    print()


if __name__ == '__main__':
    main()
