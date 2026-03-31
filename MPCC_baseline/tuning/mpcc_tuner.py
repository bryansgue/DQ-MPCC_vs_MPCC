"""
mpcc_tuner.py  –  MPCC bilevel tuner aligned with the current manual criteria.

This phase now uses:
  • multi-velocity evaluation
  • contouring error as the primary tracking metric
  • lag and attitude as secondary regularisers
  • a soft speed reward to avoid the "go-slow cheat"
  • hard penalties for incomplete or failed runs
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

from MPCC_simulation_tuner import run_simulation

# ── Shared tuning configuration ──────────────────────────────────────────────
from tuning_config import (
    W_INCOMPLETE, W_FAIL,
    DEFAULT_N_TRIALS, DEFAULT_SAMPLER, N_STARTUP_TRIALS, OPTUNA_SEED,
    Q_EC_RANGE, Q_EL_RANGE, Q_ROT_RANGE,
    U_T_RANGE, U_TAU_RANGE,
    Q_OMEGA_RANGE, Q_S_RANGE,
    TUNING_VELOCITIES,
)
from experiment_config import MPCC_MANUAL_WEIGHTS

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
    # Q_ec: contouring error [x, y, z]
    ('Q_ecx',  *Q_EC_RANGE),
    ('Q_ecy',  *Q_EC_RANGE),
    ('Q_ecz',  *Q_EC_RANGE),
    # Q_el: lag error [x, y, z]
    ('Q_elx',  *Q_EL_RANGE),
    ('Q_ely',  *Q_EL_RANGE),
    ('Q_elz',  *Q_EL_RANGE),
    # Q_q: quaternion error [roll, pitch, yaw]  (rotation log-map)
    ('Q_qx',   *Q_ROT_RANGE),
    ('Q_qy',   *Q_ROT_RANGE),
    ('Q_qz',   *Q_ROT_RANGE),
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


# Local objective composition.
W_CONTOUR_LOCAL = 1.0
W_LAG_LOCAL = 0.35
W_ATT_LOCAL = 0.10
W_VEL_SOFT = 0.6


def trial_to_weights(trial) -> dict:
    """Convert Optuna trial suggestions into a weights dict."""
    params = {}
    for name, low, high, log in SEARCH_SPACE:
        if log:
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = trial.suggest_float(name, low, high)

    return {
        'Q_ec':    [params['Q_ecx'], params['Q_ecy'], params['Q_ecz']],
        'Q_el':    [params['Q_elx'], params['Q_ely'], params['Q_elz']],
        'Q_q':     [params['Q_qx'],  params['Q_qy'],  params['Q_qz']],
        'U_mat':   [params['U_T'],   params['U_tx'],  params['U_ty'], params['U_tz']],
        'Q_omega': [params['Q_wx'],  params['Q_wy'],  params['Q_wz']],
        'Q_s':     params['Q_s'],
    }


def dict_to_weights(d: dict) -> dict:
    """Re-pack a flat dict (from best_params) into the weights dict format."""
    return {
        'Q_ec':    [d['Q_ecx'], d['Q_ecy'], d['Q_ecz']],
        'Q_el':    [d['Q_elx'], d['Q_ely'], d['Q_elz']],
        'Q_q':     [d['Q_qx'],  d['Q_qy'],  d['Q_qz']],
        'U_mat':   [d['U_T'],   d['U_tx'],  d['U_ty'], d['U_tz']],
        'Q_omega': [d['Q_wx'],  d['Q_wy'],  d['Q_wz']],
        'Q_s':     d['Q_s'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single-velocity sub-objective
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_single_velocity(weights: dict, v_max: float) -> tuple[float, dict]:
    """Run an MPCC simulation at a specific v_theta_max and return (J, info)."""
    try:
        result = run_simulation(weights=weights, verbose=False, vtheta_max=v_max)
    except Exception as e:
        return 1e6, {'error': str(e)}

    compl     = float(result['path_completed'])
    rmse_c    = float(result.get('rmse_contorno', np.inf))
    rmse_l    = float(result.get('rmse_lag', np.inf))
    rmse_att  = float(result.get('rmse_attitude', np.inf))
    mean_vt   = float(result.get('mean_vtheta', 0.0))
    success   = bool(result.get('success', True))
    fail_ratio = float(result.get('solver_fail_ratio', 0.0))

    if not np.isfinite(rmse_c) or not np.isfinite(rmse_l) or not np.isfinite(rmse_att):
        return 1e6, result

    J = (
        W_CONTOUR_LOCAL * rmse_c
        + W_LAG_LOCAL * rmse_l
        + W_ATT_LOCAL * rmse_att
    )
    J += W_VEL_SOFT * max(0.0, v_max - mean_vt) / v_max
    if compl < 0.99:
        J += W_INCOMPLETE * (1.0 - compl)
    if not success:
        J += W_FAIL
    J += W_FAIL * fail_ratio

    result['_J'] = J
    return J, result


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-velocity objective function
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial) -> float:
    """Evaluate one set of gains over the current stabilisation velocity set."""
    weights = trial_to_weights(trial)

    t0_total = time.time()
    J_per_vel = {}
    info_per_vel = {}

    for v_max in TUNING_VELOCITIES:
        J_v, info_v = _evaluate_single_velocity(weights, v_max)
        J_per_vel[v_max] = J_v
        info_per_vel[v_max] = info_v

    J_multi = np.mean(list(J_per_vel.values()))
    wall_total = time.time() - t0_total

    parts = []
    for v in TUNING_VELOCITIES:
        info = info_per_vel[v]
        if isinstance(info, dict) and 'path_completed' in info:
            parts.append(f"v={v}: J={J_per_vel[v]:.2f} "
                        f"path={info['path_completed']*100:.0f}% "
                        f"v̄θ={float(info.get('mean_vtheta', 0)):.1f} "
                        f"rmse_c={float(info.get('rmse_contorno', 0)):.3f}")
        else:
            parts.append(f"v={v}: FAIL")

    print(f"  [Trial {trial.number:3d}]  J_multi={J_multi:8.3f}  "
          f"{'  |  '.join(parts)}  ({wall_total:.1f}s)", flush=True)

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
        description='Bilevel optimisation of MPCC cost weights (multi-velocity)')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    parser.add_argument('--sampler', type=str, default=DEFAULT_SAMPLER,
                        choices=['tpe', 'cmaes'],
                        help=f'Optuna sampler: tpe or cmaes (default: {DEFAULT_SAMPLER})')
    parser.add_argument('--study-name', type=str, default='mpcc_tuning_multivel',
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
    print("  MPCC BILEVEL GAIN TUNER  (multi-velocity)")
    print("="*70)
    print(f"\n  Sampler    : {args.sampler.upper()}")
    print(f"  Trials     : {args.n_trials}")
    print(f"  Velocities : {TUNING_VELOCITIES}")
    print(f"  Study      : {args.study_name}")
    print(f"  Storage    : {args.db or 'in-memory'}\n")
    print(f"  Objective  : {W_CONTOUR_LOCAL}*rmse_c + "
          f"{W_LAG_LOCAL}*rmse_l + {W_ATT_LOCAL}*rmse_att + vel_penalty\n")

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

    if len(study.trials) == 0:
        study.enqueue_trial({
            'Q_ecx': MPCC_MANUAL_WEIGHTS['Q_ec'][0],
            'Q_ecy': MPCC_MANUAL_WEIGHTS['Q_ec'][1],
            'Q_ecz': MPCC_MANUAL_WEIGHTS['Q_ec'][2],
            'Q_elx': MPCC_MANUAL_WEIGHTS['Q_el'][0],
            'Q_ely': MPCC_MANUAL_WEIGHTS['Q_el'][1],
            'Q_elz': MPCC_MANUAL_WEIGHTS['Q_el'][2],
            'Q_qx': MPCC_MANUAL_WEIGHTS['Q_q'][0],
            'Q_qy': MPCC_MANUAL_WEIGHTS['Q_q'][1],
            'Q_qz': MPCC_MANUAL_WEIGHTS['Q_q'][2],
            'U_T': MPCC_MANUAL_WEIGHTS['U_mat'][0],
            'U_tx': MPCC_MANUAL_WEIGHTS['U_mat'][1],
            'U_ty': MPCC_MANUAL_WEIGHTS['U_mat'][2],
            'U_tz': MPCC_MANUAL_WEIGHTS['U_mat'][3],
            'Q_wx': MPCC_MANUAL_WEIGHTS['Q_omega'][0],
            'Q_wy': MPCC_MANUAL_WEIGHTS['Q_omega'][1],
            'Q_wz': MPCC_MANUAL_WEIGHTS['Q_omega'][2],
            'Q_s': MPCC_MANUAL_WEIGHTS['Q_s'],
        })

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
            for v in TUNING_VELOCITIES:
                record[f'J_v{v}']      = t.user_attrs.get(f'J_v{v}')
                record[f'path_v{v}']   = t.user_attrs.get(f'path_v{v}')
                record[f'vtheta_v{v}'] = t.user_attrs.get(f'vtheta_v{v}')
                record[f'rmse_c_v{v}'] = t.user_attrs.get(f'rmse_c_v{v}')
            history_records.append(record)

    hist_path = os.path.join(out_dir, 'tuning_history.json')
    with open(hist_path, 'w') as fp:
        json.dump({
            'controller': 'MPCC',
            'strategy': 'multi-velocity',
            'tuning_velocities': TUNING_VELOCITIES,
            'n_trials': len(history_records),
            'sampler': args.sampler,
            'objective_info': {
                'type': 'MEAN over velocities of contour-dominant local objective',
                'W_INCOMPLETE': W_INCOMPLETE,
                'W_FAIL': W_FAIL,
                'W_CONTOUR_LOCAL': W_CONTOUR_LOCAL,
                'W_LAG_LOCAL': W_LAG_LOCAL,
                'W_ATT_LOCAL': W_ATT_LOCAL,
                'W_VEL_SOFT': W_VEL_SOFT,
                'TUNING_VELOCITIES': TUNING_VELOCITIES,
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
            'type': 'MEAN over velocities of contour-dominant local objective',
            'strategy': 'multi-velocity',
            'TUNING_VELOCITIES': TUNING_VELOCITIES,
            'W_INCOMPLETE': W_INCOMPLETE,
            'W_FAIL': W_FAIL,
            'W_CONTOUR_LOCAL': W_CONTOUR_LOCAL,
            'W_LAG_LOCAL': W_LAG_LOCAL,
            'W_ATT_LOCAL': W_ATT_LOCAL,
            'W_VEL_SOFT': W_VEL_SOFT,
        },
    }

    with open(out_path, 'w') as fp:
        json.dump(save_data, fp, indent=2)
    print(f"\n[3/3] ✓ Best weights saved to {out_path}")

    # ── Print copy-pasteable format ──────────────────────────────────────
    print("\n  ── Copy-paste into mpcc_controller.py ──")
    print(f"  DEFAULT_Q_EC    = {best_weights['Q_ec']}")
    print(f"  DEFAULT_Q_EL    = {best_weights['Q_el']}")
    print(f"  DEFAULT_Q_Q     = {best_weights['Q_q']}")
    print(f"  DEFAULT_U_MAT   = {best_weights['U_mat']}")
    print(f"  DEFAULT_Q_OMEGA = {best_weights['Q_omega']}")
    print(f"  DEFAULT_Q_S     = {best_weights['Q_s']}")
    print()


if __name__ == '__main__':
    main()
