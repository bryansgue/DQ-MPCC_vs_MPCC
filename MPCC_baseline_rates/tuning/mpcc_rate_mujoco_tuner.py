"""mpcc_rate_mujoco_tuner.py — Bilevel optimisation of MPCC-rates gains (MuJoCo SiL).

Architecture
────────────
  LEVEL 1 (outer):  Optuna (TPE or CMA-ES) optimises the gain vector
                    θ_g = [Q_ec, Q_el, Q_q, U_mat, Q_s]  (Q_omega removed)
  LEVEL 2 (inner):  Full MPCC SiL simulation with acados + MuJoCo.
                    Solver compiled ONCE; MuJoCo reset between trials.

The meta-cost mirrors the MPCC cost function:
    J = ec'Q_ec·ec + el'Q_el·el + logq'Q_q·logq + u'U·u + Q_s*(v_max-v_θ)²

accumulated over the full trajectory, with additional penalties for:
    - incomplete path traversal
    - slow laps (elapsed / expected time)
    - mean v_θ below v_max
    - anisotropic XYZ tracking error

Usage
─────
    # Terminal 1: Launch MuJoCo
    mujoco_launch.sh scene:=motors

    # Terminal 2: Run tuner
    cd ~/dev/ros2/DQ-MPCC_vs_MPCC_baseline
    python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner
    python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner --n-trials 100
    python3 -m MPCC_baseline_rates.tuning.mpcc_rate_mujoco_tuner --sampler cmaes

Results saved to:  MPCC_baseline_rates/tuning/best_weights.json
                   MPCC_baseline_rates/tuning/tuning_history.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_TUNING_DIR   = os.path.dirname(os.path.abspath(__file__))
_RATES_DIR    = os.path.dirname(_TUNING_DIR)
_PROJECT_ROOT = os.path.dirname(_RATES_DIR)
for _p in (_PROJECT_ROOT, _RATES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MPCC_baseline_rates.mpcc_mujoco_tuner_runner import run_simulation_mujoco
from MPCC_baseline_rates.tuning.tuning_config import (
    W_INCOMPLETE, W_TIME, W_VEL, W_ISOTROPY, W_CONTOUR,
    FREC,
    DEFAULT_N_TRIALS, DEFAULT_SAMPLER, N_STARTUP_TRIALS, OPTUNA_SEED,
    Q_EC_RANGE, Q_EL_RANGE, Q_ROT_RANGE,
    U_T_RANGE, U_W_RANGE, Q_S_RANGE,
    TUNING_VELOCITIES,
)

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ══════════════════════════════════════════════════════════════════════════════
#  Search space
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (parameter_name, low, high, log_scale)
SEARCH_SPACE = [
    # Q_ec: contouring error [x, y, z]
    ('Q_ecx', *Q_EC_RANGE),
    ('Q_ecy', *Q_EC_RANGE),
    ('Q_ecz', *Q_EC_RANGE),
    # Q_el: lag error [x, y, z]
    ('Q_elx', *Q_EL_RANGE),
    ('Q_ely', *Q_EL_RANGE),
    ('Q_elz', *Q_EL_RANGE),
    # Q_q: quaternion log error [roll, pitch, yaw]
    ('Q_qx',  *Q_ROT_RANGE),
    ('Q_qy',  *Q_ROT_RANGE),
    ('Q_qz',  *Q_ROT_RANGE),
    # U_mat: control effort [T, wx, wy, wz]
    ('U_T',  *U_T_RANGE),
    ('U_wx', *U_W_RANGE),
    ('U_wy', *U_W_RANGE),
    ('U_wz', *U_W_RANGE),
    # Q_s: progress speed
    ('Q_s',  *Q_S_RANGE),
]


def trial_to_weights(trial) -> dict:
    """Sample parameters from Optuna trial and pack into a weights dict."""
    params = {}
    for name, low, high, log in SEARCH_SPACE:
        params[name] = trial.suggest_float(name, low, high, log=log)

    return {
        'Q_ec':  [params['Q_ecx'], params['Q_ecy'], params['Q_ecz']],
        'Q_el':  [params['Q_elx'], params['Q_ely'], params['Q_elz']],
        'Q_q':   [params['Q_qx'],  params['Q_qy'],  params['Q_qz']],
        'U_mat': [params['U_T'],   params['U_wx'],  params['U_wy'], params['U_wz']],
        'Q_s':    params['Q_s'],
    }


def dict_to_weights(d: dict) -> dict:
    """Re-pack flat best_params dict into weights dict format."""
    return {
        'Q_ec':  [d['Q_ecx'], d['Q_ecy'], d['Q_ecz']],
        'Q_el':  [d['Q_elx'], d['Q_ely'], d['Q_elz']],
        'Q_q':   [d['Q_qx'],  d['Q_qy'],  d['Q_qz']],
        'U_mat': [d['U_T'],   d['U_wx'],  d['U_wy'], d['U_wz']],
        'Q_s':    d['Q_s'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single-velocity sub-objective
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_single_velocity(weights: dict, v_max: float) -> tuple[float, dict]:
    """Run one SiL simulation at vtheta_max=v_max and return (J, info)."""
    try:
        result = run_simulation_mujoco(
            weights=weights, verbose=False, vtheta_max=v_max)
    except Exception as e:
        return 1e6, {'error': str(e)}

    mpcc_cost = result['mean_mpcc_cost']
    compl     = result['path_completed']
    rmse_c    = result['rmse_contorno']
    v_mean    = float(result['mean_vtheta'])
    N_steps   = result['N_steps']
    s_max     = result['s_max']
    crashed   = result.get('crashed', False)

    if not np.isfinite(mpcc_cost) or not np.isfinite(rmse_c):
        return 1e6, result

    # Crash → immediate large penalty, no further evaluation
    if crashed:
        return 1e6, result

    J = mpcc_cost

    # Penalty: path not completed
    if compl < 0.99:
        J += W_INCOMPLETE * (1.0 - compl)

    # Penalty: lap time relative to ideal
    t_s   = 1.0 / FREC
    t_lap = N_steps * t_s
    T_ref = s_max / v_max
    J += W_TIME * (t_lap / T_ref)

    # Penalty: mean velocity below v_max
    vel_ratio = max(0.0, (v_max - v_mean)) / v_max
    J += W_VEL * vel_ratio

    # Penalty: asymmetric XYZ tracking error
    e_cont_3 = result['e_contorno']
    e_lag_3  = result['e_lag']
    e_all_3  = np.sqrt(e_cont_3**2 + e_lag_3**2)
    if e_all_3.shape[1] > 0:
        rmse_xyz = np.sqrt(np.mean(e_all_3**2, axis=1))
        aniso    = rmse_xyz.max() / (rmse_xyz.min() + 1e-8) - 1.0
        J += W_ISOTROPY * aniso

    # Penalty: absolute contouring RMSE
    J += W_CONTOUR * rmse_c

    result['_J'] = J
    return J, result


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-velocity objective  (called by Optuna for each trial)
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial) -> float:
    """Evaluate one gain set across TUNING_VELOCITIES.

    J_multi = mean  J(weights, v)   for v in TUNING_VELOCITIES
    """
    weights = trial_to_weights(trial)

    t0_total     = time.time()
    J_per_vel    = {}
    info_per_vel = {}

    for v_max in TUNING_VELOCITIES:
        J_v, info_v = _evaluate_single_velocity(weights, v_max)
        J_per_vel[v_max]    = J_v
        info_per_vel[v_max] = info_v

    J_multi     = float(np.mean(list(J_per_vel.values())))
    wall_total  = time.time() - t0_total

    parts = []
    for v in TUNING_VELOCITIES:
        info = info_per_vel[v]
        if isinstance(info, dict) and info.get('crashed', False):
            parts.append(f"v={v:.0f}: CRASH")
        elif isinstance(info, dict) and 'path_completed' in info:
            parts.append(
                f"v={v:.0f}: J={J_per_vel[v]:.2f}  "
                f"path={info['path_completed']*100:.0f}%  "
                f"v̄θ={float(info.get('mean_vtheta', 0)):.1f}")
        else:
            parts.append(f"v={v:.0f}: FAIL")

    print(
        f"  [Trial {trial.number:3d}]  J_multi={J_multi:8.3f}  "
        f"{'  |  '.join(parts)}  ({wall_total:.0f}s)",
        flush=True)

    trial.set_user_attr('J_multi',    float(J_multi))
    trial.set_user_attr('wall_time',  wall_total)
    for v in TUNING_VELOCITIES:
        info = info_per_vel[v]
        if isinstance(info, dict) and 'path_completed' in info:
            trial.set_user_attr(f'J_v{v}',      float(J_per_vel[v]))
            trial.set_user_attr(f'path_v{v}',   float(info['path_completed']))
            trial.set_user_attr(f'vtheta_v{v}', float(info.get('mean_vtheta', 0)))
            trial.set_user_attr(f'rmse_c_v{v}', float(info.get('rmse_contorno', 0)))

    return J_multi


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Bilevel gain tuning for MPCC-rates with MuJoCo SiL')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    parser.add_argument('--sampler', type=str, default=DEFAULT_SAMPLER,
                        choices=['tpe', 'cmaes'],
                        help=f'Optuna sampler: tpe or cmaes (default: {DEFAULT_SAMPLER})')
    parser.add_argument('--study-name', type=str,
                        default='mpcc_rate_mujoco_tuning',
                        help='Optuna study name')
    parser.add_argument('--db', type=str, default=None,
                        help='Optuna storage URI (e.g. sqlite:///tuning.db). '
                             'If not set, in-memory.')
    args = parser.parse_args()

    if not HAS_OPTUNA:
        print("ERROR: optuna not installed.  Run:  pip install optuna")
        sys.exit(1)

    print("=" * 70)
    print("  MPCC RATE MuJoCo SiL — BILEVEL GAIN TUNER  (multi-velocity)")
    print("=" * 70)
    print(f"\n  Sampler    : {args.sampler.upper()}")
    print(f"  Trials     : {args.n_trials}")
    print(f"  Velocities : {TUNING_VELOCITIES}")
    print(f"  Study      : {args.study_name}")
    print(f"  Storage    : {args.db or 'in-memory'}\n")

    # ── Warm-up: initialise ROS2 + MuJoCo + solver (runs ONCE) ──────────
    print("[1/3]  Warm-up: ROS2 init + solver compilation ...")
    t0 = time.time()
    _ = run_simulation_mujoco(weights=None, verbose=False)
    print(f"[1/3]  Done.  Baseline simulation took {time.time()-t0:.1f} s\n")

    # ── Create Optuna study ───────────────────────────────────────────────
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

    # ── Run optimisation ──────────────────────────────────────────────────
    print(f"[2/3]  Running {args.n_trials} trials "
          f"(×{len(TUNING_VELOCITIES)} velocities each) ...\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ───────────────────────────────────────────────────────────
    best         = study.best_trial
    best_weights = dict_to_weights(best.params)

    print("\n" + "=" * 70)
    print("  OPTIMISATION COMPLETE  (multi-velocity)")
    print("=" * 70)
    print(f"\n  Best trial  : #{best.number}")
    print(f"  Best J_multi: {best.value:.4f}")
    print(f"  Velocities  : {TUNING_VELOCITIES}")
    for v in TUNING_VELOCITIES:
        J_v = best.user_attrs.get(f'J_v{v}',      '?')
        p_v = best.user_attrs.get(f'path_v{v}',   '?')
        vt  = best.user_attrs.get(f'vtheta_v{v}', '?')
        rc  = best.user_attrs.get(f'rmse_c_v{v}', '?')
        print(f"    v={v:4.1f}: J={J_v}  path={p_v}  v̄θ={vt}  RMSE_c={rc}")

    print("\n  Best weights:")
    for key, val in best_weights.items():
        print(f"    {key:10s} = {val}")

    # ── Save trial history ────────────────────────────────────────────────
    out_dir = _TUNING_DIR
    history_records = []
    best_so_far = float('inf')

    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            J_val       = t.value
            best_so_far = min(best_so_far, J_val)
            record = {
                'trial':         t.number,
                'J_multi':       J_val,
                'best_J_so_far': best_so_far,
                'wall_time':     t.user_attrs.get('wall_time'),
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
            'controller':       'MPCC_rates_mujoco',
            'strategy':         'multi-velocity',
            'tuning_velocities': TUNING_VELOCITIES,
            'n_trials':         len(history_records),
            'sampler':          args.sampler,
            'objective_info': {
                'type': 'MEAN over velocities of '
                        '(mpcc_cost + completion + time + vel + isotropy + contour)',
                'W_INCOMPLETE': W_INCOMPLETE,
                'W_TIME':       W_TIME,
                'W_VEL':        W_VEL,
                'W_ISOTROPY':   W_ISOTROPY,
                'W_CONTOUR':    W_CONTOUR,
                'TUNING_VELOCITIES': TUNING_VELOCITIES,
                'FREC':         FREC,
            },
            'trials': history_records,
        }, fp, indent=2)
    print(f"\n  Trial history saved to {hist_path}")

    # ── Save best weights ─────────────────────────────────────────────────
    out_path = os.path.join(out_dir, 'best_weights.json')
    save_data = {
        'best_J_multi': best.value,
        'best_trial':   best.number,
        'weights': {
            k: (v if isinstance(v, (int, float)) else list(v))
            for k, v in best_weights.items()
        },
        'flat_params': {k: float(v) for k, v in best.params.items()},
        'user_attrs':  {k: float(v) for k, v in best.user_attrs.items()
                        if isinstance(v, (int, float))},
        'objective_info': {
            'type':              'MEAN over velocities of (mpcc_cost + penalties)',
            'strategy':          'multi-velocity',
            'TUNING_VELOCITIES': TUNING_VELOCITIES,
            'W_INCOMPLETE':      W_INCOMPLETE,
            'W_TIME':            W_TIME,
            'W_VEL':             W_VEL,
            'W_ISOTROPY':        W_ISOTROPY,
            'W_CONTOUR':         W_CONTOUR,
            'FREC':              FREC,
        },
    }
    with open(out_path, 'w') as fp:
        json.dump(save_data, fp, indent=2)
    print(f"[3/3]  Best weights saved to {out_path}")

    # ── Print copy-pasteable block ────────────────────────────────────────
    print("\n  ── Copy-paste into experiment_config.py ──")
    print(f"  MPCC_Q_EC       = {best_weights['Q_ec']}")
    print(f"  MPCC_Q_EL       = {best_weights['Q_el']}")
    print(f"  MPCC_Q_Q        = {best_weights['Q_q']}")
    print(f"  MPCC_RATE_U_MAT = {best_weights['U_mat']}")
    print(f"  MPCC_Q_OMEGA    = {best_weights['Q_omega']}")
    print(f"  MPCC_Q_S        = {best_weights['Q_s']}")
    print()


if __name__ == '__main__':
    main()
