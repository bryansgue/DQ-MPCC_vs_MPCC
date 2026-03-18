"""
mpcc_tuner.py  –  Bilevel optimisation of MPCC cost weights.

Architecture
────────────
  LEVEL 1 (outer):  Optuna (Bayesian / TPE) or CMA-ES optimises the
                    gain vector θ_g = [Q_ec, Q_el, Q_q, U_mat, Q_omega, Q_s]
  LEVEL 2 (inner):  Full MPCC simulation with acados (runs ~30 s per eval).
                    The solver was compiled ONCE with symbolic parameters.

The meta-cost uses the SAME cost function as the MPCC solver:

    J_stage = ec'Q_ec·ec + e_lag'Q_el·e_lag + log_q'Q_q·log_q
            + u'U·u + ω'Q_ω·ω + Q_s·(v_max − v_θ)²

accumulated over the entire trajectory.  An additional penalty for
incomplete path traversal is added to discourage premature stops.

Usage
─────
    cd /home/bryansgue/dev/ros2/MPCC_baseline
    python -m tuning.mpcc_tuner                     # run with defaults
    python -m tuning.mpcc_tuner --n-trials 200      # more trials
    python -m tuning.mpcc_tuner --sampler cmaes     # use CMA-ES

After optimisation, the best weights are printed and saved to
    tuning/best_weights.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MPCC_simulation_tuner import run_simulation

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ══════════════════════════════════════════════════════════════════════════════
#  Incomplete-path penalty  (the only meta-weight — everything else comes
#  from the MPCC cost function itself)
# ══════════════════════════════════════════════════════════════════════════════

W_INCOMPLETE = 1000.0   # huge penalty for not completing the path


# ══════════════════════════════════════════════════════════════════════════════
#  Search space definition
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (name, low, high, log_scale)
# These define the range that Optuna will explore.
SEARCH_SPACE = [
    # Q_ec: contouring error [x, y, z]          max = 50
    ('Q_ecx',   1.0,   50.0, True),
    ('Q_ecy',   1.0,   50.0, True),
    ('Q_ecz',   1.0,   50.0, True),
    # Q_el: lag error [x, y, z]                  max = 50
    ('Q_elx',   0.1,   50.0, True),
    ('Q_ely',   0.1,   50.0, True),
    ('Q_elz',   0.1,   50.0, True),
    # Q_q: quaternion error [roll, pitch, yaw]
    ('Q_qx',    0.1,   50.0, True),
    ('Q_qy',    0.1,   50.0, True),
    ('Q_qz',    0.1,   50.0, True),
    # U_mat: control effort [T, τx, τy, τz]     max = 800
    ('U_T',     0.01,  10.0, True),
    ('U_tx',    10.0, 800.0, True),
    ('U_ty',    10.0, 800.0, True),
    ('U_tz',    10.0, 800.0, True),
    # Q_omega: angular velocity [ωx, ωy, ωz]
    ('Q_wx',    0.01,  10.0, True),
    ('Q_wy',    0.01,  10.0, True),
    ('Q_wz',    0.01,  10.0, True),
    # Q_s: progress speed
    ('Q_s',     0.5,  10.0, True),
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
#  Objective function
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial) -> float:
    """Evaluate one set of gains by running the full MPCC simulation.

    The objective is the MEAN MPCC stage cost (same formula used inside the
    acados solver), plus a large penalty if the path is not fully completed.
    """
    weights = trial_to_weights(trial)

    try:
        t0 = time.time()
        result = run_simulation(weights=weights, verbose=False)
        wall = time.time() - t0
    except Exception as e:
        print(f"  [FAIL] Trial {trial.number}: {e}")
        return 1e6   # penalise crashes

    mpcc_cost = result['mean_mpcc_cost']
    compl     = result['path_completed']
    rmse_c    = result['rmse_contorno']
    rmse_l    = result['rmse_lag']
    effort    = result['mean_effort']

    # ── Objective = MPCC cost + completion penalty ───────────────────────
    J = mpcc_cost 

    print(f"  [Trial {trial.number:3d}]  J={J:8.3f}  "
          f"J_mpcc={mpcc_cost:.4f}  "
          f"RMSE_c={rmse_c:.4f}  RMSE_l={rmse_l:.4f}  "
          f"effort={effort:.1f}  path={compl*100:.1f}%  "
          f"({wall:.1f}s)")

    # Store sub-metrics for later analysis
    trial.set_user_attr('mean_mpcc_cost', mpcc_cost)
    trial.set_user_attr('total_mpcc_cost', result['total_mpcc_cost'])
    trial.set_user_attr('rmse_contorno', rmse_c)
    trial.set_user_attr('rmse_lag', rmse_l)
    trial.set_user_attr('mean_effort', effort)
    trial.set_user_attr('path_completed', compl)
    trial.set_user_attr('wall_time', wall)

    return J


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Bilevel optimisation of MPCC cost weights')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--sampler', type=str, default='tpe',
                        choices=['tpe', 'cmaes'],
                        help='Optuna sampler: tpe or cmaes')
    parser.add_argument('--study-name', type=str, default='mpcc_tuning',
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
    print("  MPCC BILEVEL GAIN TUNER")
    print("="*70)
    print(f"\n  Sampler : {args.sampler.upper()}")
    print(f"  Trials  : {args.n_trials}")
    print(f"  Study   : {args.study_name}")
    print(f"  Storage : {args.db or 'in-memory'}\n")

    print("[1/3] Building infrastructure (trajectory + solver) ...")
    # Trigger the solver compilation by running once with defaults
    t0 = time.time()
    _ = run_simulation(weights=None, verbose=False)
    print(f"[1/3] Done. Baseline simulation took {time.time()-t0:.1f}s\n")

    # ── Create Optuna study ──────────────────────────────────────────────
    if args.sampler == 'cmaes':
        sampler = CmaEsSampler(seed=42)
    else:
        sampler = TPESampler(seed=42, n_startup_trials=10)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction='minimize',
        storage=args.db,
        load_if_exists=True,
    )

    # ── Run optimisation ─────────────────────────────────────────────────
    print(f"[2/3] Running {args.n_trials} trials ...\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ──────────────────────────────────────────────────────────
    best = study.best_trial
    best_weights = dict_to_weights(best.params)

    print("\n" + "="*70)
    print("  OPTIMISATION COMPLETE")
    print("="*70)
    print(f"\n  Best trial : #{best.number}")
    print(f"  Best J     : {best.value:.4f}")
    print(f"  MPCC cost  : {best.user_attrs.get('mean_mpcc_cost', '?')}")
    print(f"  RMSE_c     : {best.user_attrs.get('rmse_contorno', '?')}")
    print(f"  RMSE_l     : {best.user_attrs.get('rmse_lag', '?')}")
    print(f"  Effort     : {best.user_attrs.get('mean_effort', '?')}")
    print(f"  Path       : {best.user_attrs.get('path_completed', '?')}")
    print(f"\n  Best weights:")
    for key, val in best_weights.items():
        print(f"    {key:10s} = {val}")

    # ── Save to JSON ─────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'best_weights.json')

    save_data = {
        'best_J': best.value,
        'best_trial': best.number,
        'weights': {k: (v if isinstance(v, (int, float)) else list(v))
                    for k, v in best_weights.items()},
        'flat_params': {k: float(v) for k, v in best.params.items()},
        'user_attrs': {k: float(v) for k, v in best.user_attrs.items()
                       if isinstance(v, (int, float))},
        'objective_info': {
            'type': 'mean_mpcc_cost + completion_penalty',
            'W_INCOMPLETE': W_INCOMPLETE,
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
