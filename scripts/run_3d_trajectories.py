#!/usr/bin/env python3
"""
run_3d_trajectories.py — Capture 3D trajectories for a few representative
velocities (1 nominal run each) and save to .mat for isometric plotting.

Uses the Experiment 2 infrastructure (velocity sweep) but:
  • Only 1 run per velocity (nominal IC, no perturbation)
  • Saves full XYZ trajectory arrays
  • Selects a subset of representative velocities

Output: results/experiment2/trajectory_3d_data.mat
"""

import os, sys, time, math
import numpy as np
from scipy.io import savemat

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)
_DQ_DIR     = os.path.join(_ROOT, "DQ-MPCC_baseline")
_MPCC_DIR   = os.path.join(_ROOT, "MPCC_baseline")
_OUT_DIR    = os.path.join(_ROOT, "results", "experiment2")
os.makedirs(_OUT_DIR, exist_ok=True)

sys.path.insert(0, _ROOT)
from config.experiment_config import (
    P0, Q0, VALUE, trayectoria,
    S_MAX_MANUAL, FREC, T_PREDICTION, N_WAYPOINTS,
)

# ── Parameters ───────────────────────────────────────────────────────────────
VELOCITIES_3D = [4, 6, 8, 10, 12, 14, 16, 18, 20]          # representative subset
S_MAX         = S_MAX_MANUAL if S_MAX_MANUAL is not None else 80
T_FINAL       = 30                       # same as sweep
COMPLETION_RATIO = 0.95
POS_DIVERGENCE_THRESHOLD = 30.0


# ═════════════════════════════════════════════════════════════════════════════
#  Quaternion helpers
# ═════════════════════════════════════════════════════════════════════════════

def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mult(p, q):
    return np.array([
        p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
        p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
        p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
        p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0],
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  Trajectory builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_trajectory():
    t_s = 1.0 / FREC
    t = np.arange(0, T_FINAL + t_s, t_s)
    t_finer = np.linspace(0, T_FINAL, len(t))
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
    return t, t_finer, xd, yd, zd, xd_p, yd_p, zd_p


# ═════════════════════════════════════════════════════════════════════════════
#  MPCC Baseline — single run returning XYZ trajectory
# ═════════════════════════════════════════════════════════════════════════════

def _run_mpcc_traj(p0, q0, solver, f_sys,
                   position_by_arc, s_max,
                   rk4_fn, N_sim, t_s, N_prediction):
    """Run MPCC baseline, return xyz (3, K) trajectory."""
    nx, nu = 14, 5
    x = np.zeros((nx, N_sim + 1))
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    x[:, 0] = [p0[0], p0[1], p0[2],
               0, 0, 0,
               q0n[0], q0n[1], q0n[2], q0n[3],
               0, 0, 0, 0.0]

    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    k_final = 0
    for k in range(N_sim):
        if x[13, k] >= s_max - 0.01:
            k_final = k
            break
        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])
        solver.solve()
        u_k = solver.get(0, "u")
        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[13, k+1] = np.clip(x[13, k+1], 0.0, s_max)
        e_pos = np.linalg.norm(position_by_arc(x[13, k]) - x[0:3, k])
        if e_pos > POS_DIVERGENCE_THRESHOLD:
            break
        k_final = k

    return x[0:3, :k_final+1].copy()


# ═════════════════════════════════════════════════════════════════════════════
#  DQ-MPCC — single run returning XYZ trajectory
# ═════════════════════════════════════════════════════════════════════════════

def _run_dq_traj(p0, q0, solver, f_sys,
                 position_by_arc, s_max,
                 dq_from_pose_fn, dq_get_pos_fn,
                 dq_normalize_fn, dq_hemi_fn, rk4_fn,
                 N_sim, t_s, N_prediction):
    """Run DQ-MPCC, return xyz (3, K) trajectory."""
    nx, nu = 15, 5
    q0n = q0 / (np.linalg.norm(q0) + 1e-12)
    dq0 = dq_from_pose_fn(q0n, p0)

    x = np.zeros((nx, N_sim + 1))
    x[0:8, 0] = dq0
    x[14, 0] = 0.0

    for st in range(N_prediction + 1):
        solver.set(st, "x", x[:, 0])
    for st in range(N_prediction):
        solver.set(st, "u", np.zeros(nu))

    dq_prev = x[0:8, 0].copy()
    pos_log = [dq_get_pos_fn(x[0:8, 0]).copy()]
    k_final = 0

    for k in range(N_sim):
        if x[14, k] >= s_max - 0.01:
            k_final = k
            break
        solver.set(0, "lbx", x[:, k])
        solver.set(0, "ubx", x[:, k])
        solver.solve()
        u_k = solver.get(0, "u")
        x[:, k+1] = rk4_fn(x[:, k], u_k, t_s, f_sys)
        x[0:8, k+1] = dq_normalize_fn(x[0:8, k+1])
        x[0:8, k+1] = dq_hemi_fn(x[0:8, k+1], dq_prev)
        dq_prev = x[0:8, k+1].copy()
        x[14, k+1] = np.clip(x[14, k+1], 0.0, s_max)

        pos_k = dq_get_pos_fn(x[0:8, k+1])
        pos_log.append(pos_k.copy())

        e_pos = np.linalg.norm(position_by_arc(x[14, k]) - pos_k)
        if e_pos > POS_DIVERGENCE_THRESHOLD:
            break
        k_final = k

    return np.array(pos_log).T


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t_s = 1.0 / FREC
    N_prediction = int(round(T_PREDICTION / t_s))
    s_max = S_MAX

    print(f"\n{'='*70}")
    print(f"  3D Trajectory Capture — {len(VELOCITIES_3D)} velocities")
    print(f"  Velocities: {VELOCITIES_3D} m/s")
    print(f"  s_max = {s_max} m")
    print(f"{'='*70}")

    # ── Build trajectory ─────────────────────────────────────────────────
    t_vec, t_finer, xd, yd, zd, xd_p, yd_p, zd_p = _build_trajectory()
    N_sim_max = t_vec.shape[0] - N_prediction

    # ── Import helpers ───────────────────────────────────────────────────
    _CONFLICTING = ('utils', 'models', 'ocp')

    def _save_and_clear():
        for key in list(sys.modules):
            for prefix in _CONFLICTING:
                if key == prefix or key.startswith(prefix + '.'):
                    del sys.modules[key]

    # MPCC imports
    _save_and_clear()
    sys.path.insert(0, _MPCC_DIR)

    from utils.numpy_utils import (
        euler_to_quaternion as mpcc_euler2quat,
        build_arc_length_parameterisation as mpcc_build_arc,
        build_waypoints as mpcc_build_wp,
        rk4_step_mpcc as mpcc_rk4,
    )
    from utils.casadi_utils import (
        create_position_interpolator_casadi as mpcc_pos_interp,
        create_tangent_interpolator_casadi  as mpcc_tang_interp,
        create_quat_interpolator_casadi     as mpcc_quat_interp,
    )
    import ocp.mpcc_controller as mpcc_ocp_module
    build_mpcc_solver = mpcc_ocp_module.build_mpcc_solver

    sys.path.remove(_MPCC_DIR)
    _save_and_clear()

    # DQ imports
    sys.path.insert(0, _DQ_DIR)

    from utils.numpy_utils import (
        euler_to_quaternion as dq_euler2quat,
        build_arc_length_parameterisation as dq_build_arc,
        build_waypoints as dq_build_wp,
    )
    from utils.dq_numpy_utils import (
        dq_from_pose_numpy, dq_get_position_numpy, dq_get_quaternion_numpy,
        dq_normalize, dq_hemisphere_correction,
        rk4_step_dq_mpcc as dq_rk4,
    )
    from utils.casadi_utils import (
        create_position_interpolator_casadi as dq_pos_interp,
        create_tangent_interpolator_casadi  as dq_tang_interp,
        create_quat_interpolator_casadi     as dq_quat_interp,
    )
    import ocp.dq_mpcc_controller as dq_ocp_module
    build_dq_mpcc_solver = dq_ocp_module.build_dq_mpcc_solver

    sys.path.remove(_DQ_DIR)

    # ── Arc-length parameterisation ──────────────────────────────────────
    arc_lengths, pos_ref, position_by_arc, tangent_by_arc, s_max_full = \
        mpcc_build_arc(xd, yd, zd, xd_p, yd_p, zd_p, t_finer)

    s_wp, pos_wp, tang_wp, quat_wp = mpcc_build_wp(
        s_max, N_WAYPOINTS, position_by_arc, tangent_by_arc,
        euler_to_quat_fn=mpcc_euler2quat,
    )
    gamma_pos  = mpcc_pos_interp(s_wp, pos_wp)
    gamma_vel  = mpcc_tang_interp(s_wp, tang_wp)
    gamma_quat = mpcc_quat_interp(s_wp, quat_wp)

    dq_gamma_pos  = dq_pos_interp(s_wp, pos_wp)
    dq_gamma_vel  = dq_tang_interp(s_wp, tang_wp)
    dq_gamma_quat = dq_quat_interp(s_wp, quat_wp)

    # ── Build solvers once ───────────────────────────────────────────────
    vtheta_max_build = max(VELOCITIES_3D)
    p0_nom = P0.copy()
    q0_nom = Q0 / (np.linalg.norm(Q0) + 1e-12)

    print(f"\n  Building solvers with v_theta_max = {vtheta_max_build} m/s ...")

    mpcc_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max_build
    x0_mpcc = np.zeros(14)
    x0_mpcc[0:3] = p0_nom
    x0_mpcc[6:10] = q0_nom

    print(f"  [MPCC] Generating + compiling solver ...")
    os.chdir(_MPCC_DIR)
    mpcc_solver, _, mpcc_model, mpcc_f = build_mpcc_solver(
        x0_mpcc, N_prediction, T_PREDICTION, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=False,
    )
    print(f"  [MPCC] Solver ready.")

    dq_ocp_module.DEFAULT_VTHETA_MAX = vtheta_max_build
    x0_dq = np.zeros(15)
    x0_dq[0:8] = dq_from_pose_numpy(q0_nom, p0_nom)

    for key in list(sys.modules):
        if 'acados_ocp_solver_pyx' in key:
            del sys.modules[key]

    print(f"  [DQ]   Generating + compiling solver ...")
    os.chdir(_DQ_DIR)
    dq_solver, _, dq_model, dq_f = build_dq_mpcc_solver(
        x0_dq, N_prediction, T_PREDICTION, s_max=s_max,
        gamma_pos=dq_gamma_pos, gamma_vel=dq_gamma_vel,
        gamma_quat=dq_gamma_quat,
        use_cython=False,
    )
    print(f"  [DQ]   Solver ready.")
    os.chdir(_ROOT)

    # ── Online v_theta updaters ──────────────────────────────────────────
    mpcc_ubu = np.array([
        mpcc_ocp_module.DEFAULT_T_MAX,
        mpcc_ocp_module.DEFAULT_TAUX_MAX,
        mpcc_ocp_module.DEFAULT_TAUY_MAX,
        mpcc_ocp_module.DEFAULT_TAUZ_MAX,
        0.0,
    ])
    dq_ubu = np.array([
        dq_ocp_module.DEFAULT_T_MAX,
        dq_ocp_module.DEFAULT_TAUX_MAX,
        dq_ocp_module.DEFAULT_TAUY_MAX,
        dq_ocp_module.DEFAULT_TAUZ_MAX,
        0.0,
    ])

    def _set_v_mpcc(v):
        ubu = mpcc_ubu.copy(); ubu[4] = v
        for st in range(N_prediction + 1):
            mpcc_solver.set(st, "p", np.array([v]))
        for st in range(N_prediction):
            mpcc_solver.constraints_set(st, "ubu", ubu)

    def _set_v_dq(v):
        ubu = dq_ubu.copy(); ubu[4] = v
        for st in range(N_prediction + 1):
            dq_solver.set(st, "p", np.array([v]))
        for st in range(N_prediction):
            dq_solver.constraints_set(st, "ubu", ubu)

    # ── Run for each velocity ────────────────────────────────────────────
    trajs = {}  # {(ctrl, v): (3, K) array}
    t_start = time.time()

    for vi, v in enumerate(VELOCITIES_3D):
        print(f"\n  [{vi+1}/{len(VELOCITIES_3D)}] v = {v} m/s ...")

        _set_v_mpcc(v)
        _set_v_dq(v)

        # Baseline
        xyz_base = _run_mpcc_traj(
            p0_nom, q0_nom, mpcc_solver, mpcc_f,
            position_by_arc, s_max, mpcc_rk4,
            N_sim_max, t_s, N_prediction)
        trajs[('base', v)] = xyz_base
        print(f"    Baseline: {xyz_base.shape[1]} steps")

        # DQ-MPCC
        xyz_dq = _run_dq_traj(
            p0_nom, q0_nom, dq_solver, dq_f,
            position_by_arc, s_max,
            dq_from_pose_numpy, dq_get_position_numpy,
            dq_normalize, dq_hemisphere_correction, dq_rk4,
            N_sim_max, t_s, N_prediction)
        trajs[('dq', v)] = xyz_dq
        print(f"    DQ-MPCC:  {xyz_dq.shape[1]} steps")

    # ── Reference path ───────────────────────────────────────────────────
    ref_x = np.array([xd(ti) for ti in t_finer])
    ref_y = np.array([yd(ti) for ti in t_finer])
    ref_z = np.array([zd(ti) for ti in t_finer])

    # ── Save ─────────────────────────────────────────────────────────────
    mat_dict = {
        'velocities': np.array(VELOCITIES_3D),
        's_max': s_max,
        'ref_x': ref_x,
        'ref_y': ref_y,
        'ref_z': ref_z,
    }
    for (ctrl, v), xyz in trajs.items():
        vstr = f"v{v}"
        mat_dict[f'{ctrl}_{vstr}_x'] = xyz[0]
        mat_dict[f'{ctrl}_{vstr}_y'] = xyz[1]
        mat_dict[f'{ctrl}_{vstr}_z'] = xyz[2]

    out_path = os.path.join(_OUT_DIR, 'trajectory_3d_data.mat')
    savemat(out_path, mat_dict, do_compression=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE — {len(VELOCITIES_3D)*2} runs in {elapsed:.1f} s")
    print(f"  Saved to {out_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
