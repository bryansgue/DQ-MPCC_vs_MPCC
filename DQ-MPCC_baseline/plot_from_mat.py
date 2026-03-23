#!/usr/bin/env python3
"""
plot_from_mat.py  —  DQ-MPCC: Regenerate all figures from a saved .mat file.

Ejecutar desde dentro de DQ-MPCC_baseline/:
    python3 plot_from_mat.py
    python3 plot_from_mat.py Results_DQ_MPCC_baseline_1.mat
"""

import os
import sys
import numpy as np
import scipy.io as sio

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))           # DQ-MPCC_baseline/
_ROOT  = os.path.dirname(_HERE)                                # workspace root
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# ── .mat file ─────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    mat_path = os.path.abspath(sys.argv[1])
else:
    mat_path = os.path.join(_HERE, 'Results_DQ_MPCC_baseline_1.mat')

if not os.path.isfile(mat_path):
    print(f"ERROR: archivo no encontrado: {mat_path}")
    sys.exit(1)

print(f"[DQ-MPCC plot]  mat = {mat_path}")

# ── Cargar .mat ───────────────────────────────────────────────────────────────
d = sio.loadmat(mat_path)

x_std         = d['states_std']                        # (13, N+1)
T_control     = d['T_control']                         # (5, N)
time_vec      = d['time'].ravel()                      # (N+1,)
ref           = d['ref']                               # (17, N+1)
e_contorno    = d['e_contorno']                        # (3, N)
e_arrastre    = d['e_arrastre']                        # (3, N)
vel_progres   = d['vel_progres']                       # (1, N)
vel_real      = d['vel_real']                          # (1, N)
vel_tangent   = d['vel_tangent']                       # (1, N)
theta_history = d['theta_history']                     # (1, N+1)
s_max         = float(np.asarray(d['s_max']).flat[0])

N_sim  = T_control.shape[1]
t_plot = time_vec[:N_sim + 1]
print(f"  N_sim={N_sim}  s_max={s_max:.2f}  t_end={t_plot[-1]:.2f}s")

# ── Reconstruir parametrización arc-length ────────────────────────────────────
from experiment_config import T_FINAL, FREC, trayectoria
from utils.numpy_utils import build_arc_length_parameterisation, compute_curvature

t_s   = 1.0 / FREC
t_sim = np.arange(0, T_FINAL + t_s, t_s)

# Trayectoria — from experiment_config.py (single source of truth)
xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()

arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, s_max_full = \
    build_arc_length_parameterisation(xd, yd, zd, xd_p, yd_p, zd_p, t_sim)

print(f"  s_max_full={s_max_full:.2f}")
curvature = compute_curvature(position_by_arc_length, s_max, N_samples=500)

# ── Importar funciones de plot ────────────────────────────────────────────────
from utils.graficas import (
    plot_pose,
    plot_control,
    plot_vel_lineal,
    plot_vel_angular,
    plot_velocity_analysis,
    plot_3d_trajectory,
    plot_progress_velocity,
)

# ── Guardar en la misma carpeta que el .mat ───────────────────────────────────
out_dir = os.path.dirname(mat_path)

fig1 = plot_pose(x_std, ref, t_plot)
fig1.savefig(os.path.join(out_dir, '1_pose.png'), dpi=150, bbox_inches='tight')
print('✓ 1_pose.png')

fig2 = plot_control(T_control[:4, :], t_plot[:N_sim])
fig2.savefig(os.path.join(out_dir, '2_control_actions.png'), dpi=150, bbox_inches='tight')
print('✓ 2_control_actions.png')

fig3 = plot_vel_lineal(x_std[3:6, :], t_plot)
fig3.savefig(os.path.join(out_dir, '3_vel_lineal.png'), dpi=150, bbox_inches='tight')
print('✓ 3_vel_lineal.png')

fig4 = plot_vel_angular(x_std[10:13, :], t_plot)
fig4.savefig(os.path.join(out_dir, '4_vel_angular.png'), dpi=150, bbox_inches='tight')
print('✓ 4_vel_angular.png')

fig5 = plot_velocity_analysis(
    vel_progres, vel_real, vel_tangent,
    curvature, theta_history, s_max, t_plot[:N_sim])
fig5.savefig(os.path.join(out_dir, '5_velocity_analysis.png'), dpi=150, bbox_inches='tight')
print('✓ 5_velocity_analysis.png')

fig6 = plot_3d_trajectory(
    x_std, pos_ref, s_max=s_max,
    position_by_arc=position_by_arc_length, N_plot=600)
fig6.savefig(os.path.join(out_dir, '6_trajectory_3d.png'), dpi=150, bbox_inches='tight')
print('✓ 6_trajectory_3d.png')

fig8 = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
fig8.savefig(os.path.join(out_dir, '8_progress_velocity.png'), dpi=150, bbox_inches='tight')
print('✓ 8_progress_velocity.png')

print(f'\nFiguras guardadas en: {out_dir}')
