# Codex Agent Map

This repository is organised so Codex can work with the same task split as Claude, but without handoffs.

## Shared project state

- `config/experiment_config.py` is the single source of truth for trajectory, timing, limits, masses, and active tuned JSONs.
- Root `scripts/` wrappers were intentionally removed. Use the canonical scripts inside `results/.../scripts/`.
- The project currently compares:
  - `DQ-MPCC_baseline/` with active set `final_refined_oriented_weights.json`
  - `MPCC_baseline/` with active set `final_refined_relaxed_weights.json`
- Both controllers now use the same shared quaternion attitude reference built from tangent, curvature, nominal speed, and tilt saturation.
- Both controllers use second-order progress. Current dimensions are:
  - `MPCC nx=15`
  - `DQ nx=16`

## Codex role split

Use this map mentally when working in the repo:

- `math-dq`
  - DQ mathematical formulation, SE(3) log cost, dual-quaternion interpretation.
- `math-mpcc`
  - baseline MPCC mathematical formulation, Euclidean/quaternion interpretation.
- `ocp-casadi-acados`
  - central implementation layer for both controllers, including CasADi, Acados, runtime params, bounds, and baseline scripts.
- `mil-tuning`
  - local tuning, tuned JSON selection, tuning summary `.mat`, tuning LaTeX outputs.
- `experiment1`
  - one-run comparison and its outputs.
- `experiment2`
  - velocity sweep, 3D trajectory sweep, Pareto figures, sweep report.
- `experiment3`
  - Monte Carlo robustness, plots, Monte Carlo report.
- `paper-latex`
  - `paper.tex` plus consistency between implemented methodology and generated reports.

## Canonical entrypoints

- Tuning summary:
  - `python3 results/tuning/scripts/export_tuning_summary_mat.py`
  - `python3 results/tuning/reports/generate_local_refine_combined.py`
- Experiment 1:
  - `python3 results/experiment1/scripts/run_experiment1.py`
- Experiment 2:
  - `python3 results/experiment2/scripts/run_experiment2.py`
  - `python3 results/experiment2/scripts/run_3d_trajectories.py`
  - `python3 latex/generate_sweep.py`
- Experiment 3:
  - `python3 results/experiment3/scripts/run_experiment3.py`
  - `python3 latex/generate_montecarlo.py`

## Audit checklist

Before trusting any result:

1. Confirm the active gain JSON paths in `config/experiment_config.py`.
2. Confirm the saved gain labels in the relevant `.mat` via `config/tuning_registry.py`.
3. Confirm the current trajectory/timing regime:
   - `T_FINAL`
   - `TRAJECTORY_T_FINAL`
   - `S_MAX_MANUAL`
4. For Experiment 2, confirm:
   - the velocity list in `config/sweep_config.py`
   - `N_RUNS`
   - that `RMSE_ori` is interpreted from the saved `.mat`, not guessed from the plot
5. For paper/report edits, always read the generator and the `.mat` first.

## Non-negotiable rules

- Do not recreate root wrappers.
- Do not rely on deleted `verified` or `llm` report variants unless they are recreated intentionally.
- Do not assume DQ is globally better than MPCC; interpret each experiment from the saved data.
- Keep mathematical reasoning separate from implementation reasoning.
- Keep controller-level reasoning separate from experiment-level reasoning.
