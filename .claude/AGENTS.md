# Claude Agent Index

This project is organised around a small, explicit set of Claude agents.

## Current stable project state

- The repository compares `DQ-MPCC_baseline/` against `MPCC_baseline/`.
- Both controllers now use the same shared attitude-reference construction based on path tangent, curvature, nominal speed, and a tilt cap.
- The canonical comparative scripts are no longer in the root `scripts/` directory. Use the scripts inside each result package:
  - `results/experiment1/scripts/`
  - `results/experiment2/scripts/`
  - `results/experiment3/scripts/`
  - `results/tuning/scripts/`
- Experiment 2 currently uses:
  - DQ gain file: `DQ-MPCC_baseline/tuning/final_refined_oriented_weights.json`
  - MPCC gain file: `MPCC_baseline/tuning/final_refined_relaxed_weights.json`
- The active gain labels are resolved through `config/tuning_registry.py`.

## Agent map

Three agents are non-negotiable and form the core technical split:

- `math-dq`: mathematical formulation of DQ-MPCC.
- `math-mpcc`: mathematical formulation of the baseline MPCC.
- `ocp-casadi-acados`: implementation of both controllers at the OCP / CasADi / Acados layer.

These three are intended to remain reusable across projects. In this repository they use local files as entry points, but they should be understood as domain specialists first, not file-bound helpers.

The workflow agents are:

- `mil-tuning`: local tuning, tuned JSONs, tuning summary export, and tuning reports.
- `experiment1`: one-run comparison and its reportables.
- `experiment2`: velocity sweep, 3D trajectories, sweep analysis, and associated plots.
- `experiment3`: Monte Carlo experiment and its reportables.
- `paper-latex`: LaTeX generation, report assembly, and paper-facing text consistency.

## Working rules

- Do not use handoff files.
- Do not assume root wrappers exist; they were removed on purpose.
- Always read the current source files before quoting dimensions, gains, or active labels.
- Keep experiment-specific reasoning inside the corresponding experiment agent.
- Keep mathematical reasoning in `math-dq` / `math-mpcc`.
- Keep implementation reasoning in `ocp-casadi-acados`.
