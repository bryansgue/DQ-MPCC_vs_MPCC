"""
Compatibility wrapper for the canonical runtime-parameter DQ-MPCC solver.

The DQ baseline now uses a single OCP for both:
  - production simulation
  - bilevel tuning

All numeric weights are runtime parameters in p ∈ R^18, so the tuner can
compile once and update them with solver.set(stage, "p", p_vec).
"""

from ocp.dq_mpcc_controller import (
    N_PARAMS,
    DEFAULT_VTHETA_MAX,
    weights_to_param_vector,
    apply_input_bounds,
    create_dq_mpcc_ocp_description as create_dq_mpcc_ocp_description_tunable,
    build_dq_mpcc_solver as build_dq_mpcc_solver_tunable,
)
