"""
Backward-compatible wrapper around the canonical runtime-parameter MPCC OCP.

Use `ocp/mpcc_controller.py` as the source of truth.
This file exists only so older tuner imports do not break.
"""

from ocp.mpcc_controller import (
    N_PARAMS,
    DEFAULT_Q_EC,
    DEFAULT_Q_EL,
    DEFAULT_Q_Q,
    DEFAULT_U_MAT,
    DEFAULT_Q_OMEGA,
    DEFAULT_Q_S,
    DEFAULT_VTHETA_MAX,
    T_MAX,
    T_MIN,
    TAUX_MAX,
    TAUY_MAX,
    TAUZ_MAX,
    VTHETA_MIN,
    VTHETA_MAX,
    weights_to_param_vector,
    apply_input_bounds,
    create_mpcc_ocp_description as create_mpcc_ocp_description_tunable,
    build_mpcc_solver as build_mpcc_solver_tunable,
)
