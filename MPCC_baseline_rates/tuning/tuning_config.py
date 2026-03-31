"""tuning_config.py — Shared configuration for MPCC-rates MuJoCo SiL bilevel tuner.

Round 4 — Single-velocity aggressive tuning at 15 m/s.

Philosophy: maximise arc-length speed first, accept higher tracking error.
The drone must operate near its physical limit (acrobatic regime):
  - Control effort weights U_mat kept very LOW → allow full authority
  - Q_s range pushed HIGH → solver strongly incentivised to advance fast
  - W_VEL / W_TIME very HIGH → Optuna harshly penalises slow trials
  - Single velocity = 15 m/s → no averaging over slow/fast, pure focus
"""

# ── Simulation timing (must match experiment_config) ─────────────────────────
FREC         = 100      # [Hz]  control frequency
T_PREDICTION = 0.3      # [s]   MPC horizon duration

# ── PD hold convergence settings ─────────────────────────────────────────────
PD_SETTLE_DIST    = 0.30   # [m]
PD_SETTLE_TIMEOUT = 12.0   # [s]

# ── Tuner path length ─────────────────────────────────────────────────────────
# At 15 m/s, 60m ≈ 4s of flight → ~3 full figure-8 loops.
# Use the full production path so we see sustained high-speed behaviour,
# not just a 2-second sprint.
S_MAX_TUNER = 60.0      # [m]  full production path

# ── Maximum simulation duration per trial [s] ────────────────────────────────
# 60m @ 15 m/s = 4s; bad gains may hover up to 60s before STUCK fires.
T_FINAL_TUNER = 60      # [s]

# ── Tuning velocities ─────────────────────────────────────────────────────────
# Single velocity — no averaging, pure focus on the acrobatic regime.
# With 1 velocity each trial is ~3x faster than [5,12,20] → more trials.
TUNING_VELOCITIES = [15.0]   # [m/s]  target: ≥15 m/s mean v_θ

# ── Objective function penalty weights ───────────────────────────────────────
# Aggressive speed-first objective.
# W_VEL=30 and W_TIME=20 mean a drone flying at 5 m/s (33% of target)
# receives J += 30*(1-0.33) + 20*(t_lap/T_ref) — dominated by velocity deficit.
W_INCOMPLETE = 150.0    # path completion (reduced: we accept partial if fast)
W_TIME       = 20.0     # lap time  — strong penalty for slow laps
W_VEL        = 30.0     # v_θ deficit — dominant penalty: be fast or be penalised
W_ISOTROPY   = 0.5      # XYZ asymmetry (low: not the priority)
W_CONTOUR    = 5.0      # RMSE contouring (low: accept more error at high speed)

# ── Search space bounds  (low, high, log_scale) ───────────────────────────────
# Acrobatic profile:
#   Q_ec / Q_el HIGH  → strong tracking incentive even at high speed
#   U_mat       LOW   → allow full control authority (acrobatic manoeuvres)
#   Q_s         HIGH  → solver strongly rewards fast path progress
#   Q_q         LOW   → attitude tracking secondary to speed
#
# Based on R3 best weights + direction needed for speed:
#   Q_ec=[10, 9, 47]  Q_el=[36, 169, 140]  Q_q=[0.35, 13.4, 0.12]
#   U_mat=[0.17, 0.04, 0.06, 0.09]  Q_s=0.3
Q_EC_RANGE    = (10.0,  2000.0, True)   # high — strong contouring at speed
Q_EL_RANGE    = (5.0,   2000.0, True)   # high — strong lag penalty at speed
Q_ROT_RANGE   = (0.01,    20.0, True)   # allow very small (speed > attitude)
U_T_RANGE     = (0.001,    0.5, True)   # very low → full thrust authority
U_W_RANGE     = (0.001,    0.5, True)   # very low → full rate authority
Q_OMEGA_RANGE = (0.001,    5.0, True)   # (reference only, not used in tuner)
Q_S_RANGE     = (0.5,     30.0, True)   # HIGH — must strongly incentivise speed
                                         # R3 found 0.3; bump hard to drive v_θ→15

# ── Crash detection thresholds ───────────────────────────────────────────────
# At 15 m/s acrobatic flight, extreme banking and altitude variation are normal.
CRASH_MIN_Z        = 0.01  # [m]    true floor contact only
CRASH_MAX_TILT_COS = -0.4  # []     ~114° — allow extreme banking at speed
CRASH_STILL_VEL    = 0.03  # [m/s]
CRASH_STILL_STEPS  = 300   # [steps] = 3s at 100 Hz

# ── Optuna settings ───────────────────────────────────────────────────────────
# Single velocity = 3x faster per trial → can afford more trials.
DEFAULT_N_TRIALS    = 150          # more trials: single velocity is fast
DEFAULT_SAMPLER     = 'tpe'
N_STARTUP_TRIALS    = 20
OPTUNA_SEED         = 42
