"""
utils/graficas.py  –  Reusable plotting functions for NMPC and MPCC baselines.

All plot functions return a matplotlib Figure object.
Caller is responsible for fig.savefig(...) and plt.show().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_3D = True
except ImportError:
    _HAS_3D = False

plt.rc('text', usetex=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Base figure helpers
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PARAMS = {
    'backend': 'ps',
    'axes.labelsize': 20,
    'legend.fontsize': 16,
    'legend.handlelength': 2.5,
    'legend.borderaxespad': 0,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'serif',
    'font.size': 20,
    'ps.usedistiller': 'xpdf',
    'text.usetex': False,
}


def _base_figsize():
    """Standard figure size derived from LaTeX text width."""
    pts_per_inch = 72.27
    text_width_pts = 300.0
    inverse_latex_scale = 2
    csize = inverse_latex_scale * (3.0 / 3.0) * (text_width_pts / pts_per_inch)
    return (1.0 * csize, 0.7 * csize)


def _apply_rc():
    plt.rc(_DEFAULT_PARAMS)


def _new_figure(nrows=1, ncols=1, figsize=None, sharex=False):
    """Create a clean figure with subplots and standard styling."""
    _apply_rc()
    plt.clf()
    if figsize is None:
        figsize = _base_figsize()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.25, wspace=0.02)
    plt.ioff()
    return fig, axes


def _legend(ax, **kwargs):
    defaults = dict(loc="best", frameon=True, fancybox=True, shadow=False,
                    ncol=2, borderpad=0.5, labelspacing=0.5,
                    handlelength=3, handletextpad=0.1,
                    borderaxespad=0.3, columnspacing=2)
    defaults.update(kwargs)
    ax.legend(**defaults)


def _grid(ax):
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  Standard colors
# ══════════════════════════════════════════════════════════════════════════════

C_RED   = '#BB5651'
C_GREEN = '#69BB51'
C_BLUE  = '#5189BB'
C_GOLD  = '#FFD700'
C_GREY  = '#333333'
C_PURPLE = '#9B59B6'
C_ORANGE = '#E67E22'


# ══════════════════════════════════════════════════════════════════════════════
#  Existing plot functions (cleaned-up)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pose(x, xref, t):
    """Position states (x, y, z) vs desired."""
    fig, ax = _new_figure()
    colors = [C_RED, C_GREEN, C_BLUE]
    labels = [r'$x$', r'$y$', r'$z$']

    for i in range(3):
        ax.plot(t[:x.shape[1]], x[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
        ax.plot(t[:x.shape[1]], xref[i, :x.shape[1]],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$_d$')

    ax.set_ylabel(r"Position [m]")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax)
    _grid(ax)
    return fig


def plot_control(u, t):
    """Control inputs (F, τx, τy, τz) as 4 subplots."""
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    colors = [C_RED, C_GREEN, C_BLUE, C_GOLD]
    labels = [r'$F$ [N]', r'$\tau_x$ [Nm]', r'$\tau_y$ [Nm]', r'$\tau_z$ [Nm]']

    for i in range(4):
        axs[i].plot(t[:u.shape[1]], u[i, :], color=colors[i], lw=2, ls="-")
        axs[i].set_ylabel(labels[i])
        _grid(axs[i])

    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    return fig


def plot_error(error, t):
    """Tracking error (x, y, z)."""
    fig, ax = _new_figure()
    colors = [C_RED, C_GREEN, C_BLUE]
    labels = [r'$e_x$', r'$e_y$', r'$e_z$']

    for i in range(3):
        ax.plot(t[:error.shape[1]], error[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"Error [m]")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax)
    _grid(ax)
    return fig


def plot_vel_lineal(v, t):
    """Linear velocities (vx, vy, vz)."""
    fig, ax = _new_figure()
    colors = [C_RED, C_GREEN, C_BLUE]
    labels = [r'$v_x$', r'$v_y$', r'$v_z$']

    for i in range(3):
        ax.plot(t[:v.shape[1]], v[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"Velocity [m/s]")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax)
    _grid(ax)
    return fig


def plot_vel_angular(w, t):
    """Angular velocities (ωx, ωy, ωz)."""
    fig, ax = _new_figure()
    colors = [C_RED, C_GREEN, C_BLUE]
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']

    for i in range(3):
        ax.plot(t[:w.shape[1]], w[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"Angular velocity [rad/s]")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax)
    _grid(ax)
    return fig


def plot_CBF(value, t):
    """Generic scalar safety / barrier function value."""
    fig, ax = _new_figure()
    ax.plot(t[:value.shape[1]], value[0, :],
            color=C_RED, lw=2, ls="-", label=r'$h(\mathbf{x})$')
    ax.axhline(0, color=C_GREY, lw=1.2, ls='--')
    ax.set_ylabel(r"CBF value")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax, ncol=1)
    _grid(ax)
    return fig


def plot_timing(t_solver, t_loop, t_sample, t):
    """Solver / loop timing with histogram (2 subplots)."""
    n = t_solver.shape[1]
    t_axis = t[:n]

    s_ms  = t_solver[0, :] * 1e3
    l_ms  = t_loop[0, :]   * 1e3
    ts_ms = t_sample[0, :] * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # ── Time series ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t_axis, l_ms,  color=C_BLUE, lw=1.5, ls='-',  label='Loop total')
    ax1.plot(t_axis, s_ms,  color=C_RED,  lw=1.5, ls='-',  label='Solver MPC')
    ax1.plot(t_axis, ts_ms, color=C_GREY, lw=1.2, ls='--', label=r'$t_s$ nominal')
    ax1.fill_between(t_axis, s_ms, alpha=0.15, color=C_RED)
    ax1.set_ylabel('Time [ms]')
    ax1.set_title('Computation time per iteration')
    ax1.legend(loc='upper right', frameon=True, ncol=3)
    _grid(ax1)

    # ── Histogram ────────────────────────────────────────────────────────
    ax2 = axes[1]
    bins = max(30, n // 50)
    ax2.hist(s_ms, bins=bins, color=C_RED,  alpha=0.65, edgecolor='white',
             linewidth=0.4, label='Solver')
    ax2.hist(l_ms, bins=bins, color=C_BLUE, alpha=0.55, edgecolor='white',
             linewidth=0.4, label='Loop')
    ax2.axvline(np.mean(s_ms), color=C_RED,  lw=2, ls='--',
                label=f'Mean solver {np.mean(s_ms):.2f} ms')
    ax2.axvline(np.mean(l_ms), color=C_BLUE, lw=2, ls='--',
                label=f'Mean loop   {np.mean(l_ms):.2f} ms')
    ax2.axvline(ts_ms[0], color=C_GREY, lw=1.5, ls=':',
                label=f'$t_s$ = {ts_ms[0]:.2f} ms')
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right', frameon=True, ncol=2)
    _grid(ax2)

    return fig


def plot_time(ts, delta_t, t):
    """Simple sample-time vs loop-time overlay."""
    fig, ax = _new_figure()
    ax.set_xlim((t[0], t[-1]))
    ax.plot(t[:ts.shape[1]], ts[0, :] * 1e3,
            color=C_RED, lw=2, ls="--", label=r'$t_s$')
    ax.plot(t[:ts.shape[1]], delta_t[0, :ts.shape[1]] * 1e3,
            color=C_GREEN, lw=2, ls="-", label=r'$\Delta t$')
    ax.set_ylabel(r"Time [ms]")
    ax.set_xlabel(r"$t$ [s]")
    _legend(ax)
    _grid(ax)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  New MPCC-specific plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_progress_velocity(vel_progres, vel_real, theta_history, t):
    """Progress velocity (v_θ solver, v_real) and θ evolution.

    Parameters
    ----------
    vel_progres   : ndarray (1, N)  – v_θ commanded by solver
    vel_real      : ndarray (1, N)  – real progress speed (dot(tangent, v))
    theta_history : ndarray (1, N+1) – θ state history
    t             : ndarray (N,) or (N+1,) – time vector
    """
    N = vel_progres.shape[1]

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.30)

    # ── Subplot 1: velocities ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t[:N], vel_progres[0, :], color=C_RED,  lw=2, ls='-',
             label=r'$v_\theta$ (solver)')
    ax1.plot(t[:N], vel_real[0, :],    color=C_BLUE, lw=2, ls='-',
             label=r'$v_{real}$ (projected)')
    ax1.set_ylabel(r"Velocity [m/s]")
    ax1.set_title("Progress velocity")
    ax1.legend(loc='upper right', frameon=True, ncol=2)
    _grid(ax1)

    # ── Subplot 2: θ progress ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    N_theta = theta_history.shape[1]
    if len(t) >= N_theta:
        t_theta = t[:N_theta]
    else:
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        t_theta = np.append(t[:N], t[N-1] + dt)
    ax2.plot(t_theta, theta_history[0, :N_theta], color=C_GREEN, lw=2)
    ax2.set_ylabel(r"$\theta$ [m]")
    ax2.set_xlabel(r"$t$ [s]")
    ax2.set_title(r"Arc-length progress $\theta(t)$")
    _grid(ax2)

    return fig


def plot_velocity_analysis(vel_progres, vel_real, vel_tangent,
                           curvature, theta_history, s_max, t):
    """Comprehensive velocity analysis with curvature subplot.

    Three subplots stacked vertically:
      Top (1/3)    : v_θ solver, v_real drone, v_tangent speed
      Middle (1/3) : θ progress
      Bottom (1/3) : curvature κ(s) along the path

    Parameters
    ----------
    vel_progres   : ndarray (1, N)    – v_θ commanded by solver
    vel_real      : ndarray (1, N)    – real progress speed (dot(tangent, v))
    vel_tangent   : ndarray (1, N)    – ‖v(t)‖ actual UAV speed
    curvature     : ndarray (M,)      – κ(s) evaluated at uniform arc-length samples
    theta_history : ndarray (1, N+1)  – θ state history
    s_max         : float             – total arc length
    t             : ndarray (N,)      – time vector (N steps)
    """
    N = vel_progres.shape[1]

    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)

    # ── Top: velocities ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t[:N], vel_progres[0, :],  color=C_RED,    lw=2.0, ls='-',
             label=r'$v_\theta$ (solver)')
    ax1.plot(t[:N], vel_real[0, :],     color=C_BLUE,   lw=2.0, ls='-',
             label=r'$v_{real}$ (tangent proj.)')
    ax1.plot(t[:N], vel_tangent[0, :],  color=C_GREEN,  lw=1.5, ls='--',
             label=r'$\|v\|$ (speed)')
    ax1.set_ylabel(r"Velocity [m/s]")
    ax1.set_title("Velocity analysis")
    ax1.legend(loc='upper right', frameon=True, ncol=3, fontsize=12)
    _grid(ax1)

    # ── Middle: θ progress ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # theta_history has N+1 points; build matching time axis
    N_theta = theta_history.shape[1]
    if len(t) >= N_theta:
        t_theta = t[:N_theta]
    else:
        # t has N points, theta has N+1 → interpolate last time step
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        t_theta = np.append(t[:N], t[N-1] + dt)
    ax2.plot(t_theta, theta_history[0, :N_theta], color=C_PURPLE, lw=2)
    ax2.axhline(s_max, color=C_GREY, lw=1.2, ls=':', label=f'$s_{{max}}={s_max:.1f}$ m')
    ax2.set_ylabel(r"$\theta$ [m]")
    ax2.legend(loc='lower right', frameon=True, ncol=1, fontsize=11)
    _grid(ax2)

    # ── Bottom: curvature ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    s_curv = np.linspace(0, s_max, len(curvature))
    ax3.fill_between(s_curv, curvature, alpha=0.25, color=C_ORANGE)
    ax3.plot(s_curv, curvature, color=C_ORANGE, lw=1.8)
    ax3.set_ylabel(r"$\kappa$ [1/m]")
    ax3.set_xlabel(r"Arc-length $s$ [m]")
    ax3.set_title("Path curvature")
    _grid(ax3)

    fig.tight_layout()
    return fig


def plot_3d_trajectory(x_actual, pos_ref, s_max=None,
                       position_by_arc=None, N_plot=500):
    """3D trajectory visualisation: reference path vs actual flight path.

    Falls back to 2D (XY + XZ) if mpl_toolkits.mplot3d is unavailable.

    Parameters
    ----------
    x_actual        : ndarray (13+, N+1) – state trajectory (rows 0-2 = pos)
    pos_ref         : ndarray (3, M)     – reference path samples
    s_max           : float, optional    – if given, used for dense sampling
    position_by_arc : callable, optional – for denser reference sampling
    N_plot          : int                – dense ref samples (if callable given)
    """
    # Build dense reference
    if position_by_arc is not None and s_max is not None:
        s_dense = np.linspace(0, s_max, N_plot)
        ref = np.array([position_by_arc(s) for s in s_dense]).T
    else:
        ref = pos_ref

    if _HAS_3D:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(ref[0], ref[1], ref[2],
                color=C_GREY, lw=1.5, ls='--', alpha=0.6, label='Reference path')
        ax.plot(x_actual[0, :], x_actual[1, :], x_actual[2, :],
                color=C_BLUE, lw=2.0, label='Actual trajectory')

        ax.scatter(*x_actual[:3, 0],  color=C_GREEN, s=80, marker='o',
                   zorder=5, label='Start')
        ax.scatter(*x_actual[:3, -1], color=C_RED,   s=80, marker='X',
                   zorder=5, label='End')

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_zlabel(r'$z$ [m]')
        ax.set_title('3D Trajectory')
        ax.legend(loc='upper left', frameon=True, fontsize=11)
        _set_3d_equal_aspect(ax, x_actual[:3, :])
    else:
        # 2D projections: XY on top, XZ below
        # ── Pre-compute data ranges to set axis limits and figure proportions ──
        all_x = np.concatenate([ref[0], x_actual[0, :]])
        all_y = np.concatenate([ref[1], x_actual[1, :]])
        all_z = np.concatenate([ref[2], x_actual[2, :]])

        x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

        margin_x = (x_max - x_min) * 0.08 if np.isfinite(x_max - x_min) else 1.0
        margin_y = (y_max - y_min) * 0.08 if np.isfinite(y_max - y_min) else 1.0
        margin_z = (z_max - z_min) * 0.08 if np.isfinite(z_max - z_min) else 1.0

        xlim = (x_min - margin_x, x_max + margin_x)
        ylim = (y_min - margin_y, y_max + margin_y)
        zlim = (z_min - margin_z, z_max + margin_z)

        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        z_span = zlim[1] - zlim[0]

        # height_ratios proportional to Y span so 1 m = 1 m in both subplots
        fig_w   = 9.0
        px_per_m = fig_w / x_span          # pixels per metre (common X axis)
        h1 = px_per_m * y_span             # height of XY subplot in inches
        h2 = px_per_m * z_span             # height of XZ subplot in inches
        fig_h = h1 + h2 + 1.2              # +1.2 for title/xlabel/spacing

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(fig_w, fig_h),
            sharex=True,
            gridspec_kw={'height_ratios': [y_span, z_span], 'hspace': 0.08},
        )

        # ── Common tick step for BOTH Y axes (same physical scale) ────────
        # Choose a "nice" step such that the X axis gets ~5-8 ticks,
        # then use the SAME step on both Y axes so grid spacing = same metres.
        def _nice_step(span, target_ticks=6):
            raw = span / target_ticks
            exp = np.floor(np.log10(raw))
            frac = raw / 10**exp
            if frac < 1.5:   nice = 1.0
            elif frac < 3.5: nice = 2.0
            elif frac < 7.5: nice = 5.0
            else:             nice = 10.0
            return nice * 10**exp

        tick_step = _nice_step(x_span)   # one step for ALL axes (same scale)

        # ── XY (top) ──────────────────────────────────────────────────────
        ax1.plot(ref[0], ref[1], color=C_GREY, lw=1.5, ls='--',
                 alpha=0.6, label='Reference')
        ax1.plot(x_actual[0, :], x_actual[1, :], color=C_BLUE, lw=2.0,
                 label='Actual')
        ax1.scatter(x_actual[0, 0],  x_actual[1, 0],  color=C_GREEN, s=80,
                    marker='o', zorder=5, label='Start')
        ax1.scatter(x_actual[0, -1], x_actual[1, -1], color=C_RED, s=80,
                    marker='X', zorder=5, label='End')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.set_title('Trajectory Projections')
        ax1.legend(fontsize=10)
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        _grid(ax1)

        # ── XZ (bottom) ───────────────────────────────────────────────────
        ax2.plot(ref[0], ref[2], color=C_GREY, lw=1.5, ls='--',
                 alpha=0.6, label='Reference')
        ax2.plot(x_actual[0, :], x_actual[2, :], color=C_BLUE, lw=2.0,
                 label='Actual')
        ax2.scatter(x_actual[0, 0],  x_actual[2, 0],  color=C_GREEN, s=80,
                    marker='o', zorder=5)
        ax2.scatter(x_actual[0, -1], x_actual[2, -1], color=C_RED, s=80,
                    marker='X', zorder=5)
        ax2.set_xlabel(r'$x$ [m]')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.legend(fontsize=10)
        ax2.set_xlim(*xlim)
        ax2.set_ylim(*zlim)
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        _grid(ax2)

    fig.tight_layout()
    return fig


def _set_3d_equal_aspect(ax, data):
    """Set equal aspect ratio for 3D plot given (3, N) data."""
    ranges = [np.nanmax(data[i]) - np.nanmin(data[i]) for i in range(3)]
    max_range = max(ranges) / 2.0
    mid = [(np.nanmax(data[i]) + np.nanmin(data[i])) / 2.0 for i in range(3)]
    if not (np.isfinite(max_range) and all(np.isfinite(m) for m in mid)):
        return
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
