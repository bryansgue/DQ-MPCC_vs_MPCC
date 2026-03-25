import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
plt.rc('text', usetex=False)
def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': False,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def fancy_plot():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': False,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1

def plot_pose(x, xref, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$']
    
    for i in range(3):
        ax.plot(t[0:x.shape[1]], x[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
        
        ax.plot(t[0:x.shape[1]], xref[i, 0:x.shape[1]],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_control(u, t):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Colores para cada control
    labels = [r'$F$', r'$Tx$', r'$Ty$', r'$Tz$']  # Etiquetas para cada control
    
    for i in range(4):
        axs[i].plot(t[0:u.shape[1]], u[i, :], color=colors[i], lw=2, ls="-")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    axs[-1].set_xlabel(r"$[t]$", labelpad=5)
    
    fig.tight_layout()
    
    return fig

def plot_error(error, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(3):
        ax.plot(t[0:error.shape[1]], error[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_vel_lineal(v, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$x_p$', r'$y_p$', r'$z_p$']
    
    for i in range(3):
        ax.plot(t[0:v.shape[1]], v[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[vel_lineal]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_vel_angular(w, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    
    for i in range(3):
        ax.plot(t[0:w.shape[1]], w[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[vel_angular]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_CBF(value, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651']  # Add color for psi
    labels = [r'$value$']
    
    for i in range(1):
        ax.plot(t[0:value.shape[1]], value[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[CBF_value]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig


def plot_timing(t_solver, t_loop, t_sample, t):
    """
    Grafica el tiempo del solver MPC y del loop completo por iteración.

    Parámetros
    ----------
    t_solver : ndarray (1, N)  – tiempo exclusivo del solver  [s]
    t_loop   : ndarray (1, N)  – tiempo total del loop        [s]
    t_sample : ndarray (1, N)  – sample time nominal          [s]
    t        : ndarray (M,)    – vector de tiempo global      [s]
    """
    import numpy as np

    n = t_solver.shape[1]
    t_axis = t[:n]

    s_ms  = t_solver[0, :] * 1e3   # [ms]
    l_ms  = t_loop[0, :]   * 1e3   # [ms]
    ts_ms = t_sample[0, :] * 1e3   # [ms]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # ── Subplot 1: tiempos por iteración ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t_axis, l_ms,  color='#5189BB', lw=1.5, ls='-',  label=r'Loop total')
    ax1.plot(t_axis, s_ms,  color='#BB5651', lw=1.5, ls='-',  label=r'Solver MPC')
    ax1.plot(t_axis, ts_ms, color='#333333', lw=1.2, ls='--', label=r'$t_s$ nominal')
    ax1.fill_between(t_axis, s_ms, alpha=0.15, color='#BB5651')
    ax1.set_ylabel(r'Tiempo [ms]')
    ax1.set_title(r'Tiempo de cómputo por iteración')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, ncol=3,
               borderpad=0.4, labelspacing=0.4, handlelength=2)
    ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # ── Subplot 2: histograma de distribución ────────────────────────────────
    ax2 = axes[1]
    bins = max(30, n // 50)
    ax2.hist(s_ms, bins=bins, color='#BB5651', alpha=0.65,
             edgecolor='white', linewidth=0.4, label='Solver MPC')
    ax2.hist(l_ms, bins=bins, color='#5189BB', alpha=0.55,
             edgecolor='white', linewidth=0.4, label='Loop total')
    ax2.axvline(np.mean(s_ms), color='#BB5651', lw=2, ls='--',
                label=f'Media solver  {np.mean(s_ms):.2f} ms')
    ax2.axvline(np.mean(l_ms), color='#5189BB', lw=2, ls='--',
                label=f'Media loop    {np.mean(l_ms):.2f} ms')
    ax2.axvline(ts_ms[0],      color='#333333', lw=1.5, ls=':',
                label=f'$t_s$ nominal  {ts_ms[0]:.2f} ms')
    ax2.set_xlabel(r'Tiempo [ms]')
    ax2.set_ylabel(r'Frecuencia')
    ax2.set_title(r'Histograma de tiempos de cómputo')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, ncol=2,
               borderpad=0.4, labelspacing=0.4, handlelength=2)
    ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)

    return fig

def plot_time(ts, delta_t, t):
    fig, ax = fancy_plot()
    ax.set_xlim((t[0], t[-1]))
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(1):
        ax.plot(t[0:ts.shape[1]], ts[i, :],
                color='#BB5651', lw=2, ls="--", label=labels[i])
        
        ax.plot(t[0:ts.shape[1]], delta_t[i, 0:ts.shape[1]],
                color='#69BB51', lw=2, ls="-", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig
