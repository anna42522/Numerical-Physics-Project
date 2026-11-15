"""Plotting helpers: Wigner function."""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def plot_wigner(state, ax=None, lim=5.0, n=200, title=None):
    xvec = np.linspace(-lim, lim, n)
    W = qt.wigner(state, xvec, xvec)
    ax = ax or plt.gca()
    vmax = np.abs(W).max()
    im = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Re(α)")
    ax.set_ylabel("Im(α)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return im





def plot_results(times, purities, means, parities, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(times, purities, label='Purity', color='blue')
    plt.ylabel('Purity')
    plt.title('Purity over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, means, label='Mean', color='orange')
    plt.ylabel('Mean')
    plt.title('Mean over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, parities, label='Parity', color='green')
    plt.xlabel('Time')
    plt.ylabel('Parity')
    plt.title('Parity over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)


def plot_convergence(N_traj_values, mean_errors, std_errors, path):
    """Plot the MCWF convergence error against the number of trajectories."""

    import numpy as np
    import matplotlib.pyplot as plt

    N_traj_values = np.asarray(N_traj_values)
    mean_errors = np.asarray(mean_errors)
    std_errors = np.asarray(std_errors)

    # Theoretical O(M^-1/2) reference
    theoretical = (
        mean_errors[0]
        * (N_traj_values / N_traj_values[0])**(-0.5)
    )

    # Figure
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Numerical results
    ax.errorbar(
        N_traj_values,
        mean_errors,
        yerr=std_errors,
        fmt='o-',
        color='tab:blue',
        markersize=7,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label='MCWF'
    )

    # Theoretical scaling
    ax.plot(
        N_traj_values,
        theoretical,
        '--',
        color='tab:orange',
        linewidth=2,
        label=r'$O(M^{-1/2})$'
    )

    # Logarithmic axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Number of MCWF trajectories', fontsize=12)
    ax.set_ylabel('Mean Frobenius error', fontsize=12)
    ax.set_title('MCWF convergence', fontsize=14)

    # Grid
    ax.grid(True, which='major', color='gray', alpha=0.4)
    ax.grid(True, which='minor', color='gray', alpha=0.15)

    # Legend
    ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


