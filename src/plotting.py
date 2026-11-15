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

