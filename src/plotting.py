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

