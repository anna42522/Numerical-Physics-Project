"""Fig. 1: initial cat state and evolution under photon loss (mesolve)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from src.cat_states import cat_state, loss_op, check_truncation, N_CUT, coherent_state
from src.plotting import plot_wigner

# alpha, kappa = 3.0, 1.0
# psi0 = cat_state(alpha, "even")
# assert check_truncation(psi0), "Increase N_CUT"

# H = qt.qzero(N_CUT)                      # rotating frame, H0 = 0
# times = np.array([0.0, 0.02, 0.05, 0.2])  # in units of 1/kappa
# res = qt.mesolve(H, psi0, times, [loss_op(kappa)])

# fig, axes = plt.subplots(1, len(times), figsize=(14, 3.6))
# for ax, t, st in zip(axes, times, res.states):
#     plot_wigner(st, ax=ax, title=f"κt = {t:.2f}")
# fig.tight_layout()
# out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/fig01_cat_wigner.png"
# fig.savefig(out, dpi=150)
# print("Saved:", out)


def plot_cat_wigner():
    alphas=np.linspace(0,5,5)

    fig, axes= plt.subplots(1, len(alphas), figsize=(14, 3.6))
    for ax, alpha in zip(axes, alphas):
        cat= cat_state(alpha, "even")
        wig= qt.wigner(cat, np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
        ax.contourf(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200), wig, 100)
        ax.set_title(f"α = {alpha:.1f}")
    fig.tight_layout()

    plt.title("Wigner function of the cat state")   
    plt.savefig("results/figures/single_mode_state.png")

def plot_coherent_wigner(alpha=3.0, lim=5.0, n_pts=300):
    coh = coherent_state(alpha)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = plot_wigner(coh, ax=ax, lim=lim, n=n_pts, title=f"Wigner Function of Coherent State |α={alpha}⟩")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$W(x, p)$", rotation=270, labelpad=12)
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/coherent_state.png"
    fig.savefig(out, dpi=200)
    print("Saved detailed coherent Wigner function to:", out)


import pathlib
def plot_coherent_vs_cat_detailed(alpha=2.5, N=60, lim=5.0, n_pts=300):
    """
    Plots a high-resolution, side-by-side comparison of:
    1. A single coherent state |alpha>
    2. An even Schrödinger cat state |alpha> + |-alpha>
    3. A 1D cut along the axis p=0 showing the negative interference dip.
    """
    xvec = np.linspace(-lim, lim, n_pts)
    pvec = np.linspace(-lim, lim, n_pts)
    
    # 1. Generate states
    coh = qt.coherent(N, alpha).unit()
    cat = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
    
    # 2. Compute high-resolution Wigner functions
    W_coh = qt.wigner(coh, xvec, pvec)
    W_cat = qt.wigner(cat, xvec, pvec)
    
    # Symmetric color limits so white is EXACTLY zero
    vmax = max(np.abs(W_coh).max(), np.abs(W_cat).max())
    
    # 3. Create a 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # --- PANEL 1: Coherent State ---
    im1 = axes[0].contourf(xvec, pvec, W_coh, 120, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"Coherent State $|\alpha\rangle$ ($\alpha=" + f"{alpha}" + r"$)", fontsize=12)
    axes[0].set_xlabel(r"$\mathrm{Re}(\alpha)$ (Position $x$)")
    axes[0].set_ylabel(r"$\mathrm{Im}(\alpha)$ (Momentum $p$)")
    axes[0].set_aspect("equal")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.4)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # --- PANEL 2: Cat State ---
    im2 = axes[1].contourf(xvec, pvec, W_cat, 120, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"Even Cat State $|\alpha\rangle + |-\alpha\rangle$", fontsize=12)
    axes[1].set_xlabel(r"$\mathrm{Re}(\alpha)$ (Position $x$)")
    axes[1].set_ylabel(r"$\mathrm{Im}(\alpha)$ (Momentum $p$)")
    axes[1].set_aspect("equal")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.4)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # --- PANEL 3: 1D Cross-Section at p = 0 ---
    # Slice through the horizontal midline (p = 0)
    p_zero_idx = np.argmin(np.abs(pvec))
    slice_coh = W_coh[p_zero_idx, :]
    slice_cat = W_cat[p_zero_idx, :]
    
    axes[2].plot(xvec, slice_coh, label=r"Coherent $|\alpha\rangle$", color="royalblue", lw=2)
    axes[2].plot(xvec, slice_cat, label=r"Cat State $|\psi_+\rangle$", color="crimson", lw=2)
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[2].set_title(r"1D Slice along $p = 0$ (Negative Values)", fontsize=12)
    axes[2].set_xlabel(r"$\mathrm{Re}(\alpha)$")
    axes[2].set_ylabel(r"$W(x, p=0)$")
    axes[2].legend(frameon=True)
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/detailed_coherent_vs_cat.png"
    fig.savefig(out, dpi=200)


def plot_wigner_3d_wave(alpha=2.5, parity="even", N=60, lim=5.0, n_pts=200, elev=22, azim=-75, ax=None, show_cbar=True):
    """
    Plots the Wigner function as a 3D landscape/wave, with a transparent
    horizontal reference plane at W = 0.
    
    If ax is None, a new figure is created and saved automatically.
    If ax is provided, it draws directly onto that subplot!
    """
    xvec = np.linspace(-lim, lim, n_pts)
    pvec = np.linspace(-lim, lim, n_pts)
    X, P = np.meshgrid(xvec, pvec)
    
    cat = cat_state(alpha, parity=parity, N=N)
    W = qt.wigner(cat, xvec, pvec)
    vmax = np.abs(W).max()
    
    standalone = (ax is None)
    if standalone:
        fig = plt.figure(figsize=(11, 7.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    
    # 1. Plot the Wigner function as a 3D undulating wave surface
    surf = ax.plot_surface(
        X, P, W,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        linewidth=0.1,
        edgecolor="none",
        antialiased=True,
        rstride=2, cstride=2,
        alpha=0.92
    )
    
    # 2. Reference plane at W = 0 (semi-transparent gray plane)
    zero_plane = np.zeros_like(X)
    ax.plot_surface(
        X, P, zero_plane,
        color="silver",
        alpha=0.4,
        linewidth=0,
        shade=False
    )
    
    # Customize viewing angle and labels
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$\mathrm{Re}(\alpha)$", labelpad=8, fontsize=10)
    ax.set_ylabel(r"$\mathrm{Im}(\alpha)$", labelpad=8, fontsize=10)
    ax.set_zlabel(r"$W(x, p)$", labelpad=8, fontsize=10)
    
    if standalone:
        ax.set_title(f"3D Wigner Wave of {parity.capitalize()} Cat State (α = {alpha})\n"
                     r"[Left: $-\alpha$ Peak, Right: $+\alpha$ Peak, Center: Quantum Ripples below $W=0$ plane]",
                     fontsize=12, pad=15)
        if show_cbar:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
            cbar.set_label(r"$W(x, p)$", rotation=270, labelpad=15)
        fig.tight_layout()
        out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/wigner_3d_wave.png"
        fig.savefig(out, dpi=200)
        print(f"Saved 3D Wigner wave plot to: {out}")
        return fig, ax
    
    return surf


def plot_wigner_3d_alphas():
    """
    Plots a 2x2 grid of 3D Wigner functions for different alpha values,
    showing how the mountains separate and the ripples multiply as alpha increases.
    """
    alphas = [1.0, 2.0, 3.0, 4.0]
    fig = plt.figure(figsize=(15, 12))
    
    for i, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        plot_wigner_3d_wave(
            alpha=alpha,
            parity="even",
            N=N_CUT,
            lim=5.5,
            n_pts=160,
            elev=22,
            azim=-75,
            ax=ax,
            show_cbar=False
        )
        ax.set_title(f"α = {alpha:.1f} (Separation d = {2*alpha:.1f})", fontsize=12, pad=10)
    
    fig.suptitle("Evolution of Cat State 3D Wigner Wave vs Amplitude α", fontsize=15, y=0.98)
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/wigner_3d_alphas.png"
    fig.savefig(out, dpi=200)
    print(f"Saved series of 3D Wigner plots for different alphas to: {out}")


if __name__=="__main__":
    #plot_wigner_3d_alphas()
    plot_wigner_3d_wave(alpha=4, parity="even", N=N_CUT, lim=5.0, n_pts=200, elev=22, azim=0)
