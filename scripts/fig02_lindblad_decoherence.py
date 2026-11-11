"""Fig. 2: Lindblad decoherence of an even Schrödinger cat state."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from src.cat_states import N_CUT, cat_state, check_truncation
from src.lindblad import lindblad_solver
from src.operators import create_operators
from src.plotting import plot_wigner


def main():
    alpha = 3.0
    kappa = 1.0
    gamma = 0.2
    times = np.linspace(0.0, 3.5, 1751)  # dt = 0.002 in units of 1/kappa

    psi0 = cat_state(alpha, "even", N=N_CUT)
    assert check_truncation(psi0), "Increase N_CUT for chosen alpha"

    a, _, n_op, parity_op = create_operators(N_CUT)
    H = np.zeros((N_CUT, N_CUT), dtype=complex)  # rotating frame: H_0 = 0
    observables = [n_op, parity_op]

    # Run 1: pure photon loss (gamma = 0)
    loss_op = np.sqrt(kappa) * a
    rho_loss, purity_loss, n_loss, parity_loss = lindblad_solver(
        H, [loss_op], psi0, times, observables
    )

    # Run 2: photon loss + pure dephasing (gamma > 0)
    deph_op = np.sqrt(gamma) * n_op
    _, purity_both, n_both, parity_both = lindblad_solver(
        H, [loss_op, deph_op], psi0, times, observables
    )

    # Validate our RK4 Lindblad solver against QuTiP mesolve (pure photon loss)
    res_qt = qt.mesolve(
        qt.qzero(N_CUT),
        psi0,
        times,
        [qt.Qobj(loss_op)],
        e_ops=[qt.Qobj(n_op), qt.Qobj(parity_op)],
    )
    max_n_err = np.max(np.abs(n_loss - res_qt.expect[0]))
    max_p_err = np.max(np.abs(parity_loss - res_qt.expect[1]))
    print(f"Max discrepancy vs QuTiP mesolve -> <n>: {max_n_err:.2e}, <Pi>: {max_p_err:.2e}")

    tau_dec = 1.0 / (2.0 * kappa * abs(alpha) ** 2)
    tau_diss = 1.0 / kappa

    # Build figure with independent subgrids for top curves and bottom Wigner snapshots
    fig = plt.figure(figsize=(16, 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.34)
    gs_top = outer[0].subgridspec(1, 3, wspace=0.28)
    gs_bot = outer[1].subgridspec(1, 4, wspace=0.38)

    ax_pur = fig.add_subplot(gs_top[0, 0])
    ax_num = fig.add_subplot(gs_top[0, 1])
    ax_par = fig.add_subplot(gs_top[0, 2])

    # (a) Purity
    ax_pur.plot(times, purity_loss, color="crimson", lw=2.2, label=r"Photon loss ($\gamma=0$)")
    ax_pur.plot(times, purity_both, color="darkorange", lw=1.8, ls="--", label=rf"Loss + dephasing ($\gamma={gamma}\kappa$)")
    ax_pur.axhline(0.5, color="gray", ls=":", lw=1.1, label="50/50 classical mix (0.5)")
    ax_pur.axvline(tau_dec, color="navy", ls="--", lw=1.1, alpha=0.75, label=rf"$\tau_{{\mathrm{{dec}}}} = {tau_dec:.3f}$")
    ax_pur.axvline(tau_diss, color="teal", ls="-.", lw=1.1, alpha=0.75, label=rf"$\tau_{{\mathrm{{diss}}}} = {tau_diss:.1f}$")
    ax_pur.set_title(r"Purity $\mathcal{P}(t) = \mathrm{Tr}(\rho^2)$")
    ax_pur.set_xlabel(r"Dimensionless time $\kappa t$")
    ax_pur.set_ylabel(r"Purity $\mathcal{P}(t)$")
    ax_pur.set_ylim(0.0, 1.06)
    ax_pur.grid(True, alpha=0.25)
    ax_pur.legend(fontsize=8, loc="upper right")

    # (b) Mean photon number
    ax_num.plot(times, n_loss, color="royalblue", lw=2.2, label="RK4 (pure loss)")
    ax_num.plot(times, n_both, color="darkorange", lw=1.8, ls="--", label="RK4 (loss + dephasing)")
    ax_num.plot(times, n_loss[0] * np.exp(-kappa * times), color="black", ls=":", lw=1.5, label=r"Analytical $\langle\hat{n}_0\rangle e^{-\kappa t}$")
    ax_num.axvline(tau_dec, color="navy", ls="--", lw=1.1, alpha=0.75)
    ax_num.axvline(tau_diss, color="teal", ls="-.", lw=1.1, alpha=0.75)
    ax_num.set_title(r"Energy Relaxation $\langle \hat{n}(t) \rangle$")
    ax_num.set_xlabel(r"Dimensionless time $\kappa t$")
    ax_num.set_ylabel(r"Mean photon number $\langle \hat{n} \rangle$")
    ax_num.grid(True, alpha=0.25)
    ax_num.legend(fontsize=8, loc="upper right")

    # (c) Parity
    ax_par.plot(times, parity_loss, color="purple", lw=2.2, label=r"Photon loss ($\gamma=0$)")
    ax_par.plot(times, parity_both, color="darkorange", lw=1.8, ls="--", label=rf"Loss + dephasing ($\gamma={gamma}\kappa$)")
    ax_par.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax_par.axvline(tau_dec, color="navy", ls="--", lw=1.1, alpha=0.75, label=rf"$\tau_{{\mathrm{{dec}}}} = {tau_dec:.3f}$")
    ax_par.axvline(tau_diss, color="teal", ls="-.", lw=1.1, alpha=0.75, label=rf"$\tau_{{\mathrm{{diss}}}} = {tau_diss:.1f}$")
    ax_par.set_title(r"Quantum Coherence via Parity $\langle \hat{\Pi}(t) \rangle$")
    ax_par.set_xlabel(r"Dimensionless time $\kappa t$")
    ax_par.set_ylabel(r"Parity $\langle \hat{\Pi} \rangle$")
    ax_par.set_ylim(-0.08, 1.08)
    ax_par.grid(True, alpha=0.25)
    ax_par.legend(fontsize=8, loc="upper center")

    # Bottom row: Wigner snapshots at key physical times (lim=6.5 encloses blobs at +-sqrt(2)*alpha = +-4.24)
    snap_times = [0.0, 0.05, 0.25, 1.5]
    snap_labels = [
        r"$t = 0$ (Pure Cat)",
        r"$\kappa t = 0.05 \approx \tau_{\mathrm{dec}}$",
        r"$\kappa t = 0.25$ (Classical Mix)",
        r"$\kappa t = 1.50$ (Damping $\to |0\rangle$)",
    ]
    for idx_panel, (t_target, label) in enumerate(zip(snap_times, snap_labels)):
        ax_w = fig.add_subplot(gs_bot[0, idx_panel])
        idx_t = int(np.argmin(np.abs(times - t_target)))
        im = plot_wigner(qt.Qobj(rho_loss[idx_t]), ax=ax_w, lim=6.5, n=200, title=label)
        ax_w.set_xlabel(r"$x$ quadrature")
        if idx_panel == 0:
            ax_w.set_ylabel(r"$p$ quadrature")
        else:
            ax_w.set_ylabel("")
        cbar = fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        rf"Lindblad Master Equation Decoherence of Even Cat State ($\alpha = {alpha}$, $N_{{\mathrm{{cut}}}} = {N_CUT}$)",
        fontsize=14,
        y=0.96,
    )

    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/fig02_lindblad_decoherence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("Saved figure to:", out)


if __name__ == "__main__":
    main()
