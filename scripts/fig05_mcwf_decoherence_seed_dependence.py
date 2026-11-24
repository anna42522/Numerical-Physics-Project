"""Fig. 5: Mcwf decoherence of an even Schrödinger cat state for fixed N_traj and different seeds."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from src.cat_states import cat_state, check_truncation
from src.config import load_config, get_time_grid
from src.mcwf import mcwf_solver
from src.operators import create_operators
from src.plotting import plot_wigner
from src.hamiltonian import create_Heff
from src.operators import get_jump_ops

def seed_dependence():
    cfg = load_config()
    N_CUT = cfg["system"]["N_cut"]
    alpha = cfg["system"]["alpha"]
    parity = cfg["system"]["cat_parity"]
    hbar = cfg["system"].get("hbar", 1.0)
    kappa = cfg["environment"]["kappa"]
    gamma = cfg["environment"]["gamma"]
    N_traj = cfg["simulation"]["n_trajectories"]
    seeds = cfg["simulation"]["seed_list"]
    times = get_time_grid(cfg)

    psi0 = cat_state(alpha, N_CUT, parity=parity)
    assert check_truncation(psi0), "Increase N_CUT for chosen alpha"

    a, _, n_op, parity_op = create_operators(N_CUT)
    H = np.zeros((N_CUT, N_CUT), dtype=complex)  # rotating frame: H_0 = 0
    observables = [n_op, parity_op]
    loss_op, deph_op = get_jump_ops(kappa, gamma, a, n_op)

    # Run 1: pure photon loss (gamma = 0)
    Heff_loss = create_Heff(H, [loss_op], hbar=hbar)
    n_loss_seed =[]
    purity_loss_seed=[]
    parity_loss_seed=[]
    for seed in seeds:
        rho_loss, purity_loss, n_loss, parity_loss = mcwf_solver(Heff=Heff_loss, jump_ops=[loss_op], psi0=psi0,times=times,observables=observables, N_traj=N_traj, seed=seed)
        n_loss_seed.append(n_loss)
        purity_loss_seed.append(purity_loss)
        parity_loss_seed.append(parity_loss) 
    

    #Run 2: photon loss + pure dephasing (gamma > 0)
  #  deph_op = np.sqrt(gamma) * n_op
    Heff_both= create_Heff(H,[loss_op,deph_op],hbar=1)
    n_both_seed =[]
    purity_both_seed=[]
    parity_both_seed=[]
    for seed in seeds:
        rho_both, purity_both, n_both, parity_both = mcwf_solver(Heff=Heff_both, jump_ops=[loss_op,deph_op], psi0=psi0,times=times,observables=observables, N_traj=N_traj, seed=seed)
        n_both_seed.append(n_both)
        purity_both_seed.append(purity_both)
        parity_both_seed.append(parity_both)

   
    """ only dephasing to inspect what it does"""
 #   deph_op = np.sqrt(gamma) * n_op
    Heff_deph= create_Heff(H,[deph_op],hbar=1)
    n_deph_seed =[]
    purity_deph_seed=[]
    parity_deph_seed=[]
    for seed in seeds:
        rho_deph, purity_deph, n_deph, parity_deph = mcwf_solver(Heff=Heff_deph, jump_ops=[deph_op], psi0=psi0,times=times,observables=observables, N_traj=N_traj, seed=seed)
        n_deph_seed.append(n_deph)
        purity_deph_seed.append(purity_deph)
        parity_deph_seed.append(parity_deph)

    # Validate our mcwf solver against QuTiP mesolve (pure photon loss)
    # res_qt = qt.mesolve(
    #     qt.qzero(N_CUT),
    #     psi0,
    #     times,
    #     [qt.Qobj(loss_op)],
    #     e_ops=[qt.Qobj(n_op), qt.Qobj(parity_op)],
    # )
    # max_n_err = np.max(np.abs(n_loss - res_qt.expect[0]))
    # max_p_err = np.max(np.abs(parity_loss - res_qt.expect[1]))
    # print(f"Max discrepancy vs QuTiP mesolve -> <n>: {max_n_err:.2e}, <Pi>: {max_p_err:.2e}")

    #characteristic times
    tau_dec = 1.0 / (2.0 * kappa * abs(alpha) ** 2) #decoherence
    tau_diss = 1.0 / kappa #dissipation

    #Build figure
    #ROWS: Purity, MEan photon number, Parity 
    #Columns: Loss, Dephasing, Loss+ Dephasing
    fig, axes = plt.subplots(3,3,figsize=(16,12))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.30,
        hspace=0.38
    )

    #organize results by physical condition
    conditions = [
        (
            "Photon loss",
            purity_loss_seed,
            n_loss_seed,
            parity_loss_seed
        ),
        (
            "Pure dephasing",
            purity_deph_seed,
            n_deph_seed,
            parity_deph_seed
        ),
        (
            "Loss + dephasing",
            purity_both_seed,
            n_both_seed,
            parity_both_seed
        )
    ]

    #plot all seeds
    for col, (name, pur, num, par) in enumerate(conditions):

        datasets = [pur, num, par]

        for row, data in enumerate(datasets):

            data = np.asarray(data)

            # Individual seeds
            for i, seed in enumerate(seeds):
                axes[row, col].plot(
                    times,
                    data[i],
                    color=f"C{i}",
                    lw=1.2,
                    alpha=0.5,
                    label=f"Seed {seed}" if row == 0 and col == 0 else None
                )

            # Mean over seeds
            mean = np.mean(data, axis=0)

            axes[row, col].plot(
                times,
                mean,
                color="black",
                lw=1.4,
                alpha=0.65,
                linestyle="--",
                label="Mean" if row == 0 and col == 0 else None
            )
    #titles
    axes[0, 0].set_title("Photon loss")
    axes[0, 1].set_title("Pure dephasing")
    axes[0, 2].set_title("Loss + dephasing")

    #y-axis labels 
    axes[0, 0].set_ylabel(r"Purity $\mathcal{P}(t)$")
    axes[1, 0].set_ylabel(r"Mean photon number $\langle n(t)\rangle$")
    axes[2, 0].set_ylabel(r"Parity $\langle \Pi(t)\rangle$")

    #configure all subplots
    for row in range(3):
        for col in range(3):

            ax = axes[row, col]

            ax.set_xlabel(r"Dimensionless time $\kappa t$")
            ax.grid(True, alpha=0.25)

            # Characteristic times
            ax.axvline(
                tau_dec,
                color="navy",
                ls="--",
                lw=1.1,
                alpha=0.65,
                label=r"$\tau_{\mathrm{dec}}$" if row == 0 and col == 0 else None
            )

            ax.axvline(
                tau_diss,
                color="teal",
                ls="-.",
                lw=1.1,
                alpha=0.65,
                label=r"$\tau_{\mathrm{dis}}$" if row == 0 and col == 0 else None
            )

    #Global legend        
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=8,
        fontsize=9,
        frameon=True
    )

    # Panel labels
    labels = ["(a)", "(b)", "(c)",
              "(d)", "(e)", "(f)",
              "(g)", "(h)", "(i)"]

    for ax, label in zip(axes.flat, labels):
        ax.text(
            0.0, 1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            va="bottom",
            ha="left"
        )
    
    #axis limits
    for ax in axes[0, :]:
        ax.set_ylim(0.0, 1.06)

    n_max = float(abs(alpha) ** 2) * 1.15
    for ax in axes[1, :]:
        ax.set_ylim(0.0, n_max)

    for ax in axes[2, :]:
        ax.set_ylim(-0.2, 1.08)

    fig.suptitle(
        rf"MCWF Decoherence: Seed Dependence of Even Cat State "
        rf"($\alpha = {alpha}$, $N_{{\mathrm{{cut}}}} = {N_CUT}$, "
        rf"$N_{{\mathrm{{traj}}}} = {N_traj}$)",
        fontsize=14,
        y=0.99
    )

    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/fig05_mcwf_decoherence_seed_dependence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    from colorama import Fore
    print(f"{Fore.GREEN}Saved figure to: {out}")


if __name__ == "__main__":
    seed_dependence()




