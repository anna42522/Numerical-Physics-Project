"""Fig. 7: Physical and computational advantages of MCWF over the Lindblad master equation.

Panel (a): Single-trajectory quantum jump unraveling vs smooth ensemble average.
           Demonstrates discrete parity flips (+1 -> -1 -> +1) at each single-photon loss event.
Panel (b): Memory footprint O(N_cut) vs O(N_cut^2) and CPU runtime crossover vs Hilbert dimension N_cut.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from src.cat_states import cat_state
from src.config import load_config
from src.hamiltonian import create_H0, create_Heff
from src.lindblad import RK4_solver
from src.mcwf import mcwf_solver, mcwf_trajectory
from src.operators import create_operators, get_jump_ops


def mcwf_advantages():
    cfg = load_config()
    alpha = cfg["system"]["alpha"]
    N_cut = cfg["system"]["N_cut"]
    parity = cfg["system"]["cat_parity"]
    kappa = cfg["environment"]["kappa"]
    N_traj = cfg["simulation"]["n_trajectories"]
    seed = cfg["simulation"]["random_seed"]

    # --- PART 1: Single-trajectory parity jumps vs ensemble average ---
    t_max = 0.45  # Focus on the decoherence window where the first few photon jumps occur
    times = np.linspace(0.0, t_max, 451)

    psi0 = cat_state(alpha, N_cut, parity=parity).full().ravel()
    rho0 = np.outer(psi0, psi0.conj())
    a, _, n_op, parity_op = create_operators(N_cut)
    parity_diag = np.diag(parity_op).real
    loss_op, _ = get_jump_ops(kappa=kappa, gamma=0.0, a=a, n=n_op)
    H0 = create_H0(N_cut, omega=0.0)
    Heff = create_Heff(H0, [loss_op])

    # Single representative trajectories showing discrete +1 <-> -1 parity flips
    rng_single = np.random.default_rng(seed=seed)
    psi_hist_1, jumps_1 = mcwf_trajectory(Heff, [loss_op], rng_single, psi0, times)
    parity_traj_1 = (np.abs(psi_hist_1) ** 2) @ parity_diag

    psi_hist_2, jumps_2 = mcwf_trajectory(Heff, [loss_op], rng_single, psi0, times)
    parity_traj_2 = (np.abs(psi_hist_2) ** 2) @ parity_diag

    # MCWF ensemble average without constructing rho (compute_rho=False)
    _, parity_ensemble = mcwf_solver(
        Heff=Heff,
        jump_ops=[loss_op],
        psi0=psi0,
        times=times,
        observables=[n_op, parity_op],
        N_traj=N_traj,
        seed=seed,
        compute_rho=False,
    )

    # Exact Lindblad reference
    rho_lind = RK4_solver(H0, [loss_op], rho0, times)
    parity_lind = np.einsum("tii->t", rho_lind * parity_diag[None, :, None]).real
    tau_dec = 1.0 / (2.0 * kappa * abs(alpha) ** 2)

    # --- PART 2: Computational & Memory Scaling vs Hilbert-Space Dimension N_cut ---
    dims = np.array([30, 60, 120, 240, 450, 750, 1100])
    n_bench_steps = 25
    n_bench_traj = 30
    times_bench = np.linspace(0.0, 0.05, n_bench_steps)

    t_lindblad = []
    t_mcwf = []

    print(f"Benchmarking Lindblad O(N^3) vs MCWF O(N_traj * N^2) across N_cut = {dims.tolist()}...")
    for d in dims:
        psi_d = np.zeros(d, dtype=complex)
        psi_d[min(4, d - 1)] = 1.0
        rho_d = np.outer(psi_d, psi_d.conj())

        a_d, _, n_d, par_d = create_operators(d)
        L_d = np.sqrt(kappa) * a_d
        H0_d = np.zeros((d, d), dtype=complex)
        Heff_d = create_Heff(H0_d, [L_d])

        # Time Lindblad RK4
        t0 = time.perf_counter()
        _ = RK4_solver(H0_d, [L_d], rho_d, times_bench)
        t_lind = time.perf_counter() - t0
        t_lindblad.append(t_lind)

        # Time MCWF (compute_rho=False)
        t0 = time.perf_counter()
        _ = mcwf_solver(
            Heff=Heff_d,
            jump_ops=[L_d],
            psi0=psi_d,
            times=times_bench,
            observables=[n_d, par_d],
            N_traj=n_bench_traj,
            seed=seed,
            compute_rho=False,
        )
        t_mc = time.perf_counter() - t0
        t_mcwf.append(t_mc)
        print(f"  N_cut={d:4d} -> Lindblad: {t_lind:.4f}s | MCWF (compute_rho=False): {t_mc:.4f}s")

    t_lindblad = np.array(t_lindblad)
    t_mcwf = np.array(t_mcwf)

    # State memory in KB (complex128 = 16 bytes per entry)
    mem_vec_kb = (dims * 16) / 1024.0
    mem_rho_kb = (dims**2 * 16) / 1024.0

    # --- Build 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=160)
    plt.subplots_adjust(wspace=0.30, left=0.06, right=0.97, top=0.86, bottom=0.14)

    # Panel (a): Single-shot parity jumps vs ensemble
    ax1 = axes[0]
    ax1.step(times, parity_traj_1, where="post", color="royalblue", lw=1.8, alpha=0.85,
             label=f"Single Traj #1 ({len(jumps_1)} jumps)")
    ax1.step(times, parity_traj_2, where="post", color="seagreen", lw=1.5, ls="--", alpha=0.75,
             label=f"Single Traj #2 ({len(jumps_2)} jumps)")
    for jt in jumps_1:
        ax1.axvline(jt, color="royalblue", ls=":", lw=1.0, alpha=0.6)

    ax1.plot(times, parity_ensemble, color="darkorange", lw=2.4,
             label=rf"MCWF Ensemble ($N_{{\mathrm{{traj}}}}={N_traj}$)")
    ax1.plot(times, parity_lind, color="black", ls="--", lw=1.8,
             label="Lindblad Master Eq.")
    ax1.axvline(tau_dec, color="crimson", ls="-.", lw=1.2, alpha=0.8,
                label=rf"$\tau_{{\mathrm{{dec}}}} = {tau_dec:.3f}$")

    ax1.set_title("(a) Physical Unraveling: Single-Photon Parity Jumps", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel(r"Dimensionless time $\kappa t$", fontsize=11)
    ax1.set_ylabel(r"Parity $\langle \hat{\Pi}(t) \rangle$", fontsize=11)
    ax1.set_ylim(-1.15, 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, frameon=True)

    # Panel (b): Runtime crossover vs N_cut
    ax2 = axes[1]
    ax2.loglog(dims, t_lindblad, "o-", color="crimson", lw=2.2, markersize=7,
               label=r"Lindblad $\rho(t)$ [$\mathcal{O}(N_{\mathrm{cut}}^3)$]")
    ax2.loglog(dims, t_mcwf, "s-", color="royalblue", lw=2.2, markersize=7,
               label=rf"MCWF $|\psi(t)\rangle$ (`compute_rho=False`, $N_{{\mathrm{{traj}}}}={n_bench_traj}$) [$\mathcal{{O}}(N_{{\mathrm{{traj}}}} N_{{\mathrm{{cut}}}}^2)$]")

    # Reference slopes
    ref_n3 = t_lindblad[-2] * (dims / dims[-2]) ** 3.0
    ref_n2 = t_mcwf[-2] * (dims / dims[-2]) ** 2.0
    ax2.loglog(dims, ref_n3, ":", color="crimson", alpha=0.6, lw=1.4, label=r"Cubic slope $\propto N_{\mathrm{cut}}^3$")
    ax2.loglog(dims, ref_n2, "--", color="royalblue", alpha=0.6, lw=1.4, label=r"Quadratic slope $\propto N_{\mathrm{cut}}^2$")

    ax2.set_title(r"(b) CPU Runtime Crossover vs Dimension $N_{\mathrm{cut}}$", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel(r"Hilbert-space cutoff dimension $N_{\mathrm{cut}}$", fontsize=11)
    ax2.set_ylabel("Execution time [s]", fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel (c): State memory scaling O(N_cut) vs O(N_cut^2)
    ax3 = axes[2]
    ax3.loglog(dims, mem_rho_kb, "o-", color="crimson", lw=2.2, markersize=7,
               label=r"Lindblad Matrix $\rho$ ($N_{\mathrm{cut}} \times N_{\mathrm{cut}}$) [$\mathcal{O}(N_{\mathrm{cut}}^2)$]")
    ax3.loglog(dims, mem_vec_kb, "s-", color="royalblue", lw=2.2, markersize=7,
               label=r"MCWF State Vector $|\psi\rangle$ ($N_{\mathrm{cut}} \times 1$) [$\mathcal{O}(N_{\mathrm{cut}})$]")

    ax3.set_title(r"(c) Quantum State Memory Footprint", fontsize=11.5, fontweight="bold")
    ax3.set_xlabel(r"Hilbert-space cutoff dimension $N_{\mathrm{cut}}$", fontsize=11)
    ax3.set_ylabel("Memory per state [KB]", fontsize=11)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(
        rf"Advantages of MCWF Unraveling (`compute_rho=False`) over Lindblad Master Equation ($\alpha = {alpha}$)",
        fontsize=13.5,
        y=0.97,
    )

    out = pathlib.Path(__file__).resolve().parents[1] / "results/figures/fig07_mcwf_advantages.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"{Fore.GREEN}Saved figure to: {out}")


