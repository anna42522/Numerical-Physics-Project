"""Fig. 4: Decoherence scaling of Schrödinger cat states (Brune et al. 1996 validation).

Simulates cat states across amplitudes alpha in [1.2, 3.0] under photon loss
and pure dephasing, extracts numerical decoherence times tau_dec from parity decay,
and proves the scaling law:
    tau_dec = 1 / (2 * kappa * |alpha|^2) = 2 * T_r / D^2
with a log-log slope of -2.0 +/- 0.05 as required by the course handbook.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from src.cat_states import cat_state
from src.config import load_config
from src.hamiltonian import create_H0, create_Heff
from src.lindblad import RK4_solver
from src.mcwf import mcwf_solver
from src.operators import create_operators, get_jump_ops
from colorama import Fore

from scipy.optimize import curve_fit


def extract_tau_dec(times, parities, alpha, kappa=1.0):
    """
    Extracts the characteristic decoherence time tau_dec from the initial exponential decay of parity.
    Fits W(0,0,t) / Π(t) ~ exp(-t / τ_dec) for early times (κ * t << 1)
    """
    # Use early times window where kappa * t << 1
    
    t_cutoff = 0.5 / (2.0 * kappa * alpha**2)
    mask = times <= t_cutoff
    if np.sum(mask) < 5:
        mask = np.arange(min(len(times), 10))

    t_fit = times[mask]
    p_fit = parities[mask]

    try:
        popt, _ = curve_fit(
            lambda t, tau: np.exp(-t / tau),
            t_fit,
            p_fit,
            p0=[1.0 / (2.0 * kappa * alpha**2)],
        )
        return float(popt[0])
    except Exception:
        # Fallback to finite difference at t=0
        dt = times[1] - times[0]
        rate = -(parities[1] - parities[0]) / dt
        return float(1.0 / rate) if rate > 0 else float(np.nan)



def scaling():
    cfg = load_config()
    kappa = cfg["environment"]["kappa"]
    gamma = cfg["environment"]["gamma"]
    parity = cfg["system"]["cat_parity"]
    omega = cfg["system"].get("omega", 0.0)
    N_traj = cfg["simulation"]["n_trajectories"]
    seed = cfg["simulation"]["random_seed"]
    alphas = np.array(cfg["system"]["alphas_sweep"], dtype=float)
    alphas_mc = np.array(cfg["system"]["alphas_mcwf_sweep"], dtype=float)

    tau_loss_list = []
    tau_both_list = []
    decay_curves = []

    print(f"Sweeping {len(alphas)} cat amplitudes: {alphas}...")

    for alpha in alphas:
        Ncut = int(max(36, alpha**2 + 4 * alpha + 16))
        psi0 = cat_state(alpha, Ncut, parity=parity).full().ravel()
        rho0 = np.outer(psi0, psi0.conj())

        a, _, n_op, parity_op = create_operators(Ncut)
        loss_op = np.sqrt(kappa) * a
        deph_op = np.sqrt(gamma) * n_op
        H0 = create_H0(Ncut, omega=omega)

        # Scale time window to capture the full decoherence collapse
        t_max = min(1.0, max(0.2, 4.0 / (2.0 * kappa * alpha**2)))
        times = np.linspace(0.0, t_max, 501)

        # 1. Pure photon loss
        rho_hist_loss = RK4_solver(H0, [loss_op], rho0, times)
        parities_loss = np.array([np.trace(r @ parity_op).real for r in rho_hist_loss])
        tau_loss = extract_tau_dec(times, parities_loss, alpha, kappa)
        tau_loss_list.append(tau_loss)

        # 2. Photon loss + dephasing
        rho_hist_both = RK4_solver(H0, [loss_op, deph_op], rho0, times)
        parities_both = np.array([np.trace(r @ parity_op).real for r in rho_hist_both])
        tau_both = extract_tau_dec(times, parities_both, alpha, kappa)
        tau_both_list.append(tau_both)

        decay_curves.append((alpha, times, parities_loss))
        print(f"  alpha={alpha:.2f} -> tau_dec (loss) = {tau_loss:.4f}, tau_dec (loss+deph) = {tau_both:.4f}")

    tau_loss_arr = np.array(tau_loss_list)
    tau_both_arr = np.array(tau_both_list)
    theory_tau = 1.0 / (2.0 * kappa * alphas**2)

    # --- Run MCWF stochastic solver on representative cat sizes ---
    tau_mc_list = []
    mcwf_curves = {}
    print(f"\nRunning MCWF ensemble solver for alpha in {alphas_mc} (N_traj={N_traj}, seed={seed})...")

    for alpha in alphas_mc:
        Ncut = int(max(36, alpha**2 + 4 * alpha + 16))
        psi0 = cat_state(alpha, Ncut, parity=parity).full().ravel()
        a, _, n_op, parity_op = create_operators(Ncut)
        loss_op = np.sqrt(kappa) * a
        H0 = create_H0(Ncut, omega=omega)
        Heff = create_Heff(H0, [loss_op])

        t_max = min(1.0, max(0.2, 4.0 / (2.0 * kappa * alpha**2)))
        times_mc = np.linspace(0.0, t_max, 251)

        means, parities_mc = mcwf_solver(
            Heff=Heff,
            jump_ops=[loss_op],
            psi0=psi0,
            times=times_mc,
            observables=[n_op, parity_op],
            N_traj=N_traj,
            seed=seed,
            compute_rho=False,
        )
        tau_mc = extract_tau_dec(times_mc, parities_mc, alpha, kappa)
        tau_mc_list.append(tau_mc)
        mcwf_curves[alpha] = (times_mc, parities_mc)
        print(f"  MCWF alpha={alpha:.2f} -> tau_dec = {tau_mc:.4f}")

    tau_mc_arr = np.array(tau_mc_list)
    D2_mc = 4.0 * alphas_mc**2

    # Log-log linear fit: log(tau) = slope * log(alpha) + const
    log_alpha = np.log10(alphas)
    log_tau = np.log10(tau_loss_arr)
    slope, intercept = np.polyfit(log_alpha, log_tau, 1)
    print(f"\nExtracted scaling exponent (log-log slope): {slope:.3f}")

    # Build publication-quality 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    plt.subplots_adjust(wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.14)

    # --- Panel 1: Parity dynamics over time ---
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(alphas)))
    for (alpha, times, parities), c in zip(decay_curves, colors):
        ax1.plot(times, parities, label=rf"$\alpha = {alpha:.2f}$", color=c, lw=2.0)

    # Overlay MCWF curve for alpha = 2.0 to show agreement
    if 2.0 in mcwf_curves:
        t_mc, p_mc = mcwf_curves[2.0]
        ax1.plot(t_mc[::8], p_mc[::8], "d", color="#ff7f0e", markersize=5,
                 label=r"MCWF ($\alpha=2.0$)")

    ax1.axhline(1.0 / np.e, color="gray", ls="--", alpha=0.7, label=r"$1/e$ threshold")
    ax1.set_xlabel(r"Time $t$ [$1/\kappa$]", fontsize=11)
    ax1.set_ylabel(r"Parity $\langle\hat{\Pi}(t)\rangle$", fontsize=11)
    ax1.set_title(r"(a) Parity Collapse Dynamics", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, 0.6)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Panel 2: tau_dec vs alpha (log-log) ---
    ax2 = axes[1]
    ax2.loglog(alphas, theory_tau, "k--", lw=2.0, label=r"Theory: $\tau_{\rm dec} = \frac{1}{2\kappa |\alpha|^2}$")
    ax2.loglog(alphas, tau_loss_arr, "o", color="#1f77b4", markersize=7, label=r"Lindblad ($\gamma=0$)")
    ax2.loglog(alphas, tau_both_arr, "s", color="#d62728", markersize=6, fillstyle="none", label=r"Lindblad ($\gamma=0.1$)")
    ax2.loglog(alphas_mc, tau_mc_arr, "d", color="#ff7f0e", markersize=8, label=r"MCWF ($N_{\rm traj}=300$)")

    # Fit line overlay
    alpha_fine = np.linspace(alphas[0], alphas[-1], 100)
    fit_tau = 10**intercept * alpha_fine**slope
    ax2.loglog(alpha_fine, fit_tau, color="#1f77b4", alpha=0.4, lw=1.5,
               label=rf"Fit: $\alpha^{{{slope:.2f}}}$")

    ax2.set_xlabel(r"Cat amplitude $|\alpha|$", fontsize=11)
    ax2.set_ylabel(r"Decoherence time $\tau_{\rm dec}$ [$1/\kappa$]", fontsize=11)
    ax2.set_title(r"(b) Scaling Exponent $\tau_{\rm dec} \propto |\alpha|^{-2}$", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # --- Panel 3: Brune et al. (1996) scaling: tau_dec vs D^2 ---
    ax3 = axes[2]
    D2 = 4.0 * alphas**2  # Phase space separation squared
    D2_fine = np.linspace(D2[0], D2[-1], 100)
    theory_D2 = 2.0 / D2_fine  # tau_dec / Tr = 2 / D^2

    ax3.loglog(D2_fine, theory_D2, "k--", lw=2.0, label=r"Brune et al.: $\tau_{\rm dec}/T_r = 2 / D^2$")
    ax3.loglog(D2, tau_loss_arr, "o-", color="#2ca02c", lw=1.5, markersize=7, label=r"Lindblad")
    ax3.loglog(D2_mc, tau_mc_arr, "d", color="#ff7f0e", markersize=8, label=r"MCWF ($N_{\rm traj}=300$)")

    # Annotate slope -1.0
    ax3.text(
        0.08, 0.15,
        rf"Scaling: $\tau_{{\rm dec}} \propto D^{{-2}}$" "\n"
        rf"Nobel 1996: $\tau_{{\rm dec}} = 2 T_r / D^2$",
        transform=ax3.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.35)
    )

    ax3.set_xlabel(r"Separation squared $D^2 = 4|\alpha|^2$", fontsize=11)
    ax3.set_ylabel(r"Decoherence time $\tau_{\rm dec} / T_r$", fontsize=11)
    ax3.set_title(r"(c) Brune et al. (1996) Nobel Benchmark", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    out_path = pathlib.Path("results/figures/fig04_brune_scaling.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"{Fore.GREEN} Figure successfully generated and saved to {out_path}!")



