import sys
import pathlib
import numpy as np

proj_root = str(pathlib.Path(__file__).resolve().parents[1])
src_dir = str(pathlib.Path(__file__).resolve().parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cat_states import cat_state
from config import load_config
from operators import create_operators, get_jump_ops
from hamiltonian import create_H0, create_Heff
from lindblad import RK4_solver
from mcwf import mcwf
from plotting import plot_convergence

"""Parameters for the system derived from config.yaml"""
_cfg = load_config()
_conv_cfg = _cfg["simulation"]["convergence"]

Ncut = _conv_cfg["N_cut"]
alpha = _conv_cfg["alpha"]
kappa = _conv_cfg["kappa"]
gamma = _conv_cfg["gamma"]
t_final = _conv_cfg["t_final"]
n_steps = _conv_cfg["n_steps"]
parity = _cfg["system"]["cat_parity"]
omega = _cfg["system"].get("omega", 0.0)

"""Initial state: same for Lindblad and MCWF"""
psi0 = cat_state(alpha, Ncut, parity=parity).full().ravel()
rho0 = np.outer(psi0, psi0.conj())

"""Hamiltonian and jump operators"""
a, adag, n_op, parity_op = create_operators(Ncut)
jump_ops = get_jump_ops(kappa=kappa, gamma=gamma, a=a, n=n_op)
H = create_H0(Ncut, omega=omega)  # rotating frame
Heff = create_Heff(H, jump_ops)


def frobenius_error(rho_mcwf, rho_reference):
    """Measures the Frobenius error for each instant"""
    if rho_mcwf.shape != rho_reference.shape:
        raise ValueError("History should have the same size")

    errors_per_time = np.linalg.norm(rho_mcwf - rho_reference, ord="fro", axis=(1, 2))
    error_global = errors_per_time.mean()
    return errors_per_time, error_global


def benchmark_convergence(rho_reference, times, N_traj_values, seeds):
    """
    Evaluate how the MCWF error changes with the number of trajectories.

    For each value in N_traj_values, run the MCWF solver once per seed.
    Compare each MCWF density-matrix history with the Lindblad reference
    using the mean Frobenius error over all time points.
    """
    mean_errors = []
    std_errors = []

    for N_traj in N_traj_values:
        errors = []
        for seed in seeds:
            rho_mcwf = mcwf(Heff=Heff, jump_ops=jump_ops, psi0=psi0, times=times, N_traj=N_traj, seed=seed)
            errors_per_time, global_error = frobenius_error(rho_mcwf, rho_reference)
            errors.append(global_error)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors, ddof=1))

    return np.array(mean_errors), np.array(std_errors)


if __name__ == "__main__":
    times = np.linspace(0.0, t_final, n_steps)
    rho_reference = RK4_solver(H, jump_ops, rho0, times)

    N_traj_values = _conv_cfg["n_traj_values"]
    seeds = _conv_cfg["seeds"]

    mean_errors, std_errors = benchmark_convergence(
        rho_reference,
        times,
        N_traj_values,
        seeds,
    )

    plot_convergence(
        N_traj_values,
        mean_errors,
        std_errors,
        path="results/figures/mcwf_lindblad_convergence.png",
    )
