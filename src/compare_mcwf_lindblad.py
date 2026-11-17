import sys
import numpy as np

sys.path.insert(0,"src")

from cat_states import cat_state
from operators import create_operators, get_jump_ops
from hamiltonian import create_H0, create_Heff
from lindblad import RK4_solver
from mcwf import mcwf

"""Parameters for the system"""
Ncut = 12
alpha = 1.2
kappa = 0.3
gamma = 0.1
t_final = 0.5

"""Initial state: same for Lindblad and MCWF"""
#we need a numpy array for psi0
psi0 = cat_state(alpha,Ncut,parity="even").full().ravel()
rho0 = np.outer(psi0,psi0.conj())

"""Hamiltonian and jump operators"""
a,adag,n_op,parity_op=create_operators(Ncut)
jump_ops=get_jump_ops(kappa=kappa,gamma=gamma,a=a,n=n_op)
H=create_H0(Ncut,omega=0.0) #rotating frame
Heff=create_Heff(H,jump_ops)

def frobenius_error(rho_mcwf,rho_reference):
    """Measures the Frobenius error for each instant"""

    if rho_mcwf.shape != rho_reference.shape:
        raise ValueError("History should have the same size")

    errors_per_time=np.linalg.norm(rho_mcwf-rho_reference,ord="fro", axis=(1,2))

    error_global =errors_per_time.mean()

    return errors_per_time,error_global

"""Convergence Benchmark"""

def benchmark_convergence(rho_reference, times, N_traj_values,seeds):
    """
    Evaluate how the MCWF error changes with the number of trajectories.

    For each value in N_traj_values, run the MCWF solver once per seed.
    Compare each MCWF density-matrix history with the Lindblad reference
    using the mean Frobenius error over all time points.

    Parameters
    ----------
    rho_reference : np.ndarray
        Lindblad density-matrix history, with shape
        (len(times), dimension, dimension).
    times : np.ndarray
        Time points used by both the Lindblad and MCWF solvers.
    N_traj_values : iterable of int
        Numbers of MCWF trajectories to evaluate.
    seeds : iterable of int
        Random seeds used for independent MCWF runs.

    Returns
    -------
    mean_errors : np.ndarray
        Mean temporal error across seeds for each trajectory count.
    std_errors : np.ndarray
        Standard deviation across seeds for each trajectory count.
    """

    mean_errors=[]
    std_errors=[]

    for N_traj in N_traj_values:
        errors=[]

        for seed in seeds:
            rho_mcwf = mcwf (Heff=Heff,jump_ops=jump_ops,psi0=psi0,times=times,N_traj=N_traj,seed=seed)
            
            errors_per_time, global_error = frobenius_error(rho_mcwf,rho_reference)
            errors.append(global_error)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors,ddof=1))

    return np.array(mean_errors), np.array(std_errors)

from plotting import plot_convergence

times = np.linspace(0.0, t_final, 101)
rho_reference = RK4_solver(H, jump_ops, rho0, times)

N_traj_values = [100, 500, 1000, 5000, 10000]
seeds = [1, 2, 3, 4, 5]

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

