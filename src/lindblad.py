import numpy as np
import scipy 
import qutip 
from operators import conju_T, construct_rho, create_operators
from cat_states import loss_op, dephasing_op, coherent_state, cat_state
#it returns: the time derivative of the density matrix
def lindblad_function(H, jump_ops, rho):
    dissipator = 0
    for l in jump_ops:
        dissipator += l @ rho @ conju_T(l) - 0.5 * (conju_T(l) @ l @ rho + rho @ conju_T(l) @ l)
    return -1j * (H @ rho - rho @ H) + dissipator  



def RK4_step(H, jump_ops, rho, dt):
    """
    Performs a single Runge-Kutta 4th order (RK4) step for the Lindblad master equation.
    """
    k1 = lindblad_function(H, jump_ops, rho)
    k2 = lindblad_function(H, jump_ops, rho + 0.5 * dt * k1)
    k3 = lindblad_function(H, jump_ops, rho + 0.5 * dt * k2)
    k4 = lindblad_function(H, jump_ops, rho + dt * k3)

    return rho + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def RK4_solver(H, jump_ops, rho, times):
    '''
    Integrate the Lindblad master equation over `times` using RK4.

    notes: the RK4 differential works by taking 4 different slopes of 
    the function at different points that are constructed by a step height h
    and then taking a weighted average of those slopes to get a better estimate of the next value of the function. 
    '''
    rho = rho.copy()
    rho_history = [rho]

    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        rho = RK4_step(H, jump_ops, rho, dt)
        rho_history.append(rho)

    return np.array(rho_history)


def purity_mean_parity(rho, op, parity_op):
    """
    Computes the purity of a density matrix rho.
    """
    return np.trace(rho@rho).real , np.trace(rho@op).real , np.trace(rho@parity_op).real

def plot_results(times, purities, means, parities):
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
    plt.savefig("results/figures/lindblad_observables_dephasing.png", dpi=150)

def lindblad_solver(H, jump_ops, cat, times, observables=None):
    """Evolve an initial cat state under the Lindblad equation and return history and observables."""
    rho0 = construct_rho(cat)
    purity, mean, parity = purity_mean_parity(rho0, observables[0], observables[1])
    print(f"Initial purity: {purity:.4f}, mean: {mean:.4f}, parity: {parity:.4f}")
    rho_history = RK4_solver(H, jump_ops, rho0, times)

    purities = np.empty(len(times))
    means = np.empty(len(times))
    parities = np.empty(len(times))

    for i, rho in enumerate(rho_history):
        purities[i], means[i], parities[i] = purity_mean_parity(
            rho, observables[0], observables[1]
        )
    plot_results(times, purities, means, parities)
    return rho_history, purities, means, parities
if __name__ == "__main__":
    alpha = 3.0
    kappa = 1.0
    gamma = 0.1
    psi0 = cat_state(alpha, parity="even", N=100)
    H = np.zeros((psi0.shape[0], psi0.shape[0]), dtype=complex)  # rotating frame, H0 = 0
    times = np.linspace(0.0, 3.0, 1501)     
    a, adag, n, parity_op = create_operators(psi0.shape[0])
    #remember <n>=|α|^2 , Π leaves the cat state unchanged <Π>=Tr(ρΠ) and it starts from 1 and will decay to 0
    observables=[n, parity_op]

#i made the operators again ebcause we need the as numpy objects for now
    loss_op= np.sqrt(kappa) * a
    dephasing_op= np.sqrt(gamma) * n
    jump_ops = [dephasing_op] 
    lindblad_solver(H, jump_ops, psi0, times, observables=observables)
