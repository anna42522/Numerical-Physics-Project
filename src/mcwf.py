import numpy as np
from schrodinger import schrodinger_rhs, RK4_step_sch
from operators import get_jump_ops, conju_T


def jump_prob(psi_evolved ):
    #<ψ(t+δt)|ψ(t+δt)>=1-δp
    #must be δp<<1
    return 1-np.vdot(psi_evolved, psi_evolved).real

def get_epsilon(rng):
    #get a random number between 0 and 1
    return rng.uniform(0,1)

def mcwf_step(Heff, psi, dt, rng, jump_ops, hbar=1.0):
    """
    Performs a single Monte Carlo Wave Function (MCWF) step.
    - Heff: Effective Hamiltonian in the rotating frame with Hs=0 (numpy array)
    - psi: Current state vector (numpy array)
    - rng: Random number generator (numpy.random.Generator)
    - jump_ops: List of jump operators (numpy arrays) because we cannot have qt.Qobj in the MCWF function
    Returns:
    - psi_next: Next state vector after the MCWF step (numpy array)
    """
    
    # Evolve the state using RK4 method
    psi_evolved = RK4_step_sch(Heff, psi, dt, hbar)
    delta_p = jump_prob(psi_evolved)
    epsilon = get_epsilon(rng)
    
    if epsilon < delta_p:
        # A quantum jump occurs; normalize the state after the jump (eq. 12)
        #we cannot have both operators (loss + dephasing) at the same exact time. 
        #We know that δp=δp_loss + δp_dephasing, so we can choose one of the two jump operators based on their probabilities.
        #equation 12
        actual_single_operator, delta_p_op = jump_operator_from_list(jump_ops, rng, delta_p, psi, dt)
        
        #|φ(t+δt)>=C_m|φ(t)>/√(δp_m)/δt (eq. 12) 
        jumped= actual_single_operator @ psi
        #√(δp_m)/δt = sqrt(<ψ(t)|L_m^† L_m|ψ(t)>) (eq. 10) 
        psi_next = jumped / np.linalg.norm(jumped) #we do that so that we dont have to mess with δt. 
        jump_occurred = True
    else:
        # No jump; normalize the evolved state (eq.11)
        psi_next = psi_evolved / np.sqrt(1 - delta_p)
        jump_occurred = False
    
    return psi_next, jump_occurred


def jump_operator_from_list(jump_ops, rng, delta_p, psi, dt):

    if len(jump_ops) == 0:
        raise ValueError("The list of jump operators is empty.")
    elif len(jump_ops) == 1:
        return jump_ops[0], np.vdot(psi, jump_ops[0].conj().T @ jump_ops[0] @ psi).real
    else:
        probs_of_operators = []
        sum_pm = 0
        delta_p_ms = []

        for op in jump_ops:
            # Calculate the probability of each jump operator
            #δp_m=δt * <ψ(t)|L_m^† L_m|ψ(t)> >=0 (eq. 10)
            delta_p_op = np.vdot(psi, op.conj().T @ op @ psi).real # we dont use δt since it is compared to δp which is also multiplied by δt, so it cancels out.
            sum_pm+=delta_p_op
            #δp=Σδp_m (eq. 10)
            delta_p_ms.append(delta_p_op)

        #Π_m=δp_m/Σδp_m (eq. 10)
        probs_of_operators=[p_m/sum_pm for p_m in delta_p_ms] #normalized


        test_sum = sum(probs_of_operators)
        if not np.isclose(test_sum, 1.0):   
            raise ValueError("The sum of the probabilities of the jump operators does not equal 1.")

        # Choose a jump operator based on the calculated probabilities
        chosen_index = rng.choice(len(jump_ops), p=probs_of_operators)
        return jump_ops[chosen_index], delta_p_ms[chosen_index]



def mcwf_trajectory(Heff,jump_ops, rng, psi0, times):
    """
    Main function to perform the Monte Carlo Wave Function (MCWF) simulation.
    """
    psi_history=[psi0] 
    psi=psi0
    jumps_dt=[] #list that shows when jumps occured 
    for i in range(len(times) - 1):        
        dt = times[i + 1] - times[i]
        psi, jump_occured = mcwf_step(Heff=Heff, psi=psi, dt=dt, rng=rng, jump_ops=jump_ops)
        psi_history.append(psi)
        if jump_occured: 
            jumps_dt.append(times[i+1])


    return np.array(psi_history), jumps_dt 

def mcwf(Heff,jump_ops, psi0, times, N_traj, seed=42):
    rng=np.random.default_rng(seed=seed)
    rho_history = np.zeros((len(times), len(psi0), len(psi0)), dtype=complex)    
    for traj in range(N_traj):
        psi_history, jumps_dt= mcwf_trajectory(Heff=Heff, jump_ops=jump_ops, rng=rng, psi0=psi0, times=times)
        for t, psi_t in enumerate(psi_history):
            #now we have one specific psi of the trajectory at a specific time t
            rho_history[t]+=np.outer(psi_t, psi_t.conj())


    rho_history = rho_history / N_traj
    return rho_history


def mcwf_solver(Heff, jump_ops, psi0, times, observables=None, N_traj=100, seed=42):
    """
    Runs the MCWF ensemble solver and computes purity, mean photon number, and parity over time.
    """
    if hasattr(psi0, "full"):
        psi0 = psi0.full().ravel()

    rho_history = mcwf(Heff=Heff, jump_ops=jump_ops, psi0=psi0, times=times, N_traj=N_traj, seed=seed)

    purities = np.empty(len(times))
    means = np.empty(len(times))
    parities = np.empty(len(times))

    for i, rho in enumerate(rho_history):
        purities[i] = np.trace(rho @ rho).real
        if observables is not None and len(observables) >= 2:
            means[i] = np.trace(rho @ observables[0]).real
            parities[i] = np.trace(rho @ observables[1]).real

    return rho_history, purities, means, parities


if __name__ == "__main__":
    import pathlib
    import sys

    # Add project root and src to sys.path so imports work regardless of launch directory
    proj_root = str(pathlib.Path(__file__).resolve().parents[1])
    src_dir = str(pathlib.Path(__file__).resolve().parent)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from cat_states import cat_state
    from operators import create_operators, get_jump_ops
    from hamiltonian import create_H0, create_Heff
    from plotting import plot_results
    N = 25
    alpha = 3
    kappa = 1.0
    gamma = 0.1
    times = np.linspace(0.0, 1.5, 76)

    # Initial even cat state
    psi0 = cat_state(alpha, N, parity="even").full().ravel()

    # Operators
    H0 = create_H0(N, omega=0.0)
    a, adag, n_op, parity_op = create_operators(N)
    jump_ops = (get_jump_ops(kappa=kappa, gamma=gamma, a=a, n=n_op))
    Heff = create_Heff(H0, jump_ops)
    observables = [n_op, parity_op]

    print(f"Running MCWF simulation with N_traj=50...")
    rho_history, purities, means, parities = mcwf_solver(
        Heff=Heff,
        jump_ops=jump_ops,
        psi0=psi0,
        times=times,
        observables=observables,
        N_traj=500,
        seed=42,
    )
    path="results/figures/mcwf_observables_loss_dephasing.png"
    plot_results(times, purities=purities, means=means, parities=parities, path=path)
    print("Simulation complete!")
    print(f"Initial -> Purity: {purities[0]:.3f}, <n>: {means[0]:.3f}, Parity: {parities[0]:.3f}")
    print(f"Final   -> Purity: {purities[-1]:.3f}, <n>: {means[-1]:.3f}, Parity: {parities[-1]:.3f}")



