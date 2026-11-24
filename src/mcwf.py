import numpy as np
try:
    from src.schrodinger import schrodinger_rhs, RK4_step_sch
    from src.operators import get_jump_ops
    from src.config import load_config, get_time_grid
except ImportError:
    from schrodinger import schrodinger_rhs, RK4_step_sch
    from operators import get_jump_ops, conju_T
    from config import load_config, get_time_grid

_CFG = load_config()
_DEFAULT_NTRAJ = _CFG["simulation"]["n_trajectories"]
_DEFAULT_SEED = _CFG["simulation"]["random_seed"]
_DEFAULT_HBAR = _CFG["system"].get("hbar", 1.0)
import colorama

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
        # Evaluate at the midpoint state to cancel O(dt^2) bias between RK4 no-jump damping and jump projection
        psi_mid = 0.5 * (psi + psi_evolved)
        actual_single_operator, delta_p_op = jump_operator_from_list(jump_ops, rng, delta_p, psi_mid, dt)
        
        #|φ(t+δt)>=C_m|φ(t)>/√(δp_m)/δt (eq. 12) 
        jumped = actual_single_operator @ psi_mid
        #√(δp_m)/δt = sqrt(<ψ(t)|L_m^† L_m|ψ(t)>) (eq. 10) 
        psi_next = jumped / np.linalg.norm(jumped) #we do that so that we dont have to mess with δt. 
        jump_occurred = True
    else:
        # No jump; normalize the evolved state (eq.11)
        psi_next = psi_evolved / np.sqrt(1 - delta_p)
        jump_occurred = False
    
    return psi_next, jump_occurred


def jump_operator_from_list(jump_ops, rng, delta_p, psi, dt):

    """Finds the operator that will apply in that jump"""
    """Inputs:
        jump_ops: list of jumps operators
        rng: random number, used to choose the operator
        delta_p: probability of jump
        psi: state of the system
        dt: time jump
    """

    if len(jump_ops) == 0:
        raise ValueError("The list of jump operators is empty.")
    elif len(jump_ops) == 1:
        #<ψ(t)|L_m^† L_m|ψ(t)>
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

def mcwf(Heff, jump_ops, psi0, times, N_traj=_DEFAULT_NTRAJ, seed=_DEFAULT_SEED):
    """
    Runs multiple independent mcwf trajectories and averages their states to estimate the 
    density matrix, rho.
    """
    rng = np.random.default_rng(seed=seed)
    rho_history = np.zeros((len(times), len(psi0), len(psi0)), dtype=complex)
    for traj in range(N_traj):
        psi_history, jumps_dt = mcwf_trajectory(Heff=Heff, jump_ops=jump_ops, rng=rng, psi0=psi0, times=times)
        # Vectorized outer product across all time steps: |psi(t)><psi(t)|
        rho_history += psi_history[:, :, None] * psi_history[:, None, :].conj()

    rho_history /= N_traj
    return rho_history


def mcwf_solver(
    Heff,
    jump_ops,
    psi0,
    times,
    observables=None,
    N_traj=_DEFAULT_NTRAJ,
    seed=_DEFAULT_SEED,
    compute_rho=True,
):
    """
    Runs the MCWF ensemble solver.
    - Linear observables (<n>, <Pi>) are always computed directly from the state vectors
      <psi^(k)(t)|O|psi^(k)(t)> in O(N_cut) memory and time.
    - Set compute_rho=True when you need non-linear ensemble quantities like Purity Tr(rho^2)
      or phase-space Wigner plots.
    - Set compute_rho=False when you only need expectation values (<n>, <Pi>), avoiding
      any N_cut x N_cut density matrix allocation.
    """
    if hasattr(psi0, "full"):
        psi0 = psi0.full().ravel()

    rng = np.random.default_rng(seed=seed)
    n_steps = len(times)
    n_dim = len(psi0)
    rho_history = np.zeros((n_steps, n_dim, n_dim), dtype=complex) if compute_rho else None
    means = np.zeros(n_steps)
    parities = np.zeros(n_steps)

    has_obs = observables is not None and len(observables) >= 2
    if has_obs:
        obs0, obs1 = observables[0], observables[1]
        # Check if observables are diagonal (like number operator n and parity Pi) for O(N_cut) evaluation
        obs0_diag = np.diag(obs0).real if np.allclose(obs0, np.diag(np.diag(obs0))) else None
        obs1_diag = np.diag(obs1).real if np.allclose(obs1, np.diag(np.diag(obs1))) else None

    for traj in range(N_traj):
        psi_history, _ = mcwf_trajectory(Heff=Heff, jump_ops=jump_ops, rng=rng, psi0=psi0, times=times)
        if compute_rho:
            rho_history += psi_history[:, :, None] * psi_history[:, None, :].conj()

        if has_obs:
            probs = np.abs(psi_history) ** 2
            if obs0_diag is not None:
                means += probs @ obs0_diag
            else:
                means += np.einsum("ti,ij,tj->t", psi_history.conj(), obs0, psi_history).real

            if obs1_diag is not None:
                parities += probs @ obs1_diag
            else:
                parities += np.einsum("ti,ij,tj->t", psi_history.conj(), obs1, psi_history).real

    means /= N_traj
    parities /= N_traj

    if compute_rho:
        rho_history /= N_traj
        purities = np.einsum("tij,tji->t", rho_history, rho_history).real
    else:
        purities = None

    if compute_rho:
        return rho_history, purities, means, parities
    else: 
        return means, parities


def process(compute_rho=True):
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
    from plotting import plot_results, animate_wigner

    cfg = load_config()
    N = cfg["system"]["N_cut"]
    alpha = cfg["system"]["alpha"]
    parity = cfg["system"]["cat_parity"]
    omega = cfg["system"].get("omega", 0.0)
    kappa = cfg["environment"]["kappa"]
    gamma = cfg["environment"]["gamma"]
    N_traj = cfg["simulation"]["n_trajectories"]
    seed = cfg["simulation"]["random_seed"]
    times = get_time_grid(cfg)

    vis = cfg["visualization"]
    wigner_lim = vis["wigner_lim"]
    wigner_pts = vis["wigner_points"]
    n_frames = vis["animation_frames"]
    fps = vis["animation_fps"]

    # Initial cat state from config
    psi0 = cat_state(alpha, N, parity=parity).full().ravel()

    # Operators
    H0 = create_H0(N, omega=omega)
    a, adag, n_op, parity_op = create_operators(N)
    jump_ops = get_jump_ops(kappa=kappa, gamma=gamma, a=a, n=n_op)
    Heff = create_Heff(H0, jump_ops)
    observables = [n_op, parity_op]

    print(f"Running MCWF simulation with N_traj={N_traj}, seed={seed}...")
    rho_history, purities, means, parities = mcwf_solver(
        Heff=Heff,
        jump_ops=jump_ops,
        psi0=psi0,
        times=times,
        observables=observables,
        N_traj=N_traj,
        seed=seed,
        compute_rho=compute_rho
    )
    path = "results/figures/mcwf_observables_loss_dephasing.png"
    plot_results(times, purities=purities, means=means, parities=parities, path=path)
    print(f"{colorama.Fore.GREEN}Simulation complete! - results in {path}")
    print(f"Initial -> Purity: {purities[0]:.3f}, <n>: {means[0]:.3f}, Parity: {parities[0]:.3f}")
    print(f"Final   -> Purity: {purities[-1]:.3f}, <n>: {means[-1]:.3f}, Parity: {parities[-1]:.3f}")
    gif_path="results/figures/wigner_decoherence.gif"
    animate_wigner(
        states=rho_history,
        times=times,
        alpha=alpha,
        filename=gif_path,
        lim=wigner_lim,
        n_grid=wigner_pts,
        n_frames=n_frames,
        fps=fps,
    )

    print(f"{colorama.Fore.GREEN}Animation complete! - results in {gif_path}")


if __name__ == "__main__":
    process()




