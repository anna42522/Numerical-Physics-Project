import numpy as np 
import pytest

from src.config import load_config
from src.operators import create_operators, get_jump_ops
from src.hamiltonian import create_H0, create_Heff
from src.cat_states import cat_state
from src.mcwf import mcwf_trajectory

"""Check that every state in a trajectory is normalized when there are 
    jumps no jumps"""
@pytest.mark.week2
@pytest.mark.norm

def test_trajectory_states_are_normalized_with_jumps():
    config = load_config()
    ncut=config["system"]["N_cut"]
    kappa=config["environment"]["kappa"]
    gamma=config["environment"]["gamma"]
    alpha = config["system"]["alpha"]
    cat_parity = config["system"]["cat_parity"]
    seed = config["simulation"]["random_seed"]
    a, adag, n, parity = create_operators(ncut)
    jump_ops = get_jump_ops(kappa=kappa, gamma=gamma, a=a, n=n)

    H0=create_H0(Ncut=ncut,omega=0.0)
    Heff = create_Heff(H0, jump_ops)

    psi0 = cat_state(alpha=alpha,Ncut=ncut,parity=cat_parity).full().ravel()
    
    times=np.linspace(0.0, 10.0, 1001)

    psi_history, jump_times = mcwf_trajectory(
        Heff=Heff,
        jump_ops=jump_ops,
        rng=np.random.default_rng(seed=seed),
        psi0=psi0,
        times=times,
    )

    assert len(jump_times) > 0 #we want to be sure that there has been jumps
    norms = np.sum(np.abs(psi_history) ** 2, axis=1) #calculate the norm of esach state in the trajectory
    np.testing.assert_allclose(norms, 1.0, rtol=1e-9, atol=1e-9) #check that norm=1

"""Test that when the system looses a photon it changes parity.
    IMP: this test assumes that we start with parity even"""

@pytest.mark.week2

def test_photon_loss_flips_cat_state_parity():
    config=load_config()
    ncut = config["system"]["N_cut"]
    alpha = config["system"]["alpha"]
    seed = config["simulation"]["random_seed"]
    
    # we are only testing for phton loss
    kappa = 1.0
    gamma = 0.0

    a, _, n, parity_op = create_operators(ncut)
    jump_ops = get_jump_ops(kappa=kappa, gamma=gamma, a=a, n=n)

    H0 = create_H0(Ncut=ncut, omega=0.0)
    Heff = create_Heff(H0, jump_ops)

    psi0 = cat_state(
        alpha=alpha,
        Ncut=ncut,
        parity="even",
    ).full().ravel()

    times = np.linspace(0.0, 5.0, 2501)
    

    psi_history, jump_times = mcwf_trajectory(
        Heff=Heff,
        jump_ops=jump_ops,
        rng=np.random.default_rng(seed=seed),
        psi0=psi0,
        times=times,
    )

    #there has to be at least one jump
    assert jump_times 

    for jump_number,jump_time in enumerate(jump_times,start=1): #go through all jumps
        state_index=np.searchsorted(times,jump_time) #searches the position in times of jump_time  
        psi_after= psi_history[state_index] #obtain the state after the jump

        expected_parity = -1.0 if jump_number % 2 else 1.0 
        parity_after = np.vdot(psi_after, parity_op @ psi_after).real # <phi|parity|phi>
        
        assert np.isclose(parity_after, expected_parity, atol=1e-8)

    