import numpy as np
import pytest

from src.operators import create_operators
from src.config import load_config

#Load simulation configuration 
config =load_config()

NCUT = config["system"]["N_cut"]

@pytest.fixture
def operators():
    return create_operators(NCUT)

@pytest.mark.week1
def test_operator_dimensions(operators):
    a, adag, n, parity = operators

    assert a.shape == (NCUT, NCUT)
    assert adag.shape == (NCUT, NCUT)
    assert n.shape == (NCUT, NCUT)
    assert parity.shape == (NCUT, NCUT)

@pytest.mark.week1
def test_creation_is_adjoint_of_annihilation(operators):
    a, adag, n, parity = operators

    assert np.allclose(adag, a.conj().T)

@pytest.mark.week1
def test_number_operator(operators):
    a, adag, n, parity = operators

    expected = np.diag(np.arange(NCUT))

    assert np.allclose(n, expected)

@pytest.mark.week1
def test_commutation_relation(operators):
    a, adag, n, parity = operators

    commutator = a @ adag - adag @ a 

    expected = np.eye(NCUT)

    expected[-1,-1] = 1 - NCUT

    np.testing.assert_allclose(commutator, expected)

@pytest.mark.norm
def test_hermicity(operators):
    a, adag, n, parity = operators

    assert np.allclose(n, n.conj().T)
    assert np.allclose(parity, parity.conj().T)

@pytest.mark.week1
def test_parity_is_unitary(operators):
    a, adag, n, parity = operators 

    assert np.allclose(parity @ parity, np.eye(NCUT))

@pytest.mark.week1
def test_number_operator_consistency(operators):
    a, adag, n, parity = operators
    #verifies n= adag*a

    assert np.allclose(n, adag @ a )

@pytest.mark.week1
def test_annihilation_operator_action(operators):
    a, adag, n, parity = operators

    for k in range(1, NCUT):
        expected = np.sqrt(k)*np.eye(NCUT)[:, k-1]

        assert np.allclose(a[:,k], expected)



