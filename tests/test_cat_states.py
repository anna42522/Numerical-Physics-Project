import pytest
import numpy as np
import qutip as qt

from src.cat_states import cat_state, coherent_state, check_truncation
from src.config import load_config

CONFIG = load_config()
NCUT = CONFIG["system"]["N_cut"]
ALPHA = CONFIG["system"]["alpha"]
CAT_PARITY = CONFIG["system"]["cat_parity"]

@pytest.mark.week1
def test_cat_state_is_normalized():
    """Checks that the configured cat state has unit norm"""
    state=cat_state(alpha=ALPHA,Ncut=NCUT,parity=CAT_PARITY)
    assert abs(state.norm()-1)<1e-10

@pytest.mark.week1
@pytest.mark.parametrize(
    ("parity", "expected_value"),
    [("even", 1), ("odd", -1)],
)
def test_cat_state_has_correct_parity(parity,expected_value):
    """Checks that the cat state has the expected parity"""
    state=cat_state(alpha=ALPHA,Ncut=NCUT,parity=parity) #construct state
    parity_operator=(1j*np.pi*qt.num(NCUT)).expm() #construct parity operator, P=e^(i\piNcut)

    assert abs(qt.expect(parity_operator,state)-expected_value) < 1e-6

def test_coherent_state_is_normalized():
    """Checks that the configured coherent state has unit norm"""
    state=coherent_state(alpha=ALPHA,Ncut=NCUT)
    assert abs(state.norm()-1)<1e-10

def test_cat_state_truncation():
    """Checks that the configured cat state has negligible 
    population in the last five Fock levels"""
    state=cat_state(alpha=ALPHA,Ncut=NCUT,parity=CAT_PARITY)
    assert check_truncation(state)