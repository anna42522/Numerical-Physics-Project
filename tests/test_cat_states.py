import pytest
import numpy as np
import qutip as qt
from src.cat_states import cat_state, N_CUT


@pytest.mark.week1
def test_normalization():
    for p in ("even", "odd"):
        assert abs(cat_state(2.0, p).norm() - 1) < 1e-10


@pytest.mark.week1
def test_parity():
    even, odd = cat_state(2.0, "even"), cat_state(2.0, "odd")
    P = (1j * np.pi * qt.num(N_CUT)).expm()
    assert abs(qt.expect(P, even) - 1) < 1e-6
    assert abs(qt.expect(P, odd) + 1) < 1e-6
