
import numpy as np

from src.operators import conju_T, create_operators

def create_H0(Ncut, omega, hbar=1.0):
    """Create the hermitian Hamiltonian of our system
    H0 is the harmonic oscillator"""

    _, _, n, _ = create_operators(Ncut)

    H0= hbar * omega * (n+0.5*np.eye(Ncut))

    return H0

def create_Heff(H0, collapse_operators, hbar=1.0):
    """Create the effective Hamiltonian for the mcwf method.
    Set H0 = 0 if rotating frame (frame with w=0).
    H0 and collapse operators are NumPy matrices"""
    Heff = H0.astype(complex).copy()

    for L in collapse_operators:
        Heff = Heff - ( 1j * hbar/2 *conju_T(L) @ L)

    return Heff