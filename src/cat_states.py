"""Cat states and collapse operators."""
import numpy as np
import qutip as qt


def coherent_state(alpha, Ncut):
    """Normalized |alpha> in a truncated Fock space."""
    return qt.coherent(Ncut, alpha).unit()


def cat_state(alpha, Ncut, parity="even"):
    """Normalized |alpha> ± |-alpha>. parity: 'even' (+) or 'odd' (-)."""
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'")

    if alpha == 0: #for odd it aproximates |0> and for even it approximates to |1>
        fock_number =0 if parity =="even" else 1
        return qt.basis(Ncut,fock_number) 

    sign = 1 if parity == "even" else -1
    psi = qt.coherent(Ncut, alpha) + sign * qt.coherent(Ncut, -alpha)
    return psi.unit()


def loss_op(kappa, Ncut):
    """Photon loss (amplitude damping): L = sqrt(kappa) a."""
    return np.sqrt(kappa) * qt.destroy(Ncut)


def dephasing_op(gamma, Ncut):
    """Pure dephasing: L = sqrt(gamma) a†a."""
    return np.sqrt(gamma) * qt.num(Ncut)


def check_truncation(psi, tol=1e-6):
    """Population in the last five Fock levels; should be negligible."""
    p = np.abs(psi.full().ravel()) ** 2
    return p[-5:].sum() < tol