"""Cat states and collapse operators."""
import numpy as np
import qutip as qt

N_CUT = 60  # Fock-space truncation

def coherent_state(alpha, N=N_CUT):
    """Normalized |alpha>."""
    return qt.coherent(N, alpha).unit()
def cat_state(alpha, parity="even", N=N_CUT):
    """Normalized |alpha> ± |-alpha>. parity: 'even' (+) or 'odd' (-)."""
    sign = 1 if parity == "even" else -1
    psi = qt.coherent(N, alpha) + sign * qt.coherent(N, -alpha)
    return psi.unit()


def loss_op(kappa, N=N_CUT):
    """Photon loss (amplitude damping): L = sqrt(kappa) a."""
    return np.sqrt(kappa) * qt.destroy(N)


def dephasing_op(gamma, N=N_CUT):
    """Pure dephasing: L = sqrt(gamma) a^dagger a."""
    return np.sqrt(gamma) * qt.num(N)


def check_truncation(psi, tol=1e-6):
    """Population in the last 5 Fock levels; should be negligible."""
    p = np.abs(psi.full().ravel()) ** 2
    return p[-5:].sum() < tol
