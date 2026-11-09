import numpy as np
import scipy 
import qutip 
from operators import conju_T
#it returns: the time derivative of the density matrix
def lindblad_function(H, jump_ops, rho):
    dissipator = 0
    for l in jump_ops:
        dissipator += l @ rho @ conju_T(l) - 0.5 * (conju_T(l) @ l @ rho + rho @ conju_T(l) @ l)
    return -1j * (H @ rho - rho @ H) + dissipator  

