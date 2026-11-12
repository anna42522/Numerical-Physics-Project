import numpy as np
def create_operators(Ncut):
    """
    Construct the basic quantum operators 
    in a truncated Fock space (instead of 
    a Hilbert space with inifinite dimensions, 
    a Hilbert space with dimension Ncut).
     
    Parameters
    ----------
    Ncut: int
        Dimension of the truncated Hilbert space. 
    
    Returns
    -------
    a: ndarray
       Annihilation operator

    adag: ndarray
       Creation operator
    
    n: ndarray
       Number operator
    
    parity: ndarray
        Parity operator  
    
    """

    # Annihilation operator
    a = np.zeros((Ncut,Ncut),dtype=complex) #create a complex matrix and fill with 0

    for k in range (1, Ncut):
        a[k-1,k]=np.sqrt(k)

    # Creation operator
    adag = a.conj().T

    #Number operator
    n = adag @ a

    #Parity operator
    parity = np.diag((-1.0)**np.arange(Ncut))

    return a, adag, n, parity



def conju_T(op):

    #Construct the conjugate transpose of an operator.

    return op.conj().T


def construct_rho(cat):
    "cat will be a qutip Qobj, and we want to construct the density matrix rho = |cat><cat|"
    psi=cat.full().ravel()  # Convert the Qobj to a 1D numpy array
    return np.outer(psi, conju_T(psi))