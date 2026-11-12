import numpy as np
import scipy 
import qutip 
from operators import conju_T, construct_rho, create_operators

"""Want to obtain the function at t + dt """
def schrodinger_rhs(Heff, psi, hbar=1.0):
    """Right-hand-side of the Schrodinger equation"""
    return -1j / hbar * (Heff @ psi)

def RK4_step_sch(Heff, psi, dt, hbar=1.0):
    """
    Performs a single Runge-Kutta 4th order (RK4) step for Schrodinger equation.
    """
    k1 = schrodinger_rhs(Heff, psi, hbar=1.0)
    k2 = schrodinger_rhs(Heff, psi + 0.5 * dt * k1, hbar=1.0)
    k3 = schrodinger_rhs(Heff, psi + 0.5 * dt * k2, hbar=1.0)
    k4 = schrodinger_rhs(Heff, psi + dt * k3, hbar=1.0)

    return psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
