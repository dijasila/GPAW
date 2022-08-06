"""

Implementation of spherical harmonic expansion of screened Coulomb kernel.
Based on

    János G Ángyán et al 2006 J. Phys. A: Math. Gen. 39 8613


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from math import factorial
from gpaw.sphere.lebedev import weight_n, Y_nL, R_nv
from gpaw.spherical_harmonics import Y


def Hn(n, Xi, xi):
    """

    Helper function (Eq. 24)

    """
    return 1 / (2*(xi*Xi)**(n+1)) * ((Xi**(2*n+1)+xi**(2*n+1))*erfc(Xi+xi) - (Xi**(2*n+1)-xi**(2*n+1))*erfc(Xi-xi))  # noqa: E501


def Fn(n, Xi, xi):
    """

        Helper function (Eq. 22).

        It appears, that the article has a typo, because the summation
        starts at p=1, but correct results require to start at p=0.

    """
    prefactor = 2 / np.pi**0.5
    result = 0.0
    for p in range(0, n+1):
        result += (-1 / (4 * Xi * xi))**(p + 1) * factorial(n + p) / (factorial(p) * factorial(n - p)) * ((-1)**(n - p) * np.exp(-(xi + Xi)**2)-np.exp(-(xi - Xi)**2))  # noqa: E501
    return prefactor * result


def Phi(n, mu, R, r):
    """

        The official spherical kernel expansion

    """
    Rg = np.maximum.reduce([R, r])
    Rl = np.minimum.reduce([R, r])

    # Scaling as given by Eq. 16 and the text above.
    Xi = mu*Rg
    xi = mu*Rl

    # Eq. 21
    result = Fn(n, Xi, xi) + Hn(n, Xi, xi)
    for m in range(1, n+1):
        result += Fn(n-m, Xi, xi)*(Xi**(2*m)+xi**(2*m))/(xi*Xi)**m
    return result * mu
