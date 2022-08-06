"""

Implementation of spherical harmonic expansion of screened Coulomb kernel.
Based on

    János G Ángyán et al 2006 J. Phys. A: Math. Gen. 39 8613


"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, comb
from math import factorial


def doublefactorial(n):
     if n <= 0:
         return 1
     else:
         return n * doublefactorial(n-2)


def safeerfc(x):
    #return erfc(x)
    taylor = 1 - 2 * x / np.pi**0.5 + 2 * x**3 / (3*np.pi**0.5) - x**5 / (5*np.pi)**0.5
    return np.where(x < 1e-4, taylor, erfc(x))


def Dnk(n, k, Xi):
    # Eq. 28
    if k == 0:
        sum = 0
        for m in range(1, n+1):
            sum += 2**(-m)*Xi**(-2*m) / doublefactorial(2*n-2*m+1)
        return safeerfc(Xi) + np.exp(-Xi**2) / (np.pi**0.5)*2**(n+1)*Xi**(2*n+1) * sum
    # Eq. 29
    sum = 0
    for m in range(1, k+1):
        sum += comb(m-k-1, m-1)*2**(k-m)*Xi**(2*(k-m))  / doublefactorial(2*n+2*k-2*m+1)

    return np.exp(-Xi**2) * 2**(n+1)*(2*n+1)*Xi**(2*n+1) / np.pi**0.5 / factorial(k) / (2*n+2*k+1) * sum


def Phinj(n, j, Xi, xi):
    # Eq. 30
    sum = 0
    for k in range(j):
        sum += Dnk(n, k, Xi) / (Xi**(n+1))*xi**(n+2*k)
    return sum

def Hn(n, Xi, xi):
    """

    Helper function (Eq. 24)

    """
    return 1 / (2*(xi*Xi)**(n+1)) * ((Xi**(2*n+1)+xi**(2*n+1))*safeerfc(Xi+xi) - (Xi**(2*n+1)-xi**(2*n+1))*safeerfc(Xi-xi))  # noqa: E501


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
    taylor =np.exp(-Xi**2-xi**2) * 2**(n+1)*(3+2*n+2*xi**2*Xi**2)*xi**n*Xi**n/(np.pi**0.5*doublefactorial(2*n+3))
    #match = 0
    #for x1, x2, a, b in zip(Xi, xi, taylor, prefactor * result):
    #    if x1 < 4 and x2 < 4:
    #        print(n,x1,x2,a,b, b/a)
    #        match += 1
    #print(match,'matches')
    #print((Xi < 1e-2) & (xi < 1e-2),'xxx')

    return np.where((Xi * xi)**(2 * n + 1) < 1e-8,  taylor, prefactor * result)


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


    result = np.where( xi < 0.4, Phinj(n, 2, Xi, xi), result)
    result *= mu

    return result
