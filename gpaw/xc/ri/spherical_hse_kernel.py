import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from math import factorial


def Phiold(n, mu, R, r):
    Rg = np.maximum.reduce([R, r])
    Rl = np.minimum.reduce([R, r])
    Xi = mu*Rg
    xi = mu*Rl
    if n == 0:
        prefactor = -1 / (2 * np.pi**0.5 * xi * Xi)
        A = np.exp(-(xi+Xi)**2)-np.exp(-(xi-Xi)**2)
        B = -np.pi**0.5*((xi-Xi)*erfc(Xi-xi) +
                         (Xi+xi)*erfc(Xi+xi))
        return mu*prefactor*(A+B)
    if n == 1:
        prefactor = -1 / (2 * np.pi**0.5 * xi**2 * Xi**2)
        A = 1/2*((np.exp(-(xi+Xi)**2) - np.exp(-(xi-Xi)**2))*(2*xi**2+2*xi*Xi-(1-2*Xi**2))-4*xi*Xi*np.exp(-(xi+Xi)**2))-np.pi**0.5*((xi**3-Xi**3)*erfc(Xi-xi)+(xi**3+Xi**3)*erfc(xi+Xi))  # noqa: E501
        return mu*prefactor*A
    if n == 2:
        prefactor = -1 / (2 * np.pi**0.5 * xi**3 * Xi**3)
        A = 1/4*((np.exp(-(xi+Xi)**2)-np.exp(-(xi-Xi)**2))*(4*(xi**4+xi**3*Xi+Xi**4)-2*xi**2*(1-2*Xi**2)+(1-2*xi*Xi)*(3-2*Xi**2))-4*np.exp(-(xi+Xi)**2)*xi*Xi*(2*xi**2-(3-2*Xi**2)))-np.pi**0.5*((xi**5-Xi**5)*erfc(Xi-xi)+(xi**5+Xi**5)*erfc(xi+Xi))  # noqa: E501
        return mu*prefactor*A
    raise NotImplementedError


def Hn(n, Xi, xi):
    return 1 / (2*(xi*Xi)**(n+1)) * ((Xi**(2*n+1)+xi**(2*n+1))*erfc(Xi+xi) - (Xi**(2*n+1)-xi**(2*n+1))*erfc(Xi-xi))  # noqa: E501


def Fn(n, Xi, xi):
    prefactor = 2 / np.pi**0.5
    result = 0.0
    for p in range(0, n+1):
        result += (-1 / (4 * Xi * xi))**(p + 1) * factorial(n + p) / (factorial(p) * factorial(n - p)) * ((-1)**(n - p) * np.exp(-(xi + Xi)**2)-np.exp(-(xi - Xi)**2))  # noqa: E501
    return prefactor * result


def Phi(n, mu, R, r):
    Rg = np.maximum.reduce([R, r])
    Rl = np.minimum.reduce([R, r])
    Xi = mu*Rg
    xi = mu*Rl
    result = Fn(n, Xi, xi) + Hn(n, Xi, xi)
    for m in range(1, n+1):
        result += Fn(n-m, Xi, xi)*(Xi**(2*m)+xi**(2*m))/(xi*Xi)**m
    return result * mu


if __name__ == "__main__":
    Xi = np.linspace(0, 10, 1001)[1:]
    xi = np.ones((1000, ))
    mu = 0.15

    for n in range(3):
        y = Phiold(n, mu,  xi*(n+1), Xi)
        plt.plot(Xi, y, 'xk')

        y = Phi(n, mu, xi*(n+1), Xi)
        plt.plot(Xi, y, '.r')
    plt.show()
