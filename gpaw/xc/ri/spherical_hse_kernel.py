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


def Phiold(n, mu, R, r):
    """

        Explicit implementation as given by the article Eqs. A1, A2 and A3.
        These are compared to the official implementation in the test suite.

    """
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


class ScreenedCoulombKernelDebug:
    def __init__(self, rgd, omega):
        self.rgd = rgd
        h = 0.01
        r1_gg = np.zeros((rgd.N, rgd.N))
        r2_gg = np.zeros((rgd.N, rgd.N))
        d_gg = np.zeros((rgd.N, rgd.N))
        r_g = rgd.r_g.copy()
        r_g[0] = r_g[1]
        r1_gg[:] = r_g[None, :]
        r2_gg[:] = r_g[:, None]
        rmin_gg = np.where(r1_gg < r2_gg, r1_gg, r2_gg)
        rmax_gg = np.where(r1_gg < r2_gg, r2_gg, r1_gg)
        d_gg[:] = rgd.dr_g[None, :] * rgd.r_g[None, :]**2 * 4*np.pi
        self.V_lgg = []

        for ll in range(3):
            kernel_gg = Phiold(ll, rmax_gg, rmin_gg, omega) / (2*ll+1)
            self.V_lgg.append(d_gg * kernel_gg)

        Rdir_v = np.mean(R_nv[[23, 31, 16], :], axis=0)
        Q_vv, _ = np.linalg.qr(np.array([Rdir_v]).T, 'complete')
        R2_nv = R_nv @ Q_vv

        Y2_nL = np.zeros((50, 25))
        n = 0
        for x, y, z in R2_nv:
            Y2_nL[n] = [Y(L, x, y, z) for L in range(25)]
            n += 1

        def solve(r, L=0, L2=0):
            v_g = np.zeros((rgd.N,))
            w_n = (weight_n * Y_nL[:, L]).copy()
            for n2, R2_v in enumerate(R2_nv):
                for g, r2 in enumerate(rgd.r_g):
                    D_n = np.sum((R_nv*r - r2*R2_v[None, :])**2, axis=1)**0.5
                    D2_n = np.where(D_n < h, h, D_n)
                    V_n = erfc(D_n*omega) / D2_n
                    v_g[g] += weight_n[n2] * Y2_nL[n2, L2] * np.sum(w_n * V_n)
            return 4*np.pi*v_g

        import hashlib
        import pickle

        try:
            f = open('screened_kernel'+
                    hashlib.md5(rgd.xml().encode('utf-8')).hexdigest()+
                    '.pckl','rb')
            self.V_lgg = pickle.load(f)
        except IOError:
            pass

        for l in range(len(self.V_lgg), 3):
            V_gg = np.zeros( (rgd.N, rgd.N) )
            for g, r in enumerate(rgd.r_g):
                V_gg[g,:] = d_gg[g,:] * solve(rgd.r_g[g], L=l**2, L2=l**2)
            self.V_lgg.append(V_gg)

            with open('screened_kernel' + hashlib.md5(rgd.xml().encode('utf-8')).hexdigest()+'.pckl','wb') as f:
                pickle.dump(self.V_lgg, f)

    def screened_coulomb(self, n_g, l):
        v_g = self.V_lgg[l] @ n_g
        return v_g

if __name__ == '__main__':
    from gpaw.atom.radialgd import AERadialGridDescriptor
    N = 150
    rgd = AERadialGridDescriptor(0.4 / N, 1 / N, N)
    l = 1
    sck = ScreenedCoulombKernelDebug(rgd, 0.11)
    import matplotlib.pyplot as plt
    n_g = rgd.r_g*0
    n_g[40] = 1.0 / rgd.dv_g[40]
    v_g = sck.screened_coulomb(n_g, l)
    plt.plot(rgd.r_g, v_g,'x-',label='hse',linewidth=4)
    plt.plot(rgd.r_g, n_g,'o',label='density')
    v2_g = rgd.poisson(n_g,l)
    v2_g[1:] /= rgd.r_g[1:]
    v2_g[0] = v2_g[1]
    #plt.plot(rgd.r_g, v2_g,'+',label='poisson')
    #plt.plot(rgd.r_g, (v2_g-v_g),'-',label='diff x 100')
    plt.show()
    """
    data = []
    L_l = [ 0, 3 ]
    for r in rgd.r_g:
        W_LL = np.zeros((9,9))
        generated_W_LL_screening(W_LL, np.array( [[ r ]] ), np.array( [[ r ]] ), np.array([[ 0.0 ]]), np.array([[0.0]]), 0.11)
        data.append(W_LL[0,L_l[l]] / 10)
        print(W_LL)
    plt.plot(rgd.r_g, data,'o--',label='generated')
    plt.legend()
    plt.show()
    """
    #xxx

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

