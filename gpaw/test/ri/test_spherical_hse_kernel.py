"""

Test the implementation of spherical harmonic expansion of screened Coulomb kernel.
Based on

    János G Ángyán et al 2006 J. Phys. A: Math. Gen. 39 8613


"""


from gpaw.xc.ri.spherical_hse_kernel import Phi
from scipy.special import erfc
import numpy as np
from gpaw.sphere.lebedev import weight_n, Y_nL, R_nv
from gpaw.spherical_harmonics import Y


# [23, 31, 16] are indices to a triangle in the 50-point Lebedev grid.
# This is used to align the two angular grids nicely to avoid divergences.
Rdir_v = np.mean(R_nv[[23, 31, 16], :], axis=0)
# Obtain the tangent and bitangent vectors for a full basis
Q_vv, _ = np.linalg.qr(np.array([Rdir_v]).T, 'complete')

# Get the rotated second angular integration grid
R2_nv = R_nv @ Q_vv


def Phiold(n, mu, R, r):
    """

        Explicit implementation of spherical harmonic expansion up to l=2
        as given by the article Eqs. A1, A2 and A3. These are compared to
        the official implementation in `xc/ri/spherical_hse_kernel.py`.

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


def PhiLebedev(n, mu, R_x, r_x):
    # Target spherical harmonic, primary grid
    Y1_n = Y_nL[:, n**2]
    # Target spherical harmonic, secondary grid
    Y2_n = Y(n**2, *R2_nv.T)

    V_x = np.zeros_like(R_x)
    for x, (R, r) in enumerate(zip(R_x, r_x)):
        C1_nv = R * R_nv
        C2_nv = r * R2_nv

        D_nn = np.sum((C1_nv[:, None, :] - C2_nv[None, :, :])**2, axis=2)**0.5
        V_nn = erfc(D_nn*mu) / D_nn

        V_x[x] = np.einsum('n,m,nm,n,m', weight_n, weight_n, V_nn, Y1_n, Y2_n) 

    return V_x * (4 * np.pi) * (2 * n + 1)

def test_old_vs_new_spherical_kernel():
    """
        Test the explicityly hard coded implementation with the generic implementation.
    """
    for n in range(3):
        R = np.random.rand(100)*10
        r = np.random.rand(100)*10
        params = (n, 0.11, R, r)
        new, old = Phi(*params), Phiold(*params)
        assert np.allclose(new, old)

def test_wrt_lebedev_integrated_kernel():
    """
        Test a double angular numerically integrated kernel with the generic implementation.
    """
    import matplotlib.pyplot as plt
    s = 125
    for n in range(5):
        for RR in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
            R = RR*np.ones((5000,)) #np.random.rand(1000)*10
            r = np.logspace(-5, 3, 5001)[1:] # np.random.rand(1000)*10
            params = (n, 0.11, R.ravel(), r.ravel())
            new, old = Phi(*params), PhiLebedev(*params)
            #params = (n, 0.11, R.ravel(), r.ravel())
            #new2, old2 = Phi(*params), PhiLebedev(*params)
            #new -= new2
            #old -= old2
            #plt.semilogx(r, old, '-r')        
            #plt.semilogx(r, new, '--b')
            #plt.semilogx(r, np.abs(old-new), '-k')
            plt.loglog(r, np.abs(old), '-r')        
            plt.loglog(r, np.abs(new), '--b')
            plt.loglog(r, np.abs(old-new), '-k')
            plt.ylim([1e-7, 1e7])
        plt.show()

    for n in range(5):    
        t = np.logspace(-5,5, s)
        R, r = np.meshgrid(t,t)
        params = (n, 0.11, R.ravel(), r.ravel())
        new, old = Phi(*params), PhiLebedev(*params)
        plt.contourf(np.log10(r), np.log10(R), np.reshape(np.log10(np.abs(old-new)+1e-7), (s, s) ))
        plt.colorbar()
    
    
        #assert np.allclose(new, old)
        plt.show()
    
"""
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



"""
