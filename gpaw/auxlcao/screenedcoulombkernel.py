import numpy as np
from scipy.special import erfc

from gpaw.sphere.lebedev import weight_n, Y_nL, R_nv
from gpaw.spherical_harmonics import Y


def Phi0(Xi, xi):
    print(-1/(2*np.pi**0.5*xi*Xi))
    return -1/(2*np.pi**0.5*xi*Xi) * ( ( np.exp(-(xi+Xi)**2) - np.exp(-(xi-Xi)**2 ) \
        -np.pi**0.5*((xi-Xi)*erfc(Xi-xi)+(Xi+xi)*erfc(xi+Xi))))

def Phi1(Xi, xi):
    Phi = -1/(2*np.pi**0.5*xi**2*Xi**2)
    Phi *= 1/2*(( np.exp(-(xi+Xi)**2) - np.exp(-(xi-Xi)**2 ) ) \
        * (2*xi**2 + 2*xi*Xi-(1-2*Xi**2)) - 4*xi*Xi*np.exp(-(xi+Xi)**2) ) -np.pi**0.5*((xi**3-Xi**3)* \
           erfc(Xi-xi)+(xi**3+Xi**3)*erfc(xi+Xi))
    return Phi

def Phi2(Xi, xi):
    Phi = -1/(2*np.pi**0.5*xi**3*Xi**3) * \
           (1/4*((np.exp(-(xi+Xi)**2)-np.exp(-(xi-Xi)**2))*(4*(xi**4+xi**3*Xi+Xi**4) \
            -2*xi**2*(1-2*Xi**2) + (1-2*xi*Xi)*(3-2*Xi**2))-4*np.exp(-(xi+Xi)**2)*xi*Xi*(2*xi**2-(3-2*Xi**2))) \
              -np.pi**0.5*((xi**5-Xi**5)*erfc(Xi-xi)+(xi**5+Xi**5)*erfc(xi+Xi)))
    return Phi

def F0(R,r,mu):
    return mu*Phi0(mu*R, mu*r)

def F1(R,r,mu):
    return mu*Phi1(mu*R, mu*r)

def F2(R,r,mu):
    return mu*Phi2(mu*R, mu*r)

F_l = [ F0, F1, F2 ]

class ScreenedCoulombKernel:
    def __init__(self, rgd, omega):
        self.rgd = rgd
        h = 0.01
        r1_gg = np.zeros( (rgd.N, rgd.N) )
        r2_gg = np.zeros( (rgd.N, rgd.N) )
        d_gg = np.zeros( (rgd.N, rgd.N) )
        r_g = rgd.r_g.copy()
        r_g[0] = r_g[1] # XXX
        r1_gg[:] = r_g[None, :] 
        r2_gg[:] = r_g[:, None] 
        rmin_gg = np.where(r1_gg<r2_gg, r1_gg, r2_gg)
        rmax_gg = np.where(r1_gg<r2_gg, r2_gg, r1_gg)
        d_gg[:] = rgd.dr_g[None ,:] * rgd.r_g[None,:]**2 * 4*np.pi
        self.V_lgg = []

        for l in range(3):
            kernel_gg = F_l[l](rmax_gg, rmin_gg, omega) / (2*l+1)
            self.V_lgg.append(d_gg * kernel_gg)

        Rdir_v = np.mean(R_nv[[23,31,16],:],axis=0)
        Q_vv, _ = np.linalg.qr(np.array([ Rdir_v]).T,'complete')
        R2_nv = R_nv @ Q_vv

        Y2_nL = np.zeros((50, 25))
        n = 0
        for x, y, z in R2_nv:
            Y2_nL[n] = [Y(L, x, y, z) for L in range(25)]
            n += 1

        def solve(r, L=0, L2 =0):
            v_g = np.zeros((rgd.N,))
            for n2, R2_v in enumerate(R2_nv):
                for g, r2 in enumerate(rgd.r_g):
                    D_n = np.sum((R_nv*r - r2*R2_v[None, :])**2, axis=1)**0.5
                    D2_n = np.where(D_n < h, h, D_n)
                    V_n = erfc(D_n*omega) / D2_n
                    v_g[g] += weight_n[n2] * Y2_nL[n2, L2] * np.sum(weight_n * Y_nL[:, L] * V_n)
            return 4*np.pi*v_g

        def solvef(r, L=0, L2 =0):
            v_g = np.zeros((rgd.N,))
            w_n = (weight_n * Y_nL[:, L]).copy()
            for n2, R2_v in enumerate(R2_nv):
                for g, r2 in enumerate(rgd.r_g):
                    D_n = np.sum((R_nv*r - r2*R2_v[None, :])**2, axis=1)**0.5
                    D2_n = np.where(D_n < h, h, D_n)
                    V_n = erfc(D_n*omega) / D2_n
                    v_g[g] += weight_n[n2] * Y2_nL[n2, L2] * np.sum(w_n  * V_n)
            return 4*np.pi*v_g

        import hashlib
        import pickle


        try:
            f = open('screened_kernel' + hashlib.md5(rgd.xml().encode('utf-8')).hexdigest()+'.pckl','rb')
            self.V_lgg = pickle.load(f)
        except IOError:
            pass

        for l in range(len(self.V_lgg), 3):
            V_gg = np.zeros( (rgd.N, rgd.N) )
            for g, r in enumerate(rgd.r_g):
                print(l,g, len(rgd.r_g))
                V_gg[g,:] = d_gg[g,:] * solvef(rgd.r_g[g], L=l**2, L2=l**2)
                #print ( solve(rgd.r_g[g], L=l**2, L2=l**2)  - solvef(rgd.r_g[g], L=l**2, L2=l**2) )
                #assert np.linalg.norm( solve(rgd.r_g[g], L=l**2, L2=l**2)  - solvef(rgd.r_g[g], L=l**2, L2=l**2) ) <1e-6
                #print(l,  (self.V_lgg[l][g,:] - V_gg[g,:]).ravel() )
            self.V_lgg.append(V_gg)

            with open('screened_kernel' + hashlib.md5(rgd.xml().encode('utf-8')).hexdigest()+'.pckl','wb') as f:
                pickle.dump(self.V_lgg, f)
        
    def screened_coulomb(self, n_g, l):
        v_g = self.V_lgg[l] @ n_g
        return v_g

if __name__ == '__main__':
    from gpaw.atom.radialgd import AERadialGridDescriptor
    from gpaw.auxlcao.generatedcode import generated_W_LL_screening
    N = 150
    rgd = AERadialGridDescriptor(0.4 / N, 1 / N, N)
    l = 1
    sck = ScreenedCoulombKernel(rgd, 0.11)
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
