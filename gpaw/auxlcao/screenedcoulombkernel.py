import numpy as np
from scipy.special import erfc

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
            #if omega < 0.01:
            #kernel_gg = rmax_gg**(-l-1)*rmin_gg**l / (2*l+1) # DEBUG! REMOVE ME!!!! XXX
            #else:
            kernel_gg = F_l[l](rmax_gg, rmin_gg, omega) / (2*l+1)
            self.V_lgg.append(d_gg * kernel_gg)
        
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
