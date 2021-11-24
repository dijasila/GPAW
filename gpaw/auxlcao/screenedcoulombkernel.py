import numpy as np
from scipy.special import erfc

def Phi0(Xi, xi):
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
            self.V_lgg.append(d_gg * F_l[l](rmax_gg, rmin_gg, omega) / (2*l+1))
        
    def screened_coulomb(self, n_g, l):
        return self.V_lgg[l] @ n_g
