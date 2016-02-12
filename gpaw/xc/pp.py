from __future__ import division
import numpy as np
from numpy import pi, log as ln, exp, sqrt
from ase.utils import seterr

CF = 0.6 * (6 * pi**2)**(2 / 3)


class LDA:
    type = 'LDA'
    name = 'LDA'
    
    def __init__(self):
        na, nb = symbols('na, nb', positive=True)
        n = na + nb
        zeta = (na - nb) / n
        lda = (lda_x(2 * na) + lda_x(2 * nb)) / 2 + n * lda_c_pw92(n, zeta)[0]
        ldaf = lda.evalf()
        self.e = lambdify((na, nb), ldaf, 'numpy')
        self.v = lambdify((na, nb),
                          (ldaf.diff(na), ldaf.diff(nb)),
                          'numpy')
        
    def calculate(self, e, n_sg, v, s=0, ds=0, t=0, dt=0):
        e[:] = 0.0
        c = 1e-20
        if len(n_sg) == 2:
            nn_sg = n_sg.copy()
            nn_sg[nn_sg < c] = c
            e[:] = self.e(nn_sg[0], nn_sg[1])
            v[:] += self.v(nn_sg[0], nn_sg[1])
        else:
            n_g = n_sg[0] / 2
            n_g[n_g < c] = c
            e[:] = self.e(n_g, n_g)
            v[0] += self.v(n_g, n_g)[0]

            
class PBE:
    type = 'GGA'
    name = 'PBE'
    
    def e(self, na, nb, a2a, a2ab, a2b):
        n = na + nb
        zeta = (na - nb) / n
        return (#(pbe_x(2 * na, 4 * a2a) +
                 #pbe_x(2 * nb, 4 * a2b)) / 2 +
                pbe_c(n, zeta, a2a + 2 * a2ab + a2b))
            
    def calculate(self, e, n_sg, v, a2_xg=0, ds=0, t=0, dt=0):
        e[:] = 0.0
        c = 1e-20
        if len(n_sg) == 2:
            nn_sg = n_sg.copy()
            nn_sg[nn_sg < c] = c
            args = nn_sg[0], nn_sg[1], a2_xg[0], a2_xg[1], a2_xg[2]
            e[:] = self.e(*args)
            D = []
            for i, arg in enumerate(args):
                darg = 0.000001 * arg
                arg = arg + 0.5 * darg
                newargs = list(args)
                newargs[i] = arg
                d = self.e(*newargs)
                arg -= darg
                d -= self.e(*newargs)
                darg[abs(arg) < c] = np.inf
                d /= darg
                D.append(d)
            v[:] += D[:2]
            ds[:] = D[2:]
        else:
            N_sg = np.empty((2,) + n_sg.shape[1:])
            V_sg = np.zeros((2,) + n_sg.shape[1:])
            A2_xg = np.empty((3,) + n_sg.shape[1:])
            D_xg = np.empty((3,) + n_sg.shape[1:])
            N_sg[:] = n_sg[0] / 2
            A2_xg[:] = a2_xg[0] / 4
            self.calculate(e, N_sg, V_sg, A2_xg, D_xg)
            v += V_sg[0]
            ds[:] = D_xg.sum(0) / 4
            if e.ndim == 33:
                n = e.shape[0] // 2
                import matplotlib.pyplot as plt
                plt.plot(e[n,n])
                plt.plot(a2_xg[0,n,n])
                plt.plot(n_sg[0,n,n])
                plt.plot(v[0,n,n])
                plt.plot(ds[0,n,n])
                #plt.show();asdgf
            
                
class M06_L:
    type = 'MGGA'
    name = 'M06-L'
    
    def e(self, na, nb, a2a, a2ab, a2b, taua, taub):
        return (m06_l_x(na, a2a, taua) +
                m06_l_x(nb, a2b, taub) +
                m06_l_c(na, nb, a2a, a2b, taua, taub))
            
    def calculate(self, e, n_sg, v, a2_xg=0, ds=0, t=0, dt=0):
        e[:] = 0.0
        c = 1e-20
        if len(n_sg) == 2:
            nn_sg = n_sg.copy()
            bad = nn_sg < c
            nn_sg[bad] = c
            
            tt = t.copy()
            badt = tt < c
            tt[badt] = c

            aa = a2_xg.copy()
            bada = aa < 1e-5
            aa[bada] = 0#1e-5
            
            args = (nn_sg[0], nn_sg[1],
                    aa[0], aa[1], aa[2], tt[0], tt[1])
            e[:] = self.e(*args)
            #e[np.logical_bad] = 0.0
            D = []
            for i, arg in enumerate(args):
                darg = 0.000001 * arg
                arg = arg + 0.5 * darg
                newargs = list(args)
                newargs[i] = arg
                d = self.e(*newargs)
                arg -= darg
                d -= self.e(*newargs)
                darg[abs(arg) < c] = np.inf
                d /= darg
                D.append(d)
            v[:] += D[:2]
            ds[:] = D[2:5]
            dt[:] = D[5:]
        else:
            N_sg = np.empty((2,) + n_sg.shape[1:])
            V_sg = np.zeros((2,) + n_sg.shape[1:])
            A2_xg = np.empty((3,) + n_sg.shape[1:])
            D_xg = np.empty((3,) + n_sg.shape[1:])
            T_sg = np.zeros((2,) + n_sg.shape[1:])
            W_sg = np.empty((2,) + n_sg.shape[1:])
            N_sg[:] = n_sg[0] / 2
            A2_xg[:] = a2_xg[0] / 4
            T_sg[:] = t[0] / 2
            self.calculate(e, N_sg, V_sg, A2_xg, D_xg, T_sg, W_sg)
            v += V_sg[0]
            ds[:] = D_xg.sum(0) / 4
            dt[:] = W_sg.sum(0) / 2
            if e.ndim == 33:
                n = 0#e.shape[0] // 2
                import matplotlib.pyplot as plt
                plt.plot(e[n,n])
                #plt.plot(a2_xg[0,n,n])
                #plt.plot(n_sg[0,n,n])
                plt.plot(v[0,n,n])
                plt.plot(n_sg[0,0,0])
                plt.plot(a2_xg[0,0,0])
                plt.plot(t[0,0,0])
                #plt.plot(ds[0,n,n])
                plt.show()
                plt.plot(e[:,n,0])
                plt.plot(v[0,:,n,0])
                plt.show()
            
                
def seitz_radius(n):
    return (3 / (4 * pi * n))**(1 / 3)
    
    
def lda_x(n):
    return -3 / 4 / pi * (3 * pi**2)**(1 / 3) * n**(4 / 3)
    
    
def lda_c_pw92(n, zeta):
    rs = seitz_radius(n)
    ec = G(sqrt(rs), 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = G(sqrt(rs), 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    alpha = -G(sqrt(rs), 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    zp = 1.0 + zeta
    zm = 1.0 - zeta
    xp = zp**(1 / 3)
    xm = zm**(1 / 3)
    CC1 = 1.9236610509315362
    IF2 = 0.58482236226346462
    f = CC1 * (zp * xp + zm * xm - 2)
    x = 1 - zeta**4
    return ec + alpha * IF2 * f * x + (e1 - ec) * f * zeta**4, xp, xm


def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2 * gamma * (1 + alpha1 * rtrs * rtrs)
    Q1 = 2 * gamma * rtrs * (beta1 +
                             rtrs * (beta2 +
                                     rtrs * (beta3 +
                                             rtrs * beta4)))
    return Q0 * ln(1 + 1 / Q1)


def pbe_x(n, a2):
    C2 = 0.26053088059892404
    kappa = 0.804
    mu = 0.2195149727645171
    rs = seitz_radius(n)
    c = (C2 * rs / n)**2
    s2 = a2 * c
    x = 1 + mu * s2 / kappa
    Fx = 1 + kappa - kappa / x
    return lda_x(n) * Fx

    
def pbe_c(n, zeta, a2):
    BETA = 0.06672455060314922
    GAMMA = 0.0310906908697
    C3 = 0.10231023756535741
    ec, xp, xm = lda_c_pw92(n, zeta)
    rs = seitz_radius(n)
    phi = (xp**2 + xm**2) / 2
    t2 = C3 * a2 * rs / (n**2 * phi**2)
    y = -ec / (GAMMA * phi**3)
    x = exp(y)
    A = BETA / (GAMMA * (x - 1))
    At2 = A * t2
    nom = 1 + At2
    denom = nom + At2**2
    H = GAMMA * ln(1 + BETA * t2 * nom / (denom * GAMMA)) * phi**3
    return n * (ec + H)


def m06_l_h(x2, z, alpha, d0, d1, d2, d3, d4):
    gamma = 1 + alpha * (x2 + z)
    #print(x2, z, gamma)
    return (d0 / gamma +
            (d1 * x2 + d2 * z) / gamma**2 +
            (d3 * x2**2 + d4 * x2 * z) / gamma**3)
    

def m06_l_x(n, a2, tau):
    taulda = 0.5 * CF * n**(5 / 3)
    invt = tau / taulda
    z = CF * (invt - 1)
    x2 = a2 / n**(8 / 3)
    h = m06_l_h(x2, z, 0.00186726,
                0.6012244, 0.004748822, -0.008635108, -0.000009308062,
                0.00004482811)
    w = (1 - invt) / (1 + invt)
    #print(tau, taulda, 1/invt, w)
    A = [0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615,
         30.97273, -23.18489, -56.73480, 21.60364, 34.21814, -9.049762]
    f = A[-1]
    for a in A[-2::-1]:
        f *= w
        f += a
        
    #print('fh:', f, h)
    #print('f:', f, sum(a*w**i for i, a in enumerate(A)))
    #print('rttw', n, tau, 1 / invt, w)
    #print('X:', w, f, lda_x(2 * n)/2, h)
    return (pbe_x(2 * n, 4 * a2) * f + lda_x(2 * n) * h) / 2
    
    
def m06_l_g(nom, c):
    b = nom / (1 + nom)
    g = c[4]
    for ci in c[3::-1]:
        g *= b
        g += ci
    return g
    
    
def m06_l_c(na, nb, aa2, ab2, taua, taub):
    zap = 2 * taua / na**(5 / 3)
    zbp = 2 * taub / nb**(5 / 3)
    xa2 = aa2 / na**(8 / 3)
    xb2 = ab2 / nb**(8 / 3)
    xab2 = xa2 + xb2

    n = na + nb
    zeta = (na - nb) / n
    uaa = lda_c_pw92(na, 1)[0] * na
    ubb = lda_c_pw92(nb, 1)[0] * nb
    uab = lda_c_pw92(n, zeta)[0] * n - uaa - ubb
    #print(uaa,ubb,uab+uaa+ubb)
    hab = m06_l_h(xab2, zap + zbp - 2 * CF, 0.00304966,
                  0.3957626, -0.5614546, 0.01403963, 0.0009831442,
                  -0.003577176)
    gab = m06_l_g(0.0031 * xab2,
                  [0.6042374, 177.6783, -251.3252, 76.35173, -12.55699])

    #print(uab, gab, hab, xab2, za, zb)
    e = uab * (gab + hab)

    i = 0
    for x2, zp, u in [(xa2, zap, uaa), (xb2, zbp, ubb)]:
        h = m06_l_h(x2, zp - CF, 0.00515088,
                    0.4650534, 0.1617589, 0.1833657, 0.0004692100,
                    -0.004990573)
        g = m06_l_g(0.06 * x2,
                    [0.5349466, 0.5396620, -31.61217, 51.49592, -29.19613])

        #with seterr(divide='ignore'):
        try:
            D = 1 - x2 / 4 / zp
        except FloatingPointError:
            print(taua[0,0])
            print((x2/4)[0,0])
            print(zp[0,0])
            import matplotlib.pyplot as plt
            plt.plot((x2/4)[0,0])
            plt.plot(zp[0,0])
            plt.plot((x2/4)[:,0,0])
            plt.plot(zp[:,0,0])
            plt.show()
        D[zp <= x2 / 4] = 0
        e += u * (g + h) * D
        i += 1
    
    #print(g,h,D)
    return e
    
    
if __name__ == '__main__':
    xc = M06_L()
    e, n, v, t, w = np.zeros((5, 1, 1))
    s, d = np.zeros((2, 1, 1))
    n[:] = 0.270444646134
    s[:] = 0.0543050611635
    #n[1] = 0.15
    #s[1] = -0.01
    #s[2] = 0.021
    t[:] = 0.025099925748
    #t[:] = 0.55 * CF * n**(5 / 3)
    #print(t)
    #t[1] *= 2
    xc.calculate(e[0], n, v, s, d, t, w)
    print(e[0], v, d, w)
    from gpaw.xc.kernel import XCKernel
    from gpaw.xc.libxc import LibXC
    # xc = LibXC('GGA_C_PBE')
    #xc = LibXC('MGGA_X_M06_L+MGGA_C_M06_L')
    #xc = LibXC('MGGA_X_M06_L')
    #xc = LibXC('LDA_C_PW')
    xc = XCKernel('M06-L')
    v[:] = 0.0
    xc.calculate(e[0], n, v, s, d, t, w)
    print(e[0], v, d, w)
