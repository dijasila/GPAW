from __future__ import print_function
from math import pi
import numpy as np
from gpaw.atom.aeatom import GaussianBasis
from gpaw.atom.radialgd import AERadialGridDescriptor, RadialGridDescriptor
from gpaw.utilities import erf
from ase.utils import seterr


class Hydrogen:
    def __init__(self):
        rcut = 20
        alpha2 = 1550.0
        ngpts = 2000
        rcut = 50.0
        alpha1 = 0.01
        ngauss = 250
        eps = 1.0e-9
        # Use grid with r(0)=0, r(1)=a and r(ngpts)=rcut:
        a = 1 / alpha2**0.5 / 20
        b = (rcut - a * ngpts) / (rcut * ngpts)
        b = 1 / round(1 / b)
        self.gd = AERadialGridDescriptor(a, b, ngpts)
        print(alpha1, alpha2, ngauss)
        alpha_B = alpha1 * (alpha2 / alpha1)**np.linspace(0, 1, ngauss)
        self.B = GaussianBasis(0, alpha_B, self.gd, eps)
        N = len(self.B)
        print(N)
        self.e_n, self.C_bn = np.linalg.eigh(self.B.T_bb - self.B.K_bb)
        self.C_bn /= (4 * pi)**0.5
        self.p_n = self.B.expand(self.C_bn.T)
        
    def solve(self, M=5, omega=0.1):
        gd = self.gd
        r = gd.r_g
        w_i = gd.zeros(M)
        with seterr(invalid='ignore'):
            w_i[0] = erf(2 * r) / r
        w_i[0, 0] = 4 / pi**0.5
        for i in range(1, M):
            w_i[i] = w_i[0] * np.exp(-i * r)
            
        for iter in range(50):
            S = np.zeros((M, M))
            m_i = gd.zeros(M)
            for i1, w in enumerate(w_i):
                m_i[i1] = -1 / (4 * pi) * gd.laplace(w)
            for i1, w in enumerate(w_i):
                for i2, m in enumerate(m_i):
                    S[i1, i2] = gd.integrate(m * w)
            L = np.linalg.inv(np.linalg.cholesky(S))
            m_i = np.dot(L, m_i)
            w_i = np.dot(L, w_i)
            dn_i = gd.zeros(M)
            A = np.zeros((M, M))
            for i1, w in enumerate(w_i):
                W_bb = 4 * pi * self.B.calculate_potential_matrix(r * w)
                W_nn = np.dot(np.dot(self.C_bn.T, W_bb), self.C_bn)
                dn = 0.0
                for n in range(1, len(W_nn)):
                    de = self.e_n[0] - self.e_n[n]
                    dn += (4 * de * self.p_n[0] * self.p_n[n] * W_nn[0, n] /
                           (de**2 + omega**2))
                dn_i[i1] = dn
                for i2, w in enumerate(w_i):
                    A[i1, i2] = gd.integrate(dn * w)
            a_i, U = np.linalg.eigh(A)
            dn_i = np.dot(U.T, dn_i)
            w_i = np.dot(U.T, w_i)
            m_i = np.dot(U.T, m_i)
            dw_i = dn_i - a_i[:, None] * m_i
            for w, dw in zip(w_i, dw_i):
                G, f = gd.fft(r * dw)
                ggd = RadialGridDescriptor(G, G * 0 + G[1])
                R, pdw = ggd.fft(f * G / (0.02 + G**2 / 25))
                from scipy.interpolate import InterpolatedUnivariateSpline
                pdw = InterpolatedUnivariateSpline(R, pdw / (2 * pi)**3)(r)
                w -= 1 * pdw
            print('.', end='', flush=True)
        print([gd.integrate(dw**2) for dw in dw_i])
        return a_i, dn_i
        

h = Hydrogen()
a_i, dn_i = h.solve()
print()
print(a_i)
print((np.log(1 - a_i) + a_i) / 2 / pi * 0.5 * 27)

n = h.p_n[0]**2
for dn in dn_i:
    h.gd.plot(n[0] / dn[0] * dn)
h.gd.plot(n, show=1)
