from __future__ import print_function
from math import pi
import numpy as np
import scipy.linalg as sl
from scipy.sparse.linalg import bicgstab, LinearOperator
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
        B = self.B = GaussianBasis(0, alpha_B, self.gd, eps)
        self.B_gb = B.basis_bg.T / (4 * pi)**0.5
        N = len(self.B)
        print(N)
        self.H = self.B.T_bb - self.B.K_bb
        self.e_n, self.C_bn = np.linalg.eigh(self.H)
        print(self.e_n[:4])
        self.phi = np.dot(self.B_gb, self.C_bn[:, 0])
        self.alpha = 0.5
        
    def pseudize(self, rc=0.9, rcv=0.6):
        gd = self.gd
        self.phit = gd.pseudize(self.phi, gd.round(rc))[0]
        self.ds = 1 - gd.integrate(self.phit**2)
        with seterr(invalid='ignore'):
            self.vt = -erf(gd.r_g / rcv) / gd.r_g
        self.vt[0] = -2 / pi**0.5 / rcv
        self.pt = pt = (-0.5 * gd.laplace(self.phit) +
                        (self.vt - self.e_n[0]) * self.phit)
        self.pt /= gd.integrate(pt * self.phit)
        self.pt_b = gd.integrate(self.pt * self.B_gb.T)
        self.phi_b = gd.integrate(self.phi * self.B_gb.T)
        self.phit_b = gd.integrate(self.phit * self.B_gb.T)
        self.S = (np.eye(len(self.pt_b)) +
                  self.ds * np.outer(self.pt_b, self.pt_b))

        T = self.B.T_bb
        V = -self.B.K_bb
        Vt = self.B.calculate_potential_matrix(gd.r_g * self.vt)
        self.dh = (np.dot(np.dot(self.phi_b, T + V), self.phi_b) -
                   np.dot(np.dot(self.phit_b, T + Vt), self.phit_b))
        self.Ht = T + Vt + self.dh * np.outer(self.pt_b, self.pt_b)
        self.et_n, self.Ct_bn = sl.eigh(self.Ht, self.S)
        print(self.et_n[:4])
        
    def solve(self):
        B = self.B
        gd = self.gd
        N = len(B)
        
        V = np.empty((N, N))
        for b1 in range(N):
            for b2 in range(N):
                V[b1, b2] = gd.integrate(
                    gd.poisson(self.B_gb[:, b1] * self.phi) *
                    self.B_gb[:, b2] * self.phi, -1)
                
        I = np.eye(N)
        O = np.outer(self.C_bn[:, 0], self.C_bn[:, 0])
        L = self.H - self.e_n[0] * I + self.alpha * O
        self.a_i, self.dC_bi = sl.eig(-4 * np.dot(I - O, V), L)
        self.dpsi = np.dot(self.B_gb, self.dC_bi[:, 0])

    def solvepaw(self):
        B = self.B
        gd = self.gd
        N = len(B)
        
        g = np.exp(-2 * gd.r_g**2)
        g /= gd.integrate(g)
        vgr = gd.poisson(g)
        # c0 = gd.integrate(vgr * g, -1)
        c1 = gd.integrate(vgr * self.phit**2, -1)
        vr = gd.poisson(self.phi**2)
        c2 = gd.integrate(vr * self.phi**2, -1)
        vtr = gd.poisson(self.phit**2)
        c2 -= gd.integrate(vtr * self.phit**2, -1)
        K2 = c2 - 2 * self.ds * c1
        
        V = np.empty((N, N))
        for b1 in range(N):
            for b2 in range(N):
                V[b1, b2] = gd.integrate(
                    gd.poisson(self.B_gb[:, b1] * self.phit) *
                    self.B_gb[:, b2] * self.phit, -1)
                
        G = gd.integrate(self.B_gb.T * self.phit * vgr, -1)
        BB = (V +
              self.ds * np.outer(G, self.pt_b) +
              self.ds * np.outer(self.pt_b, G) +
              K2 * np.outer(self.pt_b, self.pt_b))

        I = np.eye(N)
        O = np.outer(self.Ct_bn[:, 0], self.Ct_bn[:, 0])
        L = (self.Ht - self.e_n[0] * self.S +
             self.alpha * np.dot(np.dot(self.S, O), self.S))
        self.at_i, self.dCt_bi = sl.eig(-4 * np.dot(I - np.dot(self.S, O), BB),
                                        L)
        x = np.dot(self.dCt_bi[:, 0], self.pt_b)
        self.dpsit = np.dot(self.B_gb, self.dCt_bi[:, 0])
        return x
        
h = Hydrogen()
h.solve()
h.pseudize()
x = h.solvepaw()
print(h.a_i[:4])
print(h.at_i[:4])
h.gd.plot(--h.dpsi)
h.gd.plot(h.dpsit)
h.gd.plot(h.dpsit + x * (h.phi - h.phit))
y = h.dpsi[0] / h.phi[0]
h.gd.plot(--(h.dpsi - y * (h.phi - h.phit)), show=1)
sdfg
gd.plot(dpsit * phit)
gd.plot(dpsit * phit +
        x * (phi**2 - phit**2), show=1)
asdhasdf
#h.solve1();asdgf
a_i, dn_i = h.solve()
print()
print(a_i)
print((np.log(1 - a_i) + a_i) / 2 / pi * 0.5 * 27)

n = h.psi_n[0]**2
for dn in dn_i:
    h.gd.plot(n[0] / dn[0] * dn)
h.gd.plot(n, show=1)
