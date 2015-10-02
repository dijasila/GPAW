from __future__ import print_function
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import bicgstab, LinearOperator
from ase.utils import seterr
from ase.units import Hartree, Bohr
from gpaw.wavefunctions.pw import PWDescriptor


class Eigenmodes:
    def __init__(self):
        pass
        
    def hydrogen(self, a, n, ecut):
        from gpaw.grid_descriptor import GridDescriptor
        gd = GridDescriptor([n, n, n], [a, a, a])
        r_vR = gd.get_grid_point_coordinates()
        with seterr(divide='ignore'):
            self.vt_R = -((r_vR - a / 2)**2).sum(0)**-0.5
        n2 = n // 2
        self.vt_R[n2, n2, n2] = (2 * self.vt_R[n2, n2, n2 + 1] -
                                 self.vt_R[n2, n2, n2 + 2])
        print(self.vt_R[n2, n2])
        self.pd = PWDescriptor(ecut, gd, complex)
        self.psit_nG = self.pd.zeros(1)
        self.psit_nG[0] = self.pd.fft(np.random.rand(*gd.N_c))
        psit_G = self.psit_nG[0]
        for i in range(73):
            S_nn = self.pd.integrate(psit_G, psit_G)
            psit_G *= S_nn**-0.5
            Htpsit_G = self.apply_h(psit_G)
            eps = self.pd.integrate(psit_G, Htpsit_G)
            R_G = Htpsit_G - eps * psit_G
            psit_G -= 1 * R_G / (1 + self.pd.G2_qG[0])
        print(eps, self.pd.integrate(R_G, R_G))
        self.psit_R = self.pd.ifft(psit_G)
        self.eps = eps.real
        
    def apply_h(self, psit_G, eps=0.0):
        return (0.5 * self.pd.G2_qG[0] * psit_G +
                self.pd.fft((self.vt_R - eps) * self.pd.ifft(psit_G)))
        
    def sternheimer(self, w_R, alpha=1.0):
        b_G = self.pd.fft(self.psit_R * w_R)
        psit_G = self.psit_nG[0]
        b_G -= self.pd.integrate(psit_G, b_G) * psit_G
        
        def A(x_G):
            p = self.pd.integrate(psit_G, x_G)
            return self.apply_h(x_G, self.eps) + alpha * p * psit_G
            
        def M(x_G):
            return x_G / (1 + self.pd.G2_qG[0])
            
        N = len(psit_G)
        A_GG = LinearOperator((N, N), matvec=A, dtype=complex)
        M_GG = LinearOperator((N, N), matvec=M, dtype=complex)
        dpsit_G, info = bicgstab(A_GG, -b_G, M=M_GG, maxiter=100, tol=1e-6)
        dpsit_G -= psit_G * self.pd.integrate(psit_G, dpsit_G)
        return dpsit_G
        
    def solve(self, M):
        rc = self.pd.gd.cell_cv[0, 0] / 2
        print('rcut:', rc)
        ecut2 = 0.5 * pi**2 / (self.pd.gd.h_cv**2).sum(1).max() * 0.9999
        pd2 = PWDescriptor(ecut2, self.pd.gd, dtype=complex)
        from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC
        wstc = WSTC(self.pd.gd.cell_cv, np.ones(3, int))
        iv0_g = 1 / wstc.get_potential(pd2)

        w_g = pd2.fft(self.psit_R**2)

        for i in range(33):
            w_g /= pd2.integrate(w_g * iv0_g, w_g)**0.5
            w_R = pd2.ifft(w_g)
            dpsit_G = self.sternheimer(w_R)
            dn_R = 4 * (self.psit_R.conj() * self.pd.ifft(dpsit_G))
            assert abs(dn_R.imag).max() < 1e-13
            dn_g = pd2.fft(dn_R.real)
            a = pd2.integrate(w_g, dn_g)
            dn_g -= a * iv0_g * w_g
            print('{:2} {:.6f} {:.9f}'.format(i, a, pd2.integrate(dn_g, dn_g)))
            w_g -= 1.3 * dn_g / (1 + iv0_g)
        
        print(abs(w_R.imag).max())
        print(w_R.shape)
        
        n = len(w_R) // 2
        plt.plot(w_R[:, n, n])
        plt.plot(w_R[n, :, n])
        plt.plot(w_R[0, 0, :])
        plt.show()
        plt.plot(dn_R[:, n, n])
        plt.plot(dn_R[n, :, n])
        plt.plot(dn_R[0, 0, :])
        plt.show()


m = Eigenmodes()
m.hydrogen(10.5, 48, 40)
m.solve(1)
