import numpy as np
from math import sqrt, pi

from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.sphere.lebedev import weight_n
from gpaw.xc import XC
from gpaw.xc.gllb.contribution import Contribution


class C_GLLBScr(Contribution):
    def __init__(self, weight, functional, damp=1e-10):
        Contribution.__init__(self, weight)
        self.xc = XC(functional)
        self.damp = damp

    def set_damp(self, damp):
        self.damp = damp

    def get_name(self):
        return 'SCREENING'

    def get_desc(self):
        desc = '({})'.format(self.xc.name)
        return desc

    def initialize(self, density, hamiltonian, wfs):
        Contribution.initialize(self, density, hamiltonian, wfs)
        # Always 1 spin, no matter what calculation nspins is
        self.vt_sg = self.finegd.empty(1)
        self.e_g = self.finegd.empty()

    def initialize_1d(self, ae):
        Contribution.initialize_1d(self, ae)
        self.v_g = np.zeros(self.ae.N)
        self.e_g = np.zeros(self.ae.N)

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        self.xc.calculate_spherical(self.ae.rgd, self.ae.n.reshape((1, -1)),
                                    self.v_g.reshape((1, -1)), self.e_g)
        v_g += 2 * self.weight * self.e_g / (self.ae.n + self.damp)
        Exc = self.weight * np.sum(self.e_g * self.ae.rgd.dv_g)
        return Exc

    def calculate_spinpaired(self, e_g, n_sg, v_sg):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.calculate(self.finegd, n_sg, self.vt_sg, self.e_g)
        self.e_g[:] = np.where(n_sg[0] < self.damp, 0, self.e_g)
        v_sg[0] += self.weight * 2 * self.e_g / (n_sg[0] + self.damp)
        e_g += self.weight * self.e_g

    def calculate_spinpolarized(self, e_g, n_sg, v_sg):
        # Calculate spinpolarized exchange screening
        # as two spin-paired calculations n=2*n_s
        for n, v in [(n_sg[0], v_sg[0]), (n_sg[1], v_sg[1])]:
            self.e_g[:] = 0.0
            self.vt_sg[:] = 0.0
            self.xc.calculate(self.finegd, 2 * n[None, ...],
                              self.vt_sg, self.e_g)
            self.e_g[:] = np.where(n < self.damp, 0, self.e_g)
            v += self.weight * 2 * self.e_g / (2 * n + self.damp)
            e_g += self.weight * self.e_g / 2

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        # Get the XC-correction instance
        c = setup.xc_correction
        nspins = self.nspins

        E = 0
        for D_p, dEdD_p in zip(D_sp, H_sp):
            D_Lq = np.dot(c.B_pqL.T, nspins * D_p)
            n_Lg = np.dot(D_Lq, c.n_qg)
            if addcoredensity:
                n_Lg[0] += c.nc_g * sqrt(4 * pi)
            nt_Lg = np.dot(D_Lq, c.nt_qg)
            if addcoredensity:
                nt_Lg[0] += c.nct_g * sqrt(4 * pi)
            dndr_Lg = np.zeros((c.Lmax, c.ng))
            dntdr_Lg = np.zeros((c.Lmax, c.ng))
            for L in range(c.Lmax):
                c.rgd.derivative(n_Lg[L], dndr_Lg[L])
                c.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            vt_g = np.zeros(c.ng)
            v_g = np.zeros(c.ng)
            e_g = np.zeros(c.ng)
            deda2_g = np.zeros(c.ng)
            for y, (w, Y_L) in enumerate(zip(weight_n, c.Y_nL)):
                # Cut gradient releated coefficient to match the setup's Lmax
                A_Li = rnablaY_nLv[y, :c.Lmax]

                # Expand pseudo density
                nt_g = np.dot(Y_L, nt_Lg)

                # Expand pseudo density gradient
                a1x_g = np.dot(A_Li[:, 0], nt_Lg)
                a1y_g = np.dot(A_Li[:, 1], nt_Lg)
                a1z_g = np.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= c.rgd.r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = np.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2

                vt_g[:] = 0.0
                e_g[:] = 0.0
                # Calculate pseudo GGA energy density (potential is discarded)
                self.xc.kernel.calculate(e_g, nt_g.reshape((1, -1)),
                                         vt_g.reshape((1, -1)),
                                         a2_g.reshape((1, -1)),
                                         deda2_g.reshape((1, -1)))

                # Calculate pseudo GLLB-potential from GGA-energy density
                vt_g[:] = 2 * e_g / (nt_g + self.damp)

                dEdD_p -= self.weight * w * np.dot(np.dot(c.B_pqL, Y_L),
                                                   np.dot(c.nt_qg,
                                                          vt_g * c.rgd.dv_g))

                E -= w * np.dot(e_g, c.rgd.dv_g) / nspins

                # Expand density
                n_g = np.dot(Y_L, n_Lg)

                # Expand density gradient
                a1x_g = np.dot(A_Li[:, 0], n_Lg)
                a1y_g = np.dot(A_Li[:, 1], n_Lg)
                a1z_g = np.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= c.rgd.r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = np.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2

                v_g[:] = 0.0
                e_g[:] = 0.0
                # Calculate GGA energy density (potential is discarded)
                self.xc.kernel.calculate(e_g, n_g.reshape((1, -1)),
                                         v_g.reshape((1, -1)),
                                         a2_g.reshape((1, -1)),
                                         deda2_g.reshape((1, -1)))

                # Calculate GLLB-potential from GGA-energy density
                v_g[:] = 2 * e_g / (n_g + self.damp)

                dEdD_p += self.weight * w * np.dot(np.dot(c.B_pqL, Y_L),
                                                   np.dot(c.n_qg,
                                                          v_g * c.rgd.dv_g))
                E += w * np.dot(e_g, c.rgd.dv_g) / nspins

        return E * self.weight

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        self.xc.calculate_spherical(self.ae.rgd, self.ae.nt.reshape((1, -1)),
                                    self.v_g.reshape((1, -1)), self.e_g)
        vt_g += 2 * self.weight * self.e_g / (self.ae.nt + self.damp)
        return self.weight * np.sum(self.e_g * self.ae.rgd.dv_g)
