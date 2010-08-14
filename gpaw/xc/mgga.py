import weakref

import numpy as np

from gpaw.xc.gga import GGA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient


class MGGA(GGA):
    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        
    def initialize(self, density, hamiltonian, wfs):
        self.wfs = wfs
        self.tauct = LFC(wfs.gd,
                         [[setup.tauct] for setup in wfs.setups],
                         forces=True, cut=True)
        self.restrict = hamiltonian.restrictor.apply
        self.interpolate = density.interpolator.apply
        self.tgrad_v = [Gradient(wfs.gd, v, allocate=not False).apply
                       for v in range(3)]
        print 'TODO: Make transformers use malloc/free.'

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        tau_sg = self.wfs.calculate_kinetic_energy_density(self.tauct)
        dedtau_sg = np.empty_like(tau_sg)
        self.xckernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                                tau_sg, dedtau_sg)
        self.dedtau_sG = self.wfs.gd.empty(self.wfs.nspins)
        for s in range(self.wfs.nspins):
            self.restrict(dedtau_sg[s], self.dedtau_sG[s])

    def add_non_local_terms(self, psit_nG, Htpsit_nG, kpt):
        a_G = self.wfs.gd.empty()
        for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
            for v in range(3):
                self.tgrad[v](psit_G, a_G, kpt.phase_cd)
                self.tgrad[v](self.dedtau_sG[kpt.s] * a_G, a_G, kpt.phase_cd)
                axpy(-0.5, a_G, Htpsit_G)
