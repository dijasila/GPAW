import weakref

import numpy as np

from gpaw.xc.gga import GGA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.lfc import LFC


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
        self.grad_v = [Gradient(wfs.gd, v, allocate=not False).apply
                       for v in range(3)]
        print 'TODO: Make transformers use malloc/free.'

    def set_positions(self, spos_ac):
        self.tauct.set_positions(spos_ac)
    
    def calculate_gga(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        taut_sG = self.wfs.calculate_kinetic_energy_density(self.tauct,
                                                            self.grad_v)
        taut_sg = np.empty_like(n_sg)
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            self.tauct.add(tauc_G, 1.0 / self.nspins)
            self.interpolate(taut_G, taut_g)
        dedtaut_sg = np.empty_like(nt_sg)
        self.kernel.calculate(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                                taut_sg, dedtaut_sg)
        self.dedtau_sG = self.wfs.gd.empty(self.wfs.nspins)
        for s in range(self.wfs.nspins):
            self.restrict(dedtau_sg[s], self.dedtau_sG[s])

    def add_non_local_terms(self, psit_nG, Htpsit_nG, kpt):
        a_G = self.wfs.gd.empty()
        for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
            for v in range(3):
                self.grad[v](psit_G, a_G, kpt.phase_cd)
                self.grad[v](self.dedtau_sG[kpt.s] * a_G, a_G, kpt.phase_cd)
                axpy(-0.5, a_G, Htpsit_G)

    def forces(self, F_av):
        dF_av = hamiltonian.xc.tauct.dict(derivative=True)
        dedtau_G = hamiltonian.xc.dedtau_G
        hamiltonian.xc.tauct.derivative(dedtau_G, dF_av)
        for a, dF_v in dF_av.items():
            self.F_av[a] += dF_v[0]
