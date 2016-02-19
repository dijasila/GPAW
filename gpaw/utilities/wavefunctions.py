import numpy as np
from ase.units import Hartree, Bohr

from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities import h2gpts
from gpaw.wavefunctions.pw import PWLFC, PWDescriptor


class WaveFunctions:
    def __init__(self, calc, ecut=500, ae=True):
        self.calc = calc
        self.pd1 = calc.wfs.pd
        gd1 = self.pd1.gd
        ecut /= Hartree
        h = np.pi / (2 * ecut)**0.5
        N_c = h2gpts(h, gd1.cell_cv, 1)
        N_c = np.array([get_efficient_fft_size(N) for N in N_c])
        gd2 = GridDescriptor(N_c, gd1.cell_cv)
        self.pd2 = PWDescriptor(ecut, gd2)
        if ae:
            self.initialize_corrections()
        else:
            self.dphi = None

    def initialize_corrections(self):
        done = set()
        
        for setup in self.calc.wfs.setups:
            rcut = max(self.rcut_j) * 1.1
            gcut = self.rgd.ceil(rcut)
            dphi_j = []
            for l, phi_g, phit_g in zip(setup.l_j,
                                        setup.data.phi_jg, setup.data.phit_jg):
                dphi_g = (phi_g - phit_g)[:gcut]
                dphi_j.append(setup.rgd.spline(dphi_g, rcut, l, points=200))
            dphi_aj.append(dphi_j)
        self.dphi = LFC(dphi_ai, self.pd2)
        
    def get_wave_function(self, n, k=0, s=0):
        wfs = self.calc.wfs
        psi_r = wfs.get_wave_function_array(n, k, s)
        psi_R, _ = self.pd1.interpolate(psi_r, self.pd2)
        P_ai = wfs.kpt_u[k].P_ani
        print(P_ai)
        self.dphi.add(psi_R, P_ai)
        return psi_R * Bohr
