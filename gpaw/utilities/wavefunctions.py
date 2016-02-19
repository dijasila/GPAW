import numpy as np
from ase.units import Hartree, Bohr

from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.utilities import h2gpts
from gpaw.wavefunctions.pw import PWDescriptor


class WaveFunctions:
    def __init__(self, calc, ecut=800):
        self.calc = calc
        self.pd0 = calc.wfs.pd
        gd0 = self.pd0.gd
        ecut /= Hartree
        h = np.pi / (4 * ecut)**0.5
        N_c = h2gpts(h, gd0.cell_cv, 1)
        N_c = np.array([get_efficient_fft_size(N) for N in N_c])
        self.gd = GridDescriptor(N_c, gd0.cell_cv)
        self.pd = PWDescriptor(ecut, self.gd)
        print(gd0.N_c, N_c, self.pd0.ecut, self.pd.ecut)
        self.dphi = None

    def calculate_corrections(self):
        splines = {}
        dphi_aj = []
        for setup in self.calc.wfs.setups:
            dphi_j = splines.get(setup)
            if dphi_j is None:
                rcut = max(setup.rcut_j) * 1.1
                gcut = setup.rgd.ceil(rcut)
                dphi_j = []
                for l, phi_g, phit_g in zip(setup.l_j,
                                            setup.data.phi_jg,
                                            setup.data.phit_jg):
                    dphi_g = (phi_g - phit_g)[:gcut]
                    dphi_j.append(setup.rgd.spline(dphi_g, rcut, l,
                                                   points=200))
            dphi_aj.append(dphi_j)
            
        dphi = LFC(self.gd, dphi_aj)
        dphi.set_positions(self.calc.atoms.get_scaled_positions())
        return dphi
        
    def get_wave_function(self, n, k=0, s=0, ae=True):
        wfs = self.calc.wfs
        psi_r = wfs.get_wave_function_array(n, k, s)
        psi_R, _ = self.pd0.interpolate(psi_r, self.pd)
        P_ai = dict((a, P_ni[n]) for a, P_ni in wfs.kpt_u[k].P_ani.items())
        if ae:
            if self.dphi is None:
                self.dphi = self.calculate_corrections()
            self.dphi.add(psi_R, P_ai)
        return psi_R

        
from gpaw import GPAW
c = GPAW('H2', txt=None)
wfs = WaveFunctions(c)
w0 = wfs.get_wave_function(0, ae=False)
w = wfs.get_wave_function(0)
print(wfs.gd.integrate(w0**2))
print(wfs.gd.integrate(w**2))
import matplotlib.pyplot as plt
plt.plot(w0[7,7])
plt.plot(w[7,7])
plt.show()

