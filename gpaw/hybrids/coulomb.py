from math import pi

import numpy as np

from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb as WSTC
from gpaw.kpt_descriptor import to1bz
from ase.dft import monkhorst_pack

def coulomb_interaction(omega, gd, kd):
    if omega:
        return ShortRangeCoulomb(omega)
    return PointVolumeIntegration(gd.cell_cv, kd.N_c)
    return WSTC(gd.cell_cv, kd.N_c)


class PointVolumeIntegration:
    def __init__(self, cell_cv, N_c):
        self.cell_cv = cell_cv
        self.N_c = N_c
        self.wstc = WSTC(cell_cv, N_c)

    def get_potential(self, pd):
        G2_G = pd.G2_qG[0]
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi / G2_G
        G0 = G2_G.argmin()
        if G2_G[G0] < 1e-11:
            N = 20
            print(self.N_c)
            qf_qc = monkhorst_pack(np.array([20,20,20]))
            qf_qc = to1bz(qf_qc, pd.gd.cell_cv) / self.N_c
            B_cv = 2 * np.pi * pd.gd.icell_cv
            v_G[G0] = 4 * pi * np.sum(1.0 / np.sum(np.dot(qf_qc, B_cv)**2, axis=1)) / 20**3  # * np.abs(np.linalg.det(B_cv)) / 20**3 
            print(v_G[G0], 'V0')
        print(v_G)
        print('Comparison me / wstc', v_G / self.wstc.get_potential(pd))
        return v_G


class ShortRangeCoulomb:
    def __init__(self, omega):
        self.omega = omega

    def get_description(self):
        return f'Short-range Coulomb: erfc(omega*r)/r (omega = {self.omega} ' \
               'bohr^-1)'
    
    def get_potential(self, pd):
        G2_G = pd.G2_qG[0]
        x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi * x_G / G2_G
        G0 = G2_G.argmin()
        if G2_G[G0] < 1e-11:
            v_G[G0] = pi / self.omega**2
        return v_G
