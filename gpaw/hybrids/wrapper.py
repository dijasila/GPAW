from typing import Tuple

import numpy as np

from gpaw.xc import XC
from .hybrid import EXX


def parse_name(name: str) -> Tuple[str, float, float]:
    if name == 'EXX':
        return 'null', 1.0, 0.0
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, 0.0
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, 0.0


class HybridXC:
    orbital_dependent = True
    type = 'HYB'

    def __init__(self,
                 name: str = None,
                 xc=None,
                 exx_fraction: float = None,
                 omega: float = None):
        if name is not None:
            assert xc is None and exx_fraction is None and omega is None
            xc, exx_fraction, omega = parse_name(name)
            self.name = name
        else:
            assert xc is not None and exx_fraction is not None
            self.name = '???'

        if xc:
            xc = XC(xc)

        self.xc = xc
        self.exx_fraction = exx_fraction
        self.omega = omega

        self.exx = None

        self.description = ''

        self.vlda_sR = None

    def get_setup_name(self):
        return 'PBE'

    def initialize(self, dens, ham, wfs, occupations):
        self.exx = EXX(wfs.gd, wfs.kd, wfs.nspins, wfs.pt, wfs.setups,
                       self.omega, self.exx_fraction, self.xc)
        assert wfs.world.size == wfs.gd.comm.size

    def get_description(self):
        return self.exx.description

    def set_positions(self, spos_ac):
        self.exx.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        energy = self.exx.calculate_semi_local_potential_and_energy()
        return energy

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        1 / 0
        if not self.xc:
            return 0.0
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        return self.exx.ekin

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        if kpt.f_n is None:
            # Just use LDA for first step:
            if self.vlda_sR is None:
                # First time:
                self.vlda_sR = self.calculate_lda_potential()
            pd = kpt.psit.pd
            for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
                Htpsit_G += pd.fft(self.vt_sR[kpt.s] *
                                   pd.ifft(psit_G, kpt.k), kpt.q)
        else:
            self.vlda_sR = None
            if kpt.psit.array.base is psit_xG.base:
                self.exx.apply1(kpt, psit_xG, Htpsit_xG)
            else:
                self.exx.apply2(kpt, psit_xG, Htpsit_xG)

    def calculate_lda_potential(self):
        from gpaw.xc import XC
        lda = XC('LDA')
        nt_sr = self.dens.nt_sg
        vt_sr = np.zeros_like(nt_sr)
        vlda_sR = self.dens.gd.zeros(self.wfs.nspins)
        lda.calculate(self.dens.finegd, nt_sr, vt_sr)
        for vt_R, vt_r in zip(vlda_sR, vt_sr):
            vt_R[:], _ = self.dens.pd3.restrict(vt_r, self.dens.pd2)
        return vlda_sR

    def summary(self, log):
        log(self.exx.description)

    def add_forces(self, F_av):
        pass

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        return

    def rotate(self, kpt, U_nn):
        pass  # 1 / 0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        pass  # 1 / 0

    def read(self, reader):
        pass

    def write(self, writer):
        pass

    def set_grid_descriptor(self, gd):
        pass

