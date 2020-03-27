from typing import Tuple, Union, Dict

import numpy as np

from gpaw.xc import XC
from .scf import apply1, apply2
from .coulomb import coulomb_inteaction
from .paw import calculate_paw_stuff
from .symmetry import Symmetry


class HybridXC:
    orbital_dependent = True
    type = 'HYB'

    def __init__(self,
                 kind: Union[str, Tuple[str, float, float]]):
        from . import parse_name
        if isinstance(kind, str):
            self.name = kind
            xcname, exx_fraction, omega = parse_name(kind)
        else:
            xcname, exx_fraction, omega = kind

        self.xc = XC(xcname)
        self.exx_fraction = exx_fraction
        self.omega = omega

        self.description = f'{xcname}+{exx_fraction}*EXX(omega={omega})'

        self.vlda_sR = None
        self.v_sknG: Dict[Tuple[int, int], np.ndarray] = {}

        self.ecc = np.nan
        self.evc = np.nan
        self.evv = np.nan
        self.ekin = np.nan

        self.sym = None
        self.coulomb = None

    def get_setup_name(self):
        return 'PBE'

    def initialize(self, dens, ham, wfs, occupations):
        self.dens = dens
        self.wfs = wfs
        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction
        assert wfs.world.size == wfs.gd.comm.size

    def get_description(self):
        return self.description

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        energy = self.ecc + self.evv + self.evc
        e_r = gd.empty()
        self.xc.calculate(gd, nt_sr, vt_sr, e_r)
        energy += gd.integrate(e_r)
        return energy

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        return self.ekin

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        wfs = self.wfs
        if self.coulomb is None:
            self.coulomb = coulomb_inteaction(self.omega, wfs.gd, wfs.kd)
            self.sym = Symmetry(wfs.kd)

        paw_s = calculate_paw_stuff(self.dens, wfs.setups)  # ???????

        if kpt.f_n is None:
            # Just use LDA_X for first step:
            if self.vlda_sR is None:
                # First time:
                self.vlda_sR = self.calculate_lda_potential()
            pd = kpt.psit.pd
            for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
                Htpsit_G += pd.fft(self.vlda_sR[kpt.s] *
                                   pd.ifft(psit_G, kpt.k), kpt.q)
        else:
            self.vlda_sR = None
            if kpt.psit.array.base is psit_xG.base:
                if (kpt.s, kpt.k) not in self.v_sknG:
                    assert len(self.v_sknG) == 0
                    evc, evv, ekin, v_knG = apply1(
                        kpt, Htpsit_xG,
                        wfs,
                        self.coulomb, self.sym,
                        paw_s[kpt.s])
                    if kpt.s == 0:
                        self.evc = 0.0
                        self.evv = 0.0
                        self.ekin = 0.0
                    scale = 2 / wfs.nspins * self.exx_fraction
                    self.evc += evc * scale
                    self.evv += evv * scale
                    self.v_sknG = {(kpt.s, k): v_nG
                                   for k, v_nG in enumerate(v_knG)}
                v_nG = self.v_sknG.pop((kpt.s, kpt.k))
            else:
                v_nG = apply2(kpt, psit_xG, Htpsit_xG, wfs,
                              self.coulomb, self.sym,
                              paw_s[kpt.s])
            Htpsit_xG += v_nG * self.exx_fraction

    def calculate_lda_potential(self):
        from gpaw.xc import XC
        lda = XC('LDA_X')
        nt_sr = self.dens.nt_sg
        vt_sr = np.zeros_like(nt_sr)
        vlda_sR = self.dens.gd.zeros(self.wfs.nspins)
        lda.calculate(self.dens.finegd, nt_sr, vt_sr)
        for vt_R, vt_r in zip(vlda_sR, vt_sr):
            vt_R[:], _ = self.dens.pd3.restrict(vt_r, self.dens.pd2)
        return vlda_sR * self.exx_fraction

    def summary(self, log):
        log(self.description)

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

