from math import pi
from typing import List, Dict

import numpy as np

from gpaw.lcao.tci import TCIExpansions
from gpaw.spline import Spline
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.setup import Setup
from gpaw.xc.functional import XCFunctional


class TBWaveFunctions(LCAOWaveFunctions):
    mode = 'tb'

    def __init__(self, xc, *args, **kwargs):
        LCAOWaveFunctions.__init__(self, *args, **kwargs)

        vtphit: Dict[Setup, List[Spline]] = {}
        for setup in self.setups.setups.values():
            vt = calculate_potential(setup, xc)
            vtphit_j = []
            for phit in setup.phit_j:
                rc = phit.get_cutoff()
                r_g = np.linspace(0, rc, 150)
                vt_g = vt.map(r_g) / (4 * pi)**0.5
                phir_g = phit.map(r_g)
                eig = ...
                vtphit_j.append(Spline(phit.l, rc, vt_g * phit_g))
            vtphit[setup] = vtphit_j

        self.vtciexpansions = TCIExpansions([s.phit_j for s in self.setups],
                                            [vtphit[s] for s in self.setups],
                                            self.tciexpansions.I_a)

    def set_positions(self, spos_ac, *args, **kwargs):
        LCAOWaveFunctions.set_positions(self, spos_ac, *args, **kwargs)
        manytci = self.vtciexpansions.get_manytci_calculator(
            self.setups, self.gd, spos_ac, self.kd.ibzk_qc, self.dtype,
            self.timer)
        manytci.Pindices = manytci.Mindices
        my_atom_indices = self.basis_functions.my_atom_indices
        self.Vt_qMM = []
        for Vt_MM in manytci.P_qIM(my_atom_indices):
            Vt_MM = Vt_MM.toarray()
            # print(spos_ac[1, 2], Vt_MM[0, 1])
            Vt_MM += Vt_MM.T.conj().copy()
            M1 = 0
            for m in manytci.Mindices.nm_a:
                M2 = M1 + m
                Vt_MM[M1:M2, M1:M2] *= 0.5
                M1 = M2
            self.Vt_qMM.append(Vt_MM)


def calculate_potential(setup: Setup,
                        xc: XCFunctional) -> Spline:
    phit_jg = np.array(setup.data.phit_jg)
    rgd = setup.rgd

    # Densities with frozen core:
    nt_g = np.einsum('jg, j, jg -> g',
                     phit_jg, setup.f_j, phit_jg) / (4 * pi)
    nt_g += setup.data.nct_g * (1 / (4 * pi)**0.5)

    # XC potential:
    nt_sg = nt_g[np.newaxis]
    vt_sg = np.zeros_like(nt_sg)
    xc.calculate_spherical(rgd, nt_sg, vt_sg)
    vt_sg[0] += setup.vbar.map(rgd.r_g) / (4 * pi)**0.5
    vtr_g = vt_sg[0] * rgd.r_g
    g_g = setup.ghat_l[0].map(rgd.r_g)
    Q = -rgd.integrate(nt_g) / rgd.integrate(g_g)
    nt_g += Q * g_g
    vtr_g += rgd.poisson(nt_g)
    vtr_g[1:] /= rgd.r_g[1:]
    vtr_g[0] = vtr_g[1]
    return rgd.spline(vtr_g * (4 * pi)**0.5, points=300)
