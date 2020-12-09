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

    def __init__(self,
                 xc: XCFunctional,
                 *args, **kwargs):
        LCAOWaveFunctions.__init__(self, *args, **kwargs)

        vtphit: Dict[Setup, List[Spline]] = {}
        for setup in self.setups.setups.values():
            vt = setup.vt
            print(setup, vt)
            vtphit_j = []
            for phit in setup.phit_j:
                rc = phit.get_cutoff()
                r_g = np.linspace(0, rc, 150)
                vt_g = vt.map(r_g) / (4 * pi)**0.5
                phit_g = phit.map(r_g)
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
            print(Vt_MM)
            Vt_MM += Vt_MM.T.conj().copy()
            M1 = 0
            for m in manytci.Mindices.nm_a:
                M2 = M1 + m
                Vt_MM[M1:M2, M1:M2] *= 0.5
                M1 = M2
            print(Vt_MM)
            self.Vt_qMM.append(Vt_MM)
