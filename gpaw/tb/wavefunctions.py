from math import pi

import numpy as np

from gpaw.lcao.tci import TCIExpansions
from gpaw.spline import Spline
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class TBWaveFunctions(LCAOWaveFunctions):
    mode = 'tb'

    def __init__(self, *args, **kwargs):
        LCAOWaveFunctions.__init__(self, *args, **kwargs)

        vtphit = {}  # Dict[Setup, List[Spline]]
        for setup in self.setups.setups.values():
            vt = setup.vt
            vtphit_j = []
            import matplotlib.pyplot as plt
            for phit in setup.phit_j:
                rc = phit.get_cutoff()
                r_g = np.linspace(0, rc, 150)
                vt_g = vt.map(r_g) / (4 * pi)**0.5
                plt.plot(r_g, vt_g)
                plt.plot(r_g, phit.map(r_g))
                plt.show(); asdfg
                vtphit_j.append(Spline(phit.l, rc, vt_g * phit.map(r_g)))
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



def pseudo_potential(setup: Setup,
                     xc: XCFunctional) -> Array1D:
    phit_jg = np.array(setup.data.phi_jg)
    rgd = setup.rgd

    # Densities with frozen core:
    nt_g = np.einsum('jg, j, jg -> g',
                     phit_jg, setup.f_j, phit_jg) / (4 * pi)**0.5
    nt_g += setup.data.nct_g * (1 / (4 * pi)**0.5)

    # Potential:
    vt_g = np.zeros_like(nt_sg)
    xc.calculate_spherical(rgd, nt_g, vt_g)
    vr_sg = v_sg * rgd.r_g
    vr_sg -= setup.Z
    vr_sg += rgd.poisson(n_sg.sum(axis=0))
