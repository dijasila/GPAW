# encoding: utf-8
import numpy as np
from ase.neighborlist import PrimitiveNeighborList

from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               TwoCenterIntegralCalculator,
                               ManySiteOverlapCalculator,
                               AtomicDisplacement)

def get_cutoffs(f_Ij):
    rcutmax_I = []
    for f_j in f_Ij:
        rcutmax = 0.001  # 'paranoid zero'
        for f in f_j:
            rcutmax = max(rcutmax, f.get_cutoff())
        rcutmax_I.append(rcutmax)
    return rcutmax_I


def get_lvalues(f_Ij):
    return [[f.get_angular_momentum_number() for f in f_j] for f_j in f_Ij]


class AtomPairRegistry:
    def __init__(self, cutoff_a, pbc_c, cell_cv, spos_ac):
        nl = PrimitiveNeighborList(cutoff_a, skin=0, sorted=True,
                                   self_interaction=True,
                                   use_scaled_positions=True)

        nl.update(pbc=pbc_c, cell=cell_cv, coordinates=spos_ac)
        r_and_offset_aao = {}

        def add(a1, a2, R_c, offset):
            r_and_offset_aao.setdefault((a1, a2), []).append((R_c, offset))

        for a1, spos1_c in enumerate(spos_ac):
            a2_a, offsets = nl.get_neighbors(a1)
            for a2, offset in zip(a2_a, offsets):
                spos2_c = spos_ac[a2] + offset

                R_c = np.dot(spos2_c - spos1_c, cell_cv)
                add(a1, a2, R_c, offset)
                if a1 != a2 or offset.any():
                    add(a2, a1, -R_c, -offset)
        self.r_and_offset_aao = r_and_offset_aao

    def get(self, a1, a2):
        R_c_and_offset_a = self.r_and_offset_aao.get((a1, a2))
        return R_c_and_offset_a


class Overlap:
    def __init__(self):
        self.O_qmm = None
        self.T_qmm = None
        self.P_qim = None

class TCI:
    def __init__(self, phit_Ij, pt_Ij, I_a, cell_cv, spos_ac, pbc_c, ibzk_qc,
                 dtype):
        assert len(pt_Ij) == len(phit_Ij)
        self.dtype = dtype

        # Cutoffs by species:
        pt_rcmax_I = get_cutoffs(pt_Ij)
        phit_rcmax_I = get_cutoffs(phit_Ij)
        rcmax_I = [max(rc1, rc2) for rc1, rc2
                   in zip(pt_rcmax_I, phit_rcmax_I)]
        # XXX It is somewhat nasty that rcmax depends on how long our
        # longest orbital happens to be
        transformer = FourierTransformer(rcmax=max(rcmax_I + [1e-3]), ng=2**10)
        tsoc = TwoSiteOverlapCalculator(transformer)
        msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)

        # Cutoffs by atom:
        cutoff_a = [rcmax_I[I] for I in I_a]
        self.pt_rcmax_a = np.array([pt_rcmax_I[I] for I in I_a])
        self.phit_rcmax_a = np.array([phit_rcmax_I[I] for I in I_a])

        self.a1a2 = AtomPairRegistry(cutoff_a, pbc_c, cell_cv, spos_ac)

        self.overlapcalc = TwoCenterIntegralCalculator(ibzk_qc,
                                                       derivative=False)
        phit_Ijq = msoc.transform(phit_Ij)
        pt_Ijq = msoc.transform(pt_Ij)

        pt_l_Ij = get_lvalues(pt_Ij)
        phit_l_Ij = get_lvalues(phit_Ij)

        # Avoid two-way for O and T?
        # TODO: lazy msoc
        self.O_expansions = msoc.calculate_expansions(phit_l_Ij, phit_Ijq,
                                                      phit_l_Ij, phit_Ijq)
        self.T_expansions = msoc.calculate_kinetic_expansions(phit_l_Ij,
                                                              phit_Ijq)
        self.P_expansions = msoc.calculate_expansions(pt_l_Ij, pt_Ijq,
                                                      phit_l_Ij, phit_Ijq)


    def calculate(self, a1, a2, OT=False, P=False):
        """Calculate overlap of functions between atoms a1 and a2."""

        #print('aa', a1, a2)
        o = Overlap()
        R_c_and_offset_a = self.a1a2.get(a1, a2)

        if R_c_and_offset_a is None:
            return o

        dtype = self.dtype
        get_phases = self.overlapcalc.phaseclass
        ibzk_qc = self.overlapcalc.ibzk_qc
        nq = len(ibzk_qc)
        phit_rcmax_a = self.phit_rcmax_a
        pt_rcmax_a = self.pt_rcmax_a

        if P:
            P_expansion = self.P_expansions.get(a1, a2)
            o.P_qim = P_qim = P_expansion.zeros((nq,), dtype=dtype)

        if OT:
            O_expansion = self.O_expansions.get(a1, a2)
            T_expansion = self.T_expansions.get(a1, a2)
            o.O_qmm = O_qmm = O_expansion.zeros((nq,), dtype=dtype)
            o.T_qmm = T_qmm = T_expansion.zeros((nq,), dtype=dtype)

        for R_c, offset in R_c_and_offset_a:
            norm = np.linalg.norm(R_c)
            #print(norm)
            phases = get_phases(ibzk_qc, offset)
            disp = AtomicDisplacement(None, a1, a2, R_c, offset, phases)

            if P and norm < pt_rcmax_a[a1] + phit_rcmax_a[a2]:
                disp.evaluate_overlap(P_expansion, P_qim)

            if OT and norm < phit_rcmax_a[a1] + phit_rcmax_a[a2]:
                disp.evaluate_overlap(O_expansion, O_qmm)
                disp.evaluate_overlap(T_expansion, T_qmm)

        return o
